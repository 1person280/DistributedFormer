"""运行时安全监控骨架单元测试 (RFC #1 / v0.9.0 Safety Shield)

测试对象: distributedformer/security_monitor 包的四个组件
    - RuleBasedSafetyCheck  规则版安全判定器 (P0 最小实现)
    - IntentProbe           意图探针 (P0)
    - CriticalActionCircuitBreaker  高危熔断器 (P0)
    - SecurityMonitor       组合监控面 (执行层前唯一门禁)

覆盖: 三态判定 / 规则注入 / 历史容量 / fail-closed / 组合门禁与统计一致性
"""

import time

from distributedformer.security_monitor import (
    AuditedAction, CriticalActionCircuitBreaker, IntentProbe,
    RuleBasedSafetyCheck, SafetyCheck, SafetyVerdict, SecurityMonitor,
    Verdict,
)

# ═══════════════════════════════════════════════════════════════
# RuleBasedSafetyCheck — 规则版判定器
# ═══════════════════════════════════════════════════════════════


def test_rule_based_check_verdicts():
    """三态判定的基本路径: 正常业务放行 / 高危操作熔断 / 逃逸倾向拦截"""
    checker = RuleBasedSafetyCheck()
    assert checker.check({"tool": "report_generate"}).verdict is Verdict.ALLOW
    assert checker.check({"tool": "db_write"}).verdict is Verdict.REVIEW
    assert checker.check({"tool": "port_scan"}).verdict is Verdict.DENY


def test_rule_based_check_topic_fallback():
    """无 tool 字段时回退到 topic 判定 (兼容事件总线上的纯主题事件)"""
    checker = RuleBasedSafetyCheck()
    assert checker.check({"topic": "curl"}).verdict is Verdict.REVIEW
    assert checker.check({"topic": "escalate"}).verdict is Verdict.DENY


def test_rule_based_check_empty_and_unknown():
    """空动作 / 未知工具 / 空字符串均为放行 (默认信任业务动作)"""
    checker = RuleBasedSafetyCheck()
    for action in ({}, {"tool": ""}, {"tool": "unknown_tool"}, {"args": "x"}):
        v = checker.check(action)
        assert v.verdict is Verdict.ALLOW
        # 放行时不携带任何风险标签与理由
        assert v.risk_tags == [] and v.reasons == []


def test_rule_based_check_risk_tags():
    """拦截判定必须携带风险标签与理由 (审计留痕的最低要求)"""
    v = RuleBasedSafetyCheck().check({"tool": "sudo"})
    assert v.verdict is Verdict.DENY
    assert v.risk_tags == ["privilege-escalation"]
    assert v.reasons  # 拦截必须给出理由


def test_rule_based_check_custom_rules():
    """支持注入自定义规则集 (逃逸优先于高危)"""
    checker = RuleBasedSafetyCheck(
        high_risk={"deploy-prod": ("deploy",)},
        escape={"secret-exfil": ("dump_secrets",)},
    )
    assert checker.check({"tool": "deploy"}) .verdict is Verdict.REVIEW
    assert checker.check({"tool": "dump_secrets"}).verdict is Verdict.DENY
    # 内置规则被自定义规则整体替换 (非合并), 便于场景化定制
    assert checker.check({"tool": "db_write"}).verdict is Verdict.ALLOW


def test_rule_based_check_escape_takes_precedence():
    """同一动作同时命中逃逸与高危时, 逃逸 (DENY) 优先 —— 宁可错拦不可放过"""
    checker = RuleBasedSafetyCheck(
        high_risk={"hr": ("tool",)}, escape={"esc": ("tool",)})
    assert checker.check({"tool": "tool"}).verdict is Verdict.DENY


def test_safety_check_is_abstract():
    """SafetyCheck 是抽象基类, 不能直接实例化 (强制实现 check)"""
    try:
        SafetyCheck()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError("SafetyCheck 应为抽象基类")


def test_safety_verdict_defaults():
    """SafetyVerdict 默认值: 无理由无标签的纯判定"""
    v = SafetyVerdict(verdict=Verdict.ALLOW)
    assert v.reasons == [] and v.risk_tags == []
    assert v.verdict.value == "allow"


# ═══════════════════════════════════════════════════════════════
# IntentProbe — 意图与决策探针 (P0)
# ═══════════════════════════════════════════════════════════════


def test_intent_probe_records_history():
    """基本路径: 审计一条候选动作 → 判定 + 计数 + 入历史"""
    probe = IntentProbe()
    audited = probe.inspect({"tool": "port_scan"})
    assert audited.verdict.verdict is Verdict.DENY
    assert probe.stats()["denied"] == 1
    assert len(probe.history) == 1


def test_intent_probe_history_capacity():
    """超出容量时淘汰最旧记录 (FIFO), 累计计数不随淘汰回退"""
    probe = IntentProbe(history_capacity=3)
    for i in range(5):
        probe.inspect({"tool": f"tool_{i}"})
    # 历史只留最近 3 条, 最早 2 条被淘汰
    assert len(probe.history) == 3
    assert [a.action["tool"] for a in probe.history] == ["tool_2", "tool_3", "tool_4"]
    # 累计统计不受容量淘汰影响 (inspected=5 而非 3)
    assert probe.stats()["inspected"] == 5
    assert probe.stats()["history"] == 3


def test_intent_probe_audited_action_fields():
    """审计记录携带: 原始动作 + 判定结论 + 时间戳 (审计四元组的决策依据)"""
    before = time.time()
    audited = IntentProbe().inspect({"tool": "report_generate"})
    assert isinstance(audited, AuditedAction)
    # 动作原样保留, 供事后复盘比对
    assert audited.action == {"tool": "report_generate"}
    assert audited.verdict.verdict is Verdict.ALLOW
    # 时间戳落在合理区间 (审计时序可信)
    assert before - 1 <= audited.t <= time.time() + 1


def test_intent_probe_accepts_custom_checker():
    """探针可注入任意 SafetyCheck 实现 (策略可替换)"""

    class AlwaysDeny(SafetyCheck):
        def check(self, action):
            return SafetyVerdict(Verdict.DENY, reasons=["test"])

    probe = IntentProbe(checker=AlwaysDeny())
    assert probe.inspect({"tool": "anything"}).verdict.verdict is Verdict.DENY
    assert probe.stats() == {"inspected": 1, "denied": 1, "history": 1}


def test_intent_probe_stats_zero():
    """全新探针的零状态统计 (无隐式初始化噪音)"""
    assert IntentProbe().stats() == {"inspected": 0, "denied": 0, "history": 0}


# ═══════════════════════════════════════════════════════════════
# CriticalActionCircuitBreaker — 高危工具调用熔断 (P0)
# ═══════════════════════════════════════════════════════════════


def test_circuit_breaker_fail_closed():
    """fail-closed 语义: 无审核者时一律拒绝 (安全默认)"""
    breaker = CriticalActionCircuitBreaker()  # 无审核者
    review = breaker.trip({"tool": "db_write"}, None)
    assert not breaker.resolve(review)  # fail-closed: 默认拒绝
    # 提供审核者: 批准
    ok = CriticalActionCircuitBreaker(reviewer=lambda r: True)
    assert ok.resolve(ok.trip({"tool": "db_write"}, None))


def test_circuit_breaker_review_lifecycle():
    """审核单完整生命周期: trip → 未决 → resolve 回填结论"""
    reviewer_calls = []

    def reviewer(review):
        reviewer_calls.append(review)
        # 审核策略: db_drop 拒绝, 其余批准
        return review.action["tool"] != "db_drop"

    breaker = CriticalActionCircuitBreaker(reviewer=reviewer)
    kept = breaker.trip({"tool": "db_write"}, None)
    dropped = breaker.trip({"tool": "db_drop"}, None)
    # resolve 之前: 均为未决状态
    for review in (kept, dropped):
        assert not review.resolved and review.approved is None  # 未决
    assert breaker.resolve(kept) is True
    assert kept.resolved and kept.approved is True
    assert breaker.resolve(dropped) is False
    assert dropped.resolved and dropped.approved is False
    assert len(reviewer_calls) == 2  # 每次 resolve 恰好调用一次审核者


def test_circuit_breaker_pending_capacity():
    """挂起队列超容量时淘汰最旧审核单, 统计计数保留"""
    breaker = CriticalActionCircuitBreaker(pending_capacity=2)
    for i in range(4):
        breaker.trip({"tool": f"op_{i}"}, None)
    # 队列只留最近 2 张审核单
    assert len(breaker.pending) == 2
    assert [p.action["tool"] for p in breaker.pending] == ["op_2", "op_3"]
    assert breaker.stats()["tripped"] == 4  # 累计熔断次数不回退


def test_circuit_breaker_stats_and_rejection_count():
    """统计三元组 (熔断/拒绝/挂起) 随生命周期正确演进"""
    breaker = CriticalActionCircuitBreaker(reviewer=lambda r: False)
    breaker.trip({"tool": "a"}, None)
    breaker.trip({"tool": "b"}, None)
    # resolve 前: 2 次熔断, 0 次拒绝, 2 张挂起
    assert breaker.stats() == {"tripped": 2, "rejected": 0, "pending": 2}
    for review in list(breaker.pending):
        breaker.resolve(review)
    # resolve 后: 拒绝计数 +2 (审核者全部拒绝)
    assert breaker.stats() == {"tripped": 2, "rejected": 2, "pending": 2}


# ═══════════════════════════════════════════════════════════════
# SecurityMonitor — 组合监控面
# ═══════════════════════════════════════════════════════════════


def test_security_monitor_gate():
    """gate() 三态基本路径: 放行 / 拦截 (无审核单) / 熔断 (返回审核单)"""
    monitor = SecurityMonitor()
    assert monitor.gate({"tool": "report_generate"})[0] is True
    denied = monitor.gate({"tool": "sudo"})
    assert denied[0] is False and denied[2] is None          # DENY: 无审核单
    tripped = monitor.gate({"tool": "db_write"})
    assert tripped[0] is False and tripped[2] is not None    # REVIEW: 熔断挂起
    assert monitor.stats()["breaker"]["tripped"] == 1


def test_security_monitor_gate_allow_passthrough():
    """放行时返回完整审计记录, 动作原样透传 (监控面不改写业务数据)"""
    monitor = SecurityMonitor()
    ok, audited, review = monitor.gate({"tool": "report_generate", "args": {"n": 1}})
    assert ok and review is None
    # 动作载荷逐字段原样保留, 监控面是无副作用的检查点
    assert audited.action == {"tool": "report_generate", "args": {"n": 1}}
    assert audited.verdict.verdict is Verdict.ALLOW


def test_security_monitor_gate_review_returns_pending():
    """REVIEW 返回的审核单已挂入熔断队列, 且拒绝计数不动"""
    monitor = SecurityMonitor()
    ok, _, review = monitor.gate({"tool": "curl"})
    assert not ok
    assert review in monitor.breaker.pending  # 审核单进入待审队列
    assert monitor.stats()["breaker"]["rejected"] == 0


def test_security_monitor_stats_consistency():
    """组合统计 = 探针统计 + 熔断统计的直和 (5 次混合 gate 后逐项核对)"""
    monitor = SecurityMonitor()
    # 负载: 2 放行 + 2 逃逸拦截 + 2 高危熔断
    for tool in ("report_generate", "sudo", "db_write", "curl", "sudo"):
        monitor.gate({"tool": tool})
    stats = monitor.stats()
    assert stats["probe"] == {"inspected": 5, "denied": 2, "history": 5}
    assert stats["breaker"] == {"tripped": 2, "rejected": 0, "pending": 2}


def test_security_monitor_injects_shared_checker():
    """monitor / probe / breaker 可独立注入, checker 只作用于探针侧"""

    class AllowAll(SafetyCheck):
        def check(self, action):
            return SafetyVerdict(Verdict.ALLOW)

    monitor = SecurityMonitor(checker=AllowAll())
    # 即使是逃逸/高危关键词, AllowAll 判定器也全部放行
    for tool in ("sudo", "db_drop", "port_scan"):
        assert monitor.gate({"tool": tool})[0] is True
    # 全放行 → 熔断器从未被触发
    assert monitor.stats()["breaker"]["tripped"] == 0
