"""运行时安全监控骨架测试 (RFC #1 / v0.9.0 Safety Shield)

覆盖: SafetyCheck 三态判定 / 规则注入 / IntentProbe 历史容量 /
熔断 fail-closed 与挂起容量 / SecurityMonitor 组合门禁与统计一致性
"""

import time

from distributedformer.security_monitor import (
    AuditedAction, CriticalActionCircuitBreaker, IntentProbe,
    RuleBasedSafetyCheck, SafetyCheck, SafetyVerdict, SecurityMonitor,
    Verdict,
)

# ═══════════════════════════════════════════════════════════════
# RuleBasedSafetyCheck
# ═══════════════════════════════════════════════════════════════


def test_rule_based_check_verdicts():
    checker = RuleBasedSafetyCheck()
    assert checker.check({"tool": "report_generate"}).verdict is Verdict.ALLOW
    assert checker.check({"tool": "db_write"}).verdict is Verdict.REVIEW
    assert checker.check({"tool": "port_scan"}).verdict is Verdict.DENY


def test_rule_based_check_topic_fallback():
    """无 tool 字段时回退到 topic 判定"""
    checker = RuleBasedSafetyCheck()
    assert checker.check({"topic": "curl"}).verdict is Verdict.REVIEW
    assert checker.check({"topic": "escalate"}).verdict is Verdict.DENY


def test_rule_based_check_empty_and_unknown():
    """空动作 / 未知工具 / 空字符串均为放行 (默认信任业务动作)"""
    checker = RuleBasedSafetyCheck()
    for action in ({}, {"tool": ""}, {"tool": "unknown_tool"}, {"args": "x"}):
        v = checker.check(action)
        assert v.verdict is Verdict.ALLOW
        assert v.risk_tags == [] and v.reasons == []


def test_rule_based_check_risk_tags():
    v = RuleBasedSafetyCheck().check({"tool": "sudo"})
    assert v.verdict is Verdict.DENY
    assert v.risk_tags == ["privilege-escalation"]
    assert v.reasons  # 拦截必须给出理由 (审计留痕)


def test_rule_based_check_custom_rules():
    """支持注入自定义规则集 (逃逸优先于高危)"""
    checker = RuleBasedSafetyCheck(
        high_risk={"deploy-prod": ("deploy",)},
        escape={"secret-exfil": ("dump_secrets",)},
    )
    assert checker.check({"tool": "deploy"}) .verdict is Verdict.REVIEW
    assert checker.check({"tool": "dump_secrets"}).verdict is Verdict.DENY
    # 内置规则被自定义规则整体替换
    assert checker.check({"tool": "db_write"}).verdict is Verdict.ALLOW


def test_rule_based_check_escape_takes_precedence():
    """同一动作同时命中逃逸与高危时, 逃逸 (DENY) 优先"""
    checker = RuleBasedSafetyCheck(
        high_risk={"hr": ("tool",)}, escape={"esc": ("tool",)})
    assert checker.check({"tool": "tool"}).verdict is Verdict.DENY


def test_safety_check_is_abstract():
    """SafetyCheck 是抽象基类, 不能直接实例化"""
    try:
        SafetyCheck()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError("SafetyCheck 应为抽象基类")


def test_safety_verdict_defaults():
    v = SafetyVerdict(verdict=Verdict.ALLOW)
    assert v.reasons == [] and v.risk_tags == []
    assert v.verdict.value == "allow"


# ═══════════════════════════════════════════════════════════════
# IntentProbe
# ═══════════════════════════════════════════════════════════════


def test_intent_probe_records_history():
    probe = IntentProbe()
    audited = probe.inspect({"tool": "port_scan"})
    assert audited.verdict.verdict is Verdict.DENY
    assert probe.stats()["denied"] == 1
    assert len(probe.history) == 1


def test_intent_probe_history_capacity():
    """超出容量时淘汰最旧记录, 统计计数不回退"""
    probe = IntentProbe(history_capacity=3)
    for i in range(5):
        probe.inspect({"tool": f"tool_{i}"})
    assert len(probe.history) == 3
    assert [a.action["tool"] for a in probe.history] == ["tool_2", "tool_3", "tool_4"]
    assert probe.stats()["inspected"] == 5
    assert probe.stats()["history"] == 3


def test_intent_probe_audited_action_fields():
    """审计记录携带: 原始动作 + 判定结论 + 时间戳 (审计四元组的决策依据)"""
    before = time.time()
    audited = IntentProbe().inspect({"tool": "report_generate"})
    assert isinstance(audited, AuditedAction)
    assert audited.action == {"tool": "report_generate"}
    assert audited.verdict.verdict is Verdict.ALLOW
    assert before - 1 <= audited.t <= time.time() + 1


def test_intent_probe_accepts_custom_checker():
    """探针可注入任意 SafetyCheck 实现"""

    class AlwaysDeny(SafetyCheck):
        def check(self, action):
            return SafetyVerdict(Verdict.DENY, reasons=["test"])

    probe = IntentProbe(checker=AlwaysDeny())
    assert probe.inspect({"tool": "anything"}).verdict.verdict is Verdict.DENY
    assert probe.stats() == {"inspected": 1, "denied": 1, "history": 1}


def test_intent_probe_stats_zero():
    assert IntentProbe().stats() == {"inspected": 0, "denied": 0, "history": 0}


# ═══════════════════════════════════════════════════════════════
# CriticalActionCircuitBreaker
# ═══════════════════════════════════════════════════════════════


def test_circuit_breaker_fail_closed():
    breaker = CriticalActionCircuitBreaker()  # 无审核者
    review = breaker.trip({"tool": "db_write"}, None)
    assert not breaker.resolve(review)  # fail-closed: 默认拒绝
    # 提供审核者: 批准
    ok = CriticalActionCircuitBreaker(reviewer=lambda r: True)
    assert ok.resolve(ok.trip({"tool": "db_write"}, None))


def test_circuit_breaker_review_lifecycle():
    """审核单: trip → 未决 → resolve 回填结论"""
    reviewer_calls = []

    def reviewer(review):
        reviewer_calls.append(review)
        return review.action["tool"] != "db_drop"

    breaker = CriticalActionCircuitBreaker(reviewer=reviewer)
    kept = breaker.trip({"tool": "db_write"}, None)
    dropped = breaker.trip({"tool": "db_drop"}, None)
    for review in (kept, dropped):
        assert not review.resolved and review.approved is None  # 未决
    assert breaker.resolve(kept) is True
    assert kept.resolved and kept.approved is True
    assert breaker.resolve(dropped) is False
    assert dropped.resolved and dropped.approved is False
    assert len(reviewer_calls) == 2


def test_circuit_breaker_pending_capacity():
    """挂起队列超容量时淘汰最旧审核单, 统计计数保留"""
    breaker = CriticalActionCircuitBreaker(pending_capacity=2)
    for i in range(4):
        breaker.trip({"tool": f"op_{i}"}, None)
    assert len(breaker.pending) == 2
    assert [p.action["tool"] for p in breaker.pending] == ["op_2", "op_3"]
    assert breaker.stats()["tripped"] == 4


def test_circuit_breaker_stats_and_rejection_count():
    breaker = CriticalActionCircuitBreaker(reviewer=lambda r: False)
    breaker.trip({"tool": "a"}, None)
    breaker.trip({"tool": "b"}, None)
    assert breaker.stats() == {"tripped": 2, "rejected": 0, "pending": 2}
    for review in list(breaker.pending):
        breaker.resolve(review)
    assert breaker.stats() == {"tripped": 2, "rejected": 2, "pending": 2}


# ═══════════════════════════════════════════════════════════════
# SecurityMonitor
# ═══════════════════════════════════════════════════════════════


def test_security_monitor_gate():
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
    assert audited.action == {"tool": "report_generate", "args": {"n": 1}}
    assert audited.verdict.verdict is Verdict.ALLOW


def test_security_monitor_gate_review_returns_pending():
    """REVIEW 返回的审核单已挂入熔断队列, 且拒绝计数不动"""
    monitor = SecurityMonitor()
    ok, _, review = monitor.gate({"tool": "curl"})
    assert not ok
    assert review in monitor.breaker.pending
    assert monitor.stats()["breaker"]["rejected"] == 0


def test_security_monitor_stats_consistency():
    """组合统计 = 探针统计 + 熔断统计的直和"""
    monitor = SecurityMonitor()
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
    for tool in ("sudo", "db_drop", "port_scan"):
        assert monitor.gate({"tool": tool})[0] is True
    assert monitor.stats()["breaker"]["tripped"] == 0
