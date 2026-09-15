"""运行时安全监控集成测试 (RFC #1 / v0.9.0 Safety Shield)

把 SecurityMonitor 接入真实 CuteMamenKernel 的事件总线, 验证 RFC #1 描述的
完整链路:

    内核 think (思考) → 事件总线候选动作 → SecurityMonitor.gate (审计) → 执行层

以及与真实 RustCodingPlugin 的协同 (安全面对正常业务零干扰)。
骨架阶段 gate() 尚未硬接入 kernel 主路径, 这里以"执行层订阅者"的形式
模拟真实的接入形态 —— 内核与插件不感知监控面, 监控面只拦截动作下发。
"""

import pytest

from distributedformer.cutemamen import (
    CuteMamenKernel, ExpertPlugin, RustCodingPlugin,
)
from distributedformer.security_monitor import SecurityMonitor, Verdict

# 执行层动作主题: 插件的候选动作发布到这里, 执行层订阅者经 gate 后执行
ACTION_TOPIC = "action.request"


# ═══════════════════════════════════════════════════════════════
# 测试脚手架: 会发起工具调用的思考插件 + 经安全门禁的执行层
# ═══════════════════════════════════════════════════════════════

class ToolCallingPlugin(ExpertPlugin):
    """思考插件: 收到 numeric 事件后, 把候选工具调用发布到执行主题

    模拟"内核/插件产生意图"的一侧 —— 它只负责发布候选动作,
    不知道也不关心动作最终是否被执行 (思考与执行解耦)。
    """

    def on_think(self, event, ctx):
        # event["data"] 即插件想调用的工具名, 原样作为候选动作载荷
        tool = event["data"]
        ctx.emit(ACTION_TOPIC, {"tool": tool})
        # 思考结果照常返回 (即使候选被拦截, 思考层也已完成)
        return {"requested": tool}


class ExecutionLayer:
    """模拟 Rust 执行层: 只执行通过 SecurityMonitor.gate 的动作

    对应 RFC #1 架构图右侧的执行层, 三种判定对应三种处置:
    - ALLOW → 执行并记录
    - DENY  → 拒绝执行, 记录拦截
    - REVIEW → 不执行, 挂入熔断队列等待审核 (审核通过后由 release 放行)
    """

    def __init__(self, monitor: SecurityMonitor):
        self.monitor = monitor
        self.executed = []   # 真正执行的工具
        self.denied = []     # 被 DENY 拦截的工具
        self.reviews = []    # 被熔断挂起的审核单

    def handle(self, bus_event):
        """事件总线回调: 事件信封 {"topic", "t", "payload"} → gate → 分派"""
        action = bus_event["payload"]
        ok, audited, review = self.monitor.gate(action)
        if ok:
            self.executed.append(action["tool"])
        elif review is not None:
            self.reviews.append(review)      # REVIEW: 挂起待审
        else:
            self.denied.append(action["tool"])  # DENY: 直接拒绝

    def release(self, review) -> bool:
        """审核通过后补放行 (人工/深度审核批准的调用才真正执行)"""
        if self.monitor.breaker.resolve(review):
            self.executed.append(review.action["tool"])
            return True
        return False


def build_runtime():
    """内核 (思考) + 执行层 (行动) + 监控面 (审计) 三件套接线"""
    kernel = CuteMamenKernel(dim=16)
    monitor = SecurityMonitor()
    executor = ExecutionLayer(monitor)
    # 挂载会发起工具调用的思考插件, 路由到 numeric 主题
    kernel.mount(ToolCallingPlugin("tools", route="numeric"))
    # 执行层订阅动作主题 —— 这就是"解耦的缝", 监控面插在这条通路上
    kernel.bus.subscribe(ACTION_TOPIC, executor.handle)
    return kernel, monitor, executor


# ═══════════════════════════════════════════════════════════════
# 全链路: 内核 → 总线 → 门禁 → 执行
# ═══════════════════════════════════════════════════════════════


def test_allow_flow_end_to_end():
    """正常业务动作: think → 候选 → 放行 → 执行, 全程零拦截"""
    kernel, monitor, executor = build_runtime()
    results = kernel.think({"topic": "numeric", "data": "report_generate"})

    # 思考结果与执行结果各就各位
    assert results == [{"requested": "report_generate"}]
    assert executor.executed == ["report_generate"]
    assert executor.denied == [] and executor.reviews == []
    # 监控统计: 1 次审计, 0 拦截, 0 熔断
    assert monitor.stats() == {"probe": {"inspected": 1, "denied": 0, "history": 1},
                               "breaker": {"tripped": 0, "rejected": 0, "pending": 0}}


def test_deny_flow_blocks_execution():
    """逃逸倾向动作 (sudo): 内核思考正常完成, 但执行层被门禁拦截"""
    kernel, monitor, executor = build_runtime()
    results = kernel.think({"topic": "numeric", "data": "sudo"})

    # 思考层不受影响 (解耦: 拦的是动作下发, 不是思考本身)
    assert results == [{"requested": "sudo"}]
    assert executor.executed == []                      # 未执行
    assert executor.denied == ["sudo"]                  # 被拦截
    assert monitor.probe.total_denied == 1
    assert monitor.stats()["probe"]["history"] == 1     # 审计留痕


def test_review_flow_circuit_break_then_release():
    """高危动作 (db_write): 熔断挂起 → 不执行 → 审核拒绝维持拦截"""
    kernel, monitor, executor = build_runtime()
    kernel.think({"topic": "numeric", "data": "db_write"})

    assert executor.executed == []                      # 确认合规前拒绝下发
    assert len(executor.reviews) == 1
    review = executor.reviews[0]
    # 审核单保留完整原始动作, 供深度审核 / 人工介入时复盘
    assert review.action == {"tool": "db_write"}

    # fail-closed: 无审核者 → 拒绝 → 依然不执行
    assert not executor.release(review)
    assert executor.executed == []
    assert monitor.breaker.total_rejected == 1


def test_review_flow_approved_then_executed():
    """审核者批准的高危动作才补放行执行"""
    kernel, monitor, executor = build_runtime()
    monitor.breaker.reviewer = lambda review: True       # 深度审核通过
    kernel.think({"topic": "numeric", "data": "curl"})

    assert executor.executed == []                      # 先熔断
    review = executor.reviews[0]
    assert executor.release(review)                     # 审核通过 → 放行
    assert executor.executed == ["curl"]


def test_mixed_workload_audit_trail():
    """混合负载: 全部候选动作留痕, 只有放行动作真正执行"""
    kernel, monitor, executor = build_runtime()
    # 负载: 2 放行 + 2 逃逸 + 2 高危
    for tool in ("report_generate", "sudo", "db_write",
                 "report_generate", "port_scan", "set_env"):
        kernel.think({"topic": "numeric", "data": tool})

    assert executor.executed == ["report_generate", "report_generate"]
    assert sorted(executor.denied) == ["port_scan", "sudo"]
    assert len(executor.reviews) == 2                   # db_write + set_env
    # 探针历史 = 全部 6 条候选 (审计溯源: 思考状态 → 判定结论)
    assert [a.action["tool"] for a in monitor.probe.history] == [
        "report_generate", "sudo", "db_write",
        "report_generate", "port_scan", "set_env"]


# ═══════════════════════════════════════════════════════════════
# 与内核机制的协同
# ═══════════════════════════════════════════════════════════════


def test_monitor_does_not_break_kernel_routing():
    """未路由事件与总线广播不受监控面影响 (监控面只在动作通路上)"""
    kernel, monitor, executor = build_runtime()
    unrouted = []
    kernel.bus.subscribe("kernel.unrouted", lambda e: unrouted.append(e["payload"]))

    # 未知主题: 内核正常走 unrouted 分支, 无候选动作产生
    assert kernel.think({"topic": "unknown", "data": "x"}) == []
    assert unrouted == [{"topic": "unknown"}]
    assert monitor.stats()["probe"]["inspected"] == 0    # 未产生候选动作
    assert kernel.total_thinks == 1


def test_monitor_survives_multiple_dispatch_cycles():
    """门禁状态跨多次 think 累积 (无状态泄漏 / 无重复计数)"""
    kernel, monitor, executor = build_runtime()
    # 同一逃逸动作重复 3 次: 每次都被独立拦截并留痕
    for _ in range(3):
        kernel.think({"topic": "numeric", "data": "sudo"})
    assert monitor.probe.total_denied == 3
    assert len(monitor.probe.history) == 3
    assert kernel.bus.stats()["errors"] == 0             # 门禁回调零异常


def test_wildcard_subscriber_observes_audit_side_effects():
    """通配订阅者可同时观察业务候选与生命周期广播 (审计通道可用)"""
    kernel, monitor, executor = build_runtime()
    seen = []
    kernel.bus.subscribe("*", lambda e: seen.append(e["topic"]))

    kernel.think({"topic": "numeric", "data": "sudo"})

    # 三类事件对通配订阅者均可见: 候选动作 / 内核广播 / 插件输出
    assert ACTION_TOPIC in seen                          # 候选动作可见
    assert "kernel.think" in seen                        # 内核广播可见
    assert "plugin.tools.output" in seen                 # 插件输出可见


# ═══════════════════════════════════════════════════════════════
# 与真实 RustCodingPlugin 协同 (安全面对正常业务零干扰)
# ═══════════════════════════════════════════════════════════════


def test_rust_coding_plugin_unaffected_by_monitor():
    """真实思考插件: rust 代码分类结果正常返回, 门禁全程放行"""
    kernel, monitor, executor = build_runtime()
    kernel.mount(RustCodingPlugin("rust-coding"))        # route 默认 "rust"

    # 一段真实生命周期错误代码 → 分类器返回 5 类标签之一
    code = 'fn longest(s1: &str, s2: &str) -> &str { s1 }'
    results = kernel.think({"topic": "rust", "data": code})

    assert results and results[0]["label"] in ("lifetime", "borrow", "move", "type", "ok")
    assert executor.executed == []                       # rust 分类不发起工具调用
    assert monitor.stats()["probe"]["inspected"] == 0    # 零干扰
    assert kernel.bus.stats()["errors"] == 0


def test_rust_plugin_and_tool_plugin_share_one_gate():
    """双插件并存: rust 分类照常, tools 候选照常审计, 互不干扰"""
    kernel, monitor, executor = build_runtime()
    kernel.mount(RustCodingPlugin("rust-coding"))

    # 一次 rust 分类 (不产生候选) + 两次工具调用 (一次拦截一次放行)
    kernel.think({"topic": "rust", "data": "let x = 5;"})
    kernel.think({"topic": "numeric", "data": "sudo"})
    kernel.think({"topic": "numeric", "data": "report_generate"})

    assert executor.denied == ["sudo"]
    assert executor.executed == ["report_generate"]
    # 只有 tools 候选经过门禁: inspected=2 而非 3
    assert monitor.stats()["probe"] == {"inspected": 2, "denied": 1, "history": 2}
