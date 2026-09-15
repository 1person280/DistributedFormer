"""集成链路各环节的单元级验证 (RFC #1)

集成测试 (test_security_integration.py) 端到端地验证了
内核 → 总线 → 门禁 → 执行层 链路, 本文件对链路上的每个环节做隔离验证:
- EventBus 原语: 投递 / 通配 / 退订 / 回调异常隔离 / 历史与统计
- ExecutionLayer 分派: 三种判定 → 执行/拦截/挂起 的映射, release 补放行
- ToolCallingPlugin: 候选动作载荷格式与返回值
"""

import time

from src.cutemamen import CuteMamenKernel, ExpertPlugin
from src.security_monitor import SecurityMonitor, Verdict
from tests.test_security_integration import (
    ACTION_TOPIC, ExecutionLayer, ToolCallingPlugin, build_runtime,
)

# ═══════════════════════════════════════════════════════════════
# EventBus 原语 (集成链路的通信底座)
# ═══════════════════════════════════════════════════════════════


def test_event_bus_delivers_to_exact_topic():
    """精确投递: 只有订阅了该主题的回调收到, 无关主题送达 0"""
    kernel, _, _ = build_runtime()
    got = []
    kernel.bus.subscribe("x.y", lambda e: got.append(e["payload"]))
    delivered = kernel.bus.publish("x.y", {"n": 1})
    assert delivered == 1 and got == [{"n": 1}]
    assert kernel.bus.publish("other.topic", None) == 0  # 无订阅者


def test_event_bus_wildcard_receives_all():
    """通配订阅 "*": 收到所有主题的发布 (审计通道的实现基础)"""
    kernel, _, _ = build_runtime()
    seen = []
    kernel.bus.subscribe("*", lambda e: seen.append(e["topic"]))
    kernel.bus.publish("a", None)
    kernel.bus.publish("b.c", None)
    assert seen == ["a", "b.c"]


def test_event_bus_unsubscribe():
    """退订函数: 调用后不再投递, 订阅者计数归零"""
    kernel, _, _ = build_runtime()
    got = []
    unsub = kernel.bus.subscribe("t", lambda e: got.append(1))
    kernel.bus.publish("t", None)
    unsub()  # 退订
    kernel.bus.publish("t", None)
    assert got == [1]                    # 退订后不再投递
    assert kernel.bus.subscriber_count("t") == 0


def test_event_bus_callback_error_isolation():
    """订阅回调抛异常: 不中断发布方, 其他订阅者照常收到, 计入 errors"""
    kernel, _, _ = build_runtime()
    ok_got = []
    kernel.bus.subscribe("t", lambda e: 1 / 0)           # 故意抛异常的订阅者
    kernel.bus.subscribe("t", lambda e: ok_got.append(e["payload"]))
    delivered = kernel.bus.publish("t", {"n": 1})
    assert delivered == 1                # 正常订阅者收到了
    assert ok_got == [{"n": 1}]
    # 异常被总线捕获并留痕, 不向上传播
    assert len(kernel.bus.errors) == 1
    assert kernel.bus.errors[0]["topic"] == "t"


def test_event_bus_event_envelope():
    """总线事件信封: topic / 时间戳 / payload 三元组 (审计溯源依赖)"""
    kernel, _, _ = build_runtime()
    kernel.bus.publish("t", {"a": 1})
    event = kernel.bus.recent(1)[0]
    assert event["topic"] == "t"
    assert event["payload"] == {"a": 1}
    assert isinstance(event["t"], float)  # 时间戳可用于事后排序
    assert kernel.bus.stats()["published"] >= 1


def test_event_bus_history_capacity():
    """历史环形缓冲不无限增长 (审计留痕的内存上界)"""
    kernel, _, _ = build_runtime()
    for i in range(300):                 # 默认容量 256
        kernel.bus.publish(f"t.{i}", None)
    assert len(kernel.bus.history) == 256          # 封顶在容量
    assert kernel.bus.history[-1]["topic"] == "t.299"   # 最新仍在
    assert kernel.bus.history[0]["topic"] == "t.44"   # 最旧的被淘汰


# ═══════════════════════════════════════════════════════════════
# ExecutionLayer 分派逻辑 (三态判定 → 行为映射, 与内核解耦单测)
# ═══════════════════════════════════════════════════════════════


def _bus_event(payload):
    """构造与 EventBus.publish 相同信封的事件 (不依赖真实总线)"""
    return {"topic": ACTION_TOPIC, "t": time.time(), "payload": payload}


def test_execution_layer_allow_executes():
    """ALLOW → 执行并记录, 无拦截无挂起"""
    executor = ExecutionLayer(SecurityMonitor())
    executor.handle(_bus_event({"tool": "report_generate"}))
    assert executor.executed == ["report_generate"]
    assert executor.denied == [] and executor.reviews == []


def test_execution_layer_deny_records():
    """DENY → 拒绝执行并记录拦截, 不产生审核单"""
    executor = ExecutionLayer(SecurityMonitor())
    executor.handle(_bus_event({"tool": "sudo"}))
    assert executor.executed == []
    assert executor.denied == ["sudo"]
    assert executor.reviews == []        # DENY 不产生审核单


def test_execution_layer_review_pends():
    """REVIEW → 不执行, 挂入熔断队列等待审核"""
    executor = ExecutionLayer(SecurityMonitor())
    executor.handle(_bus_event({"tool": "db_write"}))
    assert executor.executed == []       # 确认合规前拒绝下发
    assert executor.denied == []
    assert [r.action["tool"] for r in executor.reviews] == ["db_write"]


def test_execution_layer_release_fail_closed():
    """release 无审核者: fail-closed 拒绝, 不执行"""
    executor = ExecutionLayer(SecurityMonitor())
    executor.handle(_bus_event({"tool": "db_write"}))
    assert not executor.release(executor.reviews[0])   # 无审核者 → 拒绝
    assert executor.executed == []


def test_execution_layer_release_approved():
    """release 审核批准: 补放行执行, 审核单状态回填"""
    monitor = SecurityMonitor()
    monitor.breaker.reviewer = lambda r: True          # 审核者总是批准
    executor = ExecutionLayer(monitor)
    executor.handle(_bus_event({"tool": "curl"}))
    assert executor.release(executor.reviews[0])
    assert executor.executed == ["curl"]               # 批准后补放行
    assert executor.reviews[0].resolved and executor.reviews[0].approved


def test_execution_layer_state_accumulates():
    """多次 handle 状态累积且分类正确 (无串扰)"""
    executor = ExecutionLayer(SecurityMonitor())
    # 负载: 2 放行 + 1 拦截 + 1 挂起
    for tool in ("report_generate", "sudo", "db_write", "report_generate"):
        executor.handle(_bus_event({"tool": tool}))
    assert executor.executed == ["report_generate", "report_generate"]
    assert executor.denied == ["sudo"]
    assert len(executor.reviews) == 1


# ═══════════════════════════════════════════════════════════════
# ToolCallingPlugin (候选动作的产生端)
# ═══════════════════════════════════════════════════════════════


def test_tool_plugin_emits_candidate_and_returns():
    """on_think: 发布候选到 ACTION_TOPIC (载荷 = {"tool": ...}) 并返回结果"""
    kernel = CuteMamenKernel(dim=16)
    emitted = []
    kernel.bus.subscribe(ACTION_TOPIC, lambda e: emitted.append(e["payload"]))
    plugin = ToolCallingPlugin("tools", route="numeric")
    kernel.mount(plugin)

    # 直接调用 on_think (绕过路由), 验证插件自身的发布契约
    ctx = kernel._ctx_for(plugin)
    result = plugin.on_think({"topic": "numeric", "data": "db_write"}, ctx)

    assert result == {"requested": "db_write"}
    assert emitted == [{"tool": "db_write"}]           # 执行层可消费的格式


def test_tool_plugin_is_expert_plugin():
    """脚手架插件符合 CuteMamen 专家插件基类 (可被内核 mount/路由)"""
    plugin = ToolCallingPlugin("tools", route="numeric")
    assert isinstance(plugin, ExpertPlugin)
    assert plugin.route == "numeric"          # 路由主题正确
    manifest = plugin.build_manifest()
    assert manifest["name"] == "tools"        # manifest 携带插件名
