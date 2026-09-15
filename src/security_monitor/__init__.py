"""运行时安全监控模块 (v0.9.0 Safety Shield 骨架, RFC #1)

利用思考层 (CubeGPT 内核) 与执行层 (Rust 插件) 解耦的架构优势,
在内核与插件之间建立独立的运行时安全监控面:

    ┌────────────┐   候选动作   ┌──────────────────┐   放行   ┌────────────┐
    │ CubeGPT 内核 │ ──────────→ │ SecurityMonitor  │ ───────→ │ Rust 执行层 │
    │   (思考)    │             │  (意图审计/熔断)  │   拦截   │   (行动)    │
    └────────────┘             └──────────────────┘ ←─────── └────────────┘

四个方向 (RFC #1, 按优先级):
    P0  IntentProbe            意图与决策探针: 下发前扫描候选动作
    P0  CircuitBreaker         高危工具调用熔断: 深度审核确认前拒绝下发
    P1  BehavioralFingerprint  行为指纹与异常基线检测 (占位)
    P1  ActionTracer           全链路行为审计与溯源 (占位)

当前为接口骨架阶段: 定义接口与最小可用实现, 尚未接入内核主路径。
"""

from .safety_check import SafetyCheck, SafetyVerdict, Verdict, RuleBasedSafetyCheck
from .intent_probe import IntentProbe, AuditedAction
from .circuit_breaker import CriticalActionCircuitBreaker
from .monitor import SecurityMonitor

__all__ = [
    "SafetyCheck", "SafetyVerdict", "Verdict", "RuleBasedSafetyCheck",
    "IntentProbe", "AuditedAction",
    "CriticalActionCircuitBreaker",
    "SecurityMonitor",
]
