"""运行时安全监控模块 (v0.9.2 Safety Shield, RFC #1)

利用思考层 (CubeGPT 内核) 与执行层 (Rust 插件) 解耦的架构优势,
在内核与插件之间建立独立的运行时安全监控面:

    ┌────────────┐   候选动作   ┌──────────────────┐   放行   ┌────────────┐
    │ CubeGPT 内核 │ ──────────→ │ SecurityMonitor  │ ───────→ │ Rust 执行层 │
    │   (思考)    │             │ (意图审计/熔断/指纹 │   拦截   │   (行动)    │
    └────────────┘             │   /审计溯源)      │ ←─────── └────────────┘
                              └──────────────────┘

四个方向 (RFC #1, 按优先级), 已全部落地:
    P0  IntentProbe            意图与决策探针: 下发前扫描候选动作
    P0  CircuitBreaker         高危工具调用熔断: 深度审核确认前拒绝下发
    P1  BehavioralFingerprint  行为指纹与异常基线检测: 频率/熵值/链路关联
    P1  ActionTracer           全链路行为审计与溯源: 四元组绑定留存复盘
在此之前安全判定以规则 (SafetyCheck) 为初始实现, 学习型分类器为开放问题。
"""

from .safety_check import SafetyCheck, SafetyVerdict, Verdict, RuleBasedSafetyCheck
from .intent_probe import IntentProbe, AuditedAction
from .circuit_breaker import CriticalActionCircuitBreaker, PendingReview
from .behavioral_fingerprint import BehavioralFingerprint, FingerprintResult
from .action_tracer import ActionTracer, ActionTrace
from .monitor import SecurityMonitor

__all__ = [
    "SafetyCheck", "SafetyVerdict", "Verdict", "RuleBasedSafetyCheck",
    "IntentProbe", "AuditedAction",
    "CriticalActionCircuitBreaker", "PendingReview",
    "BehavioralFingerprint", "FingerprintResult",
    "ActionTracer", "ActionTrace",
    "SecurityMonitor",
]