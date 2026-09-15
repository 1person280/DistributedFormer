"""SecurityMonitor 运行时安全监控面 (RFC #1 组合入口)

把意图探针 (P0) 与高危熔断 (P0) 组合成内核与执行层之间的单一监控面。
内核下发动作前调用 gate(); 行为指纹 (P1) 与全链路审计 (P1) 接口占位。
"""

from typing import Any, Dict, Optional

from .circuit_breaker import CriticalActionCircuitBreaker, PendingReview
from .intent_probe import AuditedAction, IntentProbe
from .safety_check import RuleBasedSafetyCheck, SafetyCheck, Verdict


class SecurityMonitor:
    """组合监控面: gate() 是执行层前唯一的必经检查点

    返回 (放行与否, 审计记录, 被熔断挂起的审核单或 None)。
    """

    def __init__(self, checker: Optional[SafetyCheck] = None,
                 probe: Optional[IntentProbe] = None,
                 breaker: Optional[CriticalActionCircuitBreaker] = None):
        self.probe = probe or IntentProbe(checker=checker or RuleBasedSafetyCheck())
        self.breaker = breaker or CriticalActionCircuitBreaker()

    def gate(self, action: Dict[str, Any]) -> tuple:
        """执行前门禁: ALLOW 放行 / REVIEW 熔断挂起 / DENY 拒绝"""
        audited = self.probe.inspect(action)
        v = audited.verdict.verdict
        if v is Verdict.ALLOW:
            return True, audited, None
        if v is Verdict.REVIEW:
            review = self.breaker.trip(action, audited.verdict)
            return False, audited, review
        return False, audited, None  # DENY

    def stats(self) -> Dict[str, Any]:
        return {"probe": self.probe.stats(), "breaker": self.breaker.stats()}
