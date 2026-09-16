"""SecurityMonitor 运行时安全监控面 (RFC #1 组合入口)

把意图探针 (P0)、高危熔断 (P0)、行为指纹 (P1) 与全链路审计 (P1)
组合成内核与执行层之间的单一监控面。内核下发动作前调用 gate();
gate() 同时进行实时行为异常判定并将全链路四元组写入审计存储。
"""

from typing import Any, Dict, List, Optional

from .action_tracer import ActionTracer
from .behavioral_fingerprint import BehavioralFingerprint, FingerprintResult
from .circuit_breaker import CriticalActionCircuitBreaker, PendingReview
from .intent_probe import AuditedAction, IntentProbe
from .safety_check import RuleBasedSafetyCheck, SafetyCheck, SafetyVerdict, Verdict


class SecurityMonitor:
    """组合监控面: gate() 是执行层前唯一的必经检查点

    返回 (放行与否, 审计记录, 被熔断挂起的审核单或 None)。
    实时判定结果存于 last_fingerprint, 全链路审计写入 tracer。
    """

    def __init__(self, checker: Optional[SafetyCheck] = None,
                 probe: Optional[IntentProbe] = None,
                 breaker: Optional[CriticalActionCircuitBreaker] = None,
                 fingerprint: Optional[BehavioralFingerprint] = None,
                 tracer: Optional[ActionTracer] = None):
        self.probe = probe or IntentProbe(checker=checker or RuleBasedSafetyCheck())
        self.breaker = breaker or CriticalActionCircuitBreaker()
        self.fingerprint = fingerprint or BehavioralFingerprint()
        self.tracer = tracer or ActionTracer()
        self.last_fingerprint: Optional[FingerprintResult] = None

    def gate(self, action: Dict[str, Any],
             thinking_state: Optional[Dict[str, Any]] = None,
             task: Optional[str] = None) -> tuple:
        """执行前门禁: ALLOW 放行 / REVIEW 熔断挂起 / DENY 拒绝

        副作用: 实时行为指纹判定; ALLOW 动作回填频率基线;
        每条动作写入全链路审计 (四元组绑定)。
        """
        audited = self.probe.inspect(action)
        v = audited.verdict.verdict

        # 行为指纹: 并列判定, 与意图规则回执一并留痕 (不下发前强拦截)
        fp = self.fingerprint.check(action, task=task)
        self.last_fingerprint = fp

        if v is Verdict.ALLOW:
            self.fingerprint.observe(action)  # 正常业务回填基线
            ok, review, outcome = True, None, "ALLOW"
        elif v is Verdict.REVIEW:
            review = self.breaker.trip(action, audited.verdict)
            ok, outcome = False, "REVIEW"
        else:
            review, outcome = None, "DENY"
            ok = False

        self.tracer.record(
            thinking_state=thinking_state,
            decision_basis=_basis_of(audited, fp),
            tool_params=action,
            execution_result={"allowed": ok, "outcome": outcome,
                              "fingerprint_flags": fp.flags},
            tags=list(fp.flags),
        )
        return ok, audited, review

    def stats(self) -> Dict[str, Any]:
        return {"probe": self.probe.stats(),
                "breaker": self.breaker.stats(),
                "fingerprint": self.fingerprint.stats(),
                "tracer": self.tracer.stats()}


def _basis_of(audited: AuditedAction,
              fp: Optional[FingerprintResult]) -> Dict[str, Any]:
    """把意图判定结论与行为指纹结果合并为审计的决策依据"""
    verdict: SafetyVerdict = audited.verdict
    return {
        "verdict": verdict.verdict.value,
        "reasons": verdict.reasons,
        "risk_tags": verdict.risk_tags,
        "fingerprint_flags": fp.flags if fp else [],
        "fingerprint_metrics": fp.metrics if fp else {},
    }
