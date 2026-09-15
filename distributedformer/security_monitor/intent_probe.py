"""IntentProbe 意图与决策探针 (RFC #1 P0)

在模型生成下一步行动/工具调用参数之前, 实时扫描其"内部决策日志"
(思考状态 / 事件总线上的候选动作), 借 SafetyCheck 判断是否包含
越权、获取高权限指令或探索系统边界等逃逸倾向, 命中即在下发前拦截。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .safety_check import RuleBasedSafetyCheck, SafetyCheck, SafetyVerdict, Verdict


@dataclass
class AuditedAction:
    """一条经过审计的动作: 原始动作 + 判定结论 + 时间戳 (审计四元组的决策依据部分)"""

    action: Dict[str, Any]
    verdict: SafetyVerdict
    t: float = field(default_factory=time.time)


class IntentProbe:
    """意图探针: 挂在内核 think() 与执行层之间的审计面

    用法 (骨架, 尚未接入 CuteMamenKernel 主路径):
        probe = IntentProbe()
        audited = probe.inspect({"topic": "tool.call", "tool": "db_write", ...})
        if audited.verdict.verdict is Verdict.DENY:
            raise RuntimeError(f"意图审计拦截: {audited.verdict.reasons}")
    """

    def __init__(self, checker: Optional[SafetyCheck] = None,
                 history_capacity: int = 256):
        self.checker = checker or RuleBasedSafetyCheck()
        self.history_capacity = history_capacity
        self.history: List[AuditedAction] = []
        self.total_inspected = 0
        self.total_denied = 0

    def inspect(self, action: Dict[str, Any]) -> AuditedAction:
        """审计一条候选动作: 判定 → 记入历史 (供审计溯源/基线检测)"""
        verdict = self.checker.check(action)
        audited = AuditedAction(action=action, verdict=verdict)
        self.history.append(audited)
        if len(self.history) > self.history_capacity:
            self.history.pop(0)
        self.total_inspected += 1
        if verdict.verdict is Verdict.DENY:
            self.total_denied += 1
        return audited

    def stats(self) -> Dict[str, Any]:
        return {"inspected": self.total_inspected,
                "denied": self.total_denied,
                "history": len(self.history)}
