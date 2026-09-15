"""CriticalActionCircuitBreaker 高危工具调用熔断 (RFC #1 P0)

对高风险操作不直接交由执行层执行: 拦截该调用, 触发异步或同步深度
安全审核 (可升级为人工介入), 确认合规前拒绝下发。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .safety_check import SafetyVerdict, Verdict


@dataclass
class PendingReview:
    """一次被熔断挂起的调用"""

    action: Dict[str, Any]
    verdict: SafetyVerdict
    t: float = field(default_factory=time.time)
    resolved: bool = False
    approved: Optional[bool] = None


class CriticalActionCircuitBreaker:
    """熔断器: REVIEW 判定 → 挂起等待审核; DENY 判定 → 直接拒绝

    审核回调 (reviewer) 决定放行与否; 未提供回调时默认全部拒绝
    (fail-closed, 安全默认)。
    """

    def __init__(self, reviewer: Optional[Callable[[PendingReview], bool]] = None,
                 pending_capacity: int = 128):
        self.reviewer = reviewer
        self.pending_capacity = pending_capacity
        self.pending: List[PendingReview] = []
        self.total_tripped = 0
        self.total_rejected = 0

    def trip(self, action: Dict[str, Any], verdict: SafetyVerdict) -> PendingReview:
        """熔断一条高危调用: 记录挂起并立即拒绝下发, 审核结果另行处理"""
        review = PendingReview(action=action, verdict=verdict)
        self.pending.append(review)
        if len(self.pending) > self.pending_capacity:
            self.pending.pop(0)
        self.total_tripped += 1
        return review

    def resolve(self, review: PendingReview) -> bool:
        """执行审核并回填结论: 无审核者时 fail-closed (拒绝)"""
        review.approved = bool(self.reviewer(review)) if self.reviewer else False
        review.resolved = True
        if not review.approved:
            self.total_rejected += 1
        return review.approved

    def stats(self) -> Dict[str, Any]:
        return {"tripped": self.total_tripped,
                "rejected": self.total_rejected,
                "pending": len(self.pending)}
