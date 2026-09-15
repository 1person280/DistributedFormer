"""SafetyCheck 接口 (RFC #1 P0)

单条候选动作的安全判定接口。Python 侧抽象基类与 Rust 插件层的
`SafetyCheck` trait (见 safety_check.rs) 一一对应:

    trait SafetyCheck {
        fn check(&self, action: &Action) -> Verdict;
    }

执行层插件在处理高危操作前必须先经过某个 SafetyCheck 实例;
判定为 Deny 的动作不允许下发。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(Enum):
    """安全判定结果"""

    ALLOW = "allow"      # 放行: 正常业务动作
    REVIEW = "review"    # 熔断: 高危操作, 需深度审核 (可升级人工) 后再决定
    DENY = "deny"        # 拦截: 命中逃逸倾向, 下发前直接拒绝


@dataclass
class SafetyVerdict:
    """一次安全判定的完整结论

    - verdict: 判定结果 (ALLOW / REVIEW / DENY)
    - reasons: 命中的规则/探针说明 (审计留痕用)
    - risk_tags: 风险标签, 如 privilege-escalation / boundary-probing
    """

    verdict: Verdict
    reasons: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)


class SafetyCheck(ABC):
    """候选动作安全判定接口 (与 Rust 侧 SafetyCheck trait 对齐)"""

    @abstractmethod
    def check(self, action: Dict[str, Any]) -> SafetyVerdict:
        """对一条候选动作 (事件总线上的候选/工具调用参数) 做安全判定"""


# 高危操作模式 (P0 熔断的初始规则集, 详见 RFC #1 开放问题 1)
_HIGH_RISK_PATTERNS = (
    ("modify-system-config", ("write_config", "set_env", "edit_registry")),
    ("access-core-database", ("db_drop", "db_write", "raw_sql")),
    ("download-external-code", ("curl", "wget", "pip_install", "fetch_url")),
)

# 逃逸倾向模式 (P0 意图探针的初始规则集)
_ESCAPE_PATTERNS = (
    ("privilege-escalation", ("sudo", "run_as_admin", "escalate")),
    ("boundary-probing", ("port_scan", "enum_files", "list_secrets")),
)


class RuleBasedSafetyCheck(SafetyCheck):
    """规则版最小实现: 关键词规则先行, 学习型分类器为 RFC 开放问题"""

    def __init__(self,
                 high_risk: Optional[dict] = None,
                 escape: Optional[dict] = None):
        self.high_risk = dict(high_risk or _HIGH_RISK_PATTERNS)
        self.escape = dict(escape or _ESCAPE_PATTERNS)

    def check(self, action: Dict[str, Any]) -> SafetyVerdict:
        tool = str(action.get("tool", action.get("topic", "")))
        tags: List[str] = []

        for tag, verbs in self.escape.items():
            if any(v in tool for v in verbs):
                tags.append(tag)
        if tags:
            return SafetyVerdict(Verdict.DENY,
                                 reasons=[f"命中逃逸倾向: {tool}"],
                                 risk_tags=tags)

        for tag, verbs in self.high_risk.items():
            if any(v in tool for v in verbs):
                tags.append(tag)
        if tags:
            return SafetyVerdict(Verdict.REVIEW,
                                 reasons=[f"高危操作, 触发熔断: {tool}"],
                                 risk_tags=tags)

        return SafetyVerdict(Verdict.ALLOW)
