//! SafetyCheck trait — Rust 执行层侧的安全判定接口 (RFC #1 P0)
//!
//! 与 Python 侧 `distributedformer/security_monitor/safety_check.py`
//! 的 `SafetyCheck` 抽象基类一一对应。执行层插件处理高危操作前
//! 必须先经过某个 SafetyCheck 实现; Verdict::Deny 的动作不允许下发。
//!
//! 注: Rust 执行层尚未落地 (见路线图), 本文件先固化接口契约。

use std::collections::HashMap;

/// 一次安全判定的完整结论 (与 Python 侧 SafetyVerdict 对齐)
pub struct SafetyVerdict {
    pub verdict: Verdict,
    pub reasons: Vec<String>,
    pub risk_tags: Vec<String>,
}

/// 安全判定结果
pub enum Verdict {
    /// 放行: 正常业务动作
    Allow,
    /// 熔断: 高危操作, 深度审核确认前拒绝下发
    Review,
    /// 拦截: 命中逃逸倾向, 直接拒绝
    Deny,
}

/// 候选动作 (事件总线上的候选 / 工具调用参数)
pub struct Action {
    pub tool: String,
    pub args: HashMap<String, String>,
}

/// 候选动作安全判定接口
pub trait SafetyCheck {
    fn check(&self, action: &Action) -> SafetyVerdict;
}
