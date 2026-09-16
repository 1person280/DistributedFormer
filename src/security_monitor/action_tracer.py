"""ActionTracer 全链路行为审计与溯源 (RFC #1 P1)

对"产生想法 → 执行动作"全流程做结构化日志留存——思考状态、决策依据、
工具参数、执行结果四元组绑定存储; 安全事件发生时支持完整复盘溯源,
满足强监管行业的合规审计需求。

四元组概念:
    思考状态 thinking_state   模型内部决策现场 (想法/意图)
    决策依据 decision_basis   安全判定结论 (Rules/审计理由/风险标签)
    工具参数 tool_params      即将下发的候选动作载荷
    执行结果 execution_result 放行/拦截后的最终结果 (含审核结论)

record() 生成全局唯一 action_id 并绑定四元组; query() 支持多维回溯;
replay() 重建单个动作的完整链路以供事后复盘。
"""

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ActionTrace:
    """一条完整调用链的结构化审计记录 (四元组绑定)"""

    action_id: str
    t: float
    thinking_state: Dict[str, Any] = field(default_factory=dict)
    decision_basis: Dict[str, Any] = field(default_factory=dict)
    tool_params: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ActionTracer:
    """全链路审计存储: 绑定四元组 + 多维查询 + 溯源复盘

    用法 (骨架, 尚未接入 CuteMamenKernel 主路径):
        tracer = ActionTracer()
        aid = tracer.record(
            thinking_state={"step": "call db_write"},
            decision_basis={"verdict": "review", "reasons": [...]},
            tool_params={"tool": "db_write"},
            execution_result={"allowed": False, "outcome": "REVIEW"},
        )
        trace = tracer.replay(aid)          # 完整复盘单条链路
        logs = tracer.query(risk_tag="review")  # 多维回溯审计
    """

    def __init__(self, capacity: int = 512,
                 id_factory=None, clock=None):
        """- capacity: 内存环形容量, 超出后淘汰最旧记录 (落盘为 RFC 开放问题)
        - id_factory/clock: 可注入以支持确定性测试
        """
        self.capacity = capacity
        self._id = id_factory or _new_id
        self._clock = clock or time.time
        self.traces: List[ActionTrace] = []

    def record(self,
               thinking_state: Optional[Dict[str, Any]] = None,
               decision_basis: Optional[Dict[str, Any]] = None,
               tool_params: Optional[Dict[str, Any]] = None,
               execution_result: Optional[Dict[str, Any]] = None,
               action_id: Optional[str] = None,
               tags: Optional[Iterable[str]] = None) -> str:
        """绑定四元组并存储, 返回本次动作的 action_id (为溯源主键)"""
        trace = ActionTrace(
            action_id=action_id or self._id(),
            t=self._clock(),
            thinking_state=dict(thinking_state or {}),
            decision_basis=dict(decision_basis or {}),
            tool_params=dict(tool_params or {}),
            execution_result=dict(execution_result or {}),
            tags=list(tags or []),
        )
        self.traces.append(trace)
        if len(self.traces) > self.capacity:
            self.traces.pop(0)
        return trace.action_id

    # ── 回溯查询 ──────────────────────────────────────────────────
    def query(self,
              action_id: Optional[str] = None,
              tool: Optional[str] = None,
              tag: Optional[str] = None,
              outcome: Optional[str] = None,
              risk_tag: Optional[str] = None,
              since: Optional[float] = None,
              until: Optional[float] = None) -> List[ActionTrace]:
        """多维过滤回溯; 任一维度均视为 AND 条件 (None 表示不过滤)"""
        result: List[ActionTrace] = []
        for tr in self.traces:
            if action_id is not None and tr.action_id != action_id:
                continue
            if tool is not None and tr.tool_params.get("tool") != tool:
                continue
            if tag is not None and tag not in tr.tags:
                continue
            if outcome is not None and tr.execution_result.get("outcome") != outcome:
                continue
            if risk_tag is not None and risk_tag not in tr.decision_basis.get("risk_tags", []):
                continue
            if since is not None and tr.t < since:
                continue
            if until is not None and tr.t > until:
                continue
            result.append(tr)
        return result

    def replay(self, action_id: str) -> Dict[str, Any]:
        """溯源复盘: 返回该动作的四元组完整画像 (未找到则返回空字典)"""
        for tr in self.traces:
            if tr.action_id == action_id:
                return {
                    "action_id": tr.action_id,
                    "t": tr.t,
                    "thinking_state": tr.thinking_state,
                    "decision_basis": tr.decision_basis,
                    "tool_params": tr.tool_params,
                    "execution_result": tr.execution_result,
                    "tags": tr.tags,
                }
        return {}

    def iter_traces(self) -> Iterator[ActionTrace]:
        """按时间顺序遍历全部审计记录 (合规导出用)"""
        return iter(self.traces)

    def export(self) -> List[Dict[str, Any]]:
        """导出结构化的全量审计日志 (逐条四元组字典)"""
        return [{
            "action_id": tr.action_id,
            "t": tr.t,
            "thinking_state": tr.thinking_state,
            "decision_basis": tr.decision_basis,
            "tool_params": tr.tool_params,
            "execution_result": tr.execution_result,
            "tags": tr.tags,
        } for tr in self.traces]

    def stats(self) -> Dict[str, Any]:
        return {"traces": len(self.traces), "capacity": self.capacity}


def _new_id() -> str:
    return uuid.uuid4().hex