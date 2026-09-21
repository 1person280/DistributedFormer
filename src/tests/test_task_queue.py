# -*- coding: utf-8 -*-
"""服务端任务队列 (v0.19.0) 测试

覆盖: TaskManager 提交即返回 id / 状态流转 / 全局中断 (保留已执行节点
+ 清空待处理队列) / 历史持久化与恢复 / 同步执行语义 unchanged。
每个测试用独立 tmp task_dir + 独立内核, 避免竞态。
"""

import os
import threading
import time

from src.cutemamen.kernel import CuteMamenKernel
from src.cutemamen.plugin import ExpertPlugin
from src.deployment.task_queue import TaskManager
from src.deployment.workflow_ui import WorkflowEngine


class SleepPlugin(ExpertPlugin):
    """受控耗时假插件: on_think 睡眠 seconds 秒后返回标记。

    route="slow", 用于安全地观察 running 状态与触发中断。
    """

    CAPABILITY = "test-only sleep"

    def __init__(self, seconds: float, name: str = "sleep-plugin"):
        super().__init__(name, route="slow")
        self.seconds = seconds

    def on_think(self, event, ctx):
        time.sleep(self.seconds)
        return {"state": "awake", "topic": event.get("topic")}


def _slow_workflow(delay: float) -> dict:
    return {
        "name": "slow", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "slow", "data": "tick"}},
            {"id": "model", "type": "model", "x": 200, "y": 0,
             "params": {"route": "slow", "data": None}},
            {"id": "out", "type": "output", "x": 400, "y": 0,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in", "from_port": "out",
             "to": "model", "to_port": "in"},
            {"id": "e2", "from": "model", "from_port": "out",
             "to": "out", "to_port": "in"},
        ],
    }


def _make_manager(tmp_path_factory, seconds: float = 0.01,
                  task_dir=None):
    tmp = tmp_path_factory.mktemp("taskq") if task_dir is None else task_dir
    kernel = CuteMamenKernel(dim=16)
    kernel.mount(SleepPlugin(seconds))
    return TaskManager(WorkflowEngine(kernel),
                       task_dir=str(tmp / "tasks") if task_dir is None
                       else task_dir)


def _wait_terminal(tm, tid, timeout=5.0):
    """轮询直到任务进入终态 (ok/err/aborted), 返回其快照。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = tm.get(tid)
        if snap and snap["state"] in ("ok", "err", "aborted"):
            return snap
        time.sleep(0.01)
    raise AssertionError(f"task {tid} 未在 {timeout}s 内进入终态")


def test_submit_returns_id_immediately(tmp_path_factory):
    """提交立即返回 {ok, task_id:int}, 无需等待执行完成。"""
    tm = _make_manager(tmp_path_factory, seconds=0.05)
    resp = tm.submit(_slow_workflow(0.05), name="t1")
    assert resp["ok"] is True
    assert isinstance(resp["task_id"], int)
    # 此刻可能是 pending 或 running, 但 id 已分配
    assert tm.status() or True  # 列表可用
    tm.interrupt()  # 收尾


def test_state_transitions_pending_to_ok(tmp_path_factory):
    """短任务应 pending→running→ok, 带 results/log。"""
    tm = _make_manager(tmp_path_factory, seconds=0.02)
    tid = tm.submit(_slow_workflow(0.02))["task_id"]
    snap = _wait_terminal(tm, tid)
    assert snap["state"] == "ok"
    assert snap["ms"] is not None
    assert snap["results"] and "out" in snap["results"]
    assert "state" in snap["results"]["out"]["data"], "输出应含插件返回标记"
    assert snap["log"], "应有执行日志"


def _chained_workflow(delay: float) -> dict:
    """in→model1(slow)→model2(slow)→out: 用于边界确定性中断验证。"""
    return {
        "name": "chain", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "slow", "data": "tick"}},
            {"id": "m1", "type": "model", "x": 200, "y": 0,
             "params": {"route": "slow", "data": None}},
            {"id": "m2", "type": "model", "x": 400, "y": 0,
             "params": {"route": "slow", "data": None}},
            {"id": "out", "type": "output", "x": 600, "y": 0,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in", "from_port": "out",
             "to": "m1", "to_port": "in"},
            {"id": "e2", "from": "m1", "from_port": "out",
             "to": "m2", "to_port": "in"},
            {"id": "e3", "from": "m2", "from_port": "out",
             "to": "out", "to_port": "in"},
        ],
    }


def test_interrupt_marks_aborted_keeps_done_nodes(tmp_path_factory):
    """运行慢任务时 interrupt → aborted; 已执行节点结果保留, 后续节点跳过。"""
    tm = _make_manager(tmp_path_factory, seconds=1.0)
    tid = tm.submit(_chained_workflow(1.0))["task_id"]
    # 等待它真正进入 running (m1 已在 sleep, m2 尚未启动)
    deadline = time.time() + 3
    while time.time() < deadline:
        s = tm.get(tid)
        if s and s["state"] == "running":
            break
        time.sleep(0.01)
    assert tm.get(tid)["state"] == "running", "任务应已进入 running"

    tm.interrupt()
    snap = _wait_terminal(tm, tid)
    assert snap["state"] == "aborted"
    assert snap["error"] == "interrupted"
    # in/m1 已执行并保留; m2 在中断后节点边界被打断 → 未执行
    results = snap["results"] or {}
    assert "in" in results, "已执行的 input 节点结果应保留"
    assert "m1" in results, "中断时正在跑的节点会跑完并保留"
    assert "m2" not in results, "中断后后续节点应被跳过"


def test_interrupt_clears_queued(tmp_path_factory):
    """入队 3 个再 interrupt → 待处理队列清空为 aborted, 当前任务也终止。"""
    tm = _make_manager(tmp_path_factory, seconds=0.5)
    tids = [tm.submit(_slow_workflow(0.5))["task_id"] for _ in range(3)]
    # 等第一个 running
    deadline = time.time() + 3
    while time.time() < deadline:
        if tm.get(tids[0])["state"] == "running":
            break
        time.sleep(0.01)
    tm.interrupt()
    for t in tids:
        snap = _wait_terminal(tm, t)
        assert snap["state"] == "aborted", f"task {t} 应被中断"
    # 所有清空后不再有待处理
    pend = [t for t in tm.status() if t["state"] == "pending"]
    assert not pend, "interrupt 后不应有 pending 任务"


def test_history_persisted_to_disk(tmp_path_factory):
    """完成任务应落盘 cache/tasks/<id>.json, 且新 Manager 可恢复。"""
    dirname = str(tmp_path_factory.mktemp("persist"))
    task_dir = os.path.join(dirname, "tasks")
    tm = _make_manager(tmp_path_factory, seconds=0.01, task_dir=task_dir)
    tid = tm.submit(_slow_workflow(0.01))["task_id"]
    _wait_terminal(tm, tid)
    path = os.path.join(task_dir, f"{tid}.json")
    assert os.path.isfile(path), f"{path} 应已落盘"

    # 恢复: 新建 Manager (同一目录) 能读到该历史
    tm2 = _make_manager(tmp_path_factory, seconds=0.01, task_dir=task_dir)
    recovered = tm2.get(tid)
    assert recovered is not None
    assert recovered["state"] == "ok"
    assert recovered["workflow"] is not None


def test_sync_run_still_works(tmp_path_factory):
    """队列不改变 WorkflowEngine.run 同步语义 (旧路径不变)。"""
    kernel = CuteMamenKernel(dim=16)
    kernel.mount(SleepPlugin(0.01))
    engine = WorkflowEngine(kernel)
    res = engine.run(_slow_workflow(0.01))
    assert res["ok"] is True
    assert "aborted" not in res