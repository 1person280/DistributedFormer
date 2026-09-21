# -*- coding: utf-8 -*-
"""
服务端任务队列 (v0.19.0) — ComfyUI 风格的服务端执行模型

把原本纯前端的"入队执行"提升为服务端管理的真实任务队列:
  - POST submit 立即返回全局 task_id, 后台 daemon worker 串行消费
  - 可查询状态/结果/历史 (GET /api/tasks, /api/tasks/<id>)
  - 全局中断 (interrupt) 置位当前任务 stop_event 并清空待处理队列
  - 每个任务历史持久化到 ./cache/tasks/<id>.json, 启动可恢复

线程约定:
  CuteMamenKernel 非线程安全, 因此这里用单一 worker 串行跑 run, 绝不并行;
  所有状态字典读写经 self._cv (threading.Condition) 串行化, 与请求线程互斥。
"""

import copy
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from src.deployment.workflow_ui import WorkflowEngine

# 任务历史落盘目录 (沿用项目统一缓存根 ./cache, 与 WORKFLOW_DIR 同层级)
TASK_DIR = os.path.join("cache", "tasks")

# 状态排序权重: running > pending > ok > err > aborted
_STATE_ORDER = {"running": 0, "pending": 1, "ok": 2, "err": 3, "aborted": 4}


class TaskManager:
    """服务端任务队列: 提交 → 后台串行执行 → 状态/历史可查, 支持中断。"""

    def __init__(self, engine: WorkflowEngine, task_dir: str = TASK_DIR):
        self.engine = engine
        self.task_dir = task_dir
        self._cv = threading.Condition()
        self._tasks: Dict[int, Dict[str, Any]] = {}
        self._queue: List[int] = []
        self._seq = 0
        self._active_stop: Optional[threading.Event] = None  # 当前任务 stop event
        self._recover()
        self._worker = threading.Thread(target=self._loop,
                                        daemon=True, name="task-worker")
        self._worker.start()

    # ── 提交 ───────────────────────────────────────────────
    def submit(self, workflow: Dict[str, Any],
               name: Optional[str] = None) -> Dict[str, Any]:
        with self._cv:
            self._seq += 1
            tid = self._seq
            self._tasks[tid] = {
                "id": tid,
                "name": name or (workflow or {}).get("name") or f"task-{tid}",
                "state": "pending",
                "start": None,
                "ms": None,
                "results": None,
                "log": None,
                "error": None,
                "workflow": copy.deepcopy(workflow or {}),
            }
            self._queue.append(tid)
            self._persist(tid)
            self._cv.notify()
        return {"ok": True, "task_id": tid}

    # ── 查询 ───────────────────────────────────────────────
    def status(self) -> List[Dict[str, Any]]:
        with self._cv:
            items: List[Dict[str, Any]] = []
            for t in self._tasks.values():
                r = dict(t)
                r.pop("workflow", None)  # 列表不带大块 workflow
                items.append(r)
            items.sort(key=lambda t: (_STATE_ORDER.get(t["state"], 9), -t["id"]))
            return items

    def get(self, tid: int) -> Optional[Dict[str, Any]]:
        with self._cv:
            t = self._tasks.get(tid)
            return copy.deepcopy(t) if t else None

    def load_workflow(self, tid: int) -> Optional[Dict[str, Any]]:
        t = self.get(tid)
        return (t or {}).get("workflow")

    # ── 中断 ───────────────────────────────────────────────
    def interrupt(self) -> Dict[str, Any]:
        with self._cv:
            # 清空待处理队列, 标记为 aborted
            for tid in list(self._queue):
                task = self._tasks.get(tid)
                if task and task["state"] == "pending":
                    task["state"] = "aborted"
                    task["error"] = "interrupted"
                self._persist(tid)
            self._queue.clear()
            ev = self._active_stop
        if ev is not None:
            ev.set()  # 终止正在运行的当前任务
        return {"ok": True}

    # ── 后台 worker (单线程串行) ───────────────────────────
    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                tid = self._queue.pop(0)
                task = self._tasks.get(tid)
                if task is None or task["state"] != "pending":
                    continue
                task["state"] = "running"
                task["start"] = time.time()
                stop_event = threading.Event()
                self._active_stop = stop_event
                workflow = task["workflow"]
            try:
                result = self.engine.run(workflow, stop_event=stop_event)
            except Exception as exc:  # noqa: BLE001
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            with self._cv:
                self._active_stop = None
                task["ms"] = round((time.time() - (task["start"] or time.time()))
                                   * 1000, 2)
                task["log"] = result.get("log")
                if result.get("aborted"):
                    task["state"] = "aborted"
                    task["error"] = "interrupted"
                    task["results"] = result.get("results")
                elif result.get("ok", False):
                    task["state"] = "ok"
                    task["results"] = result.get("results")
                else:
                    task["state"] = "err"
                    task["error"] = result.get("error", "运行失败")
                    task["results"] = result.get("results")
                self._persist(tid)

    # ── 持久化 ─────────────────────────────────────────────
    def _persist(self, tid: int) -> None:
        """原子写 TASK_DIR/<id>.json (调用方需持有 self._cv)。"""
        task = self._tasks.get(tid)
        if task is None:
            return
        record = dict(task)
        record["workflow"] = self._sanitize(record.get("workflow"))
        try:
            os.makedirs(self.task_dir, exist_ok=True)
            path = os.path.join(self.task_dir, f"{tid}.json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 持久化失败不致命
            pass

    @staticmethod
    def _sanitize(w: Any) -> Any:
        """去重/瘦身超大 data, 保证可 JSON 化并控制体积。"""
        if not isinstance(w, dict):
            return w
        w = copy.deepcopy(w)
        for n in w.get("nodes", []):
            p = n.get("params") or {}
            d = p.get("data")
            if isinstance(d, (list, tuple)) and len(d) > 256:
                p["data"] = f"[大块数据 {len(d)} 项, 已省略]"
        return w

    def _recover(self) -> None:
        """启动扫描 task_dir, 恢复已完成历史 (丢弃 pending/running)。"""
        if not os.path.isdir(self.task_dir):
            return
        for fn in sorted(os.listdir(self.task_dir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.task_dir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:  # noqa: BLE001 忽略损坏文件
                continue
            if not isinstance(rec, dict):
                continue
            tid = rec.get("id")
            state = rec.get("state")
            if not isinstance(tid, int) or state not in ("ok", "err", "aborted"):
                continue
            rec.setdefault("results", None)
            rec.setdefault("log", None)
            rec.setdefault("error", None)
            rec.setdefault("workflow", None)
            self._tasks[tid] = rec
            self._seq = max(self._seq, tid)