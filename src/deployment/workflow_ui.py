# -*- coding: utf-8 -*-
"""
节点图工作流 (v0.19.0) — ComfyUI 式自定义模型工作流

零第三方依赖, 纯手写 vanilla JS/HTML/CSS 前端 + stdlib http.server 后端。
借鉴 ComfyUI 交互: 自由拼搭任意 DAG (输入→模型→输出 可扇出/汇聚/驳接),
双击画布搜索加节点, 节点端口连线, 平移/缩放, 小地图, 保存/加载/内置样例。

Node / Edge 结构:
    workflow = {
      "name": <str>, "version": 1,
      "nodes": [ {"id","type"(input|model|output),"x","y","params": {...}} ],
      "edges": [ {"id","from","from_port","to","to_port"} ]
    }

端口模型 (一致约定: 输出端口 id="out", 输入端口 id="in"):
    input  → 仅输出 "out"
    model  → 输入 "in" + 输出 "out"
    output → 仅输入 "in"

执行 (Kahn 拓扑, 支持任意无环图):
    input 节点产出事件 {topic, data};
    model 节点: data = params.data (非 null) ? 覆盖 : 入边数据;
               无入边且 route 空 → no_output; 否则 kernel.request({topic,data})
               多条入边 → 汇聚为 {"inputs": {源: data}};
               out = {topic, data: 结果}
    output 节点: 单入边取上游 data, 多入边汇聚展示。
"""

import json
import os
import re
import copy
import time
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from src.cutemamen.kernel import CuteMamenKernel

# 工作流落盘目录 (沿用项目统一缓存根 ./cache, 与 DEFAULT_PKG_DIR 同层级)
WORKFLOW_DIR = os.path.join("cache", "workflows")

_VERSION = "0.19.0"

# 名称清洗: 仅保留中英文/数字/下划线/连字符/点, 防路径穿越
_NAME_CLEAN = re.compile(r"[^\w\u4e00-\u9fff._-]+")


def _json_safe(v: Any) -> Any:
    """递归 JSON 安全化; 视频帧(ndarray 数组)过大会拖垮 payload, 统一丢弃"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items() if k != "frames"}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _clean_name(name: str) -> str:
    name = _NAME_CLEAN.sub("_", name or "").strip().strip(".")
    return name or "workflow"


def _upstream(src: str, from_port: Optional[str],
              out: Dict[str, Dict]) -> Any:
    """按端口取上游数据: 优先 out["<src>:<from_port>"], 回退 out[src]"""
    key = f"{src}:{from_port}" if from_port else src
    entry = out.get(key) or out.get(src) or {}
    return entry.get("data")


def _incoming_data(srcs: List[Any], out: Dict[str, Dict]) -> Any:
    """上游数据取法: 无/单/多入边 (srcs 为 (src, from_port, to_port) 三元组)"""
    if not srcs:
        return None
    if len(srcs) == 1:
        s, fp, _tp = srcs[0]
        return _upstream(s, fp, out)
    return {"inputs": {f"{s}:{fp}": _upstream(s, fp, out)
                       for (s, fp, _tp) in srcs}}


# ── 轻量条件谓词求值 (条件模块用) ───────────────────────────
_OP = re.compile(r"^(?P<path>[\w.]+)\s*(?P<op>==|!=|>=|<=|>|<|contains)"
                 r"\s*(?P<val>.+)$")
# 无路径表达式: 直接对整条数据求值, 如 `contains '你'`
_OP2 = re.compile(r"^(?P<op>==|!=|>=|<=|>|<|contains)\s*(?P<val>.+)$")


def _resolve_path(data: Any, path: str) -> Any:
    cur = data
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _eval_cond(data: Any, expr: str) -> bool:
    """对 data 求值一条条件。支持: `path op 字面量`, op ∈ == != >= <= > < contains;
    无运算符则视为路径是否存在且真值。"""
    expr = (expr or "").strip()
    if not expr:
        return False
    m = _OP.match(expr)
    if not m:
        m2 = _OP2.match(expr)
        if not m2:
            return bool(_resolve_path(data, expr))
        path, op, raw = None, m2.group("op"), m2.group("val")
    else:
        path, op, raw = m.group("path"), m.group("op"), m.group("val").strip()
    val = _resolve_path(data, path) if path else data
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        lit, cmp_str = raw[1:-1], True
    else:
        try:
            lit, cmp_str = float(raw), False
        except ValueError:
            lit, cmp_str = raw, True
    if op == "contains":
        return lit in (str(val) if val is not None else "")
    if val is None:
        return (lit in (None, "null", "")) if op == "==" else (op != "==")
    if cmp_str:
        sval = str(val)
        return {"==": sval == str(lit), "!=": sval != str(lit),
                ">": sval > str(lit), ">=": sval >= str(lit),
                "<": sval < str(lit), "<=": sval <= str(lit)}[op]
    try:
        nval = float(val)
    except (TypeError, ValueError):
        return False
    return {"==": nval == lit, "!=": nval != lit, ">": nval > lit,
            ">=": nval >= lit, "<": nval < lit, "<=": nval <= lit}[op]


# ── 结构化模型输出 (类别 + 内容) ───────────────────────────
_SPIKE_KEYS = ("logits", "vector", "spike", "embedding", "prediction", "output")


def _typed_outputs(route: str, res: Any) -> Optional[List[Dict[str, Any]]]:
    """把插件返回归一化为结构化输出列表 [{category, content}]:
    video → frame+text; chat → text; 分类模型 → text+spike(若有); 兜底 → text"""
    if res is None:
        return None
    route = route or ""
    safe = _json_safe(res)
    if "video" in route:
        content = ({k: v for k, v in safe.items() if k != "frames"}
                   if isinstance(safe, dict) else safe)
        summary = content.get("summary") if isinstance(content, dict) else None
        return [{"category": "frame", "content": content},
                {"category": "text", "content": summary}]
    if "chat" in route:
        text = safe.get("reply") if isinstance(safe, dict) else safe
        return [{"category": "text", "content": text}]
    spike = None
    if isinstance(res, dict):
        for k in _SPIKE_KEYS:
            if k in res:
                spike = _json_safe(res[k])
                break
    out = [{"category": "text", "content": safe}]
    if spike is not None:
        out.append({"category": "spike", "content": spike})
    return out


class WorkflowEngine:
    """节点图工作流执行引擎: 枚举模型 + 拓扑执行 + 保存/加载"""

    def __init__(self, kernel: CuteMamenKernel):
        self.kernel = kernel

    # ── 调色板: 内核中的真实模型 (插件) ────────────────────
    def list_models(self) -> List[Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, p in self.kernel.plugins.items():
            out[name] = {
                "name": name,
                "route": p.route,
                "base_model": getattr(type(p), "BASE_MODEL", None),
                "capability": getattr(p, "capability", "")
                              or getattr(type(p), "CAPABILITY", "") or "",
                "loaded": True,
            }
        for name in self.kernel.registry:
            if name not in out:
                out[name] = {"name": name, "route": name,
                             "base_model": None, "capability": "",
                             "loaded": False}
        return sorted(out.values(), key=lambda m: (m["route"] or "", m["name"]))

    # ── 拓扑执行 ───────────────────────────────────────────
    def run(self, workflow: Dict[str, Any],
            stop_event: Optional[threading.Event] = None) -> Dict[str, Any]:
        nodes: Dict[str, Dict] = {n["id"]: n for n in workflow.get("nodes", [])}
        edges: List[Dict] = workflow.get("edges", [])
        if not nodes:
            return {"ok": False, "error": "工作流为空 (没有节点)"}

        inbound: Dict[str, List[Any]] = {nid: [] for nid in nodes}
        out_adj: Dict[str, List[str]] = {nid: [] for nid in nodes}
        indeg: Dict[str, int] = {nid: 0 for nid in nodes}
        for e in edges:
            src, dst = e.get("from"), e.get("to")
            if src not in nodes or dst not in nodes:
                return {"ok": False, "error": f"边 {e.get('id')} 引用了不存在的节点"}
            out_adj[src].append(dst)
            if e.get("to_port") == "in":
                inbound[dst].append((src, e.get("from_port"),
                                     e.get("to_port")))
                indeg[dst] += 1

        queue = [nid for nid, d in indeg.items() if d == 0]
        out: Dict[str, Dict] = {}
        results: Dict[str, Any] = {}
        log: List[Dict[str, Any]] = []
        aborted = False

        while queue:
            if stop_event is not None and stop_event.is_set():
                log.append({"node_id": "_interrupt", "status": "aborted"})
                aborted = True
                break
            nid = queue.pop(0)
            node = nodes[nid]
            ntype = node.get("type")
            params = node.get("params") or {}
            try:
                _run_node(self, nid, ntype, params, inbound, out, results, log)
            except Exception as exc:
                log.append({"node_id": nid, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
                results[nid] = {"type": ntype,
                                "error": f"{type(exc).__name__}: {exc}"}
            for dst in out_adj[nid]:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    queue.append(dst)

        if (not aborted) and len(results) < len(nodes):
            return {"ok": False, "error": "工作流存在环 (无法拓扑排序)"}
        rez: Dict[str, Any] = {"ok": True, "results": results, "log": log}
        if aborted:
            rez["aborted"] = True
        return rez

    # ── 保存 / 加载 / 列出 ─────────────────────────────────
    def save(self, name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        name = _clean_name(name)
        os.makedirs(WORKFLOW_DIR, exist_ok=True)
        with open(os.path.join(WORKFLOW_DIR, f"{name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({**workflow, "name": name}, f,
                      ensure_ascii=False, indent=2)
        return {"ok": True, "name": name}

    def list_workflows(self) -> Dict[str, Any]:
        files: List[Dict[str, Any]] = []
        if os.path.isdir(WORKFLOW_DIR):
            for fn in sorted(os.listdir(WORKFLOW_DIR)):
                if fn.endswith(".json"):
                    files.append({"name": fn[:-5], "is_preset": False})
        presets = [{"name": k, "is_preset": True,
                    "category": (PRESETS[k] or {}).get("category", "")}
                   for k in PRESETS]
        return {"workflows": files, "presets": presets}

    def load(self, name: str) -> Dict[str, Any]:
        clean = _clean_name(name)
        preset = PRESETS.get(name) or PRESETS.get(clean)
        user_path = os.path.join(WORKFLOW_DIR, f"{clean}.json")
        if os.path.isfile(user_path):
            with open(user_path, encoding="utf-8") as f:
                return {"ok": True, "workflow": json.load(f),
                        "is_preset": False}
        if preset is not None:
            return {"ok": True, "workflow": copy.deepcopy(preset),
                    "is_preset": True}
        return {"ok": False, "error": f"工作流 {name!r} 不存在"}


def _run_node(engine: WorkflowEngine, nid: str, ntype: str,
              params: Dict, inbound: Dict[str, List[Any]],
              out: Dict[str, Dict], results: Dict[str, Any],
              log: List[Dict[str, Any]]) -> None:
    """执行单个节点 (被 run 循环调用); 扇出/扇入均支持

    数据取法:
      - 无入边          → 用 params.data (route 空 → no_output)
      - 单条入边        → 复用上游 data (逐节点传递, 保链式语义)
      - 多条入边(扇入)  → 汇聚为 {"inputs": {源节点: data}}

    节点类型:
      - model       通过 kernel.request 路由, 输出加结构化 typed 列表
      - condition   对入边数据按 params.conditions 求值, 满足条件 → 对应
                    端口同步扇出 out["<nid>:<label>"] (多条件满足 → 多路)
      - feedback    SNN 脉冲回传补做: 把入边数据回传给 target 插件重跑
      - output      汇聚所有入边数据展示
    """
    dt = time.time()
    if ntype == "input":
        event = {"topic": params.get("topic", ""),
                 "data": params.get("data")}
        out[nid] = event
        results[nid] = {"type": "input", "topic": event["topic"],
                        "data": _json_safe(event["data"])}
        return
    srcs = inbound.get(nid) or []
    if ntype == "model":
        route = params.get("route") or ""
        data = params.get("data")
        if data is None:
            data = _incoming_data(srcs, out)
        res = engine.kernel.request({"topic": route, "data": data}) if route else None
        ms = round((time.time() - dt) * 1000, 2)
        out[nid] = {"topic": route, "data": res if res is not None else None}
        results[nid] = {"type": "model", "route": route,
                        "output": _json_safe(res),
                        "typed": _typed_outputs(route, res)}
        log.append({"node_id": nid,
                    "status": "ok" if res is not None else "no_output",
                    "ms": ms})
        return
    if ntype == "condition":
        conds = params.get("conditions") or []
        data = _incoming_data(srcs, out)
        matched = {}
        for c in conds:
            label = (c.get("label") or "").strip()
            expr = (c.get("match") or "").strip()
            if label and expr and _eval_cond(data, expr):
                matched[label] = data
        # 仅匹配的分支才有输出; 未匹配分支无端口条目 → 下游读不到数据
        for label, d in matched.items():
            out[f"{nid}:{label}"] = {"topic": "condition",
                                     "data": d, "port": label}
        ms = round((time.time() - dt) * 1000, 2)
        results[nid] = {"type": "condition", "matched": list(matched),
                        "output": {k: _json_safe(v)
                                   for k, v in matched.items()}}
        log.append({"node_id": nid,
                    "status": "ok" if matched else "no_output",
                    "ms": ms})
        return
    if ntype == "feedback":
        target = params.get("target") or params.get("route") or ""
        data = _incoming_data(srcs, out)
        res = engine.kernel.feedback(data, target) if target else None
        ms = round((time.time() - dt) * 1000, 2)
        out[nid] = {"topic": "feedback",
                    "data": res if res is not None else None}
        results[nid] = {"type": "feedback", "target": target,
                        "output": _json_safe(res)}
        log.append({"node_id": nid,
                    "status": "ok" if res is not None else "no_output",
                    "ms": ms})
        return
    # reroute/线束: 汇聚多条入边成有序队列 (按拓扑序) 输出, 供下游排队消费
    if ntype == "reroute":
        queue = []
        for (s, fp, _tp) in srcs:
            d = _upstream(s, fp, out)
            if d is not None:
                queue.append(d)
        out[nid] = {"topic": "reroute", "data": queue}
        results[nid] = {"type": "reroute", "data": _json_safe(queue)}
        ms = round((time.time() - dt) * 1000, 2)
        log.append({"node_id": nid,
                    "status": "ok" if queue else "no_output",
                    "ms": ms})
        return
    # output: 汇聚所有入边数据展示
    if len(srcs) == 1:
        s, fp, _tp = srcs[0]
        up = _upstream(s, fp, out)
    elif srcs:
        up = {"inputs": {f"{s}:{fp}": _upstream(s, fp, out)
                         for (s, fp, _tp) in srcs}}
    else:
        up = None
    out[nid] = {"topic": "", "data": up}
    results[nid] = {"type": "output", "data": _json_safe(up)}


# ── 内置样例工作流 (只读预设, 纯真实模型, 可按 category 分类浏览) ──
PRESETS: Dict[str, Dict[str, Any]] = {
    "Rust 代码分类": {
        "name": "Rust 代码分类", "version": 1, "category": "文本",
        "nodes": [
            {"id": "in1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "rust",
                        "data": "fn main() { let s = String::from(\"a\"); println!(\"{}\", s); }"}},
            {"id": "model", "type": "model", "x": 340, "y": 80,
             "params": {"route": "rust", "data": None}},
            {"id": "out", "type": "output", "x": 640, "y": 80,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in1", "from_port": "out",
             "to": "model", "to_port": "in"},
            {"id": "e2", "from": "model", "from_port": "out",
             "to": "out", "to_port": "in"},
        ],
    },
    "Rust 双路扇出": {
        "name": "Rust 双路扇出", "version": 1, "category": "文本",
        "nodes": [
            {"id": "in1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "rust",
                        "data": "fn bad() { let x = 1 }"}},
            {"id": "m1", "type": "model", "x": 340, "y": 40,
             "params": {"route": "rust", "data": None}},
            {"id": "m2", "type": "model", "x": 340, "y": 200,
             "params": {"route": "rust", "data": "fn x(){}"}},
            {"id": "out1", "type": "output", "x": 640, "y": 40,
             "params": {}},
            {"id": "out2", "type": "output", "x": 640, "y": 200,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in1", "from_port": "out",
             "to": "m1", "to_port": "in"},
            {"id": "e2", "from": "in1", "from_port": "out",
             "to": "m2", "to_port": "in"},
            {"id": "e3", "from": "m1", "from_port": "out",
             "to": "out1", "to_port": "in"},
            {"id": "e4", "from": "m2", "from_port": "out",
             "to": "out2", "to_port": "in"},
        ],
    },
    "条件双路扇出": {
        "name": "条件双路扇出", "version": 1, "category": "文本",
        "nodes": [
            {"id": "in1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "chat", "data": "你好世界"}},
            {"id": "model", "type": "model", "x": 300, "y": 80,
             "params": {"route": "chat", "data": None}},
            {"id": "cond", "type": "condition", "x": 560, "y": 80,
             "params": {"conditions": [
                 {"label": "A", "match": "reply contains '你'"},
                 {"label": "B", "match": "reply contains '好'"},
                 {"label": "C", "match": "reply contains '不存在'"}]}},
            {"id": "outA", "type": "output", "x": 820, "y": -40,
             "params": {}},
            {"id": "outB", "type": "output", "x": 820, "y": 80,
             "params": {}},
            {"id": "outC", "type": "output", "x": 820, "y": 200,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in1", "from_port": "out",
             "to": "model", "to_port": "in"},
            {"id": "e2", "from": "model", "from_port": "out",
             "to": "cond", "to_port": "in"},
            {"id": "e3", "from": "cond", "from_port": "A",
             "to": "outA", "to_port": "in"},
            {"id": "e4", "from": "cond", "from_port": "B",
             "to": "outB", "to_port": "in"},
            {"id": "e5", "from": "cond", "from_port": "C",
             "to": "outC", "to_port": "in"},
        ],
    },
    "条件直测（无模型）": {
        "name": "条件直测（无模型）", "version": 1, "category": "文本",
        "nodes": [
            {"id": "in1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "raw", "data": "你好世界"}},
            {"id": "cond", "type": "condition", "x": 320, "y": 80,
             "params": {"conditions": [
                 {"label": "A", "match": "contains '你'"},
                 {"label": "B", "match": "contains '好'"},
                 {"label": "C", "match": "contains '不存在'"}]}},
            {"id": "outA", "type": "output", "x": 600, "y": -40,
             "params": {}},
            {"id": "outB", "type": "output", "x": 600, "y": 80,
             "params": {}},
            {"id": "outC", "type": "output", "x": 600, "y": 200,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in1", "from_port": "out",
             "to": "cond", "to_port": "in"},
            {"id": "e2", "from": "cond", "from_port": "A",
             "to": "outA", "to_port": "in"},
            {"id": "e3", "from": "cond", "from_port": "B",
             "to": "outB", "to_port": "in"},
            {"id": "e4", "from": "cond", "from_port": "C",
             "to": "outC", "to_port": "in"},
        ],
    },
    "Video 渲染": {
        "name": "Video 渲染", "version": 1, "category": "视频",
        "nodes": [
            {"id": "in1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "video",
                        "data": {
                            "keyframes": [
                                [[[255, 0, 0]] * 8 for _ in range(8)],
                                [[[0, 0, 255]] * 8 for _ in range(8)],
                            ],
                            "shots": [
                                {"motion": "pan_left", "duration_s": 0.5},
                                {"motion": "zoom_in", "duration_s": 0.5},
                            ],
                            "fps": 3,
                        }}},
            {"id": "model", "type": "model", "x": 340, "y": 80,
             "params": {"route": "video", "data": None}},
            {"id": "out", "type": "output", "x": 640, "y": 80,
             "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in1", "from_port": "out",
             "to": "model", "to_port": "in"},
            {"id": "e2", "from": "model", "from_port": "out",
             "to": "out", "to_port": "in"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# 前端: ComfyUI 式节点图编辑器 (纯手写 vanilla JS)
# ═══════════════════════════════════════════════════════════════
WORKFLOW_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><title>DistributedFormer Workflow</title>
<style>
:root{--bg:#1b1b1b;--panel:#242424;--node:#282828;--node-h:#343434;
  --line:#3c3c3c;--fg:#dcdcdc;--mut:#8f8f8f;--acc:#6bc3ff;--ok:#35d07f;
  --warn:#ffb454;--err:#ff6b6b;--blue:#3b5bff;--green:#3a9d5d;--amber:#c99a2e}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:13px/1.5 system-ui,'Segoe UI',Roboto,sans-serif;
  background:var(--bg);color:var(--fg);overflow:hidden}
/* ── 工具栏 ── */
.toolbar{height:46px;background:var(--panel);border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:8px;padding:0 12px;flex-shrink:0}
.brand{display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px}
.brand .logo{color:var(--blue);font-size:18px}
.brand .sub{color:var(--mut);font-weight:400;font-size:12px}
.spacer{flex:1}
.toolbar input{border:0;background:#1b1b1b;border:1px solid var(--line);color:var(--fg);
  padding:5px 9px;border-radius:4px;font:inherit;width:160px}
.toolbar select{border:0;background:#1b1b1b;border:1px solid var(--line);color:var(--fg);
  padding:5px;border-radius:4px;width:150px}
.btn{border:0;border-radius:4px;padding:6px 13px;font:inherit;cursor:pointer;color:#fff;
  background:#3b3b3b;border:1px solid #464646}
.btn:hover{background:#484848}
.btn.run{background:var(--blue)}.btn.run:hover{background:#4b68ff}
.btn.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
.btn.ghost:hover{border-color:var(--acc);color:#fff}
.btn:disabled{opacity:.5;cursor:default}
/* ── 布局 ── */
#stage{display:flex;height:calc(100% - 46px)}
#canvas{flex:1;position:relative;overflow:hidden;cursor:grab;
  background:radial-gradient(circle,#2b2b2b 1px,transparent 1.4px);
  background-size:22px 22px}
#canvas.dragging{cursor:grabbing}
#canvas.dragging,#canvas.dragging *{user-select:none;-webkit-user-select:none}
#world{position:absolute;left:0;top:0;transform-origin:0 0}
#world svg{position:absolute;left:0;top:0;width:200000px;height:200000px;
  overflow:visible;pointer-events:none}
.edge{fill:none;stroke:#7a7a7a;stroke-width:2;pointer-events:stroke;cursor:pointer}
.edge:hover{stroke:var(--acc)}
.edge.active{stroke:#fff;stroke-width:3}
.edge.ghost{stroke:var(--warn);stroke-dasharray:6 4;stroke-width:2}
.edge-hit{fill:none;stroke:transparent;stroke-width:16;pointer-events:stroke;cursor:pointer}
.edge-hit:hover{stroke:rgba(107,195,255,.06)}
/* ── 节点 ── */
.node{position:absolute;min-width:190px;max-width:250px;background:var(--node);
  border:1px solid #444;border-radius:7px;box-shadow:0 3px 10px rgba(0,0,0,.45);
  user-select:none;z-index:1}
.node.hiddennode{display:none}
.node.selected{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc),0 4px 14px rgba(0,0,0,.5)}
.node-head{padding:6px 10px;font-size:12px;font-weight:600;color:#fff;cursor:grab;
  border-radius:6px 6px 0 0;display:flex;justify-content:space-between}
.node-head.dragging{cursor:grabbing}
.node-head .del{border:0;background:transparent;color:rgba(255,255,255,.7);
  cursor:pointer;font-size:14px;padding:0 2px;line-height:1}
.node-head .del:hover{color:#fff}
.node-body{padding:7px 10px;font-size:12px;color:var(--mut)}
.node-body .route{color:var(--acc);font-weight:600}
.node-body pre{white-space:pre-wrap;word-break:break-all;max-height:110px;overflow:auto;
  background:#1b1b1b;border:1px solid #3a3a3a;border-radius:4px;padding:5px;font-size:11px;
  color:#b5e0b5;margin:4px 0}
.node-body .res{display:none;border-color:var(--line)}
.node.done .res{display:block}
.node-body .res .tbadge{display:inline-block;background:#2a3350;color:#cfd9ff;
  border:1px solid var(--line);border-radius:10px;padding:0 8px;font-size:11px;
  margin:2px 4px 2px 0;vertical-align:middle}
.node-body .res code{color:#ffd7a1}
/* 端口 socket */
.socket{position:absolute;width:15px;height:15px;border-radius:50%;
  border:2px solid var(--node);cursor:crosshair;left:0;right:0;margin:auto}
.socket.in{background:var(--green);left:-9px}
.socket.out{background:var(--blue);right:-9px}
.socket:hover{transform:scale(1.2)}
/* 具名端口区 (多端口按名路由) */
.node-ports{position:relative;background:rgba(255,255,255,.02);padding:2px 16px}
.node-ports .pr{position:relative;height:24px;display:flex;align-items:center;
  justify-content:space-between}
.node-ports .p{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--mut);
  min-width:0;cursor:crosshair;line-height:1}
.node-ports .p.in{justify-content:flex-start}
.node-ports .p.out{justify-content:flex-end;text-align:right}
.node-ports .p .dot{width:13px;height:13px;border-radius:50%;border:2px solid var(--node);
  flex-shrink:0;box-sizing:border-box;transition:transform .1s}
.node-ports .p .lbl{user-select:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.node-ports .p.in .dot{background:var(--green);margin-left:-5px}
.node-ports .p.out .dot{background:var(--blue);margin-right:-5px}
.node-ports .p:hover .dot{transform:scale(1.25)}
.node-ports .p.in:hover .lbl{color:var(--ok)}
.node-ports .p.out:hover .lbl{color:var(--acc)}
/* ── 状态栏 / 信息条 ── */
#statusBar{position:absolute;left:0;right:0;bottom:0;height:24px;padding:3px 10px;
  background:rgba(15,15,15,.85);color:var(--mut);font-size:11px;display:flex;gap:18px;
  align-items:center;border-top:1px solid var(--line);z-index:5;user-select:none}
#statusBar .s-ok{color:var(--ok)}#statusBar .s-err{color:var(--err)}
/* ── 鼠标工具切换 (顶部工具条) ── */
#toolToggle{margin-left:14px;display:flex;align-items:center;background:#1b1b1b;
  border:1px solid var(--line);border-radius:6px;overflow:hidden;flex-shrink:0}
#toolToggle button{background:transparent;color:var(--mut);border:0;padding:5px 10px;
  font-size:11px;cursor:pointer;line-height:1;display:flex;align-items:center;gap:4px}
#toolToggle button:hover{color:var(--fg)}
#toolToggle button.active{background:var(--acc);color:#fff}
#toolToggle .tl-ico{font-size:13px}
#canvas.tool-drag{cursor:grab}
#canvas.tool-drag.dragging{cursor:grabbing}
/* ── 小地图 ── */
#minimap{position:absolute;right:12px;bottom:36px;width:170px;height:130px;
  background:rgba(20,20,20,.85);border:1px solid var(--line);border-radius:6px;z-index:4;
  cursor:pointer}
#minimap canvas{width:100%;height:100%;display:block}
#mmLabel{position:absolute;left:8px;top:4px;font-size:9px;color:var(--mut)}
/* ── 右侧面板 ── */
#inspector{width:300px;background:var(--panel);border-left:1px solid var(--line);
  padding:12px 14px;overflow:auto;flex-shrink:0}
#inspector h3{margin:16px 0 8px;font-size:12px;color:var(--mut);
  border-bottom:1px solid var(--line);padding-bottom:4px}
#inspector label{display:block;font-size:11px;color:var(--mut);margin:8px 0 2px}
#inspector textarea{width:100%;min-height:64px;resize:vertical;background:#1b1b1b;
  border:1px solid var(--line);color:var(--fg);border-radius:4px;padding:6px;font:inherit}
#inspector input,#inspector select{width:100%;background:#1b1b1b;border:1px solid var(--line);
  color:var(--fg);border-radius:4px;padding:5px 7px;font:inherit;margin-bottom:2px}
#log{font-size:11px}
#log div{padding:2px 0;border-bottom:1px dashed #383838}
#log .ok{color:var(--ok)}#log .no_output{color:var(--warn)}#log .error{color:var(--err)}
.hint{color:var(--mut);font-size:11px}
/* ── 添加节点浮层 ── */
#addOverlay{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);
  width:380px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
  box-shadow:0 8px 30px rgba(0,0,0,.6);z-index:20;display:none}
#addOverlay.show{display:block}
#addOverlay input{width:100%;border:0;border-bottom:1px solid var(--line);
  background:transparent;color:var(--fg);padding:10px 12px;border-radius:8px 8px 0 0;font:inherit}
#addList{max-height:300px;overflow:auto;padding:6px}
.add-item{padding:7px 10px;border-radius:5px;cursor:pointer;display:flex;
  justify-content:space-between;gap:8px}
.add-item:hover{background:#3a3a3a}
.add-item small{color:var(--mut);font-size:11px}
.add-group{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;
  color:var(--acc);padding:8px 8px 3px;cursor:pointer;user-select:none}
.add-group .caret{transition:transform .12s;font-size:9px}
.add-group.open .caret{transform:rotate(90deg)}
.add-group small{color:var(--mut);font-weight:400;margin-left:auto}
.add-group-items{padding-left:6px}
.add-group.closed .add-group-items{display:none}
/* ── 框选 ── */
#rb{position:absolute;border:1px solid var(--acc);background:rgba(107,195,255,.10);
  display:none;pointer-events:none;z-index:2}
/* ── 右键菜单 ── */
.ctx-menu{position:fixed;min-width:190px;background:#2b2b2b;border:1px solid #464646;
  border-radius:6px;box-shadow:0 6px 22px rgba(0,0,0,.55);z-index:60;padding:4px;display:none}
.ctx-menu.show{display:block}
.ctx-menu .i{display:flex;gap:10px;align-items:center;padding:6px 12px;border-radius:4px;
  cursor:pointer;font-size:12px;color:var(--fg)}
.ctx-menu .i:hover{background:#3a3a3a}
.ctx-menu .i .k{margin-left:auto;color:var(--mut);font-size:10px}
.ctx-menu .i.danger:hover{background:#4a2424;color:#ff9b9b}
.ctx-menu .sep{height:1px;background:var(--line);margin:4px 6px}
/* ── 节点: 折叠 / 标题 ── */
.node .pt{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.node.collapsed .node-body{display:none}
.respre{background:#1b1b1b;border:1px solid #3a3a3a;border-radius:4px;padding:5px;
  font-size:11px;color:#8fc7ff;max-height:220px;overflow:auto;white-space:pre;margin-top:2px}
.toolbar .undo[disabled]{opacity:.35;cursor:default}
/* ══ 左侧边栏 (ComfyUI 式浮动侧栏) ══ */
#stage{position:relative}
#canvas{position:relative}
#sidebar{position:absolute;left:0;top:0;bottom:0;width:236px;z-index:10;
  background:rgba(30,30,30,.96);border-right:1px solid var(--line);
  box-shadow:3px 0 14px rgba(0,0,0,.35);display:flex;flex-direction:column}
#sidebar.hidden{display:none}
#sbTop{padding:10px 12px;border-bottom:1px solid var(--line)}
#sbTop h4{margin:0 0 8px;font-size:10px;color:var(--mut);letter-spacing:1px;
  text-transform:uppercase}
#sbTop .row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
#sbTop .row:last-child{margin-bottom:0}
#sbTop .row .q-lbl{font-size:12px;color:var(--mut)}
#queueBadge{display:inline-flex;align-items:center;gap:5px;min-width:26px;
  background:#191919;border:1px solid var(--line);border-radius:5px;padding:2px 6px;
  font-size:12px;font-weight:700}
#queueBadge.running{color:var(--warn)}
#queueBadge .qdot{width:7px;height:7px;border-radius:50%;background:var(--warn)}
#sbTabs{display:flex;padding:6px 8px;gap:4px;border-bottom:1px solid var(--line)}
#sbTabs .tb{flex:1;text-align:center;padding:6px 2px;border-radius:5px;cursor:pointer;
  font-size:11px;color:var(--mut)}
#sbTabs .tb:hover{background:#333}
#sbTabs .tb.active{background:var(--node-h);color:#fff}
#sbPanel{flex:1;overflow:auto;padding:8px}
#sbPanel .title{font-size:10px;color:var(--mut);margin:8px 0 4px;font-weight:600;
  letter-spacing:.5px;text-transform:uppercase}
.sb-item{display:flex;justify-content:space-between;align-items:center;gap:8px;
  padding:6px 8px;border-radius:5px;cursor:pointer;font-size:12px;border:1px solid transparent}
.sb-item:hover{background:#333}
.sb-item.sel{background:#2a2a2a;border-color:var(--acc)}
.sb-item small{color:var(--mut);font-size:11px}
.sb-item .st{font-size:10px;font-weight:700}
.sb-item .st.pending{color:var(--warn)}.sb-item .st.ok{color:var(--ok)}.sb-item .st.err{color:var(--err)}
.sb-item .st.run{color:var(--acc)}
.cat-head{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:var(--acc);
  padding:5px 6px;margin-top:2px;cursor:pointer;user-select:none}
.cat-head small{color:var(--mut);font-weight:400;margin-left:auto}
.cat-body{margin-left:4px}
.sb-toggle{width:26px;height:26px;border-radius:5px;background:#2f2f2f;border:1px solid var(--line);
  color:var(--mut);cursor:pointer;font-size:14px;line-height:1;padding:0}
.sb-toggle:hover{color:#fff;border-color:var(--acc)}
#sbBar{position:absolute;left:8px;top:10px;z-index:11}
#sidebar.hidden+#sbBar{display:block}
#sbBar{display:none}
/* ══ 节点内联控件 (ComfyUI widget) ══ */
.node-widget{border-top:1px solid rgba(255,255,255,.07);padding:5px 9px;font-size:11px}
.node-widget label{display:flex;justify-content:space-between;align-items:center;gap:6px;
  color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.3px;margin-bottom:3px}
.node-widget input[type=text],.node-widget input[type=number],.node-widget select,
.node-widget textarea{width:100%;background:#1d1d1d;border:1px solid #3a3a3a;color:var(--fg);
  border-radius:4px;padding:3px 6px;font:inherit;font-size:11px;box-sizing:border-box}
.node-widget textarea{min-height:26px;resize:vertical;white-space:pre-wrap;display:block}
.node-widget select{cursor:pointer}
.node-widget .w-row{display:flex;gap:6px;align-items:center}
.node-widget input[type=range]{flex:1;accent-color:var(--acc)}
.node-widget .val{color:var(--acc);font-variant-numeric:tabular-nums;min-width:34px;text-align:right}
.node-widget .lock{width:16px;height:16px;border-radius:4px;border:1px solid var(--line);
  cursor:pointer;font-size:10px;display:flex;align-items:center;justify-content:center;
  background:#1d1d1d;color:var(--mut)}
.node-widget .lock.on{background:var(--acc);color:#000;border-color:var(--acc)}
/* ══ 节点状态 (成功/失败/执行中) ══ */
.node.status-ok{border-color:var(--ok)}
.node.status-err{border-color:var(--err)}
.node.status-run{outline:2px solid var(--warn);outline-offset:2px}
/* 旁通 / 禁用 */
.node.bypass{opacity:.4}
.node.bypass .node-head,.node.bypass .dot{filter:saturate(.3) grayscale(.4)}
.node.muted .node-head .pt{text-decoration:line-through;opacity:.6}
/* 节点标题类别色 + 状态圆点徽章 */
.node-head .cat{width:8px;height:8px;border-radius:2px;display:inline-block;margin-right:6px}
.node-head .runtime{margin-left:auto;font-size:10px;font-weight:400;opacity:.9;
  white-space:nowrap}
/* 备注 subtitle */
.node-body .note{font-size:10px;color:var(--mut);font-style:italic;margin:2px 0 4px}
/* ══ 组 (选区组) ══ */
.group{position:absolute;border:1px dashed rgba(120,160,255,.55);border-radius:9px;
  background:rgba(120,160,255,.05);pointer-events:none;z-index:0}
.group>div{position:absolute;left:8px;top:6px;font-size:10px;color:#bcd0ff;pointer-events:auto;
  background:rgba(30,30,30,.85);padding:1px 8px;border-radius:8px;cursor:grab;border:1px solid rgba(120,160,255,.35)}
.group.fold .foldnodes{display:none}
.group.muted{opacity:.4}
.c-swatches{display:flex;gap:5px;padding:4px 8px}
.c-swatches span{width:14px;height:14px;border-radius:50%;cursor:pointer;border:1px solid rgba(255,255,255,.25)}
.c-swatches span:hover{transform:scale(1.25)}
/* ══ 选中浮动操作条 (点选—操作闭环) ══ */
#selBar{position:absolute;display:none;align-items:center;gap:3px;padding:3px 6px;
  background:rgba(24,24,24,.96);border:1px solid var(--acc);border-radius:8px;
  box-shadow:0 4px 14px rgba(0,0,0,.5);z-index:20;transform:translateX(-50%) translateY(-100%)}
#selBar.show{display:flex}
#selBar button{border:none;background:transparent;color:#cfe3ff;font-size:11px;cursor:pointer;
  padding:3px 7px;border-radius:5px;white-space:nowrap}
#selBar button:hover{background:rgba(107,195,255,.18)}
#selBar button.danger:hover{background:rgba(255,80,80,.22);color:#ff9a9a}
#selBar .selbar-count{font-size:11px;color:#ffd76a;padding:2px 6px;white-space:nowrap}
/* ══ 拖放高亮 ══ */
#canvas.dragdrop{outline:2px solid var(--acc);outline-offset:-4px}
#dropHud{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  z-index:30;background:rgba(27,27,27,.55);backdrop-filter:blur(2px);pointer-events:none}
#dropHud.show{display:flex}
#dropHud .box{border:2px dashed var(--acc);border-radius:12px;padding:26px 42px;color:#fff;
  background:rgba(40,40,40,.9);font-size:15px;text-align:center}
#dropHud .box small{display:block;color:var(--mut);font-size:11px;margin-top:6px}
/* 左侧tab HUD 收起后展开按钮置顶 */
.sb-inline{width:34px;height:34px;background:#2f2f2f;border:1px solid var(--line);border-radius:6px;
  color:var(--acc);font-size:16px;cursor:pointer;position:absolute;left:8px;top:10px;z-index:11}
</style></head>
<body>
<header class="toolbar">
  <button id="sbToggle" class="sb-toggle" title="侧边栏 (Ctrl+B)">☰</button>
  <div class="brand"><span class="logo">◈</span>DistributedFormer<span class="sub">workflow</span></div>
  <a href="/" class="btn wf-back" title="返回控制台 (Type here to command)" style="margin-left:6px">⌂ 控制台</a>
  <div id="toolToggle" title="鼠标工具: 框选 / 拖拽画布">
    <button class="tl active" data-t="select"><span class="tl-ico">☐</span>框选</button>
    <button class="tl" data-t="drag"><span class="tl-ico">✥</span>拖拽</button>
  </div>
  <div class="spacer"></div>
  <input id="wfName" value="my_graph" placeholder="工作流名称">
  <button id="btnRun" class="btn run" title="运行 (Ctrl+Enter)">▶ Run</button>
  <button id="btnSave" class="btn" title="保存 (Ctrl+S)">Save</button>
  <button id="btnSaveAs" class="btn" title="另存为 JSON">Save As</button>
  <select id="loadSel"></select>
  <button id="btnLoad" class="btn" title="加载所选">Load</button>
  <button id="btnImport" class="btn" title="导入 JSON">⬆ Import</button>
  <button id="btnExport" class="btn" title="导出 JSON (Ctrl+Shift+S)">⬇ Export</button>
  <button id="btnUM" class="btn undo" disabled title="撤销 (Ctrl+Z)">↩</button>
  <button id="btnRM" class="btn undo" disabled title="重做 (Ctrl+Y)">↪</button>
  <button id="btnClear" class="btn ghost">✕ Clear</button>
</header>
<div id="stage">
  <div id="canvas">
    <div id="world"><svg id="edges"></svg></div>
    <aside id="sidebar">
      <div id="sbTop">
        <h4>Generate</h4>
        <div class="row"><span class="q-lbl">队列</span><span id="queueBadge" class=""><span class="qdot"></span><span id="queueCnt">0</span></span></div>
        <div class="row">
          <button id="sbInterrupt" class="btn ghost" title="中断运行 (Esc)">■ Interrupt</button>
          <button id="sbClearQueue" class="btn ghost" title="清空队列">✕ Queue</button>
        </div>
      </div>
      <div id="sbTabs">
        <div class="tb active" data-t="queue">Queue</div>
        <div class="tb" data-t="load">Load</div>
        <div class="tb" data-t="explore">Explorer</div>
      </div>
      <div id="sbPanel"></div>
    </aside>
    <button id="sbBar" class="sb-inline" title="展开侧边栏">☰</button>
    <div id="dropHud"><div class="box">放开以加载工作流/模型文件<small>支持 .json 工作流 或 内核模型 route 文件</small></div></div>
    <div id="statusBar"><span id="stCount">0 节点 · 0 边</span><span id="stZoom"></span><span id="stMsg" class="hint">双击画布搜索添加节点 · 连接端口运行 · 框选/多选/拖拽</span></div>
    <div id="selBar"></div>
    <div id="minimap"><canvas id="mmCv"></canvas><span id="mmLabel">MAP</span></div>
  </div>
  <aside id="inspector">
    <h3>属性</h3><div id="prop" class="hint">点击画布中的节点以编辑属性</div>
    <h3>运行日志</h3><div id="log"><small>尚未运行</small></div>
  </aside>
</div>
<div id="addOverlay"><input id="addSearch" placeholder="搜索节点… (Esc 关闭)">
  <div id="addList"></div></div>
<div class="ctx-menu" id="ctxMenu"><div id="ctxItems"></div></div>
<input type="file" id="fileIn" accept=".json,application/json" style="display:none">
<input type="file" id="fileData" accept=".json,.txt,.md,.csv,.log,application/json,text/plain" style="display:none">

<script>
'use strict';
const $=id=>document.getElementById(id);
let MODELS=[];                                    // 后端模型清单
let nodes=[], edges=[], nidSeq=0, eidSeq=0;
let selection=null, selEdge=null, drag=null;      // 选择/拖拽状态
const view={ox:40,oy:40,scale:1};                 // 画布平移/缩放
const selSet=new Set();                           // 多选节点 id 集合
let clip=null;                                    // 复制/粘贴剪贴板
const hist={past:[],future:[],max:50};            // 撤销/重做栈
let spaceHeld=false;                              // 空格键按住(平移)
let tool='select';                                // 鼠标工具: select=框选 / drag=拖拽画布
let groups=[];                                    // 选区组 {id,name,nodes:[],color,fold,muted}
const grpSeq={n:0};                               // 组 id 计数器
let q=[], qCounter=1, qCurrent=null;              // 执行队列 (客户端)
let running=false, runAbort=false, runSeq=0;      // 运行状态 / 递增取消令牌
let sidebarCollapsed=false, sbTabEt='queue';      // 侧栏状态
const CAT_COLOR={input:'#3a9d5d',model:'#3b5bff',output:'#c99a2e',reroute:'#8f8f8f',condition:'#b26bf5',feedback:'#e0596f'};
const NODE_W=210, SOCK_Y=28;                       // 节点宽, socket 世界Y(标题区)
const TYPES={input:{title:'输入',color:'#3a9d5d',ins:[],outs:['数据','元信息']},
             model:{title:'模型',color:'var(--blue)',ins:['输入','配置'],outs:['脉冲 spike','文本 text','帧 frame','音频 audio']},
             condition:{title:'条件',color:'#b26bf5',ins:['输入'],outs:['A','B','C']},
             feedback:{title:'回传',color:'#e0596f',ins:['输入'],outs:['结果']},
             output:{title:'输出',color:'#c99a2e',ins:['数据','脉冲','损失'],outs:[]},
             reroute:{title:'线束',color:'#8f8f8f',ins:['输入1','输入2'],outs:['队列']}};
const DEF={input:{topic:'',data:'fn main(){}'},
           model:{route:'',data:null},
           condition:{conditions:[{label:'A',match:''},{label:'B',match:''}]},
           feedback:{target:'',data:null},
           output:{},
           reroute:{ins:['输入1','输入2']}};
const world=$('world'), canvas=$('canvas');
const rbEl=document.createElement('div');rbEl.id='rb';world.appendChild(rbEl);
const ctxMenu=$('ctxMenu'), ctxItems=$('ctxItems');

// ── 坐标换算 ─────────────────────────────
const rect=()=>canvas.getBoundingClientRect();
function worldFrom(clientX,clientY){const r=rect();return{
  x:(clientX-r.left-view.ox)/view.scale, y:(clientY-r.top-view.oy)/view.scale};}
function applyView(){world.style.transform=`translate(${view.ox}px,${view.oy}px) scale(${view.scale})`;$('stZoom').textContent=Math.round(view.scale*100)+'%';}
// socket 世界坐标 (svg 与节点同层世界系, 直接按屏幕反算)
function sockWorld(el){const r=el.getBoundingClientRect(),c=rect();
  const sx=r.left+r.width/2-c.left, sy=r.top+r.height/2-c.top;
  return [(sx-view.ox)/view.scale,(sy-view.oy)/view.scale];}

// ── 拖拽流水线 (rAF 合帧, 避免每帧重建连线/重绘小地图导致的卡顿) ──
let _mmQ=false,_edQ=false;
function scheduleMap(){if(_mmQ)return;_mmQ=true;requestAnimationFrame(()=>{_mmQ=false;updateMap();});}
function scheduleEdges(){if(_edQ)return;_edQ=true;requestAnimationFrame(()=>{_edQ=false;renderEdges();updateMap();updateSelBar();});}

// ── 渲染节点 ─────────────────────────────
const svg=$('edges');
// 节点内联控件 HTML (ComfyUI 风格 widget, 与 n.params 双向同步)
function widgetHTML(n){
  const p=n.params||{};
  if(n.type==='input'){
    let dataHTML;
    if(typeof p.data==='number'){
      dataHTML=`<div class="w-row"><input type="range" data-w="data" min="-100" max="100"
        step="0.1" value="${p.data}"><span class="val">${p.data}</span>
        <input type="number" data-w="data" value="${p.data}" style="width:66px"></div>`;
    }else{
      dataHTML=`<textarea data-w="data" rows="2" spellcheck="false">${esc(typeof p.data==='string'?p.data:JSON.stringify(p.data||''))}</textarea>`;
    }
    return `<div class="node-widget">
      <label>Topic</label>
      <input type="text" data-w="topic" value="${esc(p.topic||'')}" placeholder="topic (可选)">
      <label>自然语言 / 消息 / 数据</label>${dataHTML}
      <button type="button" class="btn ghost" style="margin-top:4px;width:100%" data-w="loadfile">📂 载入文件到 data</button></div>`;
  }
  if(n.type==='model'){
    const sel=p.route||'';
    let opts='<option value="">— 选择模型 —</option>';
    let isManual=true;
    MODELS.forEach(m=>{const r=m.route||m.name;
      if(sel===r)isManual=false;
      opts+=`<option ${(sel===r)?'selected':''} value="${esc(r)}">${esc(m.name)}</option>`;});
    opts+=`<option value="__manual__" ${(sel&&isManual)?'selected':''}>✎ 自定义 topic</option>`;
    return `<div class="node-widget">
      <label>Model</label>
      <select data-w="route">${opts}</select>
      ${(sel&&isManual)?`<input type="text" data-w="route_man" value="${esc(sel)}"
        placeholder="topic" style="margin-top:3px">`:''}
      <label>Data (override)</label>
      <textarea data-w="data" rows="1" spellcheck="false"
        placeholder="留空继承上游">${esc(p.data===null||p.data===undefined?'':(typeof p.data==='string'?p.data:JSON.stringify(p.data)))}</textarea>
      </div>`;
  }
  if(n.type==='condition'){
    const conds=n.params.conditions||[];
    let rows='';
    conds.forEach((c,i)=>{rows+=`<div class="cond-row">
      <input type="text" data-w="condlabel" data-i="${i}" value="${esc(c.label||'')}"
        placeholder="标签" style="width:44px;flex:none;margin:0">
      <input type="text" data-w="condmatch" data-i="${i}" value="${esc(c.match||'')}"
        placeholder='条件, 如 reply contains "你"' style="margin:0">
    </div>`;});
    return `<div class="node-widget">
      <label>条件 (满足哪路 → 输出到哪路, 多路同步扇出)</label>
      ${rows}
      <button type="button" class="btn ghost" style="width:100%" data-w="condadd">＋ 加条件</button>
      </div>`;
  }
  if(n.type==='feedback'){
    const sel=p.target||p.route||'';
    let opts='<option value="">— 回传目标模型 —</option>';
    MODELS.forEach(m=>{const r=m.route||m.name;
      opts+=`<option ${(sel===r)?'selected':''} value="${esc(r)}">${esc(m.name)}</option>`;});
    return `<div class="node-widget">
      <label>回传目标 (SNN 脉冲回传补做)</label>
      <select data-w="target">${opts}</select>
      <label>Data (override)</label>
      <textarea data-w="data" rows="1" spellcheck="false"
        placeholder="留空继承上游">${esc(p.data===null||p.data===undefined?'':(typeof p.data==='string'?p.data:JSON.stringify(p.data)))}</textarea>
      </div>`;
  }
  if(n.type==='output'){
    return `<div class="node-widget"><label>Output · 汇聚展示</label></div>`;
  }
  return `<div class="node-widget">
    <label>线束 · 多路输入排队</label>
    <button type="button" class="btn ghost" style="width:100%" data-w="harnessadd">＋ 加输入</button></div>`; // reroute/线束
}
function renderNodes(){
  world.querySelectorAll('.node').forEach(el=>el.remove());
  renderGroups(); // 先生成组框(底层), 再叠节点
  // 折叠组内节点需隐藏
  const hiddenIds=new Set();
  groups.forEach(g=>{if(g.fold)g.nodes.forEach(id=>hiddenIds.add(id));});
  for(const n of nodes){
    const d=TYPES[n.type]||TYPES.reroute;
    const el=document.createElement('div'); el.className='node'; el.dataset.id=n.id;
    el.style.left=n.x+'px'; el.style.top=n.y+'px';
    let preview='';
    if(n.type==='input'){const v=typeof n.params.data==='string'?n.params.data:JSON.stringify(n.params.data||'');
      preview=v.replace(/\s+/g,' ').slice(0,60);}
    const bodyTxt=n.type==='model'?`<span class="route">${n.params.route||'(未选路由)'}</span>`
      : n.type==='input'?`topic: ${JSON.stringify(n.params.topic||'')}` : '';
    const note=n.note?`<div class="note">${esc(n.note)}</div>`:'';
    // 具名端口区 (in 靠左, out 靠右, 按端口名逐行排布)
    let portsHtml='';
    let ip4=d.ins||[], op4=d.outs||[];
    if(n.type==='reroute'){ // 线束: 输入端口由 params.ins 动态决定
      ip4=(n.params.ins&&n.params.ins.length)?n.params.ins:(DEF.reroute.ins||d.ins);
    }
    if(ip4.length||op4.length){
      const rows=Math.max(ip4.length,op4.length);
      let acc='';
      for(let i=0;i<rows;i++){
        const inp=ip4[i], outp=op4[i];
        acc+=`<div class="pr">`
           +(inp?`<div class="p in" data-node="${n.id}" data-k="in" data-name="${inp}"><i class="dot"></i><span class="lbl">${inp}</span></div>`:'<span></span>')
           +(outp?`<div class="p out" data-node="${n.id}" data-k="out" data-name="${outp}"><span class="lbl">${outp}</span><i class="dot"></i></div>`:'<span></span>')
           +`</div>`;
      }
      portsHtml=`<div class="node-ports">${acc}</div>`;
    }
    el.innerHTML=
      `<div class="node-head" style="background:${d.color}">
         <span style="display:flex;align-items:center;gap:6px;flex:1;min-width:0">
           <span class="cat" style="background:${CAT_COLOR[n.type]||'#888'}"></span>
           <span class="pt" title="双击折叠/展开">${n.collapsed?'▸ ':'▾ '}${esc(n.title||d.title)}</span>
           ${(n._runMs!==undefined)?`<span class="runtime">${n._runMs}ms</span>`:''}
         </span>
         <button class="del" title="删除节点">✕</button></div>
       ${portsHtml}
       <div class="node-body">${bodyTxt}
         ${preview?`<pre class="pv">${preview}</pre>`:''}
         ${note}
         ${widgetHTML(n)}
         <pre class="res"></pre></div>`;
    if(n.collapsed)el.classList.add('collapsed');
    if(n.bypass)el.classList.add('bypass');
    if(n.mute)el.classList.add('muted');
    if(n._runStatus==='ok')el.classList.add('status-ok');
    else if(n._runStatus==='err')el.classList.add('status-err');
    if(hiddenIds.has(n.id))el.classList.add('hiddennode');
    world.appendChild(el);
    el.querySelector('.node-head').addEventListener('dblclick',ev=>{ev.stopPropagation();toggleCollapse(n);});
    el.addEventListener('mousedown',ev=>{if(ev.button!==0)return;
      if(ev.target.closest('input,textarea,select,button,a,textarea'))return;
      ev.stopPropagation();startNodeDrag(ev,n,el);});
    el.querySelector('.del').addEventListener('click',ev=>{ev.stopPropagation();deleteNodes(new Set([n.id]));});
    el.querySelectorAll('.node-ports .p').forEach(s=>s.addEventListener('mousedown',ev=>startSock(ev,s,n)));
    // 内联控件绑定 (双向同步 n.params / 属性面板)
    el.querySelectorAll('.node-widget [data-w]').forEach(w=>{
      const apply=(w2,q)=>{
        q=q||w2; const kind=q.dataset.w;
        if(kind==='route'){
          if(q.value==='__manual__'){renderNodes();return;} // 保留 route, 展示手动输入框
          n.params.route=q.value; renderNodes();
        }
        else if(kind==='route_man'){n.params.route=q.value; renderProp();}
        else if(kind==='target'){n.params.target=q.value; renderProp();}
        else if(kind==='condlabel'||kind==='condmatch'){
          const i=+q.dataset.i; const c=(n.params.conditions||[])[i];
          if(c){if(kind==='condlabel')c.label=q.value;else c.match=q.value; renderProp();}
        }
        else if(kind==='topic'){n.params.topic=q.value; renderProp();}
        else if(kind==='data'){
          if(q.tagName==='TEXTAREA'){n.params.data=parseData(q.value);}
          else{n.params.data=parseFloat(q.value)||0;
            el.querySelectorAll('.node-widget [data-w="data"]').forEach(o=>{
              if(o!==q)o.value=(o.type==='number')?n.params.data:n.params.data;});
            el.querySelector('.node-widget .val').textContent=n.params.data;}
          renderProp();
        }
      };
      if(w.dataset.w==='loadfile'){ // 文件输入: 载入文本/JSON 到 data
        w.onclick=()=>{_fileNode=n;$('fileData').click();};
        return;
      }
      if(w.dataset.w==='condadd'){ // 追加一条条件
        w.onclick=()=>{pushHist();n.params.conditions=n.params.conditions||[];
          n.params.conditions.push({label:'',match:''});renderNodes();};
        return;
      }
      if(w.dataset.w==='harnessadd'){ // 线束加一条输入端口
        w.onclick=()=>{pushHist();n.params.ins=n.params.ins||(DEF.reroute.ins.slice());
          n.params.ins.push('输入'+(n.params.ins.length+1));renderNodes();};
        return;
      }
      w.addEventListener(w.tagName==='SELECT'?'change':'input',e=>apply(e.target));
    });
    if(n._err){el.classList.add('done');el.querySelector('.res').textContent=n._err;
      el.querySelector('.res').style.color='var(--err)';}
    else if(n._result!==undefined){el.classList.add('done');
      const resEl=el.querySelector('.res');
      const typed=n._result&&n._result.typed;
      if(Array.isArray(typed)){const badge={spike:'⚡ 脉冲',text:'💬 文本',frame:'🎞 帧',audio:'🔊 音频'};
        resEl.innerHTML=typed.map(t=>`<span class="tbadge">${badge[t.category]||t.category}</span> `+
          `<code>${esc(trunc(t.content,140))}</code>`).join('<br>');}
      else{resEl.textContent=trunc(n._result,300);}
    }
  }
  renderGroups();
  applySel();
}
// ── 选区组 (ComfyUI group) ────────────────
function renderGroups(){
  world.querySelectorAll('.group').forEach(e=>e.remove());
  for(const g of groups){
    const ms=g.nodes.map(id=>nodes.find(n=>n.id===id)).filter(Boolean);
    if(!ms.length)continue;
    const x0=Math.min(...ms.map(n=>n.x))-12, y0=Math.min(...ms.map(n=>n.y))-20;
    const x1=Math.max(...ms.map(n=>n.x+NODE_W))+12;
    let y1=20;ms.forEach(n=>{const nn=world.querySelector('.node[data-id="'+n.id+'"]');const hh=nn&&!nn.classList.contains('hiddennode')?nn.offsetHeight:80;if(n.y+hh>y1)y1=n.y+hh;});
    y1+=12;
    const d=document.createElement('div'); d.className='group'+(g.fold?' fold':'')+(g.muted?' muted':'');
    d.dataset.gid=g.id;
    const gc=g.color||'#7aa2ff';
    d.style.borderColor=gc; d.style.background=gc+'18';
    d.style.left=x0+'px'; d.style.top=y0+'px';
    d.style.width=(x1-x0)+'px'; d.style.height=(y1-y0)+'px';
    const t=document.createElement('div'); t.className='g-title';
    t.style.color=gc; t.style.borderColor=gc;
    t.textContent=(g.fold?'▸ ':'▾ ')+(g.name||'组')+' ('+ms.length+')';
    t.title='拖动移动全组 · 双击折叠/展开';
    t.onmousedown=ev=>{ev.preventDefault();ev.stopPropagation();startGroupDrag(ev,g);};
    t.ondblclick=ev=>{ev.stopPropagation();toggleGroupFold(g);};
    d.appendChild(t);
    world.appendChild(d);
  }
}
function toggleGroupFold(g){pushHist();g.fold=!g.fold;renderGroups();renderNodes();updateMap();}
function startGroupDrag(ev,g){
  if(drag)finalizeDrag(true);
  const w0=worldFrom(ev.clientX,ev.clientY); const base={};
  g.nodes.forEach(id=>{const nd=nodes.find(x=>x.id===id);if(nd)base[id]={x:nd.x,y:nd.y};});
  let moved=false;
  const onm=e=>{const w=worldFrom(e.clientX,e.clientY);const dx=w.x-w0.x,dy=w.y-w0.y;
    g.nodes.forEach(id=>{const b=base[id];if(!b)return;const nd=nodes.find(x=>x.id===id);
      if(nd){const nx=Math.max(0,Math.round(b.x+dx)),ny=Math.max(0,Math.round(b.y+dy));
        if(nx!==nd.x||ny!==nd.y)moved=true;nd.x=nx;nd.y=ny;}});
    document.querySelectorAll('.node').forEach(el=>{const nd=nodes.find(x=>x.id===el.dataset.id);
      if(nd){el.style.left=nd.x+'px';el.style.top=nd.y+'px';}});
    renderGroups();renderEdges();updateMap();};
  const onu=()=>{if(moved)pushHist();canvas.classList.remove('dragging');
    window.removeEventListener('mousemove',onm);window.removeEventListener('mouseup',onu);updateSelBar();};
  canvas.classList.add('dragging');
  window.addEventListener('mousemove',onm);window.addEventListener('mouseup',onu);
}
function makeGroup(){
  const ids=[...selSet];
  if(ids.length<2){toast('请先框选/多选 ≥2 个节点');return;}
  pushHist(); const g={id:'g'+(grpSeq.n+=1),name:'组'+grpSeq.n,nodes:ids,fold:false,muted:false,color:'#7aa2ff'};
  groups.push(g);renderGroups();renderNodes();toast('已创建组');}
function removeGroup(g){pushHist();groups=groups.filter(x=>x.id!==g.id);renderGroups();renderNodes();toast('已撤分组');}
function renameGroup(g){const v=prompt('组名称',g.name||'组');if(v!==null){g.name=v.trim()||'组';renderGroups();}}
function groupOf(nid){return groups.find(g=>g.nodes.includes(nid));}
function trunc(s,n){s=typeof s==='string'?s:JSON.stringify(s);return s.length>n?s.slice(0,n)+'…':s;}

// ── 渲染连线 ─────────────────────────────
function portEl(nodeEl,kind,name){
  const list=nodeEl.querySelectorAll('.p.'+kind);
  for(const el of list)if(el.dataset.name===name)return el;
  return list[0]||null;
}
function renderEdges(){
  svg.innerHTML='';
  const els={};world.querySelectorAll('.node').forEach(e=>els[e.dataset.id]=e);
  for(const e of edges){
    const a=els[e.from], b=els[e.to]; if(!a||!b)continue;
    const co=portEl(a,'out',e.from_port), ci=portEl(b,'in',e.to_name);
    if(!co||!ci)continue;
    const [x1,y1]=sockWorld(co.querySelector('.dot'));
    const [x2,y2]=sockWorld(ci.querySelector('.dot'));
    const active=(e.id===selEdge);
    const p=path(x1,y1,x2,y2,active?'edge active':'edge');
    // 宽透明命中路径: 2px 细线难点中, 用加宽描边承接点选
    const hit=document.createElementNS('http://www.w3.org/2000/svg','path');
    hit.setAttribute('class','edge-hit');
    hit.setAttribute('d',p.getAttribute('d'));
    svg.appendChild(hit);
    hit.addEventListener('mousedown',ev=>{ev.stopPropagation();selectEdge(e.id);});
  }
}
function path(x1,y1,x2,y2,cls){
  const dx=Math.max(46,Math.abs(x2-x1)*.55);
  const p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('class',cls);
  p.setAttribute('d',`M${x1} ${y1} C${x1+dx} ${y1} ${x2-dx} ${y2} ${x2} ${y2}`);
  svg.appendChild(p);return p;
}

// ── 交互: 拖拽 / 连线 / 平移 / 缩放 ───────
function startNodeDrag(ev,n,el){
  if(drag)finalizeDrag(true);
  ev.preventDefault();ev.stopPropagation();
  if(ev.shiftKey){toggleSel(n);return;}
  if(!selSet.has(n.id))selectOne(n);
  const w=worldFrom(ev.clientX,ev.clientY);
  const base={};selSet.forEach(id=>{const nd=nodes.find(x=>x.id===id);if(nd)base[id]={x:nd.x,y:nd.y};});
  const g=base[n.id]||{x:n.x,y:n.y};
  drag={kind:'node',n,el,base,gdx:w.x-g.x,gdy:w.y-g.y,anchored:true,moved:false}; // 按下瞬间锚定抓取偏移
  canvas.classList.add('dragging');
  el.querySelector('.node-head').classList.add('dragging');
}
function startSock(ev,s,n){
  if(ev.button!==0)return;
  if(drag)finalizeDrag(true);
  ev.stopPropagation(); if(s.dataset.k!=='out')return; // 从输出发起
  drag={kind:'edge',from:n,fromName:s.dataset.name||'out',fromEl:s.closest('.node'),ghost:null,target:null};
  canvas.classList.add('dragging');
}
function startPan(ev){const r=rect();
  drag={kind:'pan',ox:view.ox,oy:view.oy,sx:ev.clientX-r.left,sy:ev.clientY-r.top};
  canvas.classList.add('dragging');}
function startRB(ev){if(drag)finalizeDrag(true);
  drag={kind:'rb',ok:false,orig:worldFrom(ev.clientX,ev.clientY),rect:null};
  rbEl.style.display='none';canvas.classList.add('dragging');}
function setTool(t){
  tool=t;
  document.querySelectorAll('#toolToggle .tl').forEach(b=>b.classList.toggle('active',b.dataset.t===t));
  canvas.classList.toggle('tool-drag',t==='drag');
  try{localStorage.setItem('wf_tool',t);}catch(e){}
}
document.querySelectorAll('#toolToggle .tl').forEach(b=>{
  b.addEventListener('click',()=>setTool(b.dataset.t));
});
try{const saved=localStorage.getItem('wf_tool');if(saved==='drag'||saved==='select')setTool(saved);}catch(e){}
canvas.addEventListener('mousedown',ev=>{
  if(ev.target.closest('.node')||ev.target.closest('.node-ports .p'))return;
  closeMenu();closeAdd();
  if(ev.button===1||(ev.button===0&&spaceHeld)){startPan(ev);}
  else if(ev.button===0){if(tool==='drag'){startPan(ev);}else{startRB(ev);selectOne(null);}}
});
canvas.addEventListener('dblclick',ev=>{if(!ev.target.closest('.node'))openAdd(worldFrom(ev.clientX,ev.clientY));});
window.addEventListener('mousemove',ev=>{
  // 小地图拖拽 → 平移大画布 (独立于主 drag 状态)
  if(mapDrag){
    const mmr=mmEl.getBoundingClientRect();
    const mx=(ev.clientX-mmr.left)*2, my=(ev.clientY-mmr.top)*2;
    const sc=mapCfg.sc;
    const vx=mapCfg.ox+(mx-mapDrag.dx)/sc;
    const vy=mapCfg.oy+(my-mapDrag.dy)/sc;
    view.ox=-vx*view.scale; view.oy=-vy*view.scale; clampView();
    applyView();scheduleMap();
    return;
  }
  if(!drag)return;
  // 残留拖拽防护: 鼠标已无按键按下但 drag 仍未结束(如鼠标在窗口外松开) → 立即终结
  if(drag.kind!=='map'&&ev.buttons===0){finalizeDrag(true);return;}
  const r=rect(),sx=ev.clientX-r.left,sy=ev.clientY-r.top,w=worldFrom(ev.clientX,ev.clientY);
  if(drag.kind==='pan'){view.ox=drag.ox+(sx-drag.sx);view.oy=drag.oy+(sy-drag.sy);applyView();scheduleMap();}
  else if(drag.kind==='rb'){
    const a=drag.orig,b=w;
    const x0=Math.min(a.x,b.x),y0=Math.min(a.y,b.y),x1=Math.max(a.x,b.x),y1=Math.max(a.y,b.y);
    drag.rect={x0,y0,x1,y1};drag.ok=Math.abs(x1-x0)>6||Math.abs(y1-y0)>6;
    if(!drag.ok){rbEl.style.display='none';return;}
    rbEl.style.display='block';
    rbEl.style.left=x0+'px';rbEl.style.top=y0+'px';
    rbEl.style.width=(x1-x0)+'px';rbEl.style.height=(y1-y0)+'px';
    document.querySelectorAll('.node').forEach(e=>e.classList.remove('selected'));
    selSet.clear();
    nodes.forEach(n=>{if(n.x<x1&&n.x+NODE_W>x0&&n.y<y1&&n.y+90>y0)selSet.add(n.id);});
    document.querySelectorAll('.node').forEach(e=>{if(selSet.has(e.dataset.id))e.classList.add('selected');});
    updateCount(); // 框选过程实时反馈
  }
  else if(drag.kind==='node'){
    const grabbed=drag.base[drag.n.id]||{x:drag.n.x,y:drag.n.y};
    // 抓取偏移在 mousedown 已锚定(drag.gdx/gdy); 每帧按当前鼠标重算抓取节点目标, 其余保持相对位差
    const gx=w.x-drag.gdx, gy=w.y-drag.gdy;
    for(const key in drag.base){const nd=nodes.find(x=>x.id===key);if(!nd)continue;
      const b=drag.base[key];
      const nx=Math.max(0,Math.round(b.x+(gx-grabbed.x)));
      const ny=Math.max(0,Math.round(b.y+(gy-grabbed.y)));
      if(nx!==nd.x||ny!==nd.y)drag.moved=true;
      nd.x=nx;nd.y=ny;}
    document.querySelectorAll('.node').forEach(el=>{const nd=nodes.find(x=>x.id===el.dataset.id);if(nd){el.style.left=nd.x+'px';el.style.top=nd.y+'px';}});
    scheduleEdges();}
  else if(drag.kind==='edge'){
    if(drag.ghost)drag.ghost.remove();
    const el=drag.fromEl; if(!el){drag=null;return;}
    const [x1,y1]=sockWorld(el.querySelector('.p.out .dot'));
    const t=document.elementFromPoint(ev.clientX,ev.clientY);
    const inp=t?t.closest('.p.in'):null;
    let x2,y2;
    if(inp){const [a,b]=sockWorld(inp.querySelector('.dot'));x2=a;y2=b;drag.target=inp.closest('.node').dataset.id;}
    else{x2=w.x;y2=w.y;drag.target=null;}
    drag.ghost=path(x1,y1,x2,y2,'edge ghost');
  }
});
window.addEventListener('mouseup',ev=>{
  if(mapDrag){mapDrag=null;scheduleMap();return;}
  if(!drag)return;
  finalizeDrag(false,ev);
});
// 拖拽统一终结: aborted=中途被打断(无有效落点/按键丢失), ev=真实 mouseup 事件(用于边缘落点判定)
function finalizeDrag(aborted,ev){
  if(!drag)return;
  if(drag.kind==='edge'){
    if(drag.ghost){drag.ghost.remove();drag.ghost=null;}
    if(!aborted&&ev){
      const t=document.elementFromPoint(ev.clientX,ev.clientY);
      const tin=t?t.closest('.p.in'):null;
      if(tin){
        const to=tin.closest('.node').dataset.id, from=drag.from.id;
        if(from!==to&&to){
          pushHist();
          const inName=tin.dataset.name||'in';
          edges=edges.filter(e=>!(e.to===to&&e.to_name===inName)); // 每个输入端口仅一连接
          edges.push({id:'e'+(++eidSeq),from,from_port:drag.fromName||'out',to,to_port:'in',to_name:inName});
        }
      }
    }
    renderEdges();updateMap();
  }
  else if(drag.kind==='rb'){
    rbEl.style.display='none';
    if(!aborted){
      if(!drag.ok){selSet.clear();selection=null;}
      else{const arr=nodes.filter(n=>selSet.has(n.id));selection=arr[0]||null;}
      applySel();renderEdges();renderProp();updateMap();
    }else{applySel();renderEdges();updateMap();}
  }
  else if(drag.kind==='node'){
    if(drag.moved&&!aborted)pushHist();
    if(drag.el){const hd=drag.el.querySelector('.node-head');if(hd)hd.classList.remove('dragging');}
    renderEdges();updateMap();}
  else if(drag.kind==='pan'){renderEdges();updateMap();}
  canvas.classList.remove('dragging');drag=null;
  updateSelBar();
}
window.addEventListener('blur',()=>{finalizeDrag(true);});
canvas.addEventListener('wheel',ev=>{
  ev.preventDefault();
  const r=rect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
  const w={x:(mx-view.ox)/view.scale,y:(my-view.oy)/view.scale};
  const f=ev.deltaY<0?1.12:0.9;
  view.scale=Math.min(2.5,Math.max(.35,view.scale*f));
  view.ox=mx-w.x*view.scale;view.oy=my-w.y*view.scale;
  applyView();updateMap();
},{passive:false});
window.addEventListener('keydown',ev=>{
  const tag=(ev.target.tagName||'').toUpperCase();
  const typing=tag==='INPUT'||tag==='TEXTAREA';
  const mod=ev.ctrlKey||ev.metaKey;
  if(mod){
    if(ev.key==='z'||ev.key==='Z'){ev.preventDefault();if(ev.shiftKey)redo();else undo();}
    else if(ev.key==='y'||ev.key==='Y'){ev.preventDefault();redo();}
    else if(ev.key==='d'||ev.key==='D'){ev.preventDefault();if(!typing)duplicateSel();}
    else if(ev.key==='c'||ev.key==='C'){ev.preventDefault();if(!typing){copySelToClip();toast('已复制 '+selSet.size+' 个节点');}}
    else if(ev.key==='v'||ev.key==='V'){ev.preventDefault();if(!typing)pasteClip(32);}
    else if(ev.key==='s'||ev.key==='S'){ev.preventDefault();if(ev.shiftKey)exportJson();else saveNow();}
    else if(ev.key==='a'||ev.key==='A'){ev.preventDefault();if(!typing)selectAllNodes();}
    else if(ev.key==='g'||ev.key==='G'){ev.preventDefault();if(!typing)makeGroup();}
    else if(ev.key==='b'||ev.key==='B'){ev.preventDefault();toggleSidebar();}
    else if(ev.key==='Enter'){ev.preventDefault();if(!typing)enqueue();}
  }
  else if(!typing&&(ev.key==='ArrowUp'||ev.key==='ArrowDown'||ev.key==='ArrowLeft'||ev.key==='ArrowRight')){
    ev.preventDefault();nudgeSel(ev);
  }
  if(!typing&&(ev.key==='Delete'||ev.key==='Backspace')){
    ev.preventDefault();
    if(selEdge)deleteEdge(selEdge);
    else if(selSet.size||selection)deleteSel();
  }
  if(ev.key==='Escape'){closeAdd();closeMenu();selectOne(null);if(running)setMsg('按 ■ Interrupt 中断');}
  if(ev.key===' '&&!typing)spaceHeld=true;
});
window.addEventListener('keyup',ev=>{if(ev.key===' ')spaceHeld=false;});
function selectAllNodes(){selSet.clear();nodes.forEach(n=>selSet.add(n.id));selection=nodes[0]||null;
  applySel();renderEdges();renderProp();}
function nudgeSel(ev){
  const ids=selSet.size?[...selSet]:(selection?[selection.id]:[]);
  if(!ids.length)return; const step=ev.shiftKey?8:1;
  const d=ev.key==='ArrowUp'?[0,-step]:ev.key==='ArrowDown'?[0,step]
    :ev.key==='ArrowLeft'?[-step,0]:[step,0];
  pushHist();
  ids.forEach(id=>{const n=nodes.find(x=>x.id===id);if(n){n.x=Math.max(0,n.x+d[0]);n.y=Math.max(0,n.y+d[1]);}});
  renderNodes();renderEdges();updateMap();};

// ── 拖放 (ComfyUI: 拖入 .json 工作流 / 内核模型文件) ──
const dropHud=$('dropHud');
// 仅当拖入的是真实文件时才展示/处理上传, 避免普通鼠标拖拽误触
function isFileDrag(e){
  const dt=e.dataTransfer;if(!dt)return false;
  const types=(dt.types)?Array.prototype.slice.call(dt.types):[];
  return types.some(t=>t==='Files'||t==='application/x-moz-file');
}
canvas.addEventListener('dragover',e=>{e.preventDefault();
  if(isFileDrag(e)){dropHud.classList.add('show');canvas.classList.add('dragdrop');}});
canvas.addEventListener('dragleave',e=>{e.preventDefault();dropHud.classList.remove('show');
  canvas.classList.remove('dragdrop');});
window.addEventListener('dragend',()=>{dropHud.classList.remove('show');canvas.classList.remove('dragdrop');});
canvas.addEventListener('drop',e=>{e.preventDefault();e.stopPropagation();
  dropHud.classList.remove('show');canvas.classList.remove('dragdrop');
  if(!isFileDrag(e))return;
  const f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];const name=f?(f.name||''):'';
  const base=(name||'').replace(/\.[^.]+$/,'').toLowerCase();
  const m=MODELS.find(x=>(x.route||'').toLowerCase()===base||(x.name||'').toLowerCase()===base);
  const w=worldFrom(e.clientX,e.clientY);addPos={x:Math.max(40,w.x),y:Math.max(40,w.y)};
  if(!f){toast('未检测到文件');return;}
  if(m){addNode({t:'model',model:m});toast('拖入模型节点: '+m.name);return;}
  const rd=new FileReader();rd.onload=()=>{try{const obj=JSON.parse(rd.result);
    if(obj&&(obj.nodes||obj.workflow)){importWorkflow(obj.nodes?obj:obj.workflow);}
    else{toast('未识别文件 (非工作流 JSON)');}}
    catch(err){toast('无法解析文件: '+err.message);}};
  rd.readAsText(f);});

// ── 选择 / 编辑 ──────────────────────────
function applySel(){
  document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected',selSet.has(el.dataset.id)));
  updateSelBar();
}
// 点选—操作闭环: 选中节点/边后在画布上弹出上下文操作条 (多选时显示计数并置于选区质心)
function updateSelBar(){
  const bar=$('selBar'); if(!bar)return;
  bar.innerHTML='';
  const mk=(label,fn,danger)=>{const b=document.createElement('button');b.textContent=label;
    if(danger)b.className='danger';
    b.onmousedown=ev=>{ev.preventDefault();ev.stopPropagation();}; // 防止按钮点击抢焦点/选中文字, 且不冒泡到画布触发 startRB/startPan 清空选择
    b.onclick=()=>{fn();updateSelBar();};bar.appendChild(b);};
  const place=(wx,wy)=>{bar.style.left=wx*view.scale+view.ox+'px';bar.style.top=wy*view.scale+view.oy+'px';};
  // 连线选中
  if(selEdge){
    const e=edges.find(x=>x.id===selEdge);
    if(e){
      const a=nodes.find(n=>n.id===e.from), b=nodes.find(n=>n.id===e.to);
      if(a&&b){
        place((a.x+b.x)/2,(a.y+b.y)/2);
        mk('✕ 断开连线',()=>deleteEdge(selEdge),true);
        bar.classList.add('show'); return;
      }
    }
    bar.classList.remove('show'); return;
  }
  // 多选: 质心 + 计数
  if(selSet.size>=2){
    const sel=nodes.filter(n=>selSet.has(n.id));
    if(sel.length){
      const cx=sel.reduce((s,n)=>s+n.x,0)/sel.length, cy=sel.reduce((s,n)=>s+n.y,0)/sel.length;
      place(cx,cy);
      const c=document.createElement('span');c.className='selbar-count';
      c.textContent=`已选中 ${selSet.size} 个对象`;bar.appendChild(c);
      mk('⇶ 转为组',()=>makeGroup());
      mk('✕ 删除',()=>deleteNodes(new Set(selSet)),true);
      bar.classList.add('show'); return;
    }
  }
  // 单选节点
  if(selection&&selSet.size){
    const n=selection;
    place(n.x,n.y);
    mk('✎ 重命名',()=>{const v=prompt('节点名称',n.title||TYPES[n.type].title);
      if(v!==null){n.title=v.trim()||'';renderNodes();renderProp();}});
    mk('⧉ 重复',()=>duplicateSel());
    mk(n.bypass?'✓旁通':'旁通',()=>{n.bypass=!n.bypass;renderNodes();renderProp();});
    mk(n.mute?'✓禁用':'禁用',()=>{n.mute=!n.mute;renderNodes();renderProp();});
    mk('✕ 删除',()=>deleteNodes(new Set(selSet)),true);
    bar.classList.add('show'); return;
  }
  bar.classList.remove('show');
}
function selectOne(n){selection=n;selEdge=null;selSet.clear();if(n)selSet.add(n.id);
  applySel();renderEdges();renderProp();updateCount();}
function selectEdge(id){selEdge=id;selection=null;selSet.clear();
  applySel();renderEdges();renderProp();updateCount();}
function toggleSel(n){if(!n)return;
  if(selSet.has(n.id)){selSet.delete(n.id);if(selection&&selection.id===n.id)selection=null;}
  else{selSet.add(n.id);selection=n;}
  if(!selSet.size)selection=null;
  applySel();renderEdges();renderProp();}
function toggleCollapse(n){pushHist();n.collapsed=!n.collapsed;renderNodes();renderEdges();}
function deleteNodes(ids){if(!ids.size)return;pushHist();
  nodes=nodes.filter(n=>!ids.has(n.id));
  edges=edges.filter(e=>!ids.has(e.from)&&!ids.has(e.to));
  groups=groups.map(g=>({...g,nodes:g.nodes.filter(id=>!ids.has(id))})).filter(g=>g.nodes.length);
  if(selection&&ids.has(selection.id))selection=null;
  ids.forEach(i=>selSet.delete(i));
  applySel();renderNodes();renderEdges();updateMap();renderProp();}
function deleteSel(){const ids=new Set(selSet.size?[...selSet]:[]);
  if(!ids.size&&selection)ids.add(selection.id);
  if(!ids.size){toast('没有选中节点');return;}
  deleteNodes(ids);}
function deleteEdge(id){pushHist();edges=edges.filter(e=>e.id!==id);selEdge=null;renderEdges();updateMap();}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function rawData(d){return typeof d==='string'?d:JSON.stringify(d!==null&&d!==undefined?d:'');}
function parseData(s){const t=(s||'').trim();if(!t)return null;try{return JSON.parse(t);}catch(e){return t;}}
function pretty(v,max){max=max||1600;if(v===undefined||v===null)return String(v);
  return typeof v==='object'?(()=>{try{const s=JSON.stringify(v,null,2);return s.length>max?s.slice(0,max)+'…':s;}catch(e){return String(v);}})():String(v);}
function renderProp(){
  const box=$('prop');
  if(!selection){box.className='hint';box.textContent='点击画布中的节点以编辑属性';return;}
  box.className='';const n=selection,p=n.params;let h='';
  h+=`<label>名称 (标题)</label><input id="p_title" value="${esc(n.title||TYPES[n.type].title)}">`;
  h+=`<label>类型</label><input value="${TYPES[n.type].title}" disabled>`;
  if(n.type==='input'){
    h+=`<label>topic (路由主题)</label><input id="p_topic" value="${esc(p.topic||'')}">`;
    h+=`<label>data (JSON 或原始文本)</label><textarea id="p_data">${esc(rawData(p.data))}</textarea>`;
  }else if(n.type==='model'){
    h+=`<label>模型 / 路由 (topic)</label><select id="p_route">`;
    const known=[];
    MODELS.forEach(m=>{known.push(m.route||m.name);
      h+=`<option ${(p.route||'')===(m.route||m.name)?'selected':''} value="${esc(m.route||m.name)}">${esc(m.name)} · ${esc(m.route||'')}${m.loaded?'':' (未载)'}</option>`;});
    h+=`<option value="" ${!(known.includes(p.route))?'selected':''}>自定义 topic</option></select>`;
    h+=`<input id="p_route_custom" value="${esc(known.includes(p.route)?'':(p.route||''))}" placeholder="自定义 topic (留空继承/不路由)">`;
    h+=`<label>data 覆盖 (可选, JSON; 留空继承上游)</label><textarea id="p_data">${esc(rawData(p.data))}</textarea>`;
  }
  h+=`<label>备注 (注释)</label><textarea id="p_note">${esc(n.note||'')}</textarea>`;
  h+=`<label>执行选项</label><div style="display:flex;gap:6px">
     <button id="p_bypass" class="btn ghost" style="flex:1" title="旁通: 不执行, 透传上游">${n.bypass?'✓ ':'○ '}旁通</button>
     <button id="p_mute" class="btn ghost" style="flex:1" title="禁用: 跳过执行">${n.mute?'✓ ':'○ '}禁用</button></div>`;
  if(n._result!==undefined){h+=`<label>结果预览</label><pre class="respre">${esc(pretty(n._result))}</pre>`;}
  box.innerHTML=h;
  const bind=(id,fn)=>{const el=box.querySelector(id);if(el)el.addEventListener('input',()=>fn(el));};
  bind('#p_title',el=>{n.title=el.value;renderNodes();renderEdges();});
  bind('#p_topic',el=>p.topic=el.value);
  bind('#p_route',el=>{p.route=el.value;renderProp();});
  bind('#p_route_custom',el=>{p.route=el.value;});
  bind('#p_data',el=>{p.data=parseData(el.value);});
  bind('#p_note',el=>{n.note=el.value;renderNodes();});
  const by=$('p_bypass'),mu=$('p_mute');
  if(by)by.onclick=()=>{n.bypass=!n.bypass;renderNodes();renderProp();};
  if(mu)mu.onclick=()=>{n.mute=!n.mute;renderNodes();renderProp();};
}

// ── 撤销 / 重做 ──────────────────────────
function snapStr(){return JSON.stringify({nodes,edges});}
function pushHist(){hist.past.push(snapStr());if(hist.past.length>hist.max)hist.past.shift();hist.future.length=0;updateHist();}
function restoreStr(s){const g=JSON.parse(s);nodes=g.nodes;edges=g.edges;
  selection=null;selSet.clear();selEdge=null;applySel();renderNodes();renderEdges();updateMap();renderProp();}
function undo(){if(!hist.past.length)return;hist.future.push(snapStr());restoreStr(hist.past.pop());}
function redo(){if(!hist.future.length)return;hist.past.push(snapStr());restoreStr(hist.future.pop());}
function updateHist(){$('btnUM').disabled=!hist.past.length;$('btnRM').disabled=!hist.future.length;}
$('btnUM').onclick=undo;$('btnRM').onclick=redo;

// ── 复制 / 粘贴 / 重复 ────────────────────
function copySelToClip(){if(!selSet.size)return;
  const ids=new Set(selSet);const sel=nodes.filter(n=>ids.has(n.id));
  const idMap={};const newNodes=sel.map(n=>{const nid='n'+(++nidSeq);idMap[n.id]=nid;
    return{...JSON.parse(JSON.stringify(formatNode(n))),id:nid};});
  const newEdges=edges.filter(e=>ids.has(e.from)&&ids.has(e.to))
    .map(e=>({id:'e'+(++eidSeq),from:idMap[e.from]||e.from,from_port:e.from_port,to:idMap[e.to]||e.to,to_port:e.to_port}));
  clip={nodes:newNodes,edges:newEdges};}
function pasteClip(offset){if(!clip||!clip.nodes.length)return;pushHist();
  const idMap={};clip.nodes.forEach(n=>{idMap[n.id]='n'+(++nidSeq);});
  const pn=clip.nodes.map(n=>({...JSON.parse(JSON.stringify(n)),id:idMap[n.id],x:Math.max(0,n.x+offset||0),y:Math.max(0,n.y+offset||0)}));
  const pe=clip.edges.map(e=>({id:'e'+(++eidSeq),from:idMap[e.from]||e.from,from_port:e.from_port,to:idMap[e.to]||e.to,to_port:e.to_port}));
  nodes.push(...pn);edges.push(...pe);
  selSet.clear();pn.forEach(n=>selSet.add(n.id));selection=pn[0]||null;
  applySel();renderNodes();renderEdges();updateMap();renderProp();}
function duplicateSel(){copySelToClip();if(clip)pasteClip(24);}

// ── 右键上下文菜单 ───────────────────────
function menuItem(label,fn,key,danger){const d=document.createElement('div');
  d.className='i'+(danger?' danger':'');d.innerHTML=`<span>${esc(label)}</span>${key?`<span class="k">${esc(key)}</span>`:''}`;
  d.onmousedown=ev=>{ev.preventDefault();ev.stopPropagation();closeMenu();fn();};ctxItems.appendChild(d);}
function menuSep(){const d=document.createElement('div');d.className='sep';ctxItems.appendChild(d);}
function openMenu(x,y){ctxMenu.style.left=Math.min(x,innerWidth-220)+'px';
  ctxMenu.style.top=Math.min(y,innerHeight-40)+'px';ctxMenu.classList.add('show');}
function closeMenu(){$('ctxMenu').classList.remove('show');}
document.addEventListener('mousedown',ev=>{if(!ev.target.closest('.ctx-menu'))closeMenu();});
document.addEventListener('contextmenu',ev=>{
  const nodeEl=ev.target.closest('.node');
  const grpEl=ev.target.closest('.group');
  ctxItems.innerHTML='';
  if(grpEl&&grpEl.dataset.gid&&!nodeEl){
    ev.preventDefault();
    const g=groups.find(x=>x.id===grpEl.dataset.gid); if(!g){closeMenu();return;}
    menuItem(g.fold?'展开组':'折叠组',()=>toggleGroupFold(g),'双击');
    menuItem('重命名组',()=>renameGroup(g));
    menuItem(g.muted?'取消禁用组':'禁用组',()=>{g.muted=!g.muted;renderGroups();renderNodes();});
    menuSep();
    { // 组着色色板
      const row=document.createElement('div');row.className='c-swatches';
      ['#7aa2ff','#ffb86c','#8be9fd','#ff79c6','#50fa7b','#f1fa8c'].forEach(c=>{
        const dot=document.createElement('span');dot.style.background=c;
        dot.onmousedown=ev=>{ev.preventDefault();ev.stopPropagation();closeMenu();
          g.color=c;renderGroups();};
        row.appendChild(dot);});
      ctxItems.appendChild(row);
    }
    menuSep();
    menuItem('撤分组 (保留节点)',()=>removeGroup(g));
    menuItem('删除组内节点',()=>deleteNodes(new Set(g.nodes)),'Del',true);
    openMenu(ev.clientX,ev.clientY);
    return;
  }
  if(nodeEl){
    ev.preventDefault();
    const n=nodes.find(x=>x.id===nodeEl.dataset.id);if(!n)return;
    if(!selSet.has(n.id)&&!ev.shiftKey)selectOne(n);
    menuItem('复制',copySelToClip,'Ctrl+C');
    menuItem('重复 (复制+粘贴)',duplicateSel,'Ctrl+D');
    menuItem('重命名',()=>{const v=prompt('节点名称',n.title||TYPES[n.type].title);if(v!==null){n.title=v.trim()||'';renderNodes();renderProp();}});
    menuItem(n.collapsed?'展开':'折叠',()=>toggleCollapse(n),'双击标题');
    menuSep();
    menuItem(n.bypass?'取消旁通':'旁通',()=>{n.bypass=!n.bypass;renderNodes();},'');
    menuItem(n.mute?'取消禁用':'禁用',()=>{n.mute=!n.mute;renderNodes();},'');
    if(selSet.size>=2)menuItem('转为组',makeGroup,'Ctrl+G');
    const g=groupOf(n.id);if(g)menuItem('撤出该组',()=>removeGroup(g));
    menuSep();
    menuItem('删除',()=>deleteNodes(new Set(selSet.size?[...selSet]:[n.id])),'Del',true);
    openMenu(ev.clientX,ev.clientY);
  }else{
    ev.preventDefault();
    menuItem('添加节点…',()=>openAdd(worldFrom(ev.clientX,ev.clientY)),'双击');
    menuItem('添加线束节点',()=>addReroute(worldFrom(ev.clientX,ev.clientY)));
    if(clip)menuItem('粘贴节点',()=>pasteClip(32),'Ctrl+V');
    if(selSet.size>=2)menuItem('转为组 (Ctrl+G)',makeGroup,'Ctrl+G');
    if(groups.length)menuItem('全部分组',()=>{groups=[];renderGroups();});
    menuSep();
    menuItem('导出 JSON',exportJson,'Ctrl+Shift+S');
    menuItem('导出 ComfyUI JSON',exportComfyJson);
    menuItem('导入 JSON',()=>$('fileIn').click());
    openMenu(ev.clientX,ev.clientY);
  }
});
function addReroute(w){pushHist();const id='n'+(++nidSeq);const n={id,type:'reroute',x:Math.max(20,w.x),y:Math.max(20,w.y),params:{ins:(DEF.reroute.ins.slice())}};nodes.push(n);selectOne(n);renderNodes();renderEdges();updateMap();}

// ── 导入 / 导出 JSON ─────────────────────
function formatNode(n){return{id:n.id,type:n.type,x:n.x,y:n.y,
  params:JSON.parse(JSON.stringify(n.params||{})),title:n.title||'',collapsed:!!n.collapsed,
  note:n.note||'',bypass:!!n.bypass,mute:!!n.mute};}
function exportJson(){const wf={name:$('wfName').value||'wf',version:1,
  nodes:nodes.map(n=>formatNode(n)),edges:JSON.parse(JSON.stringify(edges)),
  groups:JSON.parse(JSON.stringify(groups))};
  downloadJson(wf,'workflow');}
$('btnExport').onclick=exportJson;
function exportComfyJson(){
  const wf={nodes:nodes.map(n=>formatNode(n)),edges:JSON.parse(JSON.stringify(edges)),
    groups:JSON.parse(JSON.stringify(groups))};
  fetch('/api/workflow/export_comfy',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({workflow:wf})})
    .then(r=>r.json()).then(j=>{if(!j.ok||!j.workflow){toast(j.error||'导出失败');return;}
      downloadJson(j.workflow,'comfy');toast('已导出 ComfyUI 格式 JSON');})
    .catch(()=>toast('导出失败'));
}
function downloadJson(obj,prefix){
  const blob=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download=(prefix||'workflow')+'.json';a.click();URL.revokeObjectURL(a.href);}
// 判别 ComfyUI 格式: 顶层有 links 且无 edges → 走后端转换后再加载
async function importWorkflow(obj){
  let graph=obj;
  if(obj&&obj.links&&!(obj.edges)){            // ComfyUI 原生
    try{const r=await fetch('/api/workflow/import_comfy',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({wf:obj})});
      const j=await r.json();if(!j.ok||!j.workflow){toast(j.error||'ComfyUI 导入失败');return false;}
      graph=j.workflow;}catch(e){toast('ComfyUI 导入失败: '+e.message);return false;}}
  pushHist();loadGraph(graph);
  toast('已导入 '+(graph.name||'workflow'));return true;}
$('btnImport').onclick=()=>$('fileIn').click();
$('fileIn').addEventListener('change',ev=>{const f=ev.target.files[0];if(!f)return;
  const rd=new FileReader();rd.onload=()=>{try{const w=JSON.parse(rd.result);importWorkflow(w);}catch(e){toast('导入失败: '+e.message);}};
  rd.readAsText(f);ev.target.value='';});
// 文件输入: 把选中文件内容载入到某 Input 节点的 data
let _fileNode=null;
$('fileData').addEventListener('change',ev=>{const f=ev.target.files[0];if(!f){_fileNode=null;return;}
  const rd=new FileReader();rd.onload=()=>{
    const n=_fileNode;_fileNode=null;
    if(!n)return;
    const txt=String(rd.result);
    try{n.params.data=JSON.parse(txt);}catch(e){n.params.data=txt;}
    renderNodes();renderProp();toast('已载入文件到 '+n.title+' 的 data ('+txt.length+' 字符)');
  };
  rd.readAsText(f);ev.target.value='';});

// ── 添加节点 (双击画布搜索) ──────────────
let addPos={x:120,y:120};
function openAdd(w){addPos={x:Math.max(40,w.x),y:Math.max(40,w.y)};
  $('addOverlay').classList.add('show');$('addSearch').value='';filterAdd('');$('addSearch').focus();}
function closeAdd(){$('addOverlay').classList.remove('show');}
// 分组构建: 基础节点一组, 模型按 capability 分类折叠; 支持搜索过滤/打分
function addGroups(q){
  const groups=[];
  groups.push({key:'基础',items:[
    {t:'input',label:'输入 (Input)',sub:'topic + data',color:'var(--green)'},
    {t:'condition',label:'条件 (Condition)',sub:'满足条件 → 对应端口扇出',color:'#b26bf5'},
    {t:'feedback',label:'回传 (Feedback)',sub:'SNN 脉冲回传补做',color:'#e0596f'},
    {t:'output',label:'输出 (Output)',sub:'展示结果',color:'#c99a2e'},
    {t:'reroute',label:'线束 (Harness)',sub:'多路输入排队汇流',color:'#8f8f8f'},
  ]});
  const byCap={};
  MODELS.forEach(m=>{
    const cap=m.capability||m.base_model||m.route||m.name||'模型';
    (byCap[cap]=byCap[cap]||[]).push({t:'model',label:'模型 · '+m.name,sub:'route='+(m.route||''),model:m,color:'var(--blue)'});
  });
  Object.keys(byCap).sort().forEach(cap=>groups.push({key:cap,items:byCap[cap]}));
  if(!q.trim())return groups;                       // 无搜索 → 全量分组
  const out=[];
  groups.forEach(g=>{
    const hit=g.items.filter(it=>{
      const s=(it.label+' '+(it.sub||'')).toLowerCase();
      return s.includes(q);
    });
    if(hit.length)out.push({key:g.key,items:hit});
  });
  return out;
}
function filterAdd(q){
  const list=$('addList');list.innerHTML='';q=(q||'').toLowerCase();
  addGroups(q).forEach(g=>{
    const grp=document.createElement('div');grp.className='add-group open';grp.dataset.key=g.key;
    grp.innerHTML=`<span class="caret">▶</span><span>${esc(g.key)}</span><small>${g.items.length}</small>`;
    const wrap=document.createElement('div');wrap.className='add-group-items';
    g.items.forEach(it=>{
      const d=document.createElement('div');d.className='add-item';
      d.innerHTML=`<span>${esc(it.label)}</span><small>${esc(it.sub||'')}</small>`;
      d.onclick=()=>{addNode(it);closeAdd();};wrap.appendChild(d);});
    grp.onclick=()=>{const closed=grp.classList.toggle('closed');wrap.style.display=closed?'none':'';};
    list.appendChild(grp);list.appendChild(wrap);
  });
  list.scrollTop=0;
}
$('addSearch').addEventListener('input',e=>filterAdd(e.target.value));
$('addOverlay').addEventListener('keydown',e=>{if(e.key==='Escape')closeAdd();});
function addNode(it){
  pushHist();
  const id='n'+(++nidSeq);const p=Object.assign({},DEF[it.t]);
  if(Array.isArray(p.conditions))p.conditions=DEF[it.t].conditions.map(c=>Object.assign({},c)); // 深拷贝, 防共享
  if(Array.isArray(p.ins))p.ins=p.ins.slice(); // 线束输入端口深拷贝, 防共享
  if(it.t==='model'&&it.model)p.route=it.model.route||it.model.name;
  const n={id,type:it.t,x:addPos.x,y:addPos.y,params:p};
  nodes.push(n);selectOne(n);renderNodes();renderEdges();updateMap();
}

// ── 小地图 ───────────────────────────────
const mmCv=$('mmCv'), mmEl=$('minimap');
let mapCfg={sc:1,ox:0,oy:0};                       // 世界→小地图映射 (mapx=(world-ox)*sc)
let mapDrag=null;                                  // 小地图视口拖拽 {dx,dy} (抓取点到视口左上角的偏移)
function clampView(){                              // 把视口限制在内容附近, 防止拖丢图
  const cr=canvas.getBoundingClientRect();
  const vw=cr.width/view.scale, vh=cr.height/view.scale;
  let minX=0,minY=0,maxX=400,maxY=300;
  if(nodes.length){minX=Math.min(...nodes.map(n=>n.x));minY=Math.min(...nodes.map(n=>n.y));
    maxX=Math.max(...nodes.map(n=>n.x+NODE_W));maxY=Math.max(...nodes.map(n=>n.y+90));}
  const left=Math.max(minX-300,Math.min(-view.ox/view.scale,maxX+300-vw));
  const top=Math.max(minY-300,Math.min(-view.oy/view.scale,maxY+300-vh));
  view.ox=-left*view.scale; view.oy=-top*view.scale;
}
function updateMap(){
  const cv=mmCv,dpr=1;cv.width=340;cv.height=260;
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  let minX=0,minY=0,maxX=400,maxY=300;
  if(nodes.length){minX=Math.min(...nodes.map(n=>n.x));minY=Math.min(...nodes.map(n=>n.y));
    maxX=Math.max(...nodes.map(n=>n.x+NODE_W));maxY=Math.max(...nodes.map(n=>n.y+90));}
  const pad=20,sw=cv.width,sh=cv.height;
  const vw=maxX-minX+pad*2,vy=maxY-minY+pad*2;
  const sx=sw/vw,sy=sh/vy,sc=Math.min(sx,sy);
  const ox=minX-pad,oy=minY-pad;
  mapCfg={sc,ox,oy};
  ctx.fillStyle='rgba(255,255,255,0.04)';ctx.fillRect(0,0,sw,sh);
  for(const n of nodes){ctx.fillStyle=TYPES[n.type].color;ctx.globalAlpha=.85;
    ctx.fillRect((n.x-ox)*sc,(n.y-oy)*sc,NODE_W*sc,80*sc);ctx.globalAlpha=1;}
  // 视口矩形 (当前大画布可见区域 → 小地图)
  const cr=canvas.getBoundingClientRect();
  const vpX=((0-view.ox)/view.scale-ox)*sc, vpY=((0-view.oy)/view.scale-oy)*sc;
  const vw2=(cr.width/view.scale)*sc, vh2=(cr.height/view.scale)*sc;
  ctx.fillStyle='rgba(107,195,255,.10)';ctx.fillRect(vpX,vpY,vw2,vh2);
  ctx.strokeStyle='#6bc3ff';ctx.lineWidth=1.5;ctx.strokeRect(vpX,vpY,vw2,vh2);
}
// 小地图拖拽视口 → 平移大画布
mmEl.addEventListener('mousedown',ev=>{
  ev.preventDefault();ev.stopPropagation();
  const r=mmEl.getBoundingClientRect();
  const mx=(ev.clientX-r.left)*2, my=(ev.clientY-r.top)*2;   // 换算到 cv 坐标(340x260)
  const sc=mapCfg.sc;
  const vmx=(-view.ox/view.scale-mapCfg.ox)*sc;              // 视口左上角在小地图中的位置
  const vmy=(-view.oy/view.scale-mapCfg.oy)*sc;
  mapDrag={dx:mx-vmx, dy:my-vmy};
});

// ── 运行 (服务端任务队列 + 轮询 + 逐节点高亮) ────
$('btnRun').onclick=enqueue;
let pollTimer=null, lastAnimId=0, srvTasks=[];
function wfSerializeForRun(){return{name:$('wfName').value||'wf',version:1,
  nodes:nodes.map(n=>({id:n.id,type:n.type,x:n.x,y:n.y,params:n.params})),
  edges:JSON.parse(JSON.stringify(edges))};}
function enqueue(){
  const wf=wfSerializeForRun();$('btnRun').disabled=true;
  fetch('/api/workflow/submit',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({workflow:wf,name:wf.name})})
    .then(r=>r.json()).then(j=>{
      if(!j.ok){$('btnRun').disabled=false;toast(j.error||'提交失败');return;}
      setMsg(`任务 #${j.task_id} 已入队`);startPoll();})
    .catch(()=>{$('btnRun').disabled=false;toast('提交失败');});
}
function startPoll(){if(pollTimer)return;pollTimer=setInterval(pollTasks,700);pollTasks();}
async function pollTasks(){
  let j;try{const r=await fetch('/api/tasks');j=await r.json();}catch(e){return;}
  if(!j.ok)return;
  const tasks=j.tasks||[];renderQueue(tasks);
  const runn=tasks.find(t=>t.state==='running');
  const hasPending=runn||tasks.some(t=>t.state==='pending');
  $('btnRun').disabled=hasPending;
  // 最近完成且未播放的 ok 任务 → 逐节点高亮
  const newest=tasks.filter(t=>t.state==='ok'&&t.id>lastAnimId)
    .sort((a,b)=>a.id-b.id).pop();
  if(newest){lastAnimId=newest.id;animateTask(newest);}
  if(tasks.some(t=>t.state==='running'||t.state==='pending'))updateQueueBadge(tasks);
  if(!hasPending){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}updateQueueBadge(tasks);}
}
function animateTask(t){
  nodes.forEach(n=>{n._result=null;n._err=undefined;n._runStatus=undefined;n._runMs=undefined;});
  renderNodes();renderEdges();
  const log=t.log||[];const doneIds=new Set();
  (async()=>{for(const L of log){
    if(runAbort)break;
    const n=nodes.find(x=>x.id===L.node_id);
    if(n&&!n.bypass&&!n.mute&&!doneIds.has(n.id)){n._runStatus='run';renderNodes();renderEdges();}
    if(n)n._runMs=L.ms;
    await sleep(60);
    const res=t.results&&t.results[L.node_id];
    if(n){n._result=(res&&res.error)?undefined:res;n._err=res&&res.error?res.error:undefined;
      n._runStatus=(res&&res.error)?'err':((n.bypass||n.mute)?undefined:'ok');}
    doneIds.add(L.node_id);renderNodes();renderEdges();renderProp();appendLog(L);
  }
  const ok=log.filter(L=>L.status==='ok').length;
  setMsg(`任务 #${t.id} 完成 · ${ok} 节点输出结果`,'ok');})();
}
function appendLog(L){const lg=$('log');lg.innerHTML='';
  const d=document.createElement('div');d.className=L.status;
  d.textContent=`#${L.node_id} · ${L.status}${L.ms!==undefined?' · '+L.ms+'ms':''}${L.error?' · '+L.error:''}`;
  lg.appendChild(d);}

// ── 左侧边栏 (Queue / Load / Explorer) ────
function updateQueueBadge(tasks){const t=tasks||srvTasks;const running=t.some(x=>x.state==='running');
  const n=t.filter(x=>x.state==='running').length+t.filter(x=>x.state==='pending').length;
  $('queueCnt').textContent=n;$('queueBadge').classList.toggle('running',running);}
function setTab(t){
  sbTabEt=t;document.querySelectorAll('#sbTabs .tb').forEach(b=>b.classList.toggle('active',b.dataset.t===t));
  if(t==='load')renderLoadTab();else if(t==='explore')renderExploreTab();else renderQueue(srvTasks);}
function renderQueue(tasks){
  srvTasks=tasks||srvTasks;updateQueueBadge(srvTasks);
  if(sbTabEt!=='queue'){$('sbPanel').innerHTML='';return;}
  const p=$('sbPanel');let h='';
  const running=srvTasks.find(t=>t.state==='running');
  const pending=srvTasks.filter(t=>t.state==='pending');
  const done=srvTasks.filter(t=>['ok','err','aborted'].includes(t.state)).slice(0,20);
  h+=`<div class="title">执行队列 (${pending.length+(running?1:0)})</div>`;
  if(running)h+=`<div class="sb-item sel"><span>#${running.id} ${esc(running.name)}</span><span class="st run">● 运行中</span></div>`;
  pending.forEach(it=>{h+=`<div class="sb-item"><span>#${it.id} ${esc(it.name)}</span><span class="st pending">排队</span></div>`;});
  if(!running&&!pending.length)h+=`<div class="hint">队列空闲 — 点击 Run 入队</div>`;
  h+=`<div class="title">最近运行 (${done.length})</div>`;
  if(!done.length)h+=`<div class="hint">暂无记录</div>`;
  done.forEach(it=>{const st=it.state==='ok'?'ok':it.state==='err'?'err':'pending';
    const label=it.state==='ok'?('ok · '+((it.log||[]).filter(L=>L.status==='ok').length))
      :(it.error?('err · '+(it.error||'').slice(0,18)):(it.state||''));
    h+=`<div class="sb-item taskit" data-id="${it.id}" title="点击回看 / 复用"><span>#${it.id} ${esc(it.name)}</span><span class="st ${st}">${label}</span></div>`;});
  p.innerHTML=h;
  p.querySelectorAll('.taskit').forEach(el=>el.onclick=()=>viewTask(Number(el.dataset.id)));}
async function viewTask(tid){
  const r=await fetch('/api/tasks/'+tid);const j=await r.json();
  if(!j.ok||!j.task){toast('任务不存在');return;}
  const t=j.task;
  if(t.workflow&&t.workflow.nodes){pushHist();loadGraph(t.workflow);}
  runAbort=false;
  if(t.state==='ok')animateTask(t);
  setMsg(`已回看任务 #${tid}`,'ok');
}
function renderLoadTab(){
  const p=$('sbPanel');let h='';
  // 模板: 按 category 分类折叠
  const pres=loadList.presets||[];
  const cats={};
  pres.forEach(w=>{const c=w.category||'其他';(cats[c]=cats[c]||[]).push(w);});
  if(pres.length)h+=`<div class="title">模板图库 (${pres.length})</div>`;
  Object.keys(cats).forEach(c=>{
    h+=`<div class="cat-head" data-c="${esc(c)}"><span>▸</span>${esc(c)}<small>${cats[c].length}</small></div>`;
    h+=`<div class="cat-body">`;
    cats[c].forEach(w=>{h+=`<div class="sb-item loadit" data-n="${esc(w.name)}" title="点击加载"><span>★ ${esc(w.name)}</span><small>preset</small></div>`;});
    h+=`</div></div>`;
  });
  h+=`<div class="title">工作流 (本地)</div>`;
  (loadList.workflows||[]).forEach(w=>{h+=`<div class="sb-item loadit" data-n="${esc(w.name)}" title="点击加载"><span>${esc(w.name)}</span><small>json</small></div>`;});
  if(!pres.length&&!(loadList.workflows||[]).length)h+=`<div class="hint">暂无</div>`;
  p.innerHTML=h;
  p.querySelectorAll('.cat-head').forEach(el=>el.onclick=()=>{
    const body=el.nextElementSibling;body.style.display=body.style.display==='none'?'':'none';
    el.style.opacity=body.style.display==='none'?'.5':'1';});
  p.querySelectorAll('.loadit').forEach(el=>el.onclick=()=>loadByName(el.dataset.n));}
function renderExploreTab(){
  const p=$('sbPanel');let h='';
  h+=`<div class="title">模型 & 插件 (${MODELS.length})</div>`;
  MODELS.forEach(m=>{h+=`<div class="sb-item" data-r="${esc(m.route||m.name)}" title="点击添加节点"><span>${esc(m.name)}</span><small>${esc(m.route||'')}</small></div>`;});
  if(!MODELS.length)h+=`<div class="hint">无模型</div>`;
  p.innerHTML=h;
  p.querySelectorAll('.sb-item').forEach(el=>el.onclick=()=>{const r=el.dataset.r;
    const it={t:'model',model:MODELS.find(m=>(m.route||m.name)===r)};addPos={x:120+(nodes.length*10)%180,y:80+(nodes.length*16)%90};addNode(it);});}
async function loadByName(name){if(!name)return;
  const rv=await fetch('/api/workflow/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j=await rv.json();if(!j.ok){toast(j.error);return;}
  pushHist();loadGraph(j.workflow);toast('已加载 '+name);
  $('loadSel').value='';}
function toggleSidebar(force){const hidden=force!==undefined?force:!sidebarCollapsed;
  sidebarCollapsed=!hidden?false:true;$('sidebar').classList.toggle('hidden',!hidden);
  $('sbBar').style.display=hidden?'none':'block';}
$('sbToggle').onclick=()=>toggleSidebar();
$('sbBar').onclick=()=>toggleSidebar(true);
document.querySelectorAll('#sbTabs .tb').forEach(b=>b.onclick=()=>setTab(b.dataset.t));
$('sbInterrupt').onclick=()=>{runAbort=true;
  fetch('/api/workflow/interrupt',{method:'POST'}).then(r=>r.json()).then(j=>{
    setMsg((j&&j.ok)?'已中断当前任务并清空队列':'中断请求失败','warn');
    if(pollTimer){clearInterval(pollTimer);pollTimer=null;}startPoll();})
    .catch(()=>setMsg('中断请求失败','err'));};
$('sbClearQueue').onclick=()=>{
  fetch('/api/workflow/interrupt',{method:'POST'}).then(r=>r.json()).then(()=>{
    if(pollTimer){clearInterval(pollTimer);pollTimer=null;}startPoll();toast('已清空待处理队列');});
  };
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

// ── 保存 / 加载 / 清空 ───────────────────
function wfSerialize(){return{name:$('wfName').value||'wf',version:1,
  nodes:nodes.map(n=>formatNode(n)),edges:JSON.parse(JSON.stringify(edges)),
  groups:JSON.parse(JSON.stringify(groups))};}
function saveNow(){const wf=wfSerialize();
  fetch('/api/workflow/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:wf.name,workflow:wf})})
    .then(r=>r.json()).then(j=>{toast(j.ok?('已保存 '+wf.name):(j.error||'保存失败'));refreshSel();})
    .catch(()=>toast('保存失败'));}
$('btnSave').onclick=saveNow;
$('btnSaveAs').onclick=()=>{const name=prompt('工作流名称',$('wfName').value||'wf');
  if(!name)return;$('wfName').value=name;saveNow();};
$('btnLoad').onclick=async()=>{
  const name=$('loadSel').value;if(!name)return;
  const r=await fetch('/api/workflow/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j=await r.json();if(!j.ok){toast(j.error);return;}
  pushHist();loadGraph(j.workflow);toast('已加载 '+name);
};
$('btnClear').onclick=()=>{pushHist();nodes=[];edges=[];selection=null;selEdge=null;selSet.clear();
  applySel();renderNodes();renderEdges();updateMap();renderProp();$('log').innerHTML='<small>已清空</small>';};
function loadGraph(w){nodes=(w.nodes||[]).map(n=>({...n,title:n.title||'',collapsed:!!n.collapsed,
  note:n.note||'',bypass:!!n.bypass,mute:!!n.mute}));
  edges=w.edges||[];
  groups=(w.groups||[]).map(g=>({...g,nodes:[...g.nodes],fold:!!g.fold,muted:!!g.muted}))||[];
  nidSeq=nodes.reduce((m,n)=>{const k=parseInt((n.id||'').replace(/\D/g,''))||0;return Math.max(m,k);},0);
  eidSeq=edges.reduce((m,e)=>{const k=parseInt((e.id||'').replace(/\D/g,''))||0;return Math.max(m,k);},0);
  selection=null;selEdge=null;selSet.clear();applySel();
  renderNodes();renderEdges();updateMap();renderProp();updateHist();}
let loadList={presets:[],workflows:[]};
async function refreshSel(){
  const r=await fetch('/api/workflow/list');const j=await r.json();
  loadList={presets:j.presets||[],workflows:j.workflows||[]};
  const sel=$('loadSel');sel.innerHTML='';const seen=new Set();
  (j.presets||[]).forEach(p=>{sel.appendChild(new Option('★ '+p.name,p.name));seen.add(p.name);});
  (j.workflows||[]).forEach(w=>{if(!seen.has(w.name))sel.appendChild(new Option(w.name,w.name));});
  renderQueue();
}

// ── 杂项 ─────────────────────────────────
function setMsg(s,cls){const m=$('stMsg');m.textContent=s;m.className=(cls==='ok'?'s-ok':cls==='err'?'s-err':'');}
function toast(msg){const t=document.createElement('div');t.textContent=msg;
  t.style.cssText='position:fixed;right:14px;bottom:34px;background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:7px 14px;font-size:12px;z-index:50;box-shadow:0 4px 16px rgba(0,0,0,.5)';
  document.body.appendChild(t);setTimeout(()=>t.remove(),1600);}
function updateCount(){
  const selN=selSet.size;
  let txt=`${nodes.length} 节点 · ${edges.length} 边`;
  if(selN||selEdge)txt=`选中 ${selN} 节点${selEdge?' · 1 边':''} (${txt})`;
  $('stCount').textContent=txt;}

// ── 初始化 ───────────────────────────────
async function init(){
  const r=await fetch('/api/workflow/models');const j=await r.json();
  MODELS=j.models||[];
  refreshSel();
  setTab('queue');
  window.setInterval(updateCount,400);updateCount();
}
function firstRender(){
  applyView();
  const d=document.createElement('div');
  // 提示空画布辅助
}
// 首次: 加载预设 "Rust 代码分类" 作为示例画布
(async()=>{await init();
  const r=await fetch('/api/workflow/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:'Rust 代码分类'})});
  const j=await r.json();if(j.ok)loadGraph(j.workflow);
  applyView();updateMap();
})().catch(console.error);
</script></body></html>
"""