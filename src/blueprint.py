"""预制存档 (蓝图 Blueprint, v0.19.0) — 走 ComfyUI 规范的表层工作流 + 模型存档

「蓝图」是一个把表层节点图工作流 (workflow) 与模型存档打包成单个
分发文件的自包含存档。设计要点:

- 后缀: .blueprint.zip
- 压缩: ZIP_STORED (不压缩) — 内容多为已压缩模型包, 再压无益, 且便于
  直接查看/替换内部文件
- 工作流格式: 遵循 ComfyUI Workflow JSON v1.0 规范 (nodes/links/groups/
  version/config/state), 可由 ComfyUI 前端直接载入; 本模块同时提供
  与分布式项目内部节点图格式 (节点节点图引擎) 的双向转换
- 模型存档: 包内 models/ 目录, 复用 .CuteMamen / .dfpkg 独立包文件,
  原样打包不重压

包内布局:

    <name>.blueprint.zip/
        manifest.json            ← 清单 (format=comfyui-workflow-v1)
        workflow.json            ← ComfyUI v1.0 规范工作流
        models/<file>            ← 模型存档 (0..n 个, 原样拷贝)

模块为纯数据层 (仅标准库), 不依赖内核, 便于测试与复用。
"""

import json
import os
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# 蓝图规范版本 / 清单格式标记
BLUEPRINT_FORMAT = "comfyui-workflow-v1"
BLUEPRINT_SUFFIX = ".blueprint.zip"

# 默认存档目录 (与 cache/workflows 同级)
BLUEPRINT_DIR = os.path.join("cache", "blueprints")

# 名称清洗: 仅保留中英文/数字/下划线/连字符/点, 防路径穿越
_NAME_CLEAN = re.compile(r"[^\w\u4e00-\u9fff._-]+")

# ComfyUI 节点槽位定义 (与 workflow_ui.TYPES 保持一致)
_NODE_PORTS = {
    "input": {"ins": [], "outs": ["data", "meta"]},
    "model": {"ins": ["seq", "config"], "outs": ["logits", "loss"]},
    "output": {"ins": ["data", "logits", "loss"], "outs": []},
    "reroute": {"ins": ["in"], "outs": ["out"]},
}
# 兼容未知类型 → 直通
_DEFAULT_PORTS = {"ins": ["in"], "outs": ["out"]}
_NODE_W = 210


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _clean_name(name: str) -> str:
    name = _NAME_CLEAN.sub("_", name or "").strip().strip(".")
    return name or "blueprint"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ═══════════════════════════════════════════════════════════════
# 内部节点图格式 ↔ ComfyUI v1.0 规范 转换
# ═══════════════════════════════════════════════════════════════
# 内部格式 (前端节点图引擎):
#   {name, version, nodes:[{id,type,x,y,params,title,collapsed,note,bypass,mute}],
#    edges:[{id,from,to,from_port,to_port,to_name}],
#    groups:[{id,name,nodes,color,fold,muted}]}
# ComfyUI v1.0 规范格式:
#   {version:1, config, state:{lastNodeId,lastLinkId,lastGroupid,...},
#    groups:[{title,bounding:[x,y,w,h],color,font_size,locked}],
#    nodes:[{id,type,pos:[x,y],size:[w,h],flags,order,mode,
#            inputs:[{name,type,link,slot_index}],outputs:[...],
#            properties,widgets_values}],
#    links:[[id,origin_id,origin_slot,target_id,target_slot,type]]}


def to_comfy(graph: Dict[str, Any]) -> Dict[str, Any]:
    """内部节点图 → ComfyUI v1.0 规范工作流 (仅映射数据, 不写盘)"""
    nodes_in = graph.get("nodes", [])
    edges_in = graph.get("edges", [])
    groups_in = graph.get("groups", [])

    # 节点: 计算 id → 输出槽位名索引 (供 links 定位 origin_slot)
    out_slot: Dict[Any, Dict[str, int]] = {}
    comfy_nodes: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes_in):
        ports = _NODE_PORTS.get(n.get("type"), _DEFAULT_PORTS)
        ins_names, outs_names = ports["ins"], ports["outs"]
        ins = [{"name": nm, "type": "*", "link": None, "slot_index": si}
               for si, nm in enumerate(ins_names)]
        outs = [{"name": nm, "type": "*", "links": [], "slot_index": si}
                for si, nm in enumerate(outs_names)]
        out_slot[n.get("id")] = {nm: si for si, nm in enumerate(outs_names)}
        size = _node_size(n)
        comfy_nodes.append({
            "id": n.get("id"),
            "type": n.get("type", "reroute"),
            "pos": [float(n.get("x", 0)), float(n.get("y", 0))],
            "size": [size[0], size[1]],
            "flags": {"collapsed": bool(n.get("collapsed"))},
            "order": i,
            "mode": 0 if not n.get("bypass") else 2,  # 2=bypass (ComfyUI)
            "inputs": ins,
            "outputs": outs,
            "properties": {"title": n.get("title", "")},
            "widgets_values": _params_widgets(n.get("params") or {}),
        })

    # links: 内部 edge → ComfyUI link 元组
    links: List[Any] = []
    for e in edges_in:
        eid = e.get("id")
        src = e.get("from")
        dst = e.get("to")
        from_name = e.get("from_port") or "out"
        to_name = e.get("to_name") or e.get("to_port") or "in"
        oslot = (out_slot.get(src) or {}).get(from_name, 0)
        # 目标输入槽位: 按端口名在节点 inputs 中定位
        tins = _node_ins(dst, nodes_in)
        tslo = tins.index(to_name) if to_name in tins else 0
        links.append([eid, src, oslot, dst, tslo, "*"])

    # groups
    comfy_groups: List[Dict[str, Any]] = []
    for gi, g in enumerate(groups_in):
        ms = [n for n in nodes_in if n.get("id") in (g.get("nodes") or [])]
        if not ms:
            continue
        x0 = min(n.get("x", 0) for n in ms) - 12
        y0 = min(n.get("y", 0) for n in ms) - 20
        x1 = max(n.get("x", 0) + _NODE_W for n in ms) + 12
        y1 = max(n.get("y", 0) + _node_size(n)[1] for n in ms) + 12
        comfy_groups.append({
            "title": g.get("name", "组"),
            "bounding": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "color": g.get("color", "#7aa2ff"),
            "font_size": 12,
            "locked": False,
        })

    return {
        "version": 1,
        "config": {"links_ontop": False, "align_to_grid": False},
        "state": {
            "lastNodeId": _max_numeric_id(nodes_in),
            "lastLinkId": _max_numeric_id(edges_in),
            "lastGroupid": len(comfy_groups),
            "lastRerouteId": 0,
        },
        "groups": comfy_groups,
        "nodes": comfy_nodes,
        "links": links,
    }


def from_comfy(wf: Dict[str, Any]) -> Dict[str, Any]:
    """ComfyUI v1.0 规范工作流 → 内部节点图格式"""
    comfy_nodes = wf.get("nodes", [])
    comfy_links = wf.get("links", []) or []
    comfy_groups = wf.get("groups", []) or []

    # 节点: 还原输入槽位名 (按 slots), 输出槽位名
    nodes: List[Dict[str, Any]] = []
    for n in comfy_nodes:
        ports = _NODE_PORTS.get(n.get("type"), _DEFAULT_PORTS)
        ins_names, outs_names = ports["ins"], ports["outs"]
        # 优先取节点自带槽位名, 否则用默认
        real_ins = _slot_names(ins_names, n.get("inputs"))
        real_outs = _slot_names(outs_names, n.get("outputs"))
        params = _widgets_to_params(n.get("widgets_values"))
        nodes.append({
            "id": n.get("id"),
            "type": n.get("type", "reroute"),
            "x": float((n.get("pos") or [0, 0])[0]),
            "y": float((n.get("pos") or [0, 0])[1]),
            "params": params,
            "title": (n.get("properties") or {}).get("title", ""),
            "collapsed": bool((n.get("flags") or {}).get("collapsed")),
            "note": (n.get("properties") or {}).get("note", ""),
            "bypass": bool(n.get("mode") == 2),
            "mute": False,
            "_ins": real_ins,
            "_outs": real_outs,
        })

    # 节点 id → (输入槽位名, 输出槽位名)
    in_name: Dict[Any, List[str]] = {}
    out_name: Dict[Any, List[str]] = {}
    for n in nodes:
        in_name[n["id"]] = n["_ins"]
        out_name[n["id"]] = n["_outs"]

    # links → edges
    edges: List[Dict[str, Any]] = []
    for lnk in comfy_links:
        eid, src, oslot, dst, tslo = lnk[0], lnk[1], lnk[2], lnk[3], lnk[4]
        from_port = _safe_idx(out_name.get(src), oslot, "out")
        to_name_ = _safe_idx(in_name.get(dst), tslo, "in")
        edges.append({
            "id": eid, "from": src, "from_port": from_port,
            "to": dst, "to_port": "in", "to_name": to_name_,
        })

    # groups
    groups: List[Dict[str, Any]] = []
    for gi, g in enumerate(comfy_groups):
        b = g.get("bounding") or [0, 0, 0, 0]
        # 组内节点: 落在 bounding 内的节点
        x0, y0, w, h = b
        member_ids = [n["id"] for n in nodes
                      if x0 - 1 <= n["x"] <= x0 + w + 1
                      and y0 - 1 <= n["y"] <= y0 + h + 1]
        groups.append({
            "id": "g" + str(gi + 1), "name": g.get("title", "组"),
            "nodes": member_ids, "color": g.get("color", "#7aa2ff"),
            "fold": False, "muted": False,
        })

    return {
        "name": (wf.get("_meta") or {}).get("name", "workflow"),
        "version": 1,
        "nodes": nodes, "edges": edges, "groups": groups,
    }


# ── 转换辅助 ─────────────────────────────────────────────

def _node_ins(nid: Any, nodes_in: List[Dict[str, Any]]) -> List[str]:
    """取节点的输入端口名列表"""
    for n in nodes_in:
        if n.get("id") == nid:
            ports = _NODE_PORTS.get(n.get("type"), _DEFAULT_PORTS)
            return list(ports["ins"])
    return []


def _node_size(n: Dict[str, Any]) -> Tuple[float, float]:
    """估算节点高 (标题 + 端口行 + 控件)"""
    ports = _NODE_PORTS.get(n.get("type"), _DEFAULT_PORTS)
    rows = max(len(ports["ins"]), len(ports["outs"]))
    h = 30 + rows * 24 + 10
    if n.get("collapsed"):
        h = 30
    return float(_NODE_W), float(h)


def _params_widgets(params: Dict[str, Any]) -> List[Any]:
    """内部 params → ComfyUI widgets_values (扁平标量列表)"""
    vals: List[Any] = []
    for k in ("topic", "route", "data"):
        if k in params:
            v = params[k]
            vals.append(None if v is None else v)
    return vals


def _widgets_to_params(vals: Any) -> Dict[str, Any]:
    """ComfyUI widgets_values → 内部 params (尽力还原)"""
    if not isinstance(vals, list):
        return {}
    keys = ("topic", "route", "data")
    params: Dict[str, Any] = {}
    for i, v in enumerate(vals[:3]):
        if i < len(keys) and v is not None:
            params[keys[i]] = v
    return params


def _slot_names(defaults: List[str], slots: Any) -> List[str]:
    """取节点 inputs/outputs 的 name 列表, 缺则用默认"""
    if not slots:
        return list(defaults)
    names = [s.get("name") for s in slots if isinstance(s, dict) and s.get("name")]
    return names if names else list(defaults)


def _safe_idx(lst: Any, i: Any, fallback: str) -> str:
    try:
        idx = int(i)
    except (TypeError, ValueError):
        return fallback
    if lst is None or idx >= len(lst) or idx < 0:
        return fallback
    return lst[idx]


def _max_numeric_id(items: List[Dict[str, Any]]) -> int:
    m = 0
    for it in items:
        try:
            m = max(m, int(it.get("id")))
        except (TypeError, ValueError):
            pass
    return m


# ═══════════════════════════════════════════════════════════════
# 蓝图打包 / 解包
# ═══════════════════════════════════════════════════════════════

def save_blueprint(name: str, workflow: Dict[str, Any],
                   models: Optional[List[str]] = None,
                   out_dir: str = BLUEPRINT_DIR,
                   core_version: str = "0.19.0") -> str:
    """把工作流(内部格式)+模型存档打包成 .blueprint.zip (ZIP_STORED 不压缩)

    workflow: 内部节点图 dict (或已是 ComfyUI 规范, 由 keys 自动识别)
    models:   可选, 模型存档文件路径列表 (.CuteMamen / .dfpkg), 原样拷入
    out_dir:  输出目录 (默认 cache/blueprints)
    返回: 生成的蓝图文件绝对路径
    """
    name = _clean_name(name)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, name + BLUEPRINT_SUFFIX)

    comfy = workflow if ("links" in workflow and "nodes" in workflow
                         and workflow.get("version") in (1, 1.0)) \
        else to_comfy(workflow)

    manifest = {
        "format": BLUEPRINT_FORMAT,
        "name": name,
        "archived_at": _now(),
        "core_version": core_version,
        "models": [],
    }

    model_pairs: List[Tuple[str, str]] = []
    if models:
        for mp in models:
            if not os.path.isfile(mp):
                continue
            model_pairs.append((mp, "models/" + os.path.basename(mp)))
    manifest["models"] = [arc for _, arc in model_pairs]

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("workflow.json",
                    json.dumps(comfy, ensure_ascii=False, indent=2))
        for src, arc in model_pairs:
            zf.write(src, arc)
    return dest


def load_blueprint(path: str) -> Tuple[Dict[str, Any], Dict[str, Any],
                                       Dict[str, bytes]]:
    """解包蓝图 → (workflow_internal, manifest, models:{arc:bytes})

    workflow 始终转回内部节点图格式 (可直接交给节点图引擎/前端载入)。
    models 返回包内字节 (arc → bytes), 由调用方决定落盘位置。
    """
    path = str(path)
    if not path.endswith(BLUEPRINT_SUFFIX):
        path += BLUEPRINT_SUFFIX
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        comfy = json.loads(zf.read("workflow.json").decode("utf-8"))
        workflow = from_comfy(comfy)
        models: Dict[str, bytes] = {}
        for m in manifest.get("models") or []:
            try:
                models[m] = zf.read(m)
            except KeyError:
                pass
    return workflow, manifest, models


def list_blueprints(directory: str = BLUEPRINT_DIR) -> List[Dict[str, Any]]:
    """列出目录下的蓝图 (读 manifest, 不落地解包)"""
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(BLUEPRINT_SUFFIX):
            continue
        p = os.path.join(directory, fn)
        try:
            with zipfile.ZipFile(p, "r") as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            out.append({"file": fn, "name": manifest.get("name", fn),
                        "format": manifest.get("format"),
                        "archived_at": manifest.get("archived_at"),
                        "core_version": manifest.get("core_version"),
                        "models": manifest.get("models") or []})
        except Exception:
            out.append({"file": fn, "name": fn[:-len(BLUEPRINT_SUFFIX)],
                        "error": "manifest 不可读"})
    return out
