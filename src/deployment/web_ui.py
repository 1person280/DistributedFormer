# -*- coding: utf-8 -*-
"""
Web 图形界面 (v0.19.0) — dformer ui

零第三方依赖的本地浏览器控制台, 复用 openai_server 的 stdlib http.server
风格 (ThreadingHTTPServer + BaseHTTPRequestHandler), 单页 HTML/JS 内嵌,
无外部资源/框架。

路由:
  GET  /                       → 单页图形化控制台
  GET  /workflow               → ComfyUI 式节点图工作流页
  GET  /api/status             → {version, dim, plugins:[{name,route,base_model}]}
  GET  /api/benchmarks         → 三真实任务基准汇总 (benchmark_all_results.json)
  POST /api/think              → body {topic, data}; kernel.think(...) → 结果
  GET  /api/workflow/models    → 节点调色板 (内核中的真实模型/插件)
  GET  /api/workflow/list      → {workflows, presets}
  POST /api/workflow/run       → body {workflow}; 拓扑执行 → {results, log}
  POST /api/workflow/save      → body {name, workflow}
  POST /api/workflow/load      → body {name} → {workflow, is_preset}

服务端任务队列 (v0.19.0):
  POST /api/workflow/submit    → body {workflow, name} → {ok, task_id} 立即返回
  GET  /api/tasks              → 任务列表 (运行中 + 已完成), 按状态/新→旧
  GET  /api/tasks/<id>         → 单任务详情 (含 results/log/error/workflow)
  POST /api/workflow/interrupt → 终止当前 + 清空待处理队列
  POST /api/workflow/export_comfy → body {workflow} → ComfyUI {nodes,links,groups}
  POST /api/workflow/import_comfy → body {wf} → 内部 {nodes,edges,groups}

启动: dformer ui --host 127.0.0.1 --port 8011 --depth 1 --dim 16
"""

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import numpy as np

from src.cutemamen.kernel import CuteMamenKernel, DEFAULT_PKG_DIR
from src.deployment.workflow_ui import WorkflowEngine, WORKFLOW_PAGE
from src.deployment.task_queue import TaskManager
from src.blueprint import to_comfy, from_comfy

_UI_VERSION = "0.19.0"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _coerce_data(s: str) -> Any:
    """命令里 `topic: 数据` 的数据侧: 尝试 JSON 解析, 否则按字符串"""
    if not s:
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


class UIServer:
    """持有内核/插件/基准数据, 供 HTTP handler 读写"""

    def __init__(self, depth: int = 1, dim: int = 16):
        self.depth = depth
        self.dim = dim
        self.kernel = CuteMamenKernel(dim=dim, pkg_dir=DEFAULT_PKG_DIR)
        self.started = time.time()
        self.plugins = self._build_plugins()
        self.workflow = WorkflowEngine(self.kernel)
        self.tasks = TaskManager(self.workflow)

    def _build_plugins(self) -> List[Dict[str, str]]:
        from src.cutemamen.rust_coding import RustCodingPlugin
        from src.cutemamen.video_making import VideoMakingPlugin
        from src.cutemamen.chat import ChatPlugin

        out = []
        for p in (RustCodingPlugin(name="ui-rust"),
                  VideoMakingPlugin(name="ui-video"),
                  ChatPlugin(name="ui-chat")):
            self.kernel.mount(p)
            out.append({
                "name": p.name,
                "route": p.route,
                "base_model": getattr(p, "base_model", None),
            })
        return out

    def status(self) -> Dict[str, Any]:
        return {
            "version": _UI_VERSION,
            "depth": self.depth,
            "dim": self.dim,
            "uptime_s": int(time.time() - self.started),
            "plugins": self.plugins,
        }

    def think(self, topic: str, data: Any) -> Dict[str, Any]:
        try:
            result = self.kernel.think({"topic": topic, "data": data})
            return {"ok": True, "result": result}
        except Exception as exc:  # 异常也返回给前端展示
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def command(self, text: str) -> Dict[str, Any]:
        """命令输入(回车执行): 直接文本→chat; `topic: 数据` 指定路由; /命令"""
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "空命令"}
        low = text.lower()
        if low in ("/help", "help", "帮助", "/?"):
            return {"ok": True, "result": {
                "reply": "直接输入文本→chat 模型; `topic: 数据` 指定路由;\n"
                         "命令: /stats 统计 · /plugins 插件 · /help 帮助 · /exit 退出"}}
        if low in ("/stats", "stats", "统计"):
            return {"ok": True, "result": self.status()}
        if low in ("/plugins", "plugins", "插件"):
            return {"ok": True, "result": {"plugins": self.plugins}}
        if low in ("/exit", "/quit", "exit", "退出"):
            return {"ok": True, "result": {"reply": "再见"}}
        if ":" in text:
            topic, _, data = text.partition(":")
            topic = topic.strip()
            if topic and not any(ch.isspace() for ch in topic[:1]):
                return self.think(topic, _coerce_data(data.strip()))
        return self.think("chat", text)

    def benchmarks(self) -> Dict[str, Any]:
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "experiments")
        for name in ("benchmark_all_results.json",
                     "multi_task_results.json"):
            path = os.path.join(base, name)
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return {"source": name, "data": json.load(f)}
                except Exception:
                    continue
        return {"source": None, "data": None}


_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><title>DistributedFormer v0.19.0 图形控制台</title>
<style>
  :root{--bg:#0f1420;--card:#1a2133;--line:#2a3350;--fg:#e6ecff;--mut:#8aa0c8;
        --acc:#5b8cff;--ok:#35d07f;--warn:#ffb454;}
  *{box-sizing:border-box}body{margin:0;font:14px/1.6 system-ui,Segoe UI,sans-serif;
    background:var(--bg);color:var(--fg)}
  header{padding:18px 28px;border-bottom:1px solid var(--line);
    display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header h1{margin:0;font-size:18px}
  .badge{background:var(--acc);color:#fff;border-radius:12px;padding:2px 10px;
    font-size:12px}
  main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:22px 28px}
  @media(max-width:900px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
    padding:16px 18px}
  .card h2{margin:0 0 10px;font-size:15px;color:var(--mut);font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
  th{color:var(--mut);font-weight:600}
  input,select,textarea{width:100%;background:#0d1220;color:var(--fg);
    border:1px solid var(--line);border-radius:8px;padding:8px 10px;
    font:inherit;margin-bottom:10px}
  button{background:var(--acc);color:#fff;border:0;border-radius:8px;
    padding:9px 16px;font:inherit;cursor:pointer}
  button:disabled{opacity:.5}
  pre{background:#0d1220;border:1px solid var(--line);border-radius:8px;
    padding:10px;overflow:auto;max-height:340px;font-size:12px}
  .ok{color:var(--ok)}.err{color:#ff6b6b}
  .kv{display:flex;gap:8px;flex-wrap:wrap}.kv span{background:#0d1220;
    border:1px solid var(--line);padding:4px 10px;border-radius:8px}
  .pill{padding:1px 8px;border-radius:10px;font-size:11px;border:1px solid var(--line)}
  #cmdbar{display:flex;align-items:center;gap:10px;margin:0 28px;padding:8px 14px;
    background:#0d1220;border:1px solid var(--line);border-radius:10px}
  #cmdbar .gt{color:var(--ok);font-weight:700;font-size:16px;user-select:none}
  #cmdbar input{flex:1;background:transparent;border:0;outline:0;color:var(--fg);
    font:inherit;margin:0;padding:4px 0}
  #cmdbar input::placeholder{color:var(--mut)}
  #cmdOut{white-space:pre-wrap;margin:8px 28px 0;font-size:12px}
</style></head>
<body>
<header>
  <h1>DistributedFormer <span class="badge">v0.19.0 · 图形化</span></h1>
  <nav style="display:flex;gap:8px;margin-left:8px">
    <a href="/" style="color:var(--acc);padding:2px 10px;text-decoration:none;border:1px solid var(--line);border-radius:8px">控制台</a>
    <a href="/workflow" style="color:var(--fg);padding:2px 10px;text-decoration:none;border:1px solid var(--line);border-radius:8px">工作流</a>
  </nav>
  <div class="kv" id="status"></div>
</header>
<div id="cmdbar"><span class="gt">›</span>
  <input id="cmd" autocomplete="off" spellcheck="false"
    placeholder="Type here to command · 输入指令回车执行（如 你好 / chat: 你好 / /stats /plugins /help）">
</div>
<pre id="cmdOut" style="max-height:120px;color:var(--ok)"></pre>
<main>
  <div class="card">
    <h2>思考控制台 (CuteMamen 内核)</h2>
    <input id="topic" list="topics" placeholder="topic, 如 rust / video / chat">
    <datalist id="topics">
      <option value="rust"><option value="video"><option value="chat">
    </datalist>
    <textarea id="data" rows="4" placeholder='data, 如 {"code": "fn main(){}"}'></textarea>
    <button id="thinkBtn">思考 (think)</button>
    <pre id="out">// 结果将显示在这里</pre>
  </div>
  <div class="card">
    <h2>全面训练 · 三真实任务基准 (benchmark-all)</h2>
    <div id="bench"><table><tr><th>任务</th><th>随机基线</th>
      <th>val_acc</th><th>超基线折数</th></tr><tr><td colspan="4">
      未找到基准结果 (运行 dformer benchmark-all)</td></tr></table></div>
  </div>
</main>
<script>
const $=id=>document.getElementById(id);
async function j(url,opt){const r=await fetch(url,opt);return r.json();}
async function loadStatus(){
  const s=await j('/api/status');
  $('status').innerHTML=
    `<span>depth=${s.depth}</span><span>dim=${s.dim}</span>`+
    `<span>插件: ${s.plugins.map(p=>p.route).join(', ')}</span>`+
    `<span>up ${s.uptime_s}s</span>`;
}
async function loadBench(){
  const b=await j('/api/benchmarks');
  if(!b.data){return;}
  const tasks=b.data.tasks||[];
  $('bench').innerHTML='<table><tr><th>任务</th><th>随机基线</th><th>val_acc</th>'+
    '<th>超基线折数</th></tr>'+tasks.map(t=>
    `<tr><td>${t.task||''}</td><td>${(t.random_baseline*100).toFixed(0)}%</td>`+
    `<td><b>${(t.val_acc_mean*100).toFixed(1)}%</b> ± ${(t.val_acc_std*100).toFixed(1)}%</td>`+
    `<td>${t.folds_beat_random}/${t.total_folds}</td></tr>`).join('')+'</table>';
}
$('thinkBtn').onclick=async()=>{
  $('thinkBtn').disabled=true; $('out').textContent='思考中…';
  let data=$('data').value.trim();
  try{ data=data?JSON.parse(data):null; }catch(e){ /* 保留字符串 */ }
  const r=await j('/api/think',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({topic:$('topic').value,data})});
  $('out').textContent=JSON.stringify(r.result!==undefined?r.result:r,null,2);
  $('out').className=r.ok?'ok':'err';
  $('thinkBtn').disabled=false;
};
// 命令输入: 回车执行, 回显到 cmdOut
async function runCmd(){
  const c=$('cmd').value.trim(); if(!c)return;
  $('cmd').value='';
  $('cmdOut').textContent='› '+c;
  const r=await j('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({command:c})});
  const show=r.result!==undefined?r.result:(r.error?{error:r.error}:r);
  $('cmdOut').textContent='› '+c+'\\n'+(typeof show==='string'?show:JSON.stringify(show,null,2));
  $('cmdOut').className=r.ok?'ok':'err';
}
$('cmd').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();runCmd();}});
loadStatus();loadBench();
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "DistributedFormerUI/0.19.0"

    @property
    def ui(self) -> UIServer:
        return self.server.ui  # type: ignore[attr-defined]

    # ── 工具 ──────────────────────────────────────────────
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any):
        payload = json.dumps(obj, ensure_ascii=False,
                             default=_json_default).encode("utf-8")
        self._send(code, payload)

    def log_message(self, fmt, *args):  # 精简日志
        print(f"[ui] {self.command} {self.path} — {fmt % args}")

    def _read_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            return {"_bad": f"bad json: {exc}"}

    # ── GET ───────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _PAGE.encode("utf-8")
            self._send(200, body, "text/html")
        elif self.path == "/workflow":
            self._send(200, WORKFLOW_PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/status":
            self._json(200, self.ui.status())
        elif self.path == "/api/benchmarks":
            self._json(200, self.ui.benchmarks())
        elif self.path == "/api/workflow/models":
            self._json(200, {"ok": True, "models": self.ui.workflow.list_models()})
        elif self.path == "/api/workflow/list":
            self._json(200, {"ok": True, **self.ui.workflow.list_workflows()})
        elif self.path == "/api/tasks":
            self._json(200, {"ok": True, "tasks": self.ui.tasks.status()})
        elif re.fullmatch(r"/api/tasks/\d+", self.path):
            tid = int(self.path.rsplit("/", 1)[1])
            task = self.ui.tasks.get(tid)
            if task is None:
                self._json(404, {"ok": False, "error": f"任务 {tid} 不存在"})
            else:
                self._json(200, {"ok": True, "task": task})
        else:
            self._json(404, {"error": "not found"})

    # ── POST ──────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/api/think":
            payload = self._read_body()
            topic = payload.get("topic", "")
            data = payload.get("data")
            return self._json(200, self.ui.think(topic, data))
        if self.path == "/api/command":
            payload = self._read_body()
            return self._json(200, self.ui.command(payload.get("command", "")))
        if self.path == "/api/workflow/run":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, self.ui.workflow.run(payload.get("workflow", {})))
        if self.path == "/api/workflow/save":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, self.ui.workflow.save(
                payload.get("name", ""), payload.get("workflow", {})))
        if self.path == "/api/workflow/load":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, self.ui.workflow.load(payload.get("name", "")))
        if self.path == "/api/workflow/submit":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, self.ui.tasks.submit(
                payload.get("workflow", {}), payload.get("name")))
        if self.path == "/api/workflow/interrupt":
            return self._json(200, self.ui.tasks.interrupt())
        if self.path == "/api/workflow/export_comfy":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, {"ok": True,
                                    "workflow": to_comfy(payload.get("workflow", {}))})
        if self.path == "/api/workflow/import_comfy":
            payload = self._read_body()
            if "_bad" in payload:
                return self._json(400, {"ok": False, "error": payload["_bad"]})
            return self._json(200, {"ok": True,
                                    "workflow": from_comfy(payload.get("wf", {}))})
        return self._json(404, {"error": "not found"})


def serve(host: str = "127.0.0.1", port: int = 8011,
          depth: int = 1, dim: int = 16) -> None:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.ui = UIServer(depth=depth, dim=dim)  # type: ignore[attr-defined]
    print(f"DistributedFormer v{_UI_VERSION} Web 图形界面:")
    print(f"  → http://{host}:{port}/  (Ctrl+C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] 已退出")
        server.shutdown()


def main(host: str = "127.0.0.1", port: int = 8011,
         depth: int = 1, dim: int = 16) -> None:
    serve(host=host, port=port, depth=depth, dim=dim)


if __name__ == "__main__":
    main()
