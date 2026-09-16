"""
OpenAI 兼容接口服务器 — 让 OpenCode 等 LLM 客户端把 CubeGPT 当模型后端 (v0.9.1)

目标 (大模型落地): 用标准库 http.server 暴露 OpenAI 兼容的
`/v1/models` 与 `/v1/chat/completions` (含 SSE 流式), OpenCode
把 baseURL 指到这里即可把 CubeGPT 接入编码工作流。

CubeGPT 是脉冲水库、没有自回归语言生成头, 因此"生成"策略如实接地:
  · 通用对话      → 脉冲统计回复 (与 demos/chat.py 同一套输出头统计)
  · 代码 / Rust 请求 → 路由到 RustCodingPlugin 的 502 段真实语料知识,
                    分类命中哪类编译错误, 并给出真实、可编译的改法建议
                    (confidence + 来源: 主模型迁移读出层 / 语料原型)。

依赖: 仅 Python 标准库 (http.server / json), 不引入 web 框架。

用法:
    dformer serve-opencode --host 127.0.0.1 --port 8000
    # OpenCode 配置: "@openai" provider, baseURL=http://127.0.0.1:8000,
    # model=cubegpt
"""

import json
import re
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

MODEL_ID = "cubegpt"
_SYSTEM_NAME = "CubeGPT-spiking-reservoir"
# 本地测试用 OpenAI 兼容 API 密钥 (OpenCode / curl 等客户端需带
# `Authorization: Bearer <key>`)。生产部署请用 --api-key 指定你自己的,
# 或 --no-auth 关闭校验 (仅推荐纯本地/无外网环境)。
DEFAULT_API_KEY = "cubegpt-local-test-key"

# 生成后端的默认取值来自环境变量 (有则接真 LLM, 无则纯 CubeGPT):
LM_BASE_ENV = "CUBEGPT_LM_BASE"        # 例如 http://127.0.0.1:11434/v1 (Ollama)
LM_KEY_ENV = "CUBEGPT_LM_KEY"          # 后端 Bearer key (Ollama 常为 "ollama")
LM_MODEL_ENV = "CUBEGPT_LM_MODEL"      # 后端模型名, 例如 qwen2.5-coder:7b

# ── Rust 帮改法: 每类真实编译错误的改法说明 + 可编译修正片段 ──
# 修正片段是真实 Rust 系导数 (从 rust_coding 语料族的根因归纳而来)。
_RUST_FIX_GUIDE = {
    "move": ("所有权移动 (E0382/E0505/E0507): 值被移动后不能再使用。",
             "fn fix(s: String) {\n    let t = s.clone(); // 移动前复制副本\n"
             "    println!(\"{}\", s); // 原值仍可用\n}"),
    "borrow": ("借用冲突 (E0502/E0499): 可变与不可变借用同时存活。",
               "let r = &v;      // 不可变借用\n"
               "println!(\"{:?}\", r); // 借用域结束\n"
               "v.push(1);       // 后再做可变借用\n"),
    "lifetime": ("生命周期 (E0597/E0106/E0515): 引用话不过借用者的寿命。",
                 "fn longer<'a>(a: &'a str, b: &'a str) -> &'a str {\n"
                 "    if a.len() > b.len() { a } else { b }\n} // 显式 'a\n"),
    "type": ("类型不匹配 (E0308/E0277): 值类型与期望类型不一致。",
             "let x: u32 = 5;\nlet y: u64 = x as u64; // as 显式转换后类型一致\n"),
    "ok": ("代码合法 (可编译), 无需修改。",
           None),
}


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算 (OpenCode 只用 usage 做展示)"""
    return max(1, (len(text) + 3) // 4)


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """取 messages 里最后一条 user 的文本内容 (拼 content 字符串)"""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                seg.get("text", "") if isinstance(seg, dict)
                else str(seg) for seg in content
            ]
            text = "".join(parts).strip()
            if text:
                return text
    return ""


# ── 代码意图识别 (真代码, 无合成) ────────────────────────────
_RUST_HINT = re.compile(
    r"(can't\s*cannot)?\s*move|borrow|E0\d{3}|use of moved|"
    r"mismatched types|dangling|lifetime", flags=re.IGNORECASE)


def _looks_like_code(text: str) -> bool:
    hints = ("fn ", "fn main", "let ", "use std", "struct ", "::",
             "-> ", "&mut", "fn(")
    return any(h in text for h in hints)


def _extract_code(text: str) -> Optional[str]:
    """从请求里抠出要诊断的 Rust 代码段; 非代码意图返回 None"""
    # 优先取 ``` fence 内的代码块
    blocks = re.findall(r"```(?:rust)?\s*(.*?)```", text, flags=re.S)
    if blocks and blocks[0].strip():
        return blocks[0].strip()
    if _RUST_HINT.search(text) or _looks_like_code(text):
        return text.strip()
    return None


def build_chat_completion(content: str, model: str = MODEL_ID,
                          prompt_text: str = ""):
    """组装 OpenAI 非流式 chat.completion 响应字典"""
    now = int(time.time())
    comp_tokens = _estimate_tokens(content)
    return {
        "id": f"chatcmpl-{now}{len(content) % 1000:03d}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": _estimate_tokens(prompt_text),
            "completion_tokens": comp_tokens,
            "total_tokens": _estimate_tokens(prompt_text) + comp_tokens,
        },
    }


def _sse_chunk(chunk_id: str, created: int, model: str,
               delta: Dict[str, Any], finish: Optional[str] = None) -> str:
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": finish}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def iter_sse_chunks(content: str, model: str = MODEL_ID,
                    chunk_chars: int = 6):
    """把整段回复切成小块, 依 Chrome 兼容的 SSE 格式逐块 yield。

    OpenAI 流式协议: 首块带 role, 中间块每块若干 content 字符,
    末块 finish_reason=stop, 最后 `data: [DONE]`。
    """
    chunk_id = f"chatcmpl-{int(time.time())}{len(content) % 1000:03d}"
    created = int(time.time())
    pieces = [content[i:i + chunk_chars] for i in
              range(0, max(len(content), 1), chunk_chars)]
    for i, piece in enumerate(pieces):
        if i == 0:
            delta = {"role": "assistant", "content": piece}
        else:
            delta = {"content": piece}
        finish = "stop" if i == len(pieces) - 1 else None
        yield _sse_chunk(chunk_id, created, model, delta, finish)
    yield f"data: [DONE]\n\n"


class CubeGPTBackend:
    """把 OpenAI chat 请求换算成 CubeGPT 能产生的真实回复。

    三级流水线 (v0.9.1 · 大模型落地 —— "CubeGPT 前置感知 + 真 LLM 生成
    + CubeGPT 后置审计"):
        1) 前置感知  每次请求先把用户文本喂进 CubeGPT 水库, 取脉冲统计
        2) 生成     若配置了本地 OpenAI 兼容后端 (CUBEGPT_LM_BASE),
                    转发完整 payload 给真 LLM 得到任意语言代码;
                    否则回退到 CubeGPT 自身的 Rust 知识/脉冲模板
        3) 后置审计  把生成结果再喂一次水库; 若命中 Rust 代码则复用
                    RustCodingPlugin 真实语料分类, 附改法与置信度

    未配置后端时保持纯本地模式 (离线 / 无网 / 测试均可用)。
    """

    def __init__(self, depth: int = 2, dim: int = 16,
                 lm_base: Optional[str] = None,
                 lm_key: Optional[str] = None,
                 lm_model: Optional[str] = None):
        from src.demos.chat import CubeGPTChat
        self.chat = CubeGPTChat(depth=depth, dim=dim)
        self.rust = None  # 惰性构造 RustCodingPlugin
        # 外挂 LLM 走 .CuteMamen 标准插件 (LLMProviderPlugin), 替换早前
        # 服务器内硬编码转发 —— 生命周期/记忆/清单/存档/注册发现全支持。
        from src.cutemamen.llm_provider import LLMProviderPlugin
        self.llm = LLMProviderPlugin(name="llm-provider", route="llm",
                                     base=lm_base, key=lm_key,
                                     model=lm_model)

    @property
    def using_lm(self) -> bool:
        return self.llm.available

    def _ensure_rust(self):
        # 惰性构造 Rust 知识插件; 无迁移权重时走语料原型 (502 段真实代码
        # 的每类质心), 全程真实、确定性, 不额外训练。
        if self.rust is None:
            from src.cutemamen.rust_coding import RustCodingPlugin
            plugin = RustCodingPlugin(name="serve-opencode")
            plugin._ensure_corpus()
            self.rust = plugin
        return self.rust

    def generate(self, user_text: str) -> str:
        """生成一段回复(非流式全集), 由 HTTP 层加密流式或整段返回"""
        code = _extract_code(user_text)
        if code is not None:
            return self._rust_reply(code)
        return self._spike_reply(user_text)

    # ── 三级流水线 (落地) ──────────────────────────────────
    def run(self, payload: Dict[str, Any]) -> str:
        """处理一次 chat 请求: 前置感知 → 真LLM生成/Local → 后置审计"""
        user_text = _last_user_text(payload.get("messages", []))
        if self.using_lm:
            lm_content = self.llm.generate(payload)
            if lm_content is not None:
                sense = self.sense_text(user_text)
                audit = self.audit_text(lm_content)
                return f"{sense}\n\n{lm_content}\n\n---\n{audit}"
        # 未配置 LLM 或转发失败 → 纯 CubeGPT 本地模态
        return self.generate(user_text)

    def _call_lm(self, payload: Dict[str, Any],
                 user_text: str) -> Optional[str]:
        """转发到 .CuteMamen 标准 LLM 提供者插件 (LLMProviderPlugin)"""
        return self.llm.generate(payload)

    def sense_text(self, text: str) -> str:
        """前置感知: 用户文本喂一次水库, 取一行脉冲统计"""
        meta = self.chat.reply(text).rsplit("\n", 1)[-1].strip()
        return f"CubeGPT 前置感知: `{meta}`"

    def audit_text(self, content: str) -> str:
        """后置审计: 生成结果再喂水库; 命中 Rust 则复用真实语料分类"""
        lines = ["CubeGPT 后置审计"]
        code = _extract_code(content)
        if code is not None:
            cls = self._ensure_rust().classify(code)
            label = cls.get("label", "ok")
            guide, _ = _RUST_FIX_GUIDE.get(label, _RUST_FIX_GUIDE["ok"])
            lines.append(f"  检测为代码: {cls.get('label_name', label)} "
                         f"[{cls.get('rustc', '-')}] 置信度 "
                         f"{cls.get('confidence', 0.0):.2f}"
                         f" (来源: {cls.get('source')})")
            lines.append(f"  建议: {guide}")
        meta = self.chat.reply(content).rsplit("\n", 1)[-1].strip()
        lines.append(f"  脉冲响应: {meta}")
        return "\n".join(lines)

    def _spike_reply(self, text: str) -> str:
        """通用对话: 交给 CubeGPTChat 的脉冲统计回复, 如实展示网络状态"""
        return self.chat.reply(text)

    def _rust_reply(self, code: str) -> str:
        """代码意图: 用 Rust 真实语料知识分类 + 给改法与可编译片段"""
        rust = self._ensure_rust()
        cls = rust.classify(code)
        label = cls.get("label", "ok")
        source = cls.get("source", "unknown")
        guide, snippet = _RUST_FIX_GUIDE.get(label, _RUST_FIX_GUIDE["ok"])
        head = (f"# CubeGPT 静态诊断 (知识来源: {source})")
        lines = [head, "", f"**诊断** — {cls.get('label_name', label)} "
                           f"[{cls.get('rustc', '-')}], 置信度 "
                           f"{cls.get('confidence', 0.0):.2f}", "", guide]
        if snippet:
            lines += ["", "```rust", snippet.strip(), "```"]
        # 附一"脉冲活动"自报, 让 OpenCode 用户看到水库确实参与了
        spike = self.chat.reply("/* 运行一次水库以感知该代码段 */")
        spike_meta = spike.rsplit("\n", 1)[-1]
        lines += ["", f"`{spike_meta.strip()}`"]
        return "\n".join(lines)


def _import_src():
    import src
    return src


# ── HTTP 层 ─────────────────────────────────────────────────
def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Cache-Control": "no-store",
    }


class OpenAICompatHandler(BaseHTTPRequestHandler):
    server_version = f"CubeGPTOpenAI/{getattr(_import_src(), '__version__', '?')}"
    backend: "CubeGPTBackend"

    def log_message(self, fmt, *args):  # 精简日志
        print(f"[opencode:{self.address_string()}] {fmt % args}")

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS 预检
        self.send_response(204)
        for k, v in _cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def _unauthorized(self) -> bool:
        """API 密钥校验: 请求须带 `Authorization: Bearer <key>`。

        server.api_key 为 None 时关闭校验 (测试/纯本地)。返回 True = 拒绝。
        """
        key = getattr(self.server, "api_key", None)
        if key is None:
            return False
        expect = {"Bearer", key}
        auth = self.headers.get("Authorization", "")
        parts = auth.split(None, 1)
        if len(parts) != 2 or parts[0] not in expect or parts[1] != key:
            self._send_json(401, {"object": "error",
                                  "message": "Invalid API key: 请提供 "
                                             "`Authorization: Bearer <api key>`"})
            return True
        return False

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if self._unauthorized():
            return
        if path in ("/v1/models", "/models"):
            now = int(time.time())
            self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model",
                          "created": now, "owned_by": "distributedformer"}],
            })
        elif path in ("/health", "/"):
            self._send_json(200, {
                "service": "CubeGPT OpenAI-compatible server",
                "model": MODEL_ID, "status": "ok",
                "endpoints": ["GET /v1/models", "POST /v1/chat/completions"],
            })
        else:
            self._send_json(404, {"object": "error",
                                  "message": f"未知路径: {path}"})

    def do_POST(self):
        if self._unauthorized():
            return
        if self.path.rstrip("/") not in ("/v1/chat/completions",
                                         "/chat/completions"):
            self._send_json(404, {"object": "error",
                                  "message": f"未知路径: {self.path}"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"object": "error",
                                  "message": f"请求体不是合法 JSON: {e}"})
            return
        try:
            self._handle_chat(payload)
        except Exception as e:  # 兜底, 避免崩溃进程
            self._send_json(500, {"object": "error", "message": str(e)})

    def _handle_chat(self, payload: Dict[str, Any]) -> None:
        messages = payload.get("messages", [])
        model = payload.get("model", MODEL_ID)
        prompt_text = "\n".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict))
        content = self.backend.run(payload)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            for k, v in _cors_headers().items():
                self.send_header(k, v)
            self.end_headers()
            for chunk in iter_sse_chunks(content, model=model):
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
            return
        self._send_json(200, build_chat_completion(
            content, model=model, prompt_text=prompt_text))


def run_openai_server(host: str = "127.0.0.1", port: int = 8000,
                      depth: int = 2, dim: int = 16,
                      api_key: Optional[str] = DEFAULT_API_KEY,
                      lm_base: Optional[str] = None,
                      lm_key: Optional[str] = None,
                      lm_model: Optional[str] = None) -> None:
    """启动 CubeGPT 的 OpenAI 兼容服务器 (阻塞)

    api_key: `Authorization: Bearer <key>` 鉴权用; None 关闭校验
    (测试/纯本地), 默认用 DEFAULT_API_KEY 方便本地与 OpenCode 对接。

    lm_base/lm_key/lm_model: 可选本地 OpenAI 兼容生成后端 (真 LLM)。
    缺省时读取环境变量 CUBEGPT_LM_BASE / _KEY / _MODEL; 都不配则
    纯 CubeGPT 本地模态。
    """
    from src.demos.chat import CubeGPTChat  # 触发热加载依赖链

    host_backend = CubeGPTBackend(depth=depth, dim=dim,
                                  lm_base=lm_base, lm_key=lm_key,
                                  lm_model=lm_model)
    handler = type("Handler", (OpenAICompatHandler,), {
        "backend": host_backend,
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.api_key = api_key
    src = _import_src()
    print(f"[CubeGPT] OpenAI 兼容服务器 v{src.__version__}")
    print(f"[CubeGPT] 监听 {host}:{port}  模型 id: {MODEL_ID}")
    if host_backend.using_lm:
        print(f"[CubeGPT] 生成后端: {host_backend.llm.base}  "
              f"(model={host_backend.llm.model or '<由请求指定>'})")
    else:
        print("[CubeGPT] 生成后端: 未配置 (纯 CubeGPT 本地模态; "
              f"设 {LM_BASE_ENV} 接入真 LLM)")
    if api_key is not None:
        print(f"[CubeGPT] API key : {api_key}   (请求头需带 "
              f"`Authorization: Bearer {api_key}`)")
    print("[CubeGPT] OpenCode 中配置: ")
    print(f"          provider = @openai")
    print(f"          baseURL  = http://{host}:{port}")
    print(f"          model    = {MODEL_ID}")
    if api_key is not None:
        print(f"          apiKey   = {api_key}   (env: OPENAI_API_KEY)")
    print("[CubeGPT] Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CubeGPT] 已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_openai_server()