"""OpenAI 兼容接口服务器测试 (v0.9.1)

验证: OpenAI 报文格式、代码意图识别、Rust 帮改法生成、SSE 流式、
以及真实 HTTP 层 (/v1/models, /v1/chat/completions 流式与非流式)。
"""

import json
import threading
import urllib.request
import urllib.error

import pytest

from src.deployment.openai_server import (
    MODEL_ID, DEFAULT_API_KEY, CubeGPTBackend, build_chat_completion,
    iter_sse_chunks, _extract_code, _last_user_text,
    OpenAICompatHandler,
)
from src.deployment.openai_server import _cors_headers


# ── 报文格式 ─────────────────────────────────────────────────
def test_chat_completion_openai_shape():
    resp = build_chat_completion("你好", model=MODEL_ID, prompt_text="hi")
    assert resp["object"] == "chat.completion"
    assert resp["model"] == MODEL_ID
    assert resp["choices"][0]["message"]["role"] == "assistant"
    assert resp["choices"][0]["message"]["content"] == "你好"
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["completion_tokens"] >= 1


def test_last_user_text_with_lists():
    messages = [
        {"role": "system", "content": "你是个 Rust 助手"},
        {"role": "user", "content": [
            {"type": "text", "text": "帮我看看这段: "},
            {"type": "text", "text": "fn f() {}"}]},
    ]
    text = _last_user_text(messages)
    assert "fn f()" in text and "帮我看看" in text


# ── 代码意图识别 ─────────────────────────────────────────────
def test_extract_code_fenced():
    text = "请诊断：\n```rust\nlet s = String::from(\"hi\");\nt let = s;\n```"
    assert _extract_code(text).startswith("let s")

def test_extract_code_plain_rust_hint():
    text = "use of moved value: `s` while calling\nlet s = a;"
    assert _extract_code(text) is not None

def test_extract_code_non_code_returns_none():
    assert _extract_code("今天天气如何？") is None


# ── 后端生成 (脉冲统计 + Rust 帮改法) ────────────────────────
def test_backend_spike_reply_for_plain_text():
    backend = CubeGPTBackend(depth=1, dim=16)
    out = backend.generate("介绍一下你自己")
    assert isinstance(out, str) and out

def test_backend_rust_reply_for_code():
    backend = CubeGPTBackend(depth=1, dim=16)
    bad = ('let s = String::from("hi");\n'
           'let t = s;\nprintln!("{}", s);')  # E0382: use of moved value
    out = backend.generate(bad)
    # 命中 Rust 知识路径: 含诊断与 ```rust 修正片段
    assert "静态诊断" in out
    assert "```rust" in out
    assert "E0382" in out or "所有权" in out or "move" in out


# ── 三级流水线 (前置感知 → 真LLM → 后置审计) ─────────────────
def test_run_without_lm_falls_back_to_local():
    backend = CubeGPTBackend(depth=1, dim=16)  # 未配后端
    assert not backend.using_lm
    out = backend.run({"messages": [
        {"role": "user", "content": "介绍一下你自己"}]})
    assert isinstance(out, str) and out


def test_run_with_lm_bridges_sense_and_audit():
    # 假 LLM 后端: 返回固定 Python 代码
    import http.server

    class FakeLM(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            self.rfile.read(n)
            body = json.dumps({
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {
                    "role": "assistant",
                    "content": "def hello():\n    return 42  # 由假 LLM 生成"}}],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # noqa
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLM)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        backend = CubeGPTBackend(depth=1, dim=16,
                                 lm_base=f"http://127.0.0.1:{port}/v1",
                                 lm_key="ollama", lm_model="fake")
        assert backend.using_lm
        out = backend.run({"messages": [
            {"role": "user", "content": "写一个 hello 函数"}]})
    finally:
        srv.shutdown()
        srv.server_close()
    # 前置感知 + 假LLM内容 + 后置审计三段都出现
    assert "CubeGPT 前置感知" in out
    assert "def hello():" in out
    assert "CubeGPT 后置审计" in out


def test_call_lm_invalid_model_injected():
    backend = CubeGPTBackend(depth=1, dim=16,
                             lm_base="http://127.0.0.1:1/v1",
                             lm_key="x", lm_model="mock")
    # 不可用后端 → None → run 回退本地模态
    assert backend._call_lm({"messages": []}, "hi") is None


# ── 外挂 LLM 是标准 CuteMamen 插件 (注册/清单/权重/打包) ─────
from src.cutemamen.pkg import native_registry, save_pkg, load_pkg
from src.cutemamen.llm_provider import LLMProviderPlugin


def test_llm_plugin_registered_and_manifest():
    assert native_registry()["llm.provider"] is LLMProviderPlugin
    p = LLMProviderPlugin(name="llm-provider", base="http://x/v1",
                          model="qwen:7b")
    m = p.build_manifest()
    assert m["base_model"] == "llm.provider"
    assert m["format"] == "CuteMamen"
    assert m["route"] == "llm"
    assert m["llm_base"] == "http://x/v1"
    assert "on_think" in m["lifecycle"]


def test_llm_plugin_weights_roundtrip():
    p = LLMProviderPlugin(name="llm-provider", base="http://x/v1",
                          model="qwen:7b")
    w = p.save_weights()
    q = LLMProviderPlugin(name="llm-provider")  # 无配置
    q.load_weights(w, {})
    assert q.base == "http://x/v1"
    assert q.model == "qwen:7b"


def test_llm_plugin_package_roundtrip(tmp_path):
    p = LLMProviderPlugin(name="llm-provider", base="http://x/v1",
                          model="qwen:7b")
    pkg = tmp_path / "llm-provider.CuteMamen"
    save_pkg(p, str(pkg))
    q, manifest = load_pkg(str(pkg))
    assert isinstance(q, LLMProviderPlugin)
    assert q.base == "http://x/v1"
    assert q.model == "qwen:7b"
    assert manifest["base_model"] == "llm.provider"


# ── SSE 流式 ─────────────────────────────────────────────────
def test_sse_stream_ends_with_done():
    chunks = list(iter_sse_chunks("一段较长的代码修改建议", model=MODEL_ID))
    body = "".join(chunks)
    assert body.endswith("data: [DONE]\n\n")
    first = json.loads(chunks[0][len("data: "):].strip())
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"


# ── 真实 HTTP 层 (临时起服) ───────────────────────────────────
@pytest.fixture()
def server(tmp_path):
    import http.server
    handler = type("H", (OpenAICompatHandler,), {
        "backend": CubeGPTBackend(depth=1, dim=16)})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, port
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def auth_server(tmp_path):
    import http.server
    handler = type("H", (OpenAICompatHandler,), {
        "backend": CubeGPTBackend(depth=1, dim=16)})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.api_key = DEFAULT_API_KEY
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, port
    srv.shutdown()
    srv.server_close()


def _get(port, path, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 headers=headers)
    return urllib.request.urlopen(req)


def test_401_without_api_key(auth_server):
    _, port = auth_server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(port, "/v1/models")
    assert ei.value.code == 401


def test_401_with_wrong_api_key(auth_server):
    _, port = auth_server
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(port, "/v1/models", token="wrong-key")
    assert ei.value.code == 401


def test_200_with_correct_api_key(auth_server):
    _, port = auth_server
    with _get(port, "/v1/models", token=DEFAULT_API_KEY) as r:
        body = json.loads(r.read().decode("utf-8"))
    assert body["object"] == "list"


def _post(port, path, payload, stream_out=False):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        if stream_out:
            return r.read().decode("utf-8")
        return json.loads(r.read().decode("utf-8"))


def test_http_models_endpoint(server):
    _, port = server
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models") as r:
        body = json.loads(r.read().decode("utf-8"))
    assert body["object"] == "list"
    assert any(m["id"] == MODEL_ID for m in body["data"])


def test_http_chat_completion(server):
    _, port = server
    resp = _post(port, "/v1/chat/completions", {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "写一个 hello world"}]})
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"]


def test_http_chat_stream(server):
    _, port = server
    body = _post(port, "/v1/chat/completions", {
        "model": MODEL_ID, "stream": True,
        "messages": [{"role": "user", "content": "fn main() {} 有问题吗"}]},
        stream_out=True)
    assert body.endswith("data: [DONE]\n\n")
    assert "data: {" in body