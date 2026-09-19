"""v0.14.5 插件互通信 + 新插件 (TagSearching / Chat) 测试

覆盖:
1. 插件↔插件通信: PluginContext.ask 同步请求/应答 (可重入路由)
2. 插件↔CubeGPT 通信: PluginContext.run_gpt 驱动全模型 step
3. TagSearchingPlugin: 代码引用检索 (定义 + 引用 + 行号片段), 排除 == 误报
4. ChatPlugin: 类人对话 (优先外挂 LLM / 回退本地真人感引擎)
5. 仓库交付的 TagSearching.CuteMamen / Chat.CuteMamen 可被 CubeGPT 路由
"""

import os
import sys

import numpy as np
import pytest

# src/tests/<file> → 上溯 3 级到仓库根 (导入 src 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.cutemamen import (
    ChatPlugin,
    CubeGPTKernel,
    CuteMamenKernel,
    ExpertPlugin,
    TagSearchingPlugin,
    load_pkg,
    save_pkg,
)

_REPO_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "plugin")


@pytest.fixture
def kernel():
    return CuteMamenKernel(dim=16)


# ═══════════════════════════════════════════════════════════
# 1. 插件↔插件: 同步请求/应答 (PluginContext.ask)
# ═══════════════════════════════════════════════════════════

def test_plugin_ask_plugin_sync(kernel):
    """插件 A 在 on_think 中 ask 插件 B → B 的结果直接返回 (双向通信)"""
    class Echo(ExpertPlugin):
        def on_think(self, event, ctx):
            return {"echo": event.get("data")}

    class Caller(ExpertPlugin):
        def on_think(self, event, ctx):
            relay = ctx.ask("echo", event.get("data"))
            return {"relayed": relay, "plugin": ctx.plugin_name}

    kernel.mount(Echo("echo"))
    kernel.mount(Caller("caller"))
    out = kernel.think({"topic": "caller", "data": 42})
    assert out == [{"relayed": {"echo": 42}, "plugin": "caller"}]


def test_plugin_ask_unrouted_returns_none(kernel):
    """ask 未注册/未加载的目标 → None (不抛错)"""
    class Probe(ExpertPlugin):
        def on_think(self, event, ctx):
            return {"got": ctx.ask("no-such-plugin", "x")}

    kernel.mount(Probe("probe"))
    out = kernel.think({"topic": "probe", "data": None})
    assert out == [{"got": None}]


# ═══════════════════════════════════════════════════════════
# 2. 插件↔CubeGPT: 全模型推理 (PluginContext.run_gpt)
# ═══════════════════════════════════════════════════════════

def test_plugin_run_gpt_drives_cubegpt():
    """插件内 run_gpt({模态: 数据}) → CubeGPTKernel 执行一次 step"""
    class GptDriver(ExpertPlugin):
        def on_think(self, event, ctx):
            ctx.run_gpt({"numeric": 0.5, "text": "hi"})
            return {"total_steps": ctx.kernel.total_steps}

    np.random.seed(7)
    ck = CubeGPTKernel(depth=1, dim=16, modalities=["numeric", "text"])
    ck.mount(GptDriver("driver"))
    before = ck.total_steps
    out = ck.think({"topic": "driver", "data": None})
    assert out and out[0]["total_steps"] == before + 1
    assert set(ck.faces) == {"numeric", "text"}


# ═══════════════════════════════════════════════════════════
# 3. TagSearchingPlugin: 代码引用检索
# ═══════════════════════════════════════════════════════════

_SAMPLE = "def foo(x):\n    return x\ny = foo(1)\nif foo == 2:\n    pass\n"


def test_tagsearch_finds_defs_and_refs(tmp_path):
    """给定符号 → 定义点 + 引用点 (文件/行号/片段), 区分定义与调用"""
    (tmp_path / "sample.py").write_text(_SAMPLE)
    plugin = TagSearchingPlugin("tag-search")
    report = plugin.search("foo", root=str(tmp_path), exts=[".py"])
    assert report["definition_count"] == 1   # def foo(x):
    assert report["reference_count"] == 2    # y = foo(1)  和  if foo == 2:
    d = report["definitions"][0]
    assert d["file"].endswith("sample.py") and d["line"] == 1
    # == 是引用而非定义 (排除比较误报)
    ctxs = {r["context"] for r in report["references"]}
    assert any("foo(1)" in c for c in ctxs)
    assert all("==" not in r["context"] for r in report["definitions"])


def test_tagsearch_route_and_semantic_memory(kernel, tmp_path):
    """路由 tagsearch + 语义记忆留痕 (recall_search 可复查)"""
    (tmp_path / "a.py").write_text("def alpha():\n    pass\nalpha()\n")
    plugin = TagSearchingPlugin("tag-search")
    kernel.mount(plugin)
    out = kernel.think({"topic": "tagsearch",
                        "data": {"tag": "alpha", "root": str(tmp_path)}})
    assert out and out[0]["tag"] == "alpha"
    assert out[0]["definition_count"] == 1 and out[0]["reference_count"] == 1
    rec = plugin.recall_search("alpha")
    assert rec["definition_count"] == 1 and rec["reference_count"] == 1


def test_tagsearch_invalid_tag_rejected():
    plugin = TagSearchingPlugin("tag-search")
    with pytest.raises(ValueError):
        plugin.search("not an identifier", root=".")
    assert plugin.on_think({"topic": "tagsearch", "data": 123}, None) is None


# ═══════════════════════════════════════════════════════════
# 4. ChatPlugin: 类人对话
# ═══════════════════════════════════════════════════════════

def test_chat_local_reply_colloquial(kernel):
    """无 LLM 后端时回退本地真人感引擎 (口语化、有来有回)"""
    kernel.mount(ChatPlugin("chat"))
    out = kernel.think({"topic": "chat", "data": "你好"})
    assert out and out[0]["provider"] == "local"
    assert len(out[0]["reply"]) > 0
    assert "你好" in out[0]["reply"] or "嗨" in out[0]["reply"]


def test_chat_chains_llm_via_ask(kernel):
    """ChatPlugin 经 ctx.ask 请求外挂 LLM (route llm) → provider=llm"""
    class FakeLLM(ExpertPlugin):
        def on_think(self, event, ctx):
            return {"content": "我在听，你说？(LLM)"}

    kernel.mount(FakeLLM("llm-provider", route="llm"))
    kernel.mount(ChatPlugin("chat"))
    out = kernel.think({"topic": "chat", "data": {"message": "聊聊"}})
    assert out and out[0]["provider"] == "llm"
    assert "(LLM)" in out[0]["reply"]
    assert kernel.plugins["chat"].llm_replies == 1


def test_chat_emotion_and_question_reflection(kernel):
    """本地引擎: 情绪共情 + 问句反射 (有来有回)"""
    plugin = ChatPlugin("chat")
    kernel.mount(plugin)
    emo = plugin.reply("今天好累")
    assert "歇" in emo["reply"] or "累" in emo["reply"]
    q = plugin.reply("这个函数为什么这么慢?")
    assert len(q["reply"]) > 0


# ═══════════════════════════════════════════════════════════
# 5. 交付包: TagSearching.CuteMamen / Chat.CuteMamen 可路由
# ═══════════════════════════════════════════════════════════

def test_shipped_tagsearch_pkg_routed_by_cubegpt():
    pkg = os.path.join(_REPO_PLUGIN_DIR, "TagSearching.CuteMamen")
    assert os.path.isfile(pkg)
    np.random.seed(3)
    gpt = CubeGPTKernel(depth=1, dim=16, modalities=["numeric"])
    found = gpt.discover_plugins(_REPO_PLUGIN_DIR)
    assert found["tag-search"]["route"] == "tagsearch"
    out = gpt.think({"topic": "tagsearch", "data": "TagSearchingPlugin"})
    assert out and isinstance(gpt.plugins["tag-search"], TagSearchingPlugin)
    assert out[0]["definition_count"] >= 1


def test_shipped_chat_pkg_routed_by_cubegpt():
    pkg = os.path.join(_REPO_PLUGIN_DIR, "Chat.CuteMamen")
    assert os.path.isfile(pkg)
    np.random.seed(5)
    gpt = CubeGPTKernel(depth=1, dim=16, modalities=["numeric"])
    found = gpt.discover_plugins(_REPO_PLUGIN_DIR)
    assert found["chat"]["route"] == "chat"
    out = gpt.think({"topic": "chat", "data": "在吗"})
    assert out and isinstance(gpt.plugins["chat"], ChatPlugin)
    assert out[0]["provider"] in ("local", "llm")


def test_new_plugins_pkg_roundtrip(tmp_path):
    """两个新插件 base_model 注册表往返 (save_pkg → load_pkg)"""
    for plugin, name in [(TagSearchingPlugin("tag-search"), "tag.searching"),
                         (ChatPlugin("chat"), "chat.persona")]:
        pkg = str(tmp_path / f"{name}.CuteMamen")
        save_pkg(plugin, pkg)
        loaded, manifest = load_pkg(pkg)
        assert manifest["base_model"] == name
        assert isinstance(loaded, type(plugin))
        assert manifest["min_core_version"] == "0.14.6"
