# -*- coding: utf-8 -*-
"""节点图工作流 (v0.16.0) 测试

覆盖: WorkflowEngine 拓扑执行 (真实插件), 环检测, 模型枚举, 预设样例。
"""

from src.cutemamen.kernel import CuteMamenKernel
from src.deployment.workflow_ui import WorkflowEngine, PRESETS


def _kernel_with_rust():
    from src.cutemamen.rust_coding import RustCodingPlugin
    kernel = CuteMamenKernel(dim=16)
    kernel.mount(RustCodingPlugin(name="rust-coding", route="rust"))
    return kernel


def _kernel_with_chat():
    from src.cutemamen.chat import ChatPlugin
    kernel = CuteMamenKernel(dim=16)
    kernel.mount(ChatPlugin(name="ui-chat", route="chat"))
    return kernel


def test_engine_lists_models():
    kernel = _kernel_with_rust()
    engine = WorkflowEngine(kernel)
    models = engine.list_models()
    routes = [m["route"] for m in models]
    assert "rust" in routes


def test_input_model_output_chain():
    """真实内核: input→rust→output, 输出应含 label"""
    kernel = _kernel_with_rust()
    engine = WorkflowEngine(kernel)
    wf = {
        "name": "t", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "rust", "data": "fn main(){}"}},
            {"id": "model", "type": "model", "x": 200, "y": 0,
             "params": {"route": "rust", "data": None}},
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
    res = engine.run(wf)
    assert res["ok"] is True
    assert "label" in res["results"]["out"]["data"]
    assert res["log"][0]["status"] == "ok"


def test_cycle_detection():
    engine = WorkflowEngine(_kernel_with_rust())
    wf = {
        "name": "cycle", "version": 1,
        "nodes": [
            {"id": "a", "type": "model", "x": 0, "y": 0, "params": {"route": "rust"}},
            {"id": "b", "type": "model", "x": 200, "y": 0, "params": {"route": "rust"}},
        ],
        "edges": [
            {"id": "e1", "from": "a", "from_port": "out", "to": "b", "to_port": "in"},
            {"id": "e2", "from": "b", "from_port": "out", "to": "a", "to_port": "in"},
        ],
    }
    res = engine.run(wf)
    assert res["ok"] is False
    assert "环" in res["error"]


def test_missing_node_edge_rejected():
    engine = WorkflowEngine(_kernel_with_rust())
    wf = {"name": "bad", "version": 1,
          "nodes": [{"id": "a", "type": "input", "x": 0, "y": 0,
                     "params": {"topic": "rust", "data": "x"}}],
          "edges": [{"id": "e1", "from": "a", "from_port": "out",
                     "to": "ghost", "to_port": "in"}]}
    res = engine.run(wf)
    assert res["ok"] is False


def test_presets_runnable():
    """内置 Rust 预设 (真实内核) 应一键跑通"""
    kernel = _kernel_with_rust()
    engine = WorkflowEngine(kernel)
    assert "Rust 代码分类" in PRESETS
    res = engine.run(PRESETS["Rust 代码分类"])
    assert res["ok"] is True
    assert "label" in res["results"]["out"]["data"]


def test_chat_input_produces_natural_language():
    """UI 输入自然语言 (topic=chat) → 模型应输出自然语言回复, 而非空/编码"""
    kernel = _kernel_with_chat()
    engine = WorkflowEngine(kernel)
    wf = {
        "name": "t-chat", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "chat", "data": "你好"}},
            {"id": "model", "type": "model", "x": 200, "y": 0,
             "params": {"route": "chat", "data": None}},
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
    res = engine.run(wf)
    assert res["ok"] is True
    reply = res["results"]["out"]["data"]
    # 自然语言回复应为非空字符串 (字典含 "reply" 字段)
    assert reply is not None
    if isinstance(reply, dict):
        assert isinstance(reply.get("reply"), str) and reply["reply"].strip()
    else:
        assert isinstance(reply, str) and reply.strip()
    assert res["log"][0]["status"] == "ok"  # model 节点有输出, 非 no_output


def test_model_output_has_typed_structure():
    """模型节点输出应带结构化 typed 列表 (类别+内容)"""
    kernel = _kernel_with_chat()
    engine = WorkflowEngine(kernel)
    wf = {
        "name": "t-typed", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "chat", "data": "你好"}},
            {"id": "model", "type": "model", "x": 200, "y": 0,
             "params": {"route": "chat", "data": None}},
        ],
        "edges": [
            {"id": "e1", "from": "in", "from_port": "out",
             "to": "model", "to_port": "in"},
        ],
    }
    res = engine.run(wf)
    assert res["ok"] is True
    typed = res["results"]["model"]["typed"]
    assert isinstance(typed, list) and len(typed) >= 1
    assert typed[0]["category"] == "text"
    assert isinstance(typed[0]["content"], str) and typed[0]["content"].strip()


def test_condition_node_fanout():
    """条件模块: 满足多条条件 → 同步扇出到对应输出端口"""
    kernel = _kernel_with_chat()
    engine = WorkflowEngine(kernel)
    wf = {
        "name": "t-cond", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "chat", "data": "你好"}},
            {"id": "model", "type": "model", "x": 200, "y": 0,
             "params": {"route": "chat", "data": None}},
            {"id": "cond", "type": "condition", "x": 400, "y": 0,
             "params": {"conditions": [
                 {"label": "A", "match": 'reply contains "你"'},
                 {"label": "B", "match": 'reply contains "好"'},
             ]}},
            {"id": "outA", "type": "output", "x": 600, "y": -60, "params": {}},
            {"id": "outB", "type": "output", "x": 600, "y": 60, "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in", "from_port": "out", "to": "model", "to_port": "in"},
            {"id": "e2", "from": "model", "from_port": "out", "to": "cond", "to_port": "in"},
            {"id": "e3", "from": "cond", "from_port": "A", "to": "outA", "to_port": "in"},
            {"id": "e4", "from": "cond", "from_port": "B", "to": "outB", "to_port": "in"},
        ],
    }
    res = engine.run(wf)
    assert res["ok"] is True
    matched = res["results"]["cond"]["matched"]
    assert "A" in matched
    assert "B" in matched            # 多条件满足 → 同步扇出
    assert res["results"]["outA"]["data"] is not None
    assert res["results"]["outB"]["data"] is not None


def test_feedback_node_redo():
    """回传节点: 把入边数据回传给目标插件补做, 返回新结果"""
    kernel = _kernel_with_chat()
    engine = WorkflowEngine(kernel)
    wf = {
        "name": "t-fb", "version": 1,
        "nodes": [
            {"id": "in", "type": "input", "x": 0, "y": 0,
             "params": {"topic": "chat", "data": "再想想"}},
            {"id": "fb", "type": "feedback", "x": 200, "y": 0,
             "params": {"target": "chat", "data": None}},
            {"id": "out", "type": "output", "x": 400, "y": 0, "params": {}},
        ],
        "edges": [
            {"id": "e1", "from": "in", "from_port": "out", "to": "fb", "to_port": "in"},
            {"id": "e2", "from": "fb", "from_port": "out", "to": "out", "to_port": "in"},
        ],
    }
    res = engine.run(wf)
    assert res["ok"] is True
    fb = res["results"]["fb"]
    assert fb["target"] == "chat"
    assert fb["output"] is not None  # 补做返回了新结果
    assert res["results"]["out"]["data"] is not None
    assert res["log"][0]["status"] == "ok"