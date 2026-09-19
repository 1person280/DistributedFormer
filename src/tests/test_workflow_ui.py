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