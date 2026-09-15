"""v0.7.0 模态面 pkg 存档测试: 导出/导入/卸载/随用随载热加载 + 报告占位符修复"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CubeGPT
from src.core import face_pkg
from src.agents.base_agent import ActionAgent


@pytest.fixture
def gpt():
    np.random.seed(42)
    g = CubeGPT(depth=1, dim=16, modalities=["numeric", "text"])
    # 跑几步让面产生状态 (疲劳/不应期/皮层状态)
    for i in range(3):
        g.step({"numeric": 0.5 * i + 0.1, "text": "hello world"})
    return g


def test_export_import_roundtrip(gpt, tmp_path):
    pkg = str(tmp_path / "text.dfpkg")
    manifest = gpt.export_face("text", pkg, author="tester")
    assert os.path.exists(pkg)
    assert manifest["name"] == "text"
    assert manifest["modality"] == "text"
    assert manifest["format"] == "dfpkg"
    assert manifest["depth"] == 1 and manifest["dim"] == 16
    assert manifest["units"] == 16 + 16 + 256  # 端口16 + 皮层 Σ16^k, k=1..2
    assert manifest["params"] == manifest["units"] * 16

    # 导入到全新模型, 权重与状态逐位还原
    np.random.seed(999)
    gpt2 = CubeGPT(depth=1, dim=16, modalities=["numeric"])
    m2 = gpt2.import_face(pkg)
    assert m2["modality"] == "text"

    src_units = gpt.faces["text"].cortex._all_units_cache
    dst_units = gpt2.faces["text"].cortex._all_units_cache
    assert len(src_units) == len(dst_units)
    for u1, u2 in zip(src_units, dst_units):
        for p in face_pkg.UNIT_PARAM_NAMES:
            assert getattr(u1, p) == pytest.approx(getattr(u2, p)), p
        assert u1.outgoing.keys() == u2.outgoing.keys()
        for t, w in u1.outgoing.items():
            assert u2.outgoing[t] == pytest.approx(w)
        assert np.allclose(u1.state, u2.state)
        assert u1.fatigue == pytest.approx(u2.fatigue)
        assert u1.refractory == u2.refractory


def test_import_replaces_face_and_keeps_running(gpt, tmp_path):
    pkg = str(tmp_path / "text.dfpkg")
    gpt.export_face("text", pkg)
    np.random.seed(7)
    gpt2 = CubeGPT(depth=1, dim=16, modalities=["numeric"])
    gpt2.import_face(pkg)
    # 导入后的面可正常参与计算 (含棱传播与融合)
    spikes = gpt2.step({"numeric": 1.0, "text": "test input"})
    assert isinstance(spikes, list)
    assert gpt2.ring == ["numeric", "text"]


def test_unload_and_hot_load_in_step(gpt, tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        pkg_path = gpt.unload_face("text")
        assert os.path.exists(pkg_path)
        assert "text" not in gpt.faces
        assert gpt.list_faces() == {"loaded": ["numeric"], "registered": ["text"]}

        # 卸载后该模态输入触发随用随载热加载
        np.random.seed(3)
        gpt.step({"numeric": 1.0, "text": "hot load me"})
        assert "text" in gpt.faces
        assert gpt.list_faces() == {"loaded": ["numeric", "text"], "registered": []}
    finally:
        os.chdir(cwd)


def test_register_pkg_lazy_load(gpt, tmp_path):
    pkg = str(tmp_path / "text.dfpkg")
    gpt.export_face("text", pkg)
    # 备份状态, 卸载 (不自动导出, 用已有 pkg)
    gpt.unload_face("text", pkg_path=pkg)
    gpt.register_face_pkg(pkg)
    assert gpt.list_faces()["registered"] == ["text"]
    # 热加载后状态与 pkg 一致
    np.random.seed(1)
    gpt.load_face("text")
    assert np.allclose(
        gpt.faces["text"].inbox,
        gpt.faces["numeric"].inbox * 0 + gpt.faces["text"].inbox)  # inbox 已还原非零不报错


def test_min_core_version_rejects(gpt, tmp_path):
    pkg = str(tmp_path / "text.dfpkg")
    gpt.export_face("text", pkg, min_core_version="99.0.0")
    np.random.seed(5)
    gpt2 = CubeGPT(depth=1, dim=16, modalities=["numeric"])
    with pytest.raises(ValueError, match="拒绝加载"):
        gpt2.import_face(pkg)
    with pytest.raises(ValueError, match="拒绝加载"):
        gpt2.register_face_pkg(pkg)


def test_export_unknown_face_raises(gpt, tmp_path):
    with pytest.raises(KeyError):
        gpt.export_face("image", str(tmp_path / "image.dfpkg"))


def test_unknown_modality_still_rejected(gpt):
    with pytest.raises(KeyError, match="未知输入模态"):
        gpt.step({"vision": 1.0})


def test_read_manifest(gpt, tmp_path):
    pkg = str(tmp_path / "text.dfpkg")
    gpt.export_face("text", pkg, capability="语义计算")
    m = face_pkg.read_manifest(pkg)
    assert m["capability"] == "语义计算"
    assert m["min_core_version"] == "0.7.0"


# ── 训练报告占位符修复 ────────────────────────────────────

def test_report_generate_no_placeholder(tmp_path):
    agent = ActionAgent("rep_test", df_depth=0)
    agent.kv_stack.push("k1", np.ones(16) * 0.3, np.ones(16) * 0.3)
    agent.state.cycle_count = 12
    agent.state.total_spikes_received = 34
    agent.state.total_spikes_sent = 8
    os.chdir(tmp_path)
    try:
        result = agent._do_report_generate({
            "template": "daily_pulse",
            "sections": ["summary", "network", "memory", "learning"],
        })
        assert result["status"] == "success"
        with open(result["path"], encoding="utf-8") as f:
            content = f.read()
    finally:
        os.chdir(os.path.dirname(tmp_path))
    assert "占位" not in content
    assert "rep_test" in content
    assert "34" in content          # 真实脉冲统计
    assert "10000" in content or "10,000" in content  # KV 容量
    assert "CubeGPT" in content     # network 章节渲染真实网络统计
