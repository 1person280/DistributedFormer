"""v0.18.1 预制存档 (蓝图 Blueprint) 测试

覆盖:
1. 内部节点图 → ComfyUI v1.0 规范 转换 (nodes/links/groups/version)
2. 蓝图打包: .blueprint.zip, ZIP_STORED 不压缩 (compress_type=0)
3. 蓝图解包 round-trip: 工作流 + 模型存档字节一致
4. list_blueprints 列出 (读 manifest 不落地)
"""

import os
import sys
import tempfile
import zipfile

# src/tests/<file> → 上溯 3 级到仓库根 (导入 src 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src import blueprint as bp


def _sample_workflow():
    return {
        "name": "t", "version": 1,
        "nodes": [
            {"id": "n1", "type": "input", "x": 40, "y": 80,
             "params": {"topic": "rust", "data": "fn main(){}"},
             "title": "输入"},
            {"id": "n2", "type": "model", "x": 400, "y": 80,
             "params": {"route": "rust"}, "title": "模型"},
            {"id": "n3", "type": "output", "x": 800, "y": 80,
             "params": {}, "title": "输出"},
        ],
        "edges": [
            {"id": "e1", "from": "n1", "from_port": "data", "to": "n2",
             "to_port": "in", "to_name": "seq"},
            {"id": "e2", "from": "n2", "from_port": "logits", "to": "n3",
             "to_port": "in", "to_name": "data"},
        ],
        "groups": [
            {"id": "g1", "name": "主链", "nodes": ["n1", "n2"],
             "color": "#7aa2ff", "fold": False, "muted": False},
        ],
    }


def test_to_comfy_schema():
    """内部图 → ComfyUI v1.0 规范字段齐全"""
    comfy = bp.to_comfy(_sample_workflow())
    assert comfy["version"] == 1
    assert len(comfy["nodes"]) == 3
    assert len(comfy["links"]) == 2
    assert len(comfy["groups"]) == 1
    # 节点必备字段 (ComfyUI v1.0 required)
    n = comfy["nodes"][0]
    for key in ("id", "type", "pos", "size", "flags", "order", "mode",
                "inputs", "outputs", "properties", "widgets_values"):
        assert key in n
    # 端口槽位
    assert [o["name"] for o in comfy["nodes"][0]["outputs"]] == ["data", "meta"]
    assert [i["name"] for i in comfy["nodes"][1]["inputs"]] == ["seq", "config"]
    # link: [id, origin_id, origin_slot, target_id, target_slot, type]
    lnk = comfy["links"][0]
    assert lnk[:5] == ["e1", "n1", 0, "n2", 0] and lnk[5] == "*"


def test_save_blueprint_zip_stored():
    """蓝图后缀 .blueprint.zip, 内部 ZIP_STORED 不压缩"""
    with tempfile.TemporaryDirectory() as d:
        path = bp.save_blueprint("t_blp", _sample_workflow(),
                                 out_dir=os.path.join(d, "out"))
        assert path.endswith(bp.BLUEPRINT_SUFFIX)
        assert os.path.isfile(path)
        with zipfile.ZipFile(path) as zf:
            names = {i.filename: i.compress_type for i in zf.infolist()}
            assert set(names) == {"manifest.json", "workflow.json"}
            for ctype in names.values():
                assert ctype == zipfile.ZIP_STORED  # 0


def test_blueprint_roundtrip_with_models():
    """打包→解包: 工作流结构 + 模型存档字节一致"""
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, "fake.CuteMamen")
        payload = b"MODELDATA" * 100
        with open(mp, "wb") as f:
            f.write(payload)
        path = bp.save_blueprint("t_mdl", _sample_workflow(),
                                 models=[mp], out_dir=os.path.join(d, "out"))
        wf, manifest, models = bp.load_blueprint(path)
        # 工作流 round-trip
        assert len(wf["nodes"]) == 3
        assert len(wf["edges"]) == 2
        assert len(wf["groups"]) == 1
        assert wf["edges"][0]["from_port"] == "data"
        assert wf["edges"][0]["to_name"] == "seq"
        assert set(wf["groups"][0]["nodes"]) == {"n1", "n2"}
        # 模型
        assert manifest["format"] == bp.BLUEPRINT_FORMAT
        assert "models/fake.CuteMamen" in models
        assert models["models/fake.CuteMamen"] == payload
        assert manifest["models"] == ["models/fake.CuteMamen"]


def test_list_blueprints():
    """list_blueprints 读 manifest, 不落地"""
    with tempfile.TemporaryDirectory() as d:
        bp.save_blueprint("t_lst", _sample_workflow(),
                          out_dir=os.path.join(d, "out"))
        bls = bp.list_blueprints(os.path.join(d, "out"))
        assert len(bls) == 1
        assert bls[0]["name"] == "t_lst"
        assert bls[0]["format"] == bp.BLUEPRINT_FORMAT


def test_from_comfy_backward():
    """ComfyUI 规范 → 内部格式 (load 路径)"""
    comfy = bp.to_comfy(_sample_workflow())
    wf = bp.from_comfy(comfy)
    assert len(wf["nodes"]) == 3
    assert wf["nodes"][0]["type"] == "input"
    assert len(wf["edges"]) == 2
