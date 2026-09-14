"""
DistributedFormer — 事件驱动的脉冲神经网络智能体框架

内嵌模型 CubeGPT: 立方体连接的多模态脉冲大模型
(4 个模态面 × ~4,400 单元 × 16 参数 ≈ 281K 参数),
配合 KV 堆记忆与多智能体工作流, 面向流式监控 / 异常检测等
持续在线场景。

v0.7.2: CuteMamen 插件标准落地 — 通用固定内核 (CuteMamenKernel)
+ .CuteMamen 专家插件包; 模型精简为 CubeGPTKernel
(必要思考留内核, 其余思考由插件实现)。
"""

__version__ = "0.7.2"

__all__ = [
    "__version__",
    "CubeGPT",
    "DistributedFormer",
    "KVStack",
    "SpikingUnit",
    "SpikeMessage",
    "MultiModalCodec",
    "export_face",
    "import_face",
    "read_manifest",
    # CuteMamen (v0.7.2)
    "CuteMamenKernel",
    "CubeGPTKernel",
    "ExpertPlugin",
    "FacePlugin",
    "LoRABridgePlugin",
    "EventBus",
    "save_pkg",
    "load_pkg",
]

_CUTEMAMEN_EXPORTS = {
    "CuteMamenKernel", "CubeGPTKernel", "ExpertPlugin", "FacePlugin",
    "LoRABridgePlugin", "LoRAAdapter", "EventBus", "ExpertPlugin",
    "PluginMemory", "PluginContext", "save_pkg", "load_pkg",
    "read_manifest", "decode_manifest", "apply_lora", "lora_from_weight",
    "migrate_main",
}


def __getattr__(name):
    # 延迟导入, 避免在 import 包时就加载 numpy 重型依赖链
    if name in ("CubeGPT", "DistributedFormer"):
        from distributedformer.core import distributedformer as _core
        return getattr(_core, name)
    if name == "KVStack":
        from distributedformer.core.distributedformer import KVStack
        return KVStack
    if name == "SpikingUnit":
        from distributedformer.core.distributedformer import SpikingUnit
        return SpikingUnit
    if name == "SpikeMessage":
        from distributedformer.core.distributedformer import SpikeMessage
        return SpikeMessage
    if name == "MultiModalCodec":
        from distributedformer.codec.spike_codec import MultiModalCodec
        return MultiModalCodec
    if name in ("export_face", "import_face", "read_manifest"):
        from distributedformer.core import face_pkg
        return getattr(face_pkg, name)
    if name in _CUTEMAMEN_EXPORTS:
        from distributedformer import cutemamen
        return getattr(cutemamen, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
