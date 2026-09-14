"""
DistributedFormer — 事件驱动的脉冲神经网络智能体框架

内嵌模型 CubeGPT: 立方体连接的多模态脉冲大模型
(4 个模态面 × ~4,400 单元 × 16 参数 ≈ 281K 参数),
配合 KV 堆记忆与多智能体工作流, 面向流式监控 / 异常检测等
持续在线场景。

v0.7.2: CuteMamen 插件标准落地 — 通用固定内核 (CuteMamenKernel)
+ .CuteMamen 专家插件包; 模型精简为 CubeGPTKernel
(必要思考留内核, 其余思考由插件实现)。
v0.7.3: Rust coding 思考插件 (RustCodingPlugin) — 内嵌 100 段真实
Rust 语料作训练材料, 填补训练监督脱离实际的空白。
v0.7.4: KV 堆注意力检索向量化 — 打分与 top-k 全程 numpy 批量计算,
主计算路径检索提速约 40 倍。
v0.7.5: 训练数据真实化 — 彻底移除合成训练数据, 训练/评估管道
100% 采用真实 Rust 编码基准语料, 并建立随机/多数类评估基线。
v0.8.0: P0 双模态注入 — static_metrics 语法特征走 numeric 通路 +
代码原文走 text 通路, 消除词袋编码信息瓶颈, 真实数据读出层
验证准确率 36% → 57.6%。
v0.8.1: P1 结构感知编码 — 新增 structure_metrics 6 维结构特征
(&mut/返回引用/类型标注等), 与 static_metrics 拼成 16 维填满
numeric 通路, 读出层 57.6% → 61.6%。
v0.8.2: P1 修端到端权重更新 — 注入-恢复启发式替换为持久输出头,
w_in 按每单元感受野投影向量化更新 (消除均值池化的无差异更新),
端到端监督训练首次稳定超过基线。
"""

__version__ = "0.8.2"

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
    "RustCodingPlugin", "LoRABridgePlugin", "LoRAAdapter", "EventBus",
    "ExpertPlugin", "PluginMemory", "PluginContext", "save_pkg", "load_pkg",
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
