"""
DistributedFormer — 事件驱动的脉冲神经网络智能体框架

以 16 参数脉冲神经元为基本单元, 通过分形递归扩展规模,
配合 KV 堆记忆与多智能体工作流, 面向流式监控 / 异常检测等
持续在线场景。
"""

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "DistributedFormer",
    "KVStack",
    "SpikingUnit",
    "SpikeMessage",
    "MultiModalCodec",
]


def __getattr__(name):
    # 延迟导入, 避免在 import 包时就加载 numpy 重型依赖链
    if name == "DistributedFormer":
        from distributedformer.core.distributedformer import DistributedFormer
        return DistributedFormer
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
