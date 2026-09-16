"""
DistributedFormer — 事件驱动的脉冲神经网络智能体框架 (顶层包 src/)

════════════════════════════════════════════════════════════════
 本包是什么 / 目录怎么读 (一图流)
════════════════════════════════════════════════════════════════

DistributedFormer 是"固定内核 + 可插拔思考插件"的分布式智能体框架:

    ┌─────────────────────────────────────────────────────┐
    │  事件 {"topic": ..., "data": ...}                     │
    └───────────────────────┬─────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │  固定内核 (cutemamen/kernel.py)                        │
    │  · 路由: topic → 插件 (route_index 随用随载热加载)      │
    │  · 工作记忆: 固定容量 KV 堆注意力                       │
    │  · 调度: 内存预算 LRU 淘汰 (淘汰前自动存档 .CuteMamen)  │
    │  · 必要思考: 立方体棱路由 / 输出头 / 节律调制            │
    └───────────────────────┬─────────────────────────────┘
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │  思考插件 (.CuteMamen 独立包文件, ./plugin/ 交付)       │
    │  · on_load / on_think / on_unload 生命周期            │
    │  · 三级记忆 (working/episodic/semantic) 随包存档       │
    │  · 例: RustCodingPlugin — 携带主模型迁移的 Rust 知识    │
    └─────────────────────────────────────────────────────┘

子目录一览 (每个目录的 __init__.py 有更细的职责说明):

    core/               神经底座: SpikingUnit 脉冲单元 / 分形皮层层 /
                        KVStack 堆记忆 / CubeGPT 主模型 / 模态面 pkg
    cutemamen/          插件标准: 固定内核 / ExpertPlugin 基类 /
                        .CuteMamen 包格式 / 事件总线 / LoRA 桥接 /
                        RustCodingPlugin (主模型知识迁移宿主)
    data/               真实数据: Rust 编码基准语料 (502 段, 无合成)
    codec/              编解码: 数值/文本/时序 → 脉冲, 脉冲 → 动作
    agents/             5 类脉冲智能体基类
    workflow/           工作流引擎 + 消息路由
    training/           训练: 监督/STDP 训练器 / 读出层验证协议
                        (主模型知识迁移的源头, readout.py)
    deployment/         部署: Docker / K8s / Redis / Prometheus
    security_monitor/   安全: 意图探针 / 决策审计 / 熔断
    demos/              演示: 股票监控 / CubeGPT 终端聊天
    cli.py / selfcheck.py   dformer 命令行 / 模块自检

关键数据通路 (Rust 知识, v0.8.6 新阶段 · 分布式架构):

    真实语料 (data/) → 双模态注入 → 冻结 CubeGPT 水库 (core/)
        → 线性读出层训练 (training/readout.py)
        → migrate_from_main_model() 迁移为插件可存档权重
        → plugin/RustCoding.CuteMamen 独立交付
        → 内核 think(topic="rust") 路由即服务 (76.7%, 与主模型无损一致)

════════════════════════════════════════════════════════════════
 版本历史 (摘要)
════════════════════════════════════════════════════════════════

v0.7.2: CuteMamen 插件标准落地 — 通用固定内核 (CuteMamenKernel)
+ .CuteMamen 专家插件包; 模型精简为 CubeGPTKernel
(必要思考留内核, 其余思考由插件实现)。
v0.7.3: Rust coding 思考插件 (RustCodingPlugin) — 内嵌真实 Rust
语料作训练材料, 填补训练监督脱离实际的空白。
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
v0.8.3: P2 交叉验证评估 — 读出层验证协议由单次 75/25 划分改为
5 折分层交叉验证 (× 5 种子), 评估结论不再依赖划分运气。
v0.8.4: P2 扩真实语料 — Rust 编码基准语料 100 → 502 段
(rustc 错误索引官方样例 + 真实 crate 编译失败样本), 读出层
75.6%, 种子间评估方差 ±5% → ±0.8%, 准确率瓶颈方案收官。
v0.8.5: .CuteMamen 插件标准落地 ./plugin 目录 — 思考插件以独立
包文件交付 (plugin/RustCoding.CuteMamen), 内核 discover_plugins()
扫描注册, think() 按路由主题随用随载热加载。
v0.8.6: 新阶段 · 分布式架构 — 主模型 Rust 知识迁移到思考插件
(76.7%, 与主模型直评逐折一致); 修复 reset_state 不重置融合输入
与皮层布线未 seed Python random 两处不可复现问题; 代码扁平化
到 src/ 顶层包, 文档整理到 docs/。
v0.8.7: 更整洁的项目 + 轻量视频生成内核插件 — 运行时缓存统一到
./cache (解码/卸载/淘汰自动存档 + pytest 缓存 + Python 字节码缓存
集中一处, src/ 不再散落 __pycache__); 清理上个版本废弃路径与
一次性临时脚本; 新增 plugin/VideoMaking.CuteMamen — 关键帧 +
镜头运动曲线 → 缓动仿射帧序列 + 转场合成, 纯 numpy 零 GPU,
供宿主 (OmniSpace 等) 作轻量视频生成档位。
"""

__version__ = "0.8.7"

# ── Python 字节码缓存统一 (v0.8.7) ──────────────────────────
# 在导入任何子模块前设置 sys.pycache_prefix, 使 src/ 各子包的
# __pycache__ 统一写入根目录 ./cache/pycache, 不再散落各子目录。
# (pytest 场景由根 conftest.py 提前设置; 仅本文件自身的 .pyc
# 仍写 src/__pycache__, 可用环境变量 PYTHONPYCACHEPREFIX 消除)
import os as _os
import sys as _sys

if _sys.pycache_prefix is None:
    _sys.pycache_prefix = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "cache", "pycache")
    _os.makedirs(_sys.pycache_prefix, exist_ok=True)

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
        from src.core import distributedformer as _core
        return getattr(_core, name)
    if name == "KVStack":
        from src.core.distributedformer import KVStack
        return KVStack
    if name == "SpikingUnit":
        from src.core.distributedformer import SpikingUnit
        return SpikingUnit
    if name == "SpikeMessage":
        from src.core.distributedformer import SpikeMessage
        return SpikeMessage
    if name == "MultiModalCodec":
        from src.codec.spike_codec import MultiModalCodec
        return MultiModalCodec
    if name in ("export_face", "import_face", "read_manifest"):
        from src.core import face_pkg
        return getattr(face_pkg, name)
    if name in _CUTEMAMEN_EXPORTS:
        from src import cutemamen
        return getattr(cutemamen, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
