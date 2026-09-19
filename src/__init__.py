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
v0.8.7: 更整洁的项目 + 轻量视频生成内核插件 — 运行时产物统一到
./cache (解码/卸载/淘汰自动存档 + pytest 缓存), 仓库内零
__pycache__ (关闭字节码落地); tests 与 experiments 收进 src/,
清理上个版本废弃路径与一次性临时脚本; 新增
plugin/VideoMaking.CuteMamen — 关键帧 + 镜头运动曲线 → 缓动
仿射帧序列 + 转场合成, 纯 numpy 零 GPU, 供宿主 (OmniSpace 等)
作轻量视频生成档位。
v0.9.1: OpenAI 兼容接口服务器 (落地 · OpenCode 对接) — 纯标准库
http.server 暴露 /v1/models 与 /v1/chat/completions (含 SSE 流式),
OpenCode 配置 baseURL 即把 CubeGPT 当编码模型后端; 三级流水线
(CubeGPT 前置感知 → 真 LLM 生成 → CubeGPT 后置审计), 外挂 LLM 封装为
标准 CuteMamen 插件 LLMProviderPlugin (base_model=llm.provider), 配置
本地 OpenAI 兼容后端 (Ollama/llama.cpp) 即可做任意语言代码生成, 未配置
则回退纯 CubeGPT Rust 知识/脉冲模板 (离线可用); 支持 Bearer API 密钥。
新增 dformer serve-opencode 子命令。
v0.9.2: 安全AI P1 落地 — Safety Shield 四方向齐全 (意图探针/熔断 P0,
行为指纹与异常基线检测/全链路行为审计与溯源 P1); gate() 接入行为指纹
判定与四元组审计存储, stats() 并入 fingerprint/tracer 统计。
v0.10.0: 分布式多节点落地 — RedisKVStack 补齐与内存 KVStack 相同的协议
(含主计算路径 retrieve()), 工作流引擎/记忆智能体可按后端切换内存或
Redis 全局 KV; 多节点经 Redis 共享同一租户记忆, 租户 key 前缀隔离;
dformer serve 支持 --kv-backend, docker-compose 双节点共享记忆。
v0.10.1: 准确率基准同步与 L2 选优 — rust_benchmark 从 100 段单次 75/25
升级为 502 段 5 折分层 CV (消除脚本/结果与 README 脱节), 读出层 L2 网格
选优确认 1e-3 最优, 复测 25 折 76.7%, 与插件知识迁移逐折一致。
v0.10.2: 修复端到端后期漂移 — DFTrainer 新增 LR 调度 (默认 cosine) 与
水库冻结; cosine 后期降低 w_in 学习率, 非平稳水库不再使验证准确率回落,
12 轮在 502 段语料上最终 epoch 稳定 (含深度2复评)。
v0.13.0: Safety Shield 接入内核主路径 — SecurityMonitor 经 CuteMamenKernel
enable_security() 以执行层订阅者形态接进动作主通路 (ALLOW 转发执行主题,
DENY/REVIEW 拦截留痕); ActionTracer 落盘持久化 (persist_path, JSONL, 默认关);
修复 tests 收进 src 后遗留的旧 import 路径 bug。
v0.13.1: 真实时序异常检测 (P0 立项) — 框架主场"流式监控/异常检测"首块真实
时序基准: NAB 真实运维指标序列 (real*, 纯真实无合成) 固化 src/data/metrics_ts,
OperationalMetricsDataset 窗口化 + 正常/异常二元标签 + training_data() 导出;
实验 R5 窗口级异常检测基准 (5种子×5折, 窗口 ACC + 异常检出率); 实验 R6 时序流
在线持续学习漂移/稳定性验证 (时间顺序前后划分, cosine/冻结缓解后期漂移);
CLI 新增 --dataset ts / benchmark-anomaly / benchmark-stream; LinearReadout 增
可选类别权重; benchmark-all 扩展为四真实任务。
"""

__version__ = "0.16.0"

# ── 仓库内零 __pycache__ (v0.8.7) ────────────────────────────
# 在导入任何子模块前关闭字节码落地, 使直接运行 (python -m src /
# python src/...) 与 pytest 一样不在 src/ 各子目录生成
# __pycache__。本文件自身的 .pyc 写入早于本开关, 随手清掉;
# 如需字节码缓存, 运行前设 PYTHONPYCACHEPREFIX 指向仓库外目录。
import os as _os
import shutil as _shutil
import sys as _sys

if not _sys.dont_write_bytecode:
    _sys.dont_write_bytecode = True
    _shutil.rmtree(_os.path.join(_os.path.dirname(
        _os.path.abspath(__file__)), "__pycache__"), ignore_errors=True)

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
    "RustCodingPlugin", "JavaCodingPlugin", "LoRABridgePlugin",
    "LoRAAdapter", "EventBus",
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
