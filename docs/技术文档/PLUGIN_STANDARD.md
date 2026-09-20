# CuteMamen 插件标准（技术文档）

> 本文档是 README「插件标准」相关技术细节的完整归档：`.dfpkg` 模态面存档（见
> [MODALITY_ARCHIVE.md](./MODALITY_ARCHIVE.md)）的通用化——`.CuteMamen` 专家
> 插件标准、`CubeGPTKernel` 内核形态，以及内嵌思考插件的用法示例。

## 核心机制一览

|机制|说明|
|-|-|
|**CubeGPT**|4 个模态面（每面 = 16 单元输入端口 + 深度 2 分形皮层 4,368 单元）环形侧连 + 顶层 OutputModule 头部，4×4,400×16 ≈ 281K 参数|
|**CubeGPTKernel** (v0.7.2)|模型精简形态：内核只留必要思考（棱路由 / KV 记忆 / 输出头 / 节律），皮层计算全部外移为 FacePlugin 思考插件|
|**CuteMamen 内核** (v0.7.2)|通用固定内核（Nest）：路由器 + 工作记忆 + 插件注册表 + 内存预算 LRU 淘汰，随用随载热加载|
|**RustCodingPlugin** (v0.7.3)|内嵌 100 段真实 Rust 语料的思考插件：训练材料 `training_data()` 导出 + 最近原型分类（每类原型权重随包往返）|
|**16 参数脉冲单元**|集成放电模型：输入门控 + 状态反馈 + 疲劳/不应期 + 自发放电（默认模式网络，无输入仍"持续思考"）|
|**多模态顶层模块**|numeric / text / timeseries / image 四种模态各拥有独立的 16 单元输入模块（含绑定编码器），输出为独立顶层 OutputModule，模态按权重融合进思考层|
|**分形递归**|每层 16 单元，深度 d 的思考层含 Σ16^k (k=1..d+1) 个单元，深度 2 ≈ 4,368 单元 / 69K 参数|
|**KV 堆记忆**|分布式持久 KV 存储（余弦相似度检索 + 时间衰减 + LRU 淘汰），替代 Transformer 的 KV Cache|
|**5 类脉冲智能体**|感知 / 推理 / 动作 / 记忆 / 节律，共享全局 KV 堆，脉冲每跳衰减 ×0.7|
|**节律调制**|120 步周期（80 步思考 + 40 步抑制），模拟昼夜节律的全局兴奋/抑制切换|
|**STDP 可塑性**|脉冲时序依赖学习（LTP/LTD），与监督信号协同|

## .CuteMamen 标准 (v0.7.2)

v0.7.0 的模态面 pkg 是首个特例；v0.7.2 把它泛化为完整的插件标准实现：

**一个轻量级固定内核（Nest）+ N 个独立训练的专家思考插件。**
只有被路由激活的插件消耗算力，空闲的专家零成本——通过彻底消除空闲计算，
与每次请求激活全部参数的单体模型形成架构级差异。

```python
from distributedformer.cutemamen import (
    CuteMamenKernel, ExpertPlugin, LoRAAdapter, LoRABridgePlugin,
    save_pkg, load_pkg,
)

# ── 1. 写一个思考插件: 只需实现生命周期钩子 ──────────────
class TrendPlugin(ExpertPlugin):
    BASE_MODEL = "generic"          # manifest.base_model (加载注册表按它分发)

    def on_think(self, event, ctx): # 事件到达、被路由激活 → 核心计算
        value = event["data"]
        history = ctx.working_memory.kv_stack  # 内核工作记忆
        ctx.emit("trend.computed", value)      # 事件总线: 插件间通信
        return {"trend": value > 0}

# ── 2. 内核只做路由 / 工作记忆 / 内存调度, 不含任何专家计算 ──
kernel = CuteMamenKernel(dim=16, memory_budget_mb=64)
kernel.mount(TrendPlugin("trend", route="numeric"))
kernel.bus.subscribe("trend.computed", lambda e: print("事件总线:", e))

kernel.think({"topic": "numeric", "data": 1.5})   # 路由 → on_think
# 生命周期广播: plugin.loaded / plugin.unloaded / plugin.evicted / kernel.think

# ── 3. .CuteMamen 专家插件包: 存档 / 传输 / 随用随载 ─────
ad = LoRAAdapter("wq", a=..., b=..., alpha=2.0)    # ΔW = (α/r)·B@A
plugin = LoRABridgePlugin("lora-wq", adapter=ad)   # LoRA ↔ 插件桥接
kernel.mount(plugin)
save_pkg(plugin, "lora-wq.CuteMamen")              # manifest + weights + 三级记忆
kernel.unmount("lora-wq")                          # 卸载自动存档并注册
kernel.think({"topic": "lora-wq", "data": x})      # 用到时现场热加载
```

标准要点（全部已实现，106 项测试覆盖）：

|规范条款|实现|
|-|-|
|生命周期钩子 `on_load` / `on_think` / `on_unload`|`ExpertPlugin` 基类托管，内核按序调用|
|三级记忆存档 working / episodic / semantic|`PluginMemory`，随包 `memory/` 目录存档还原，情景→语义自动蒸馏|
|事件总线通信|`EventBus` pub/sub + `*` 通配 + 生命周期广播 + 插件间消息|
|内存预算淘汰|manifest 声明占用，超预算 LRU 淘汰（先自动存档，绝不淘汰正在思考的专家）|
|`.CuteMamen` 包格式|单个 tar.gz：`manifest.json` + `weights/` + `memory/`|
|解码器层集中兼容|v1 清单自动适配（`model_type`→`base_model`、`on_init`→`on_load`），未知字段保留|
|LoRA/Adapter 兼容桥接|`LoRABridgePlugin`（ΔW·x 低秩贡献）+ `lora_from_weight` SVD 导出|
|`min_core_version` 检查|内核过旧拒绝加载|

**演进规则**：允许破坏性变更，解码器吸收迁移成本（内核永远只看一种内部格式），
`migrate-v1-to-v2` 命令自动转换旧版插件包（随 `pip install -e .` 安装）：
`migrate-v1-to-v2 ./plugins/ --recursive --dry-run`（预览）/ `--strict`（无法
映射字段时报错）/ `--backup`（迁移前备份）。

> **原则：可以往插座上多加孔，但不能把已有的孔堵上。**
> 完整兼容性规范见 [COMPATIBILITY.md](../规范文档/COMPATIBILITY.md)。
> DistributedFormer v0.7.2 即是一个完整参考实现。

## 模态面内核：CubeGPTKernel (v0.7.2)

按"模型只保留必要思考，其余思考交给插件"的原则，CubeGPT 有了内核形态
`CubeGPTKernel`——经典 CubeGPT 的职责被重新划分：

|经典 CubeGPT 职责|CubeGPTKernel 中的去向|
|-|-|
|模态面皮层计算|**外移** → `FacePlugin` 思考插件（可卸载 / 热加载 / 预算淘汰）|
|立方体棱（邻面脉冲注入）|内核路由（必要思考）|
|KV 堆注意力|内核工作记忆（必要思考）|
|输出头（16 单元）|内核最小读出（必要思考）|
|节律调制|内核（必要思考）|
|STDP 协调 / 内存预算 / 随用随载|内核生命周期职责（新增）|

```python
from distributedformer import CubeGPT
from distributedformer.cutemamen import CubeGPTKernel

gpt = CubeGPT(depth=1, dim=16)          # 经典形态
kernel = gpt.to_kernel()                # 零拷贝转换: 同一面对象 / KV 堆 / 输出头
kernel.step({"numeric": 1.0, "text": "hello"})   # API 与经典形态一致

# 直接构造: 每个面挂载为思考插件, 内核自身只有 16 个输出头单元
kernel = CubeGPTKernel(depth=2, dim=16, memory_budget_mb=64)
kernel.step({"numeric": 1.0, "text": "..."})     # 预算紧张时面被淘汰→自动存档
kernel.step({"text": "..."})                     # 再用到时从注册表热加载
```

经典 `CubeGPT` 保留为训练基底与一体化形态，两者共享同一套面 pkg
（`.dfpkg` ⇄ `.CuteMamen` 双向可载）。

## 内嵌思考插件用法示例

### RustCodingPlugin（route=`rust`, base_model=`rust.coding`）

内置真实语料作训练/评估材料，内核无需额外数据源：

```python
from distributedformer.cutemamen import CuteMamenKernel, RustCodingPlugin
kernel = CuteMamenKernel(dim=16)
plugin = RustCodingPlugin("rust-coding")            # route 默认 "rust"
kernel.mount(plugin)

# 训练材料: 直接导出真实特征 + 标签, 供端到端训练 (路线图数据通路)
X, y = plugin.training_data()                       # (502, 特征维) / (502,) (v0.8.4 扩 502 段)

# 现场思考: 分类一段 Rust 代码命中的编译错误类别
result = kernel.think({"topic": "rust",
                       "data": "fn longest(s1: &str, s2: &str) -> &str {\n  ...\n}"})
# [{'label': 'lifetime', 'label_name': '生命周期', 'rustc': 'E0597/E0106/E0515/E0716', 'confidence': ...}]

# 存档 / 随用随载: 原型权重 (每类 static_metrics 质心) 随包往返
save_pkg(plugin, "plugin/MyRust.CuteMamen")
loaded, manifest = load_pkg("plugin/MyRust.CuteMamen")  # base_model=rust.coding
```

### VideoMakingPlugin（route=`video`, base_model=`video.making`）

轻量视频生成内核，纯 numpy、零 GPU：

```python
from distributedformer.cutemamen import CuteMamenKernel, VideoMakingPlugin
kernel = CuteMamenKernel(dim=16)
plugin = VideoMakingPlugin("video-making")          # route 默认 "video"
kernel.mount(plugin)

# 训练材料: 导出真实运镜曲线稠密采样 (X, y), 供端到端训练
X, y = plugin.training_data()                       # (200, 5) / (200,) (10 类 × 20 点)

# 现场生成: 关键帧 + 镜头运动 → 帧序列 (无 GPU)
result = kernel.think({"topic": "video",
                       "data": {"keyframes": [frame_hw3, frame_hw3],
                                "shots": [{"motion": "zoom_in", "duration_s": 2.0}]}})
# {'summary': {'n_frames': ..., 'shots': ['zoom_in'], 'shape': [...], 'kernel': 'lightweight-numpy-v1'}, ...}
```

> 各插件对应的真实数据集定义见 `src/data/`：`rust_coding.py`、
> `video_motion.py`、`metrics_time_series.py`。