<div align="center">

# DistributedFormer

**事件驱动的脉冲神经网络智能体框架** · 内嵌模型 **CubeGPT** · 插件标准 **CuteMamen**

以 16 参数脉冲神经元为基本单元 · CubeGPT 立方体连接多模态脉冲大模型 · KV 堆工作记忆 · 多智能体脉冲工作流
· CuteMamen 固定内核 + 专家思考插件
面向流式监控、异常检测等持续在线场景

[![CI](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml)
[![PyPI - Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.8.2-orange)](CHANGELOG.md)

</div>

---

## 这是什么

DistributedFormer 探索一条不同于 Transformer 的路线：**用超简单的神经元（每个恰好 16 个标量参数）+
事件驱动的异步脉冲传播**，构建可以 7×24 持续在线、按事件触发计算的智能体网络。它的目标场景是
"永远在线的流式监控"——市场异动检测、指标巡检、IoT 阈值告警——这类任务不需要大模型的重算力，
需要的是低延迟、事件驱动和持久的工作记忆。

框架内嵌的模型是 **CubeGPT**：
**Cube** 指连接方式——numeric / text / timeseries / image 四个模态面构成立方体的四个侧面，
以"棱"环形侧连（每步把本面脉冲聚合注入邻面）；**GPT** 则是向 ChatGPT 致敬的命名
（这里不妨戏称其为 Generative Pulse Transformer）。
规模为 4 面 × \~4,400 单元 × 16 参数 ≈ **281K 参数**，纯 numpy 即可运行。

核心机制一览：

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

## 安装

```bash
pip install -e .
# 可选: 真实行情数据源
pip install -e ".\\\[realtime]"
```

依赖极轻：核心只需要 `numpy`。

## 快速开始

```bash
# 端到端演示: 3 只股票的持续监控脉冲网络（模拟数据）
dformer demo --cycles 10 --interval 0.5

# 长驻监控服务（Docker/生产默认入口）
dformer serve --tickers AAPL TSLA NVDA --interval 300

# 使用真实行情 (yfinance)
dformer serve --realtime --interval 300

# 真实数据训练 (Rust 编码基准, v0.7.5)
dformer train --depth 1 --epochs 10

# 模块自检
dformer test

# 终端聊天: 与 CubeGPT 对话 (v0.7.1)
dformer chat

作为库使用（v0.3.0 多模态 API）：

```python
from distributedformer import CubeGPT, KVStack

kv = KVStack(capacity=10000, dim=16)
gpt = CubeGPT(depth=2, dim=16)   # 深度2 = 281K 参数

# 4 个模态面独立编码, 立方体棱环形侧连, 顶层输出模块生成动作脉冲
spikes = gpt.step({
    "numeric": 1.5,              # 标量或向量
    "text": "market surges",     # TF-IDF 稀疏编码
    "timeseries": \\\[1, 2, 4, 3],  # 差分编码
    "image": gray\\\_image\\\_2d,      # 4×4 池化空间编码
})
```

## 模态面 pkg 存档: 随用随载热加载 (v0.7.0)

每个模态面可以独立打包为 `.dfpkg` 存档（遵循 CuteMamen 插件规范 v0.1.0 的包格式精神：
单个 tar.gz，内含 `manifest.json` 清单 + `weights/` 权重 + `memory/` 状态），
支持自由导入导出与随用随载热加载：

```python
gpt = CubeGPT(depth=1, dim=16, modalities=["numeric", "text"])

# 导出: 把 text 面打包成独立存档
gpt.export_face("text", "text.dfpkg", author="me", capability="文本语义计算")

# 卸载: 自动先导出 pkg 再从内存移除 (可随时恢复)
gpt.unload_face("text")                    # 默认存到 face_pkgs/text.dfpkg
gpt.list_faces()                           # {'loaded': ['numeric'], 'registered': ['text']}

# 随用随载: 注册后不占内存, step() 用到该模态时现场热加载
gpt.register_face_pkg("face_pkgs/text.dfpkg")
spikes = gpt.step({"text": "hello"})       # 触发热加载, 之后常驻

# 或显式加载 / 导入到另一个模型
gpt2 = CubeGPT(depth=1, dim=16, modalities=["numeric"])
gpt2.import_face("text.dfpkg")             # 权重 + 状态逐位还原
```

`manifest.json` 含模态/深度/dim/单元与参数规模/内存占用，带 `min_core_version`
兼容性检查（内核过旧拒绝加载）。权重与状态逐位可复现：感受野投影由 layer_id
的 crc32 种子重建，小世界连接随 STDP 训练后的真值一起存档。

## CuteMamen 插件标准落地 (v0.7.2)

v0.7.0 的模态面 pkg 是首个特例；v0.7.2 把它泛化为完整的插件标准实现：

**一个轻量级固定内核（Nest）+ N 个独立训练的专家思考插件。**
只有被路由激活的插件消耗算力，空闲的专家零成本。

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

标准要点（全部已实现，97 项测试覆盖）：

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

## 模型精简: CubeGPTKernel (v0.7.2)

按"模型只保留必要思考，其余思考交给插件"的原则，CubeGPT 有了内核形态
`CubeGPTKernel`——经典 CubeGPT 的四类职责被重新划分：

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

## CuteMamen 标准: 架构哲学与演进规则

上一节是 v0.7.2 的**落地实现**；本节说明标准本身的哲学与演进规则。

> 完整兼容性规范见 [COMPATIBILITY.md](./COMPATIBILITY.md)

### 架构哲学

不同于每次请求都激活全部参数的单体模型，CuteMamen 采用了一种根本不同的方式：

- **N 个独立训练的专家插件** —— 每个插件是一个自包含的、面向特定任务的模块
- **一个轻量级固定内核（Nest）** —— 仅负责路由、生命周期管理和内存调度
- **可热加载的 `.CuteMamen` 插件包** —— 插件可在运行时加载、卸载和替换，无需重启宿主系统

只有被激活的插件消耗算力，空闲的专家零成本。

### 成本优势

| 方案 | 每万亿输出 token 成本 |
|------|---------------------|
| 单体 Transformer（如 DeepSeek） | ~$1,500,000 |
| CuteMamen Rust 插件 | ~$1.5 |

这不是渐进式优化 —— 而是**六个数量级**的架构差异，通过彻底消除空闲计算实现。

### 兼容性策略

#### 解码器层集中兼容

所有向前/向后兼容逻辑集中在单一层 —— **解码器**，其他模块无需感知插件版本。

```text
v1 插件 → 解码器（翻译为内部统一格式）→ 内核 → 各模块
v2 插件 → 解码器（直接透传，零开销）  → 内核 → 各模块
v3 插件 → 解码器（直接透传，零开销）  → 内核 → 各模块
```

- **内核和各模块**永远只看到一种内部格式，对版本无感知
- **旧插件**只要解码器保留适配逻辑就能继续运行
- **新插件**不经过任何翻译层，性能无损
- **移除旧版支持**只需删除解码器中的一个函数，其他代码一行不动

#### 标准演进规则

当标准发生演进时（如 CuteMamen v1 → v2）：

1. **允许破坏性变更** —— 新标准可以丢弃遗留字段、简化接口
2. **解码器吸收迁移成本** —— 通过 `manifest.json` 检测插件版本并自动适配
3. **提供迁移工具** —— `migrate-v1-to-v2` 自动转换旧版插件包
4. **公布废弃时间窗口** —— 旧插件触发警告，附带明确的停止支持日期

> **原则：可以往插座上多加孔，但不能把已有的孔堵上。**
> 新增字段和钩子永远欢迎。删除或修改已有契约需要版本号升级，并在解码器层做适配。

### 与 Transformer 系统的集成

CuteMamen 设计了多种与现有 Transformer 架构兼容的集成路径：

- **LoRA 适配器映射** —— 每个 `.CuteMamen` 插件映射为一个 LoRA 适配器，Transformer 基础模型充当内核
- **外部服务桥接** —— 插件作为独立进程运行，通过标准 API（OpenAI 兼容或 gRPC）调用
- **MoE 专家注册** —— 插件注册为可动态加载的混合专家（MoE）单元，在推理时按需调度

任何声称"CuteMamen 兼容"的系统，必须实现生命周期钩子（`on_load`、`on_unload`、`on_think`），遵守 `manifest.json` 格式规范，并支持 `.CuteMamen` 插件包格式。完整规范见 [COMPATIBILITY.md](./COMPATIBILITY.md)。DistributedFormer v0.7.2 即是一个完整参考实现。

---

## 迁移工具：migrate-v1-to-v2

用于将 CuteMamen v1 插件包自动迁移至 v2 标准的命令行工具（v0.7.2 起随
`pip install -e .` 一起安装，同时可 `python -m distributedformer.cutemamen.migrate` 调用）。

### 基本用法

```bash
# 迁移单个插件
migrate-v1-to-v2 ./my-plugin.CuteMamen

# 迁移整个目录下的所有插件
migrate-v1-to-v2 ./plugins/ --recursive

# 指定输出目录（不覆盖原文件）
migrate-v1-to-v2 ./my-plugin.CuteMamen --output ./migrated/

# 预览模式（只检查，不实际修改）
migrate-v1-to-v2 ./my-plugin.CuteMamen --dry-run
```

### 完整参数列表

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--output <dir>` | `-o` | 输出目录，不指定则覆盖原文件 | 覆盖原文件 |
| `--recursive` | `-r` | 递归处理目录下所有 `.CuteMamen` 文件 | 否 |
| `--dry-run` | `-n` | 预览模式，只报告变更内容，不实际写入 | 否 |
| `--verbose` | `-v` | 显示详细的迁移日志 | 否 |
| `--strict` | | 严格模式：遇到无法自动迁移的字段直接报错退出 | 否（默认跳过并警告） |
| `--backup` | `-b` | 迁移前自动备份原文件为 `.CuteMamen.bak` | 否 |
| `--from <version>` | | 指定源版本号（默认自动检测） | 自动检测 |
| `--to <version>` | | 指定目标版本号 | `2.0.0` |

### 输出示例

#### 正常迁移

```text
$ migrate-v1-to-v2 ./plugins/ -r -v

[1/3] Migrating: plugins/text-gen.CuteMamen
  ✓ manifest.json: standard_version 1.2.0 → 2.0.0
  ✓ manifest.json: renamed field "model_type" → "base_model"
  ✓ manifest.json: removed deprecated field "legacy_mode"
  ✓ lifecycle: renamed hook "on_init" → "on_load"
  ✓ lifecycle: added missing hook "on_unload" (stub)
  ✓ packaged: plugins/text-gen.CuteMamen (v2)

[2/3] Migrating: plugins/code-review.CuteMamen
  ✓ manifest.json: standard_version 1.0.3 → 2.0.0
  ✓ manifest.json: renamed field "model_type" → "base_model"
  ⚠ lifecycle: hook "on_warmup" not found in v1, skipped
  ✓ packaged: plugins/code-review.CuteMamen (v2)

[3/3] Migrating: plugins/old-vision.CuteMamen
  ✗ manifest.json: field "pipeline_config" has no v2 equivalent
    → Use --strict to fail on unresolvable fields
    → Manual migration required for this plugin

Done. 2 migrated, 1 needs manual intervention.
```

#### 预览模式

```text
$ migrate-v1-to-v2 ./my-plugin.CuteMamen --dry-run

[DRY RUN] No files will be modified.

Would apply the following changes to my-plugin.CuteMamen:
  - manifest.json: standard_version "1.2.0" → "2.0.0"
  - manifest.json: rename "model_type" → "base_model"
  - manifest.json: remove "legacy_mode" (deprecated)
  - lifecycle: rename "on_init" → "on_load"
  - lifecycle: inject stub "on_unload"

No errors. Safe to migrate.
```

### 迁移规则清单

工具内部按以下规则逐条执行：

| 变更类型 | v1 | v2 | 处理方式 |
|---------|-----|-----|---------|
| 字段重命名 | `model_type` | `base_model` | 自动重命名 |
| 字段删除 | `legacy_mode` | （已移除） | 自动删除，记录警告 |
| 钩子重命名 | `on_init` | `on_load` | 自动重命名 |
| 钩子新增 | （不存在） | `on_unload` | 自动注入空实现 |
| 版本号更新 | `1.x.x` | `2.0.0` | 自动更新 |
| 无法映射的字段 | 自定义字段 | 无对应 | 跳过并警告（`--strict` 下报错） |

### 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 全部迁移成功 |
| `1` | 部分插件需要手动迁移 |
| `2` | 严重错误（文件不存在、格式损坏等） |

## Docker 部署

```bash
cd distributedformer/deployment
docker compose up -d          # Redis + 2 个监控节点
docker compose --profile monitoring up -d   # 加 Prometheus
```

Kubernetes 清单（Deployment / Service / HPA / ConfigMap）见
[`distributedformer/deployment/k8s/`](distributedformer/deployment/k8s/)。

## 项目结构

```
DistributedFormer/
├── distributedformer/           # Python 包
│   ├── core/                    # 脉冲单元 / 分形层 / KV 堆 / CubeGPT / 模态面 pkg
│   ├── cutemamen/               # CuteMamen 插件标准 (v0.7.2): 内核 / 插件 / 事件总线 / 包格式 / LoRA 桥接 / 迁移工具 / Rust coding 插件
│   ├── codec/                   # 数值·文本·时序 → 脉冲编码; 脉冲 → 动作解码
│   ├── agents/                  # 5 类脉冲智能体
│   ├── workflow/                # 工作流引擎 + 消息路由
│   ├── data/                     # 真实数据集: Rust 编码基准 + 纯真实训练集 (v0.7.5)
│   ├── training/                 # 监督/STDP 训练器 + 真实数据读出验证 (纯真实数据)
│   ├── deployment/              # Docker / K8s / Redis / Prometheus
│   ├── demos/                   # 股票监控端到端演示 / CubeGPT 终端聊天
│   ├── cli.py                   # dformer 命令行入口
│   └── selfcheck.py             # 模块自检套件
├── tests/                       # pytest 测试 (104 项)
├── experiments/                 # 消融实验脚本、结果与报告 (E1-E5, R1-R2)
├── reports/                     # 历史训练与实验报告
├── visualization/               # 训练曲线 / 准确率图
└── RELEASE\\\_NOTES.md             # Pre0.1 归档版说明
```

## 实验记录（诚实公开）

Pre0.1 归档了全部研发记录，包括**负结果**：在合成 4 分类股票任务上，消融实验
（[`experiments/ablation\\\_report\\\_extended.md`](experiments/ablation_report_extended.md)）显示
基线准确率 30%、深度 2 大网络 22.5%——**扩大分形规模与思考层监督在该任务上没有正向增益**，
模型停留在随机水平附近。我们认为如实归档这些结果对这个方向的研究有价值。
v0.2.0 的工程化重构（可安装包、CLI、测试、部署链路修复）不改变这一结论。

## 真实数据基准: Rust Coding (v0.6.0)

针对"训练监督以合成数据为主、知识脱离实际"的限制, 新增首个真实需求基准:
**静态识别 Rust 代码命中的编译错误类别**——lints / IDE 提示 / 自动修复工具
的基础能力。

* **数据**: 100 段真实风格 Rust 代码 × 5 类 (所有权移动 E0382 / 借用冲突
  E0502·E0499 / 生命周期 E0597·E0106 / 类型不匹配 E0308·E0277 / 合法代码),
  每条附真实 rustc 错误码与报错信息, 见
  [`distributedformer/data/rust\\\_coding.py`](distributedformer/data/rust_coding.py)
* **输入模态**: text (代码原文, 代码感知分词) + numeric (静态扫描特征)
* **结果**: CubeGPT 读出层 5 种子验证准确率 **57.6% ± 10.3%**, 全部超过
  随机基线 20% (v0.6.0 初版为 54.4% ± 5.4%, 修复 CubeFeatureExtractor
  可复现性后复测提升), 见
  [`experiments/rust\\\_report.md`](experiments/rust_report.md)
  * 各类别 (种子均值): 借用冲突 76% / 生命周期 88% / 所有权移动 48% /
    类型不匹配 44% / 合法代码 32%
* 复现: `python experiments/rust\_\_benchmark.py`

## Rust coding 思考插件 (v0.7.3)

把 v0.6.0 的真实语料装进一个 CuteMamen 专家插件, **填补训练材料空白**:
训练监督不再只有合成随机数据, 插件自带 100 段真实 Rust 代码 × 5 类真实
rustc 错误作为可调用的训练/评估材料, 内核无需额外数据源。

```python
from distributedformer.cutemamen import CuteMamenKernel, RustCodingPlugin
kernel = CuteMamenKernel(dim=16)
plugin = RustCodingPlugin("rust-coding")            # route 默认 "rust"
kernel.mount(plugin)

# 训练材料: 直接导出真实特征 + 标签, 供端到端训练 (路线图数据通路)
X, y = plugin.training_data()                       # (100, 10) / (100,)

# 现场思考: 分类一段 Rust 代码命中的编译错误类别
result = kernel.think({"topic": "rust",
                       "data": "fn longest(s1: &str, s2: &str) -> &str {\n  ...\n}"})
# [{'label': 'lifetime', 'label_name': '生命周期', 'rustc': 'E0597/E0106/E0515/E0716', 'confidence': ...}]

# 存档 / 随用随载: 原型权重 (每类 static_metrics 质心) 随包往返
save_pkg(plugin, "cutemamen_pkgs/rust_coding.CuteMamen")
loaded, manifest = load_pkg("cutemamen_pkgs/rust_coding.CuteMamen")  # base_model=rust.coding
```

可学习权重 = 每类别原型特征向量, 经 `.CuteMamen` 包 weights/ 存档还原;
分类用最近原型 + softmax 置信度, 结果广播 `rust.classified` 事件供插件间订阅。

## 路线图

- [x] v0.2.0 — 产品化重构：可安装包 / CLI / serve 长驻服务 / 真实行情数据源 / 测试与 CI / 修复 Docker 链路
- [x] v0.3.0 — 多模态顶层模块化：四种模态独立输入模块 + 独立输出模块 + 确定性图像编码
- [x] v0.4.0 — 内嵌模型正式命名为 **CubeGPT**（立方体棱连接，4 面 × 4,400 单元 ≈ 281K 参数），智能体框架全面切换
- [x] v0.5.0 — 三项核心修复：KV 注意力接入主计算路径 / 训练方法学验证
  （读出层 64.8% vs 随机 25%，实验 R1）/ 真实桌面通知与 HTTP 动作
- [x] v0.6.0 — 真实数据基准起步：Rust coding（100 段真实代码 × 5 类真实 rustc 错误），
  读出层 57.6% ± 10.3% vs 随机 20%（复测值，初版 54.4%）；同步修复文本编码器的进程随机哈希（不可复现）与丢 token 问题
- [x] v0.7.0 - 模态面独立化与 **pkg存档**：随用随载热加载 / 自由导入导出具体模态面
  （`.dfpkg`，兼容 CuteMamen 包格式）；同步修复训练报告占位符问题
- [x] v0.7.1 — 简单的聊天框：`dformer chat` 与 CubeGPT 对话
- [x] v0.7.2 - **CuteMamen 插件标准落地**：通用固定内核（路由器 + 工作记忆
  + 插件注册表）与 `.CuteMamen` 专家插件包（on_load/on_think/on_unload 生命周期钩子、
  三级记忆存档、事件总线通信、内存预算淘汰、LoRA/Adapter 兼容桥接）；
  **模型精简**为 CubeGPTKernel——必要思考留内核，皮层计算全部由思考插件实现
- [x] v0.7.3 - **Rust coding 思考插件**：内嵌 100 段真实 Rust 语料作训练材料
  填补训练数据空白（`training_data()` 数据通路），每类原型权重随包往返热加载
- [x] v0.7.4 - **KV 堆注意力检索向量化**：打分与 top-k 全程 numpy 批量计算
  （`KVStack` 增量矩阵索引 + swap-remove 淘汰），主计算路径检索提速约 40 倍
  （4096 条堆 54ms → 1.3ms），行为与旧逐条打分实现数值一致
- [x] v0.7.5 - **训练数据真实化**：彻底移除合成训练数据（删除
  `data_generator.py`），训练/评估管道 100% 采用真实 Rust 编码基准语料
  （`RustCodingTrainingDataset`），训练器泛化为 5 类，统一输出随机/多数类
  评估基线；真实数据读出层验证 28%–44%（5 种子均值 36%）vs 随机 20%
- [x] v0.8.0 - **P0 双模态注入**：诊断定位 36% 瓶颈在词袋编码阶段
  （类间/类内距离比 0.68 负可分），`static_metrics` 语法特征走 numeric
  通路 + 代码原文走 text 通路，真实数据读出层验证 **57.6%**（52%–64%，
  5 种子），超随机基线 2.9 倍
- [ ] CubeGPT 端到端可学习：在真实数据集上端到端训练（Rust 基准已提供数据通路, RustCodingPlugin 导出 X/y）
- [ ] Redis 分布式 KV 堆在多节点工作流中实际启用
- [ ] 学习规则改进：目标是在 ≥2 个真实任务上显著超过随机基线

### 36% 瓶颈提升方案 (v0.7.5 诊断结论)

特征通路诊断（同一真实语料、同一线性读出层、5 种子分层划分）定位了
读出层 36% 的瓶颈：**判别信息在编码阶段即丢失**——词袋 TF-IDF 哈希的
类间/类内距离比 0.68（负可分），而纯静态语法特征 `static_metrics`
可达 71.2%。据此排定五个提升方向（按优先级）：

1. **P0 · 双模态注入** ✅ **已完成（v0.8.0）**：`RustCodingTrainingDataset`
   把 `static_metrics`（语法扫描特征）走 numeric 通路 + 代码原文走
   text 通路，读出层用 `CubeFeatureExtractor`（v0.6.0 双模态曾达
   57.6%，纯静态特征诊断 71.2%）——实测 **57.6%**（52%–64%，5 种子）
2. **P1 · 结构感知编码** ✅ **已完成（v0.8.1）**：新增 `structure_metrics`
   6 维结构特征（`&mut` 位置 / 返回引用 `-> &` / 类型标注 / println /
   let 绑定 / 防御调用），与 10 维 `static_metrics` 拼成 16 维恰好
   填满 numeric 通路——直击 move↔lifetime 互混与 type→ok 误判的
   混淆源，读出层 **57.6% → 61.6%**（48%–72%，5 种子）
3. **P1 · 修端到端权重更新** ✅ **已完成（v0.8.2）**：注入-恢复启发式
   （临时改 gain/threshold 再还原，学习信号不累积）替换为**持久输出头**
   （在线 softmax 线性头，权重跨样本/epoch 持久）；`w_in` 更新从
   `error × mean(input)` 均值池化改为**每单元感受野投影**
   `error × (receptive @ input)`——端到端监督训练从刚至基线（20%）跃升到
   **67.2%**（56%–80%，5 种子最佳验证），全部种子超随机基线 3 倍以上
4. **P2 · 扩真实语料**：100 → 500+ 段（rustc 错误索引真实样例、真实
   crate 编译失败样本），验证集 25 → 125，把评估方差从 ±6–9% 降到 ±2%
5. **P2 · 5 折交叉验证**：替代单次 75/25 划分，评估结论不再依赖
   划分运气

## 已知问题（v0.7.2 状态）

* ~~KV 堆注意力未接入主计算路径~~ **已修复（v0.5.0）**：`KVStack.retrieve()` 现为
FractalLayer / 输入端口 / 输出头的真实注意力来源，记忆影响网络动力学；
推理时输出状态持续写入 KV 堆，形成闭环。带 scan\_limit 限额防止大堆拖慢。
* ~~训练方法学未验证~~ **已验证（v0.5.0，实验 R1）**：修复两个动力学缺陷
（均值池化 → 感受野投影；恒定调制淹没输入 → 权重配平）后，水库内部状态 +
线性读出层在合成 4 分类任务上取得 **64.8% ± 4.5%** 验证准确率（5 种子，
随机基线 25%），全部种子稳定超过基线。见
[`experiments/readout\\\_report.md`](experiments/readout_report.md)。
* ~~桌面通知 / API 调用为模拟~~ **已实现（v0.5.0）**：Windows 真实系统通知
（PowerShell toast，`DF\\\_NOTIFY\\\_MODE=sim` 可切回打印）；真实 HTTP POST
（stdlib urllib，endpoint 或 `DF\\\_WEBHOOK\\\_URL` 配置，10s 超时，失败降级不中断）。
* ~~训练监督以合成数据为主~~ **已起步（v0.6.0）**：新增 Rust coding 真实需求
  基准（真实代码 + 真实 rustc 错误类别），读出层 57.6% ± 10.3% vs 随机 20%
  （复测值）；扩展更多真实数据集与真实代码语料仍在路线图中。
* ~~训练材料只有合成数据~~ **已根除（v0.7.5）**：合成训练数据集已整体删除，
  训练/评估管道 100% 采用真实 Rust 编码基准语料
  （`RustCodingTrainingDataset`，100 段真实代码 × 5 类真实 rustc 错误），
  训练入口 / CLI / 消融实验统一携带随机 20% 与多数类评估基线
* ~~训练报告只有占位符~~ **已修复（v0.7.0）**：`report_generate` 动作现在渲染
真实统计数据（智能体状态 / CubeGPT 网络统计 / KV 堆记忆 / STDP 学习统计），
不再产生 `[自动生成内容占位]`。
* ~~CuteMamen 插件标准只有规范文档~~ **已落地（v0.7.2）**：`distributedformer/cutemamen`
包实现内核 / 生命周期钩子 / 三级记忆 / 事件总线 / 内存预算淘汰 / LoRA 桥接 /
迁移工具，`migrate-v1-to-v2` 命令随包安装。
* **端到端训练存在后期漂移（v0.8.2）**：在线持久输出头 + 非平稳水库
  （w_in 持续受监督更新）的组合使训练后期验证准确率回落——最佳验证
  67.2%（56%–80%，5 种子，早停选取），最终 epoch 回落至 36%–60%。
  拟议方向：降低后期学习率 / 冻结水库只调输出头（同离线读出范式）/
  P2 扩语料缓解小样本过拟合。

## 文档

* [架构说明](docs/ARCHITECTURE.md)
* [更新日志](CHANGELOG.md)
* [参与贡献](CONTRIBUTING.md)

## License

[MIT](LICENSE)

