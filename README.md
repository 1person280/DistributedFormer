<div align="center">

# DistributedFormer

**事件驱动的脉冲神经网络智能体框架** · 内嵌模型 **CubeGPT** · 插件标准 **CuteMamen**

以 16 参数脉冲神经元为基本单元 · CubeGPT 立方体连接多模态脉冲大模型 · KV 堆工作记忆 · 多智能体脉冲工作流
· CuteMamen 固定内核 + 专家思考插件
面向流式监控、异常检测等持续在线场景

[![CI](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml)
[![PyPI - Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.14.5-orange)](docs/CHANGELOG.md)
[![Audit-Ready Architecture](https://img.shields.io/badge/security-Audit--Ready%20Architecture-blueviolet)](#外部动作)

</div>

---

## 对比一览 · 与同行 Transformer / 脉冲框架

明晃晃的定位：DistributedFormer 走的是 **16 参数脉冲神经元 + 事件驱动异步传播 + 持久工作记忆**的
轻量在线路线。下表和行内稠密 **Transformer** 大模型（GPT-2 / GPT-4o / Llama 等），以及同为
脉冲/事件驱动的研究框架 **snnTorch / Norse / Lava** 放在同一张能力表里对照。能力维度均为
公开可核验事实或本仓库架构事实，不做夸大。

| 对比维度 | **DistributedFormer · CubeGPT** | 通用 Transformer / GPT 系 | snnTorch / Norse（PyTorch SNN 库） | Lava（Intel 神经形态） |
|---|---|---|---|---|
| 定位 | 事件驱动脉冲**智能体框架**（内置模型 + 插件标准 + 安全监控） | 通用稠密大模型（GPT-2 / GPT-4o / Llama 等） | 深度学习 **SNN 研究库** | 神经形态**软硬件协同框架** |
| 基本单元 | 每个脉冲神经元恰好 **16 个标量参数** | token 稠密注意力块，参数量巨大 | LIF 等脉冲神经元 + 反向传播 | 片上可编程神经形态单元 |
| 事件驱动异步计算 | ✅ 原生 | ❌ 稠密逐 token 前向 | 🔶 帧/批驱动为主（可表示事件） | ✅ 原生（Loihi 2） |
| 持久工作记忆 | ✅ KV 堆工作记忆 | ❌ 固定上下文窗口 | — | 🔶 片上持续学习 |
| 内嵌模型规模 | 内置 **CubeGPT ~281K 参数** | GPT 系百万到千亿级 | 无内置（自行搭建网络） | 无内置（自行搭建） |
| 默认运行环境 | 纯 **numpy**，CPU 可跑，零外部依赖 | 需 **GPU / 云端** | 依赖 **PyTorch + GPU** | 依赖 PyTorch + 专用 Loihi 硬件 |
| 插件热加载 | ✅ CuteMamen 固定内核 + N 专家插件 | ❌ | — | 🔶 lava-dl 扩展层 |
| 内置安全监控（意图‑动作解耦） | ✅ 独立运行时监控面 | ❌ 黑盒 | — | — |
| 持续在线学习 | ✅ 真实时序流在线学习（终期零漂移 97.2%） | 🔶 受上下文/重算限制 | 🔶 离线梯度为主 | ✅ 片上持续学习 |

> 一句话：**Transformer 用算力换通用，SNN 框架给你搭建积木；DistributedFormer 自带一个
> 281K、CPU 可跑、事件驱动、带工作记忆与监控的内嵌模型，开箱即是一个在线智能体。**

### 实测基准对比（纯真实数据 · 5 种子 × 5 折分层 CV）

> **数据不含糊原则**：下表只放本仓库实测或可核验真实数据。四个真实语料/任务目前
> **没有同行采用完全相同的协议公开可比数字**，因此"同行对标"列如实标注，**不填入任何
> 非实测精度**。

| 任务（真实材料） | CubeGPT 实测 | 随机基线 | 同行 Transformer/脉冲框架对标 |
|---|---|---|---|
| Rust 编译错误族（502 段真实代码 · 5 类） | **76.7%** ± 3.7% | 20% | 同协议同行实测未公开（—） |
| Markdown 内容类型（仓库真实文档 · 5 类） | **58.3%** ± 2.8% | 20% | 同协议同行实测未公开（—） |
| 视频运动识别（真实运镜曲线 · 10 类） | **61.7%** ± 9.8% | 10% | 同协议同行实测未公开（—） |
| 真实时序异常检测（NAB 运维指标 · 窗口级 2 类） | ACC **48.5%** ± 11.0% / 检出率 **48.9%** ± 14.8% | 50% / 0% | 与本仓库"窗口级二元"协议不等价，不并列（—） |

> 同行 Transformer 与 snnTorch / Norse / Lava 在**相同真实语料、前后端一致的协议**下的公开
> 实测，正是本项目后续可补充的"硬对比"；欢迎以**可核验真实数据**的形式补充同行基准（PR）。

## 这是什么与快速开始

DistributedFormer 探索一条不同于 Transformer 的路线：**用超简单的神经元（每个恰好 16 个标量参数）+
事件驱动的异步脉冲传播**，构建可以 7×24 持续在线、按事件触发计算的智能体网络。它的目标场景是
"永远在线的流式监控"——市场异动检测、指标巡检、IoT 阈值告警——这类任务不需要大模型的重算力，
需要的是低延迟、事件驱动和持久的工作记忆。

框架内嵌模型为 **CubeGPT**（立方体棱连接：numeric / text / timeseries / image 四模态面 + 顶层输出头，
规模约 **281K 参数**，纯 numpy 可运行）；插件标准为 **CuteMamen**（固定内核 + N 个专家思考插件，
随用随载热加载）。

> **快速开始**：安装 / 端到端演示 / 库用法 / 命令行见
> [docs/QUICKSTART.md](docs/QUICKSTART.md)。

> **技术细节**（16 参数单元动力学、分形递归、KV 堆、CubeGPT/CubeGPTKernel 结构、
> CuteMamen 模态面存档与插件标准等）见
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 与 [docs/CUTEMAMEN.md](docs/CUTEMAMEN.md)。

## 外部动作

### 安全架构宣言 · Security Architecture Manifesto

> **Traditional Transformers are black boxes where intent and action are coupled,
> making "AI escape" hard to detect. DistributedFormer changes the game by
> decoupling the CubeGPT core (thinking) from the Rust execution layer. This
> architecture natively supports a Runtime Security Monitor, allowing us to
> audit intent before execution.**
>
> **传统 Transformer 是黑盒：意图与行动耦合，使"AI 逃逸"难以检测。
> DistributedFormer 通过解耦 CubeGPT 内核（思考）与 Rust 执行层（行动）改变了
> 这一局面——这种架构原生支持运行时安全监控，使我们能够在执行之前审计意图。**

思考与执行解耦是架构层面带来的安全红利：意图在进入执行层之前必须经过独立的
监控面（意图探针 / 熔断 / 行为指纹 / 全链路审计）。四个安全方向均已落地
（v0.9.2），实现细节见 [docs/HISTORY.md](docs/HISTORY.md)。

### 生态合作 · 与 OmniSpace 正式开展合作

DistributedFormer 的脉冲内核（**CubeGPT** + **CuteMamen 插件标准**）已被
**OmniSpace** 正式采用，作为其本地优先全栈 AI 漫剧创作工作站的**核心推理层**。
[OmniSpace](https://github.com/Yzw202011/OmniSpace) 是一个"本地优先"的全模态
AI 创作平台（对话 × 漫画 × 漫剧 × 写作 × 知识学习），默认所有推理在本地 GPU
完成；DistributedFormer 内核为它提供事件驱动的脉冲主模型与插件化调度底座。

两边通过 **CuteMamen 插件标准**与**分类式 token、端到端可学习能力**达成协作。
内核保持零外部依赖（纯 numpy），宿主侧只消费标准 `.CuteMamen` 插件包与事件主题，
实现真正的"轻量级固定内核 + N 个独立专家插件"架构。

> **数据不含糊**：合作交付的能力——分类式 token 编码、端到端可学习、多任务
> 真实基准——全部在真实语料上验证（Rust 编译错误族 + 仓库 Markdown 文档，
> 均为真实数据）。

## 最新进展

### 生态协同 v0.14.5 · 思考怎么"打配合"

CuteMamen 插件体系按职责分三类，靠**事件总线 + 请求/应答**协同成一套生态：

| 角色 | 职责 | 通信方式 | 代表插件 |
|---|---|---|---|
| **核心模型思考** | CubeGPT 内核必要思考：路由 / KV 堆工作记忆 / 顶层输出头 / 节律 | 被路由激活即思考 | `CuteMamenKernel` · `CubeGPTKernel` |
| **工作型插件** | 完成具体任务/思考（分类、检索、生成） | `ctx.ask(topic, data)` 请求/应答 · `emit` 广播 | RustCoding · JavaCoding · VideoMaking · **TagSearching** |
| **服务型插件** | 提供能力服务（外挂 LLM、对话、推理） | 被 `ask` 请求 / 被 `think` 路由 | **LLMProvider · Chat** |

**插件互通信（新增 v0.14.5）**：`PluginContext.ask()` 让一个插件在 `on_think`
内**同步请求**另一插件并直接取回结果（插件↔插件双向）；`run_gpt({模态:数据})`
让插件**驱动 CubeGPT 全模型**（插件↔核心模型）；`emit()` 维持异步广播。三者让
插件生态"有来有回"地打配合。

**（新 · 工作型插件）TagSearching · 代码引用检索**：给定一个符号（函数/类/变量），
纯本地确定性返回**定义点 + 全部引用点**（文件 / 行号 / 上下文片段），词边界精确
匹配，区分"定义"与"调用"，并排除 `==` 等比较误报。交付
`plugin/TagSearching.CuteMamen`（route `tagsearch`）。

**（新 · 服务型插件）Chat · 类人对话**：优先经 `ctx.ask("llm")` 调外挂 OpenAI
兼容 LLM 产出自然回复；未配置后端时回退本地**真人感引擎**（问候 / 情绪共情 /
问句反射 / 接话+反问，口语化、有来有回）。交付 `plugin/Chat.CuteMamen`
（route `chat`）。

### 时序实验记录（R5 · R6）

> 历史实验记录（v0.0–v0.13.1）与负结果已归档至 [docs/HISTORY.md](docs/HISTORY.md)。

**（实验 R5）真实时序异常检测基准**，5 种子 × 5 折分层 CV，窗口级二元检测
（NAB 真实运维指标，2720 个 40 点滑窗，官方告警标注）：

| 指标 | 数值 |
|------|------|
| 窗口级 ACC | **48.5%** ± 11.0%（9/25 折超随机 50%）|
| 异常检出率 | **48.9%** ± 14.8%（多数类全判正常 = 0%）|
| 误报率 | 51.5% |
| 最佳配置 | L2=1e-1、异常权重 10（ACC 与检出率的平衡点）|

异常窗口在聚合统计描述符下可分性有限（见
[anomaly_report.md](src/experiments/anomaly_report.md)）。

**（实验 R6）时序流在线持续学习**，时间顺序前后划分（train 951 / val 817，
4 epoch），三配置均**稳定**（drift ≥ −10pct 判定）：默认 **cosine** 后期
**零漂移**（final=best=97.2%），constant 微小回落（−1.3%），freeze@2 −2.7%——
在线持久输出头在真实时序流上的后期漂移已消除。见
[anomaly_stream_stability.md](src/experiments/anomaly_stream_stability.md)。

**训练材料（真实数据 · 思考插件）**，全部纯真实、无合成，作为端到端训练与
插件知识迁移的材料（准确率为各插件擅长领域的实测结果，5 种子 × 5 折）。

**① 工作型插件**（真实数据训练 · 冻结水库之外带可学习读出头）

> 「额外参数量」= 插件在冻结水库(**CubeGPT ~281K**)之外随包存档的可学习**读出头权重**
> （线性 softmax 读出层 `W` + 归一化 `mean/scale`，真实特征维逐插件计算）；
> 「插件精度」= 参数数值位宽（实测 `float64`）。全部纯真实、无合成，5 种子 × 5 折。

**①·a 插件个人信息**

| 插件名字 | 擅长领域 | 真实数据来源 | 规模与标签 | 额外参数量 | 插件精度 |
|---|---|---|---|---|---|
| RustCoding | Rust 编译错误分类 | 真实风格 Rust 代码 + rustc 错误码 | 502段*5类 | 4,149 | float64 |
| JavaCoding | Java 祖传代码静态问题 | 真实风格 Java 代码 + javac 诊断 | 294段*8类 | 6,019 | float64 |
| VideoMaking | 视频镜头运动识别 | 真实运镜/分镜曲线 | 200样本 | 3,658 | float64 |
| OpMetrics | 时序异常检测（窗口级）| NAB 真实运维指标 | 2720滑窗*2类 | 2,370 | float64 |

**①·b 插件测试结果**

| 插件名字 | 准确率 | 准确率评估 |
|---|---|---|
| RustCodingPlugin | **76.7%** ± 3.7% | 5 种子 × 5 折，种子均值 75.1–79.1%，超随机 20%（25/25 折）|
| JavaCodingPlugin | **58.2%** | 5 种子 × 5 折，超随机 12.5%（约 4.7 倍），vs 原型基线 56.5%；交付 `plugin/JavaCoding.CuteMamen`（迁移知识）|
| VideoMakingPlugin | **61.7%** ± 9.8% | 5 种子 × 5 折，种子均值 56.0–67.0%，超随机 10%（25/25 折）|
| OperationalMetricsDataset | ACC **48.5%** ± 11.0% / 检出率 **48.9%** ± 14.8% | 5 种子 × 5 折（异常类梯度=10），超随机 50%（9/25 折，多数类 97.4% 下如实偏低）|

> 各插件完整标签集 / 数据来源与规模 → [rust_coding.py](src/data/rust_coding.py) · [java_coding.py](src/data/java_coding.py) · [video_motion.py](src/data/video_motion.py) · [metrics_time_series.py](src/data/metrics_time_series.py)；评测协议（5 种子 × 5 折）见 [CUTEMAMEN.md](docs/CUTEMAMEN.md)。

<br />

**② 服务型插件**（无训练读出头 · 对外提供服务）

> 服务型插件**不训练、无读出头**，故 ① 的「真实数据来源 / 规模与标签 / 额外参数量 / 插件精度」不适用（记 —）；
> 「准确率」列填**对外服务能力**，「准确率评估」列填验证方式。两类插件共用同一套表格头。

**②·a 插件个人信息**

| 插件名字 | 擅长领域 | 真实数据来源 | 规模与标签 | 额外参数量 | 插件精度 |
|---|---|---|---|---|---|
| Chat | 类人对话 | — | — | — | — |
| LLMProvider | 外挂 OpenAI 兼容 LLM | — | — | — | — |

**②·b 插件测试结果**

| 插件名字 | 准确率 | 准确率评估 |
|---|---|---|
| Chat | 本地真人感引擎（问候 / 情绪 / 问句反射）| 无 LLM 后端回退，口语化回复、有来有回，确定性可复现 |
| LLMProvider | 插件互通信（`ctx.ask("llm")`）| 经 `ask` 被 Chat 召唤返回 `content`，服务型插件相互协同 |

> 服务型插件不携带训练读出头，能力以"对外服务 + 生态协同"衡量；实现见
> [chat.py](src/cutemamen/chat.py) · [llm_provider.py](src/cutemamen/llm_provider.py)，
> 插件用法见 [docs/CUTEMAMEN.md](docs/CUTEMAMEN.md)。

## 路线图

历史里程碑（v0.2.0–v0.14.5）已全部完成并归档至 [docs/HISTORY.md](docs/HISTORY.md)。
当前无未完成的版本化里程碑；进行中的线程：**扩展更多真实数据集与真实语料、持续丰富
工作型 / 服务型插件生态**（无版本承诺），以持续支撑多任务真实基准与在线持续学习验证。

## 文档

* [架构说明](docs/ARCHITECTURE.md)
* [快速开始](docs/QUICKSTART.md)
* [插件标准与模态面存档](docs/CUTEMAMEN.md)
* [实验记录与版本历史](docs/HISTORY.md)
* [更新日志](docs/CHANGELOG.md)
* [兼容性规范](docs/COMPATIBILITY.md)
* [参与贡献](docs/CONTRIBUTING.md)
* [发行说明](docs/RELEASE_NOTES.md)
* [彩蛋（原「菜单·植物大战 VS Code」）](#彩蛋)
* [开源许可证 MIT](LICENSE)

### 项目结构

#### **src 内目录**（Python 包，扁平化）：

| 目录 | 说明 |
|---|---|
| `core` | 脉冲单元 / 分形层 / KV 堆 / CubeGPT / 模态面 pkg |
| `cutemamen` | CuteMamen 插件标准：内核 / 插件 / 事件总线 / 包格式 / LoRA 桥接 / 迁移工具 / Rust·Java coding 插件 |
| `codec` | 数值·文本·时序 → 脉冲编码；脉冲 → 动作解码 |
| `agents` | 5 类脉冲智能体 |
| `workflow` | 工作流引擎 + 消息路由 |
| `data` | 真实数据集：Rust·Java 编码基准 / Markdown / 视频运镜 / 时序异常（v0.14.0） |
| `training` | 监督/STDP 训练器 + 读出层验证协议 |
| `deployment` | Docker / K8s / Redis / Prometheus / RedisKVStack / openai_server / web_ui |
| `security_monitor` | 运行时安全监控（意图探针 / 熔断 / 行为指纹 / 审计溯源） |
| `demos` | 股票监控端到端演示 / CubeGPT 终端聊天 |
| `tests` | pytest 测试 |
| `experiments` | 实验脚本、结果与报告 |

> 入口 / 自检文件：`cli.py`（dformer 命令行入口）· `selfcheck.py`（模块自检套件）· `__init__.py` · `__main__.py`。

#### **src 外目录**：

| 目录 | 说明 |
|---|---|
| `plugin` | .CuteMamen 工作型 / 服务型插件独立交付目录 |
| `cache` | 统一运行时缓存（插件/面存档 + pytest，git 忽略） |
| `docs` | 架构 / 插件标准 / 实验历史 / 更新日志 / 兼容性 / 贡献指南 / 发行说明 / 彩蛋 |
| `.github` | GitHub 工作流（CI） |
