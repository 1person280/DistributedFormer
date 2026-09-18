<div align="center">

# DistributedFormer

**事件驱动的脉冲神经网络智能体框架** · 内嵌模型 **CubeGPT** · 插件标准 **CuteMamen**

以 16 参数脉冲神经元为基本单元 · CubeGPT 立方体连接多模态脉冲大模型 · KV 堆工作记忆 · 多智能体脉冲工作流
· CuteMamen 固定内核 + 专家思考插件
面向流式监控、异常检测等持续在线场景

[![CI](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml)
[![PyPI - Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.13.1-orange)](docs/CHANGELOG.md)
[![Audit-Ready Architecture](https://img.shields.io/badge/security-Audit--Ready%20Architecture-blueviolet)](#外部动作)

</div>

---

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

## 最新进展（实验记录 · v0.13.1）

> 最新版本为 **v0.13.1**（真实时序异常检测）。全部历史实验记录（v0.0–v0.12.0）
> 与负结果已归档至 [docs/HISTORY.md](docs/HISTORY.md)。

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
插件知识迁移的材料（准确率为各插件擅长领域的实测结果，5 种子 × 5 折）：

「额外参数量」= 插件在冻结水库(**CubeGPT ~281K**)之外随包存档的可学习**读出头权重**
（线性 softmax 读出层 `W` + 归一化 `mean/scale`，真实特征维逐插件计算）；
「插件精度」= 参数数值位宽（实测 `float64`）。全部纯真实、无合成，5 种子 × 5 折。

**① 插件个人信息**

| 插件名字 | 擅长领域 | 真实数据来源 | 规模与标签 | 额外参数量 | 插件精度 |
|---|---|---|---|---|---|
| RustCodingPlugin | Rust 编译错误分类 | 真实风格 Rust 代码 + 真实 rustc 错误码 | 502 段 × 5 类（move/borrow/lifetime/type/ok）| 4,149 | float64 |
| JavaCodingPlugin | Java 祖传代码静态问题 | 真实风格 Java 代码 + 真实 javac/静态分析诊断 | 294 段 × 8 类（null/rawtype/deprecated/generic/type/symbol/override/ok）| 6,019 | float64 |
| VideoMakingPlugin | 视频镜头运动识别 | 真实运镜 / 分镜惯例曲线 | 10 类 × 20 点 = 200 样本 | 3,658 | float64 |
| OperationalMetricsDataset | 时序异常检测（窗口级）| NAB 真实运维指标（real* 系列，官方告警标注）| 2720 滑窗 × 2 类（异常 2.6%）| 2,370 | float64 |

**② 插件测试结果**

| 插件名字 | 准确率 | 准确率评估 |
|---|---|---|
| RustCodingPlugin | **76.7%** ± 3.7% | 5 种子 × 5 折，种子均值 75.1–79.1%，超随机 20%（25/25 折）|
| JavaCodingPlugin | **58.2%** | 5 种子 × 5 折，超随机 12.5%（约 4.7 倍），vs 原型基线 56.5%；交付 `plugin/JavaCoding.CuteMamen`（迁移知识）|
| VideoMakingPlugin | **61.7%** ± 9.8% | 5 种子 × 5 折，种子均值 56.0–67.0%，超随机 10%（25/25 折）|
| OperationalMetricsDataset | ACC **48.5%** ± 11.0% / 检出率 **48.9%** ± 14.8% | 5 种子 × 5 折（异常类梯度=10），超随机 50%（9/25 折，多数类 97.4% 下如实偏低）|

数据集定义见 `src/data/`（`rust_coding.py` / `java_coding.py` / `video_motion.py` /
`metrics_time_series.py`），插件用法见 [docs/CUTEMAMEN.md](docs/CUTEMAMEN.md)。

## 路线图

历史里程碑（v0.2.0–v0.13.1）已全部完成并归档至 [docs/HISTORY.md](docs/HISTORY.md)。
当前无未完成的版本化里程碑；进行中的线程：**扩展更多真实数据集与真实语料**
（无版本承诺），以持续支撑多任务真实基准与在线持续学习验证。

## 文档

* [架构说明](docs/ARCHITECTURE.md)
* [快速开始](docs/QUICKSTART.md)
* [插件标准与模态面存档](docs/CUTEMAMEN.md)
* [实验记录与版本历史](docs/HISTORY.md)
* [更新日志](docs/CHANGELOG.md)
* [兼容性规范](docs/COMPATIBILITY.md)
* [参与贡献](docs/CONTRIBUTING.md)
* [发行说明](docs/RELEASE_NOTES.md)
* [开源许可证 MIT](LICENSE)

### 项目结构

```
DistributedFormer/
├── src/                        # Python 包 (v0.8.6 扁平化)
│   ├── core/                    # 脉冲单元 / 分形层 / KV 堆 / CubeGPT / 模态面 pkg
│   ├── cutemamen/               # CuteMamen 插件标准: 内核 / 插件 / 事件总线 / 包格式 / LoRA 桥接 / 迁移工具 / Rust·Java coding 插件
│   ├── codec/                   # 数值·文本·时序 → 脉冲编码; 脉冲 → 动作解码
│   ├── agents/                  # 5 类脉冲智能体
│   ├── workflow/                # 工作流引擎 + 消息路由
│   ├── data/                    # 真实数据集: Rust·Java 编码基准 / Markdown / 视频运镜 / 时序异常 (v0.14.0)
│   ├── training/                # 监督/STDP 训练器 + 读出层验证协议
│   ├── deployment/              # Docker / K8s / Redis / Prometheus / RedisKVStack / openai_server / web_ui
│   ├── security_monitor/        # 运行时安全监控 (意图探针 / 熔断 / 行为指纹 / 审计溯源)
│   ├── demos/                   # 股票监控端到端演示 / CubeGPT 终端聊天
│   ├── tests/                   # pytest 测试
│   ├── experiments/             # 实验脚本、结果与报告
│   ├── cli.py                   # dformer 命令行入口
│   └── selfcheck.py             # 模块自检套件
├── plugin/                      # .CuteMamen 思考插件独立交付目录
├── cache/                       # 统一运行时缓存 (插件/面存档 + pytest, git 忽略)
└── docs/                        # 架构 / 插件标准 / 实验历史 / 更新日志 / 兼容性 / 贡献指南 / 发行说明
```