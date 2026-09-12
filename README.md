<div align="center">

# DistributedFormer

**事件驱动的脉冲神经网络智能体框架** · 内嵌模型 **CubeGPT**

以 16 参数脉冲神经元为基本单元 · CubeGPT 立方体连接多模态脉冲大模型 · KV 堆工作记忆 · 多智能体脉冲工作流
面向流式监控、异常检测等持续在线场景

[!\[CI](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/1person280/DistributedFormer/actions/workflows/ci.yml)
[!\[PyPI - Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[!\[License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[!\[Version](https://img.shields.io/badge/version-0.7.0-orange)](CHANGELOG.md)

</div>

\---

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

# 合成数据训练
dformer train --depth 1 --epochs 10

# 模块自检
dformer test
```

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
│   ├── core/                    # 脉冲单元 / 分形层 / KV 堆 / 完整网络
│   ├── codec/                   # 数值·文本·时序 → 脉冲编码; 脉冲 → 动作解码
│   ├── agents/                  # 5 类脉冲智能体
│   ├── workflow/                # 工作流引擎 + 消息路由
│   ├── training/                # 合成数据生成 + 监督/STDP 训练器
│   ├── deployment/              # Docker / K8s / Redis / Prometheus
│   ├── demos/                   # 股票监控端到端演示 (模拟器 + yfinance 真实行情)
│   ├── cli.py                   # dformer 命令行入口
│   └── selfcheck.py             # 模块自检套件
├── tests/                       # pytest 测试
├── experiments/                 # 消融实验脚本、结果与报告 (E1-E5)
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
* **结果**: CubeGPT 读出层 5 种子验证准确率 **54.4% ± 5.4%**, 全部超过
随机基线 20%, 见 [`experiments/rust\\\_report.md`](experiments/rust_report.md)
* 复现: `python experiments/rust\\\_benchmark.py`

## 路线图

- [x] v0.2.0 — 产品化重构：可安装包 / CLI / serve 长驻服务 / 真实行情数据源 / 测试与 CI / 修复 Docker 链路
- [x] v0.3.0 — 多模态顶层模块化：四种模态独立输入模块 + 独立输出模块 + 确定性图像编码
- [x] v0.4.0 — 内嵌模型正式命名为 **CubeGPT**（立方体棱连接，4 面 × 4,400 单元 ≈ 281K 参数），智能体框架全面切换
- [x] v0.5.0 — 三项核心修复：KV 注意力接入主计算路径 / 训练方法学验证
  （读出层 64.8% vs 随机 25%，实验 R1）/ 真实桌面通知与 HTTP 动作
- [x] v0.6.0 — 真实数据基准起步：Rust coding（100 段真实代码 × 5 类真实 rustc 错误），
  读出层 54.4% vs 随机 20%；同步修复文本编码器的进程随机哈希（不可复现）与丢 token 问题
- [x] v0.7.0 - 模态面独立化与 **pkg存档**：随用随载热加载 / 自由导入导出具体模态面
  （`.dfpkg`，兼容 CuteMamen 包格式）；同步修复训练报告占位符问题
- [ ] 简单的聊天框，实现向CubeGPT进行对话
- [ ] **CuteMamen 插件标准落地**：按 CuteMamen 规范 v0.1.0 实现通用固定内核（路由器 + 工作记忆
  + 插件注册表）与 `.CuteMamen` 专家插件包（on_load/on_think/on_unload 生命周期钩子、
  三级记忆存档、事件总线通信、内存预算淘汰、LoRA/Adapter 兼容桥接）——v0.7.0 的模态面 pkg 是其首个特例
- [ ] CubeGPT 端到端可学习：在真实数据集上端到端训练（Rust 基准已提供数据通路）
- [ ] KV 堆注意力检索向量化（当前 python 循环打分，大堆场景有 scan_limit 限额）
- [ ] 真实数据集基准（替代纯合成数据），建立有意义的评估基线
- [ ] Redis 分布式 KV 堆在多节点工作流中实际启用
- [ ] 学习规则改进：目标是在 ≥2 个真实任务上显著超过随机基线

## 已知问题（v0.5.0 状态）

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
基准（真实代码 + 真实 rustc 错误类别），读出层 54.4% vs 随机 20%；扩展更多
真实数据集与真实代码语料仍在路线图中。
* ~~训练报告只有占位符~~ **已修复（v0.7.0）**：`report_generate` 动作现在渲染
真实统计数据（智能体状态 / CubeGPT 网络统计 / KV 堆记忆 / STDP 学习统计），
不再产生 `[自动生成内容占位]`。

## 文档

* [架构说明](docs/ARCHITECTURE.md)
* [更新日志](CHANGELOG.md)
* [参与贡献](CONTRIBUTING.md)

## License

[MIT](LICENSE)

