# CubeGPT 路线落地: 64 比特分类式 token + 端到端可学习 + 学习规则改进

## Context(为什么做)

用户给出三条技术路线,并要求做成(纯真实数据约束下):

1. **分类式 token** — 一个 token 占 64 比特。纯文本场景:前 32 比特为 token 组,后 32 比特直接为 utf8-mb4 字符。设硬性分组,如 `0x00000000xxxxxxxx` 保留为 utf8-mb4 字符 token 组。
2. **CubeGPT 端到端可学习** — 在真实数据集上端到端训练。Rust 基准已提供数据通路,`RustCodingPlugin` 导出 X/y。
3. **学习规则改进** — 目标:在 ≥2 个真实任务上显著超过随机基线。

已和用户确认两点关键取舍:
- **第二真实任务** = 新增一份真实数据集(纯真实、无合成)。
- **token 集成深度** = 实现 codec 并**完整接入现有训练**(DFTrainer 文本通路)。

现状梳理:
- 现有编码是 `SpikeEncoder.encode_text`(TF-IDF 词袋哈希),无 64 比特 token 概念。
- 训练已有两条路径:`readout.py`(冻结水库 + 线性读出,v0.10.1 5 折 CV 约 75%)与 `trainer.py` DFTrainer(端到端,persistent head,v0.10.2 cosine/freeze 已消除后期漂移)。真实数据仅 Rust 编码 502 段 5 类。
- 版本需 4 处同步: `pyproject.toml#7`、`src/__init__.py#119`、`src/cutemamen/pkg.py#43`(CORE_VERSION)、`src/cutemamen/plugin.py#287`(min_core_version="0.10.2")。

---

## 变更概览

| 路线 | 落地产物 |
|------|----------|
| A. 分类式 token | 新模块 `src/codec/class_token.py` + 单测 |
| B. 第二真实数据集 | 新模块 `src/data/md_text.py`(真实多字节 markdown 语料)→ 文本内容类型分类 |
| C. 端到端 + 学习规则 | DFTrainer 增加可训练 class-token 嵌入层(端到端);新增多任务基准,两份真实任务均显著超随机 |

---

## A. 64 比特分类式 token 编解码(`src/codec/class_token.py`)

遵循项目扁平结构 + 模块头注释约定,新增:

- **常量 / 分组(硬性)**:
  - `TOKEN_BITS=64`、`GROUP_MASK=0xFFFFFFFF00000000`、`LOW32_MASK=0x00000000FFFFFFFF`
  - `CHAR_GROUP=0x00000000` — 保留组:utf8-mb4 字符 token,低 32 位 = Unicode 码点
  - `RESERVED_GROUPS` dict:其余分组说明(如 `0x00000001` 子词/词 token、`0x0000007E` 控制端,文档化保留,不在本版实现具体语义)
- **建/拆函数**: `make_token(group, low) -> int`、`group_of(tok) -> int`、`low_of(tok) -> int`、`is_char_token(tok) -> bool`
- **utf8-mb4 字符 token**: `char_token(cp) -> int`(校验 `0 <= cp <= 0x10FFFF`,Python `str` 迭代即码点,天然支持 4 字节 astral/emoji)、`char_group_token(ch: str) -> int`
- **文本 ⇄ token 序列**: `encode(text) -> List[int]`(每个字符 → `char_token(ord(ch))`)、`decode(tokens) -> str`(仅接受 CHAR_GROUP token,其余组抛明确异常)、`tokenize(text) -> List[int]` 别名
- **token → 脉冲信号桥(接入网络关键)**: `embed_tokens(tokens, dim) -> np.ndarray` — 对每个 64 比特 token 做确定性折叠(zlib.crc32 于 token 字节 + 组位混合)→ 映射到 dim 维度累加并 L2 归一化。使 CubeGPT 文本通路可消费 class-token,且逐比特确定(与现有 encode_text 确定性一致)。
- `src/codec/__init__.py` 导出新符号。

`src/tests/test_class_token.py`(新增):
- 往返: ASCII / 中文(utf8 3 字节)/ emoji(utf8 4 字节= utf8-mb4)encode→decode 一致
- 组校验: 字符 token 组恒为 `CHAR_GROUP`;低 32 位恒为该字符码点
- `make_token`/`group_of`/`low_of` 互逆
- `embed_tokens`: 逐比特确定(两次结果一致)、维度非零、归一化

---

## B. 第二真实数据集(真实多字节 markdown 文本)(`src/data/md_text.py`)

满足「纯真实」+「utf8-mb4 多字节」:语料 = 仓库内真实已撰写的中文/英文 `*.md` 文档(`docs/*.md`、`README.md`、`CHANGELOG.md`),**非生成、源自项目真实文字**。对每行真实文本标注其**真实内容类型**(由行首真实语法判定):
- `heading`(标题), `code`(代码块), `list`(列表), `table`(表格行), `paragraph`(正文段落) → 5 类真实标注

- 模块提供 `load_md_text()`,复用 `rust_coding.py` 的 `stratified_split` / `stratified_kfold` 做分层划分/K 折(保持评估约定 5 种子 × 5 折)。
- 类 `MarkdownTextDataset(dim=16)`,对齐 `RustCodingTrainingDataset` 接口:
  - `input_signal` = `char_token` 序列经 `embed_tokens` 的脉冲向量;**`token_seq` 元数据**保留 class-token 序列(供端到端嵌入层消费)
  - `target_pattern` = 5 类 one-hot 监督模式;`static_signal` 可选(行特征,可缺省)
  - `generate_dataset(train_ratio, seed)`、`kfold_datasets(n_folds, seed)`、`RANDOM_BASELINE=1/5`、`majority_baseline()`
- 语料规模守护:若真实文档过少导致过拟合,沿用 `RustCodingTrainingDataset` 已知经验(K 折 + L2),报告中如实给出 seed 区间。

`src/data/__init__.py` 更新说明。新增 `src/tests/test_md_dataset.py`:数据真实、分布均衡、与 Rust 数据集接口一致。

---

## C. CubeGPT 端到端可学习 + 学习规则改进

**核心:让网络与编码一同端到端可学习(不止冻结水库 + 线性读出)。**

1. **可训练 class-token 嵌入层(新增到 `src/training/trainer.py`)**
   - `EmbeddingHead`(暂并入 DFTrainer):维度 = `token_low32 哈希 → n_features` 的可训练线性嵌入,权重随 persistent head 梯度更新 —— 这是「端到端学习信号回传至输入编码」的规则改进。
   - 保留 v0.10.2 的 cosine LR 调度 + `freeze_reservoir_epoch` 稳定机制(记忆经验:非平稳水库导致后期漂移)。
   - `_forward_features` 支持 token 输入:文本样本取 `metadata["token_seq"]`,经嵌入层得特征参与分类。

2. **Rust 数据通路接通 token**(`src/data/real_dataset.py`)
   - `TrainingSample` 增加 `token_seq: Optional[List[int]] = None`(由 `metadata["code"]` 经 `class_token.encode` 得到)。
   - `multimodal_input()` 在启用 token 时以 `{"numeric": static, "text": token_seq}` 输入,`SpikeEncoder` 侧由 token 桥代替纯 TF-IDF(向后兼容:未启用 token 时保持原行为)。

3. **学习规则改进 + 多任务基准验证**(`src/experiments/multi_task_benchmark.py`)
   - 目标:在 **Rust(5 类)** 与 **Markdown(5 类)** **两份真实任务**上,均显著超过随机基线(> 随机 +5pct),并对比多数类基线。
   - 复用并扩展 `readout.run_cross_validation` 协议:5 种子 × 5 折分层 CV + L2 网格选优;为 token 特征新增一条抽取通道。
   - 输出 `multi_task_results.json` + `multi_task_report.md`(含各任务 seed min/max 区间、超基线折数、train/val gap)。

4. **CLI 接线**(`src/cli.py` + `src/__main__.py`)
   - `train`: 增加 `--dataset {rust,md}`(默认 rust);MD 走 `MarkdownTextDataset`。
   - 新增 `benchmark-multi`: 跑上面多任务基准。

5. **版本同步**: pyproject / `__version__` / `CORE_VERSION` / `min_core_version` 统一到 `0.11.0`。

---

## 关键复用(避免重复造轮子)

- `src/codec/spike_codec.SpikeEncoder` — 既有 TF-IDF 文本编码(回退路径)
- `src/data/rust_coding.{LABELS, static_metrics, structure_metrics, stratified_split, stratified_kfold}` — 特征与划分/折工具
- `src/data/real_dataset.RustCodingTrainingDataset / TrainingSample` — Rust 样本与数据集模板
- `src/training/readout.{CubeFeatureExtractor, LinearReadout, run_cross_validation}` — 读出层与 CV 协议
- `src/training/trainer.DFTrainer` — 端到端训练骨架(LR 调度 / 冻结 / persistent head)
- 现有的 `CuteMamen` 版本同步 4 点

## 验证方式

1. `python -c "from src.codec.class_token import encode, decode; ..."` 往返一致(含中文/emoji)
2. `python -m pytest src/tests/test_class_token.py src/tests/test_md_dataset.py -q` 新单测全绿
3. `python -m pytest src/tests -q` 全量回归(当前 209 测试应保持全绿)
4. `python src/experiments/multi_task_benchmark.py` → 检查 `multi_task_results.json`:Rust 与 MD 两任务 val_acc 均值均 > 随机+5pct,seed min/max 区间如实记录
5. `dformer train --dataset md --epochs 5` 与 `dformer train --dataset rust` 冒烟跑通;`dformer --version` 显示 0.11.0
6. 四个版本同步点 grep 一致(0.11.0)

## 待确认 / 风险

- **MD 语料规模**:README+docs 约几百行真实文本,接近 Rust 502 的规模量级,但仍可能有限。若某类行过少,报告中如实说明并使用 K 折评估(不做任何合成补样,严守项目硬约束)。
- **token 嵌入是"端到端回溯到输入编码"的最小改动**;不新增独立 next-token LM 头(用户未要求,避免超范围)。