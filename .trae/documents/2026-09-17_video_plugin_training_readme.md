# 训练视频生成插件 + 翻新 README (v0.12.0)

## Context

项目当前为 v0.11.0。用户希望：
1. **训练视频生成插件** —— 给 `VideoMakingPlugin` 补训练数据接口 + 跑一个真实基准（已确认：加 `training_data()` 接口 + 基准脚本），并把结果写进 README。
2. **翻新 README**，三点需求：
   - 实验记录准确率表**区分思考插件**（已确认：**拆成两张表**：主模型直评 / 思考插件推理）。
   - **训练材料章节的 Rust 两节合二为一**（`Rust coding 真实基准` + `RustCoding 思考插件` 合并）。
   - **新增"视频生成训练"**小节。
3. **版本升级到 v0.12.0**（已确认：4 处同步 + CHANGELOG）。

硬约束（项目记忆）：纯真实数据、无合成数据生成路径；版本号 4 处同步。

## 视频生成训练的设计（真实数据、可复用现有范式）

`VideoMakingPlugin` 的可学习/可存档权重 = `motion_profiles`（真实运镜曲线，[video_making.py](file:///c:/Users/yxhcf/Desktop/DistributedFormer/src/cutemamen/video_making.py) 的 `DEFAULT_MOTION_PROFILES`，10 种真实镜头运动）。视频生成本身不是分类任务，因此"训练"落地为：

**真实运动类型识别基准**——以插件自带的 10 条**真实运镜曲线**为训练材料，把每条真实曲线按其真实插值进度稠密采样为描述符样本 `[dx,dy,zoom,rot]`，标签 = 真实运动类别（static/pan_left/pan_right/tilt_up/tilt_down/zoom_in/zoom_out/dolly_in/orbit_right/orbit_left）。沿用现有"冻结水库特征 + 线性读出层 + 5 种子 × 5 折分层 CV"协议，评估插件是否从真实运镜数据学到可分信号（vs 随机基线 10%）。数值如实上报，不追求虚高。

这与 `MarkdownTextDataset`（第二真实任务）的数据通路、`stratified_kfold` 复用方式完全对齐。

## 改动清单

### 1. 给视频插件补训练数据接口
文件：`src/cutemamen/video_making.py`
- 新增 `VideoMakingPlugin.training_data()` → 返回 `(X, y)`：X = 10 条真实运镜曲线稠密采样的描述符特征矩阵，y = 运动类别索引。镜像 `RustCodingPlugin.training_data()`。
- 新增辅助 `motion_descriptors()`（导出真实曲线参数表）。

### 2. 新增真实视频数据集
新文件：`src/data/video_motion.py`
- `VideoMotionDataset`（镜像 `MarkdownTextDataset`）：
  - `LABELS` = 10 种真实运动；`LABEL_NAMES` 中文名。
  - 从 `DEFAULT_MOTION_PROFILES` 提取真实曲线，按真实插值进度稠密采样为 `VideoSample`（含 `sample_id / category / input_signal / target_pattern / static_signal / multimodal_input()`）。
  - `RANDOM_BASELINE = 1/10 = 10%`；`majority_baseline()`。
  - `kfold_datasets(n_folds=5, seed)` 复用 `src.data.rust_coding.stratified_kfold`。
- 文件头部架构注释（符合项目约定）。

### 3. 新增视频基准脚本 + 报告
新文件：`src/experiments/video_motion_benchmark.py`（镜像 `multi_task_benchmark.py`）
- `CubeFeatureExtractor` 冻结水库特征 + `LinearReadout`（L2 网格选优）做 5 种子 × 5 折分层 CV。
- 写 `video_motion_results.json` + `video_motion_report.md`（表格格式仿 `rust_report.md`）。
- 在报告中如实说明：真实运镜集规模小，本基准验证"训练材料数据通路 + 真实运镜信号可学"，数值如实上报。

### 4. 新增测试
新文件：`src/tests/test_video_training.py`
- 测 `VideoMakingPlugin.training_data()` 形状/类别、`VideoMotionDataset.kfold_datasets` 折数、读出层在视频任务上超随机基线（仿 `test_token_training_beats_random_on_md`）。
- 在 `src/tests/__init__.py` 或现有测试收集处登记（若项目按目录收集则无需）。

### 5. 翻新 README.md
- **实验记录准确率表拆成两张**：
  - 表 1「主模型直评」：v0.7.5 → v0.11.0（v0.8.6 之前主模型读出层 + v0.11.0 多任务主模型）。
  - 表 2「思考插件推理」：v0.8.6（RustCodingPlugin 知识迁移 76.7%）、v0.12.0（VideoMakingPlugin 运动识别结果）。
- **训练材料章节**：把 `### Rust coding 真实基准 (v0.6.0)` 与 `### RustCoding 思考插件 (v0.7.3)` **合并为一节** `### Rust coding 真实基准 + 思考插件`；**新增 `### 视频生成训练 (v0.12.0)`** 小节（数据集 / training_data() / 基准结果 / 复现命令 `python src/experiments/video_motion_benchmark.py`）。
- 路线图新增 `[x] v0.12.0` 条目；版本徽章 `0.11.0 → 0.12.0`；顶部"这是什么"若提及插件数可顺手同步（视实际内容）。

### 6. 版本 4 处同步 → 0.12.0
- `pyproject.toml`（L5-8 附近 version）
- `src/__init__.py`（L119-120 `__version__`）
- `src/cutemamen/pkg.py`（L42-43 `CORE_VERSION`）
- `src/cutemamen/plugin.py`（L276-290 `min_core_version`）

### 7. 更新 CHANGELOG
文件：`docs/CHANGELOG.md`
- 新增 v0.12.0 条目（仿 v0.11.0 格式：标题/日期/变更摘要/测试数/版本同步路径），内容：训练材料 Rust 合二为一、新增视频生成训练、实验记录区分思考插件、版本同步 0.11.0 → 0.12.0。

## 验证

1. 运行视频基准：`python src/experiments/video_motion_benchmark.py` → 确认生成 `video_motion_results.json` / `video_motion_report.md`，结果显著超随机 10%。
2. 跑测试：`python -m pytest src/tests/ -q`（预期现 231 项 + 新增通过；若用 `dformer test` 亦可）。
3. 版本一致性：`python -c "import src; from src.cutemamen.pkg import CORE_VERSION; print(src.__version__, CORE_VERSION)"` → 全 0.12.0。
4. README 渲染检查：准确率两表、Rust 训练材料合并、视频生成训练小节、路线图/徽章一致。
