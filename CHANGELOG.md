# 更新日志

## v0.4.0 (2026-09-11)

内嵌模型正式命名为 **CubeGPT** (cube = 立方体棱连接方式, GPT = 致敬 ChatGPT):

### 新增
- `CubeGPT` 模型类: 4 个模态面 (CubeFace) 构成立方体侧面, 以环形棱侧连
  (每步将本面脉冲聚合注入邻面下一拍输入), 顶层 OutputModule 生成头部
- 规模 (默认深度2): 4 面 × ~4,400 单元 × 16 参数 ≈ **281K 参数**
  (精确: 4×4,384+16 = 17,552 单元 / 280,832 参数), `calculate_cube_scale()` 可查
- CubeGPT 保持 v0.3.0 多模态 API 不变 (`step({模态: 数据})`),
  未提供输入的面仍消费侧向脉冲 (持续思考)
- 新增 tests/test_cube_gpt.py (9 项), 测试总计 30 项全绿

### 变更
- 智能体框架 (5 类智能体) 内嵌网络由 DistributedFormer 切换为 CubeGPT;
  DistributedFormer 保留为训练基底 (trainer 的思考层监督实验仍基于它)
- 推理智能体特殊化 (trend/anomaly) 改为作用于 CubeGPT 面皮层单元

## v0.3.0 (2026-09-11)

多模态顶层模块化改造 (breaking change):

### 新增
- **独立顶层输入模块**: numeric / text / timeseries / image 四种模态, 每种模态拥有自己的
  16 个脉冲单元与绑定编码器 (`InputModule`), 模态间互不干扰
- 新多模态 API: `df.step({"numeric": 1.5, "text": "...", "timeseries": [...], "image": 2D数组})`
  直接接受原始数据, 编码下沉到各输入模块内部
- **独立顶层输出模块** (`OutputModule`): 16 单元 + 模式提取, 与输入/思考层解耦
- 模态融合层: 激活模块按 `modality_weights` 加权平均 → tanh → 思考层输入
- 真实确定性图像编码器 (4×4 平均池化), 替代原随机占位实现
- 网络统计新增各输入模块脉冲统计

### 变更 (Breaking)
- `DistributedFormer.step(vector)` 移除, 一律使用模态字典; 全部调用方
  (trainer / agents / demos / selfcheck / experiments) 已迁移
- 未启用的模态模块静默; 未知模态抛 KeyError, 非字典输入抛 TypeError

### 兼容保留
- `get_output_pattern()` / `output_units` 属性别名等读取接口不变

## v0.2.0 (2026-09-11)

产品化重构版本。核心算法保持 Pre0.1 原貌, 工程链路全面修复:

### 新增
- 可安装的 `distributedformer` Python 包 (`pip install -e .`), `dformer` 命令行入口
- `dformer serve` 长驻监控服务 (容器部署默认入口, 支持 `--realtime` 切换 yfinance 真实行情)
- 可插拔行情数据源接口: 内置模拟器 / `YFinanceDataSource`
- pytest 测试套件 (12 项) 与 GitHub Actions CI (Python 3.9 / 3.11 / 3.12)
- MIT License, CHANGELOG, 本 README

### 修复
- Docker 镜像 CMD 引用不存在的 `python main.py server` 模式 → 改为 `dformer serve`, 镜像可正常启动
- docker-compose 构建上下文路径失效 → 指向包化后的 Dockerfile
- 所有模块导入路径统一为 `distributedformer.*`

### 变更
- 训练权重 (~240MB npz) 不再随仓库分发, 通过 GitHub Release 附件提供
- 历史实验报告 (experiments/, reports/) 作为归档保留

## Pre0.1 (2025-07-16)

首次对外公开发行版, 完整归档实验日志、训练权重、消融实验记录 (详见 RELEASE_NOTES.md)。
