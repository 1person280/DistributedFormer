# 更新日志

## v0.7.0 (2026-09-13)

模态面独立化与 pkg 存档: 每个模态面成为可独立打包、传输、加载的单元,
支持随用随载热加载与自由导入导出。

### 新增
- `distributedformer/core/face_pkg.py`: 模态面 pkg 存档 `.dfpkg` —
  遵循 CuteMamen 插件规范 v0.1.0 包格式精神 (单个 tar.gz:
  manifest.json 清单 + weights/ 权重 + memory/ 记忆状态)。
  权重 (16 标量参数 + STDP 后小世界连接) 与状态 (state/fatigue/
  refractory/学习计数) 逐位可复现; 感受野投影由 layer_id 的 crc32
  种子重建, 无需存档
- CubeGPT 面 pkg API:
  - `export_face(modality, path)`: 打包单个模态面为独立存档
  - `import_face(path)`: 导入/替换模态面 (可跨模型、可换模态名)
  - `unload_face(modality)`: 卸载释放内存, 默认先自动导出 pkg 保证可恢复
  - `register_face_pkg(path)` + `load_face(modality)`: 随用随载注册表
  - `step()` 热加载: 输入用到已注册未加载的模态时现场加载后常驻
  - `list_faces()`: 已加载 / 已注册未加载视图
  - `min_core_version` 兼容性检查 (内核过旧拒绝加载, CuteMamen §11)
- 路线图收录 **CuteMamen 插件标准 v0.1.0** (通用固定内核 + .CuteMamen
  专家插件包 + 生命周期钩子 + 三级记忆存档 + 事件总线 + 内存预算淘汰
  + LoRA/Adapter 兼容桥接); v0.7.0 的模态面 pkg 是其首个特例

### 修复
- **训练报告占位符** (已知问题清单最后一项): `report_generate` 动作
  原先对每个章节只写 `[自动生成内容占位]`, 现渲染真实统计数据 —
  智能体状态 (脉冲收发/周期) / CubeGPT 网络统计 (各面单元/脉冲/节律) /
  KV 堆记忆 (用量/利用率/访问) / STDP 学习统计, 支持章节自由组合

## v0.6.0 (2026-09-11)

真实数据基准起步, 解决"知识脱离实际"问题。首个基准: Rust coding 真实需求。

### 新增
- `distributedformer/data/rust_coding.py`: Rust Coding 基准数据集 v1 —
  100 段真实风格 Rust 代码 × 5 类 (move E0382 / borrow E0502·E0499 /
  lifetime E0597·E0106 / type E0308·E0277 / ok), 每条附真实 rustc
  错误码与报错信息; 提供分层划分与静态扫描特征 (借用符/mut/生命周期符等)
- `experiments/rust_benchmark.py` (实验 R2): CubeGPT (numeric+text 双面)
  冻结特征 + 线性读出层, 5 种子验证准确率 **54.4% ± 5.4%**, 全部超过
  随机基线 20% (报告: experiments/rust_report.md)

### 修复
- 文本编码器分词从 `\b[a-zA-Z]+\b` 改为代码感知分词 (保留 & ' -> ::
  数字等代码关键 token); 词→维度哈希从进程随机的 `hash()` 改为 crc32
  (跨进程可复现) — 两者对真实代码数据都是致命缺陷
- `LinearReadout` 类数硬编码 4 → 泛化为任意类数
- CubeFeatureExtractor 关闭自发率后重建向量化数组 (否则皮层仍有随机脉冲,
  特征不可复现)

## v0.5.0 (2026-09-11)

修复 README "已知问题" 的全部三项:

### 1. KV 堆注意力接入主计算路径
- 新增 `KVStack.retrieve()`: 聚合 top-k 检索向量, tanh 归一化, 带 scan_limit 限额
- FractalLayer / 输入端口单元 / 输出头单元的 `attn` 输入全部改为真实检索结果
  (原先恒为零向量); DistributedFormer 与 CubeGPT 双路径生效
- 推理模式输出状态持续写入 KV 堆, 记忆 → 注意力形成闭环

### 2. 训练方法学验证 (实验 R1)
定位并修复了历史消融停留在随机水平的两个动力学根因:
- **均值池化瓶颈**: 所有单元接收同一 mean(input) 标量 → 改为每单元固定随机
  感受野投影 (crc32 种子, 跨进程可复现)
- **调制淹没输入**: w_global≈0.5 的恒定加性调制比输入路径大一个数量级 →
  配平为 w_global≈0.1
- 新增 `training/readout.py` (reservoir 读出层范式) 与
  `experiments/readout_validation.py`: 5 种子验证集准确率
  **64.8% ± 4.5%**, 全部超过随机基线 25% (报告: experiments/readout_report.md)

### 3. 真实桌面通知与 HTTP 动作
- `_do_notification`: Windows 真实系统通知 (PowerShell 气泡/toast),
  `DF_NOTIFY_MODE=sim` 回退打印; 非 Windows 打印
- `_do_api_call`: 真实 HTTP POST (stdlib urllib, 10s 超时), endpoint 或
  `DF_WEBHOOK_URL` 配置, 失败降级不中断工作流

### 质量
- 测试 36 项全绿 (新增注意力闭环 / 检索限额 / API mock / 通知模拟等 6 项)

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
