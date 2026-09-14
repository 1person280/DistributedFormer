# 更新日志

## v0.7.3 (2026-09-14)

Rust coding 思考插件: 内嵌真实语料填补训练材料空白。

### 新增
- `distributedformer/cutemamen/rust_coding.py`: **RustCodingPlugin** —
  静态识别一段 Rust 代码命中的编译错误类别 (move / borrow / lifetime /
  type / ok)。后台携带 v0.6.0 的 Rust coding 真实需求语料 (100 段真实
  风格 Rust 代码 × 5 类真实 rustc 错误) 作为**训练材料**, 直接消费真实
  代码文本, 不再依赖合成随机数据
- `training_data()`: 导出真实训练材料 (100×10 特征矩阵 + 标签索引),
  供端到端训练 / 评估 (呼应路线图 "真实数据集端到端训练" 的数据通路)
- 可学习权重 = 每类别 static_metrics 原型 (类标质心), 随 `.CuteMamen`
  包 weights/ 存档往返; on_think 最近原型分类 + softmax 置信度, 总线广播
  `rust.classified` 供插件间通信; route 默认 "rust"
- `native_registry` 新增 `"rust.coding"` → RustCodingPlugin, 包可随用随载
  热加载; 附 `cutemamen_pkgs/rust_coding.CuteMamen` 现成包

### 变更
- 版本 0.7.2 → 0.7.3 (pyproject / `__version__` / pkg.CORE_VERSION /
  manifest.min_core_version)

### 质量
- 新增 tests/test_cutemamen.py 3 项 (训练材料导出 / 路由分类与总线 /
  pkg 往返), 测试总计 97 项全绿 (94 → 97)

## v0.7.2 (2026-09-14)

CuteMamen 插件标准落地 + 模型精简。新增 `distributedformer/cutemamen` 包
（约 1,600 行）：通用固定内核（路由器 + 工作记忆 + 插件注册表）与
`.CuteMamen` 专家插件包；v0.7.0 的模态面 pkg 是其首个特例。
模型精简为 CubeGPTKernel——必要思考留内核，其余思考由插件实现。

### 新增: CuteMamen 插件标准 (`distributedformer/cutemamen/`)
- `kernel.py`: **CuteMamenKernel 通用固定内核 (Nest)** — 路由器 (主题 →
  插件, 未路由事件广播 kernel.unrouted) + 工作记忆 (近期事件 + KV 堆注意力)
  + 插件注册表 (挂载/卸载/随用随载热加载) + 内存预算 LRU 淘汰
  (超预算先自动存档 .CuteMamen 再卸载, 绝不淘汰本轮激活的专家,
  淘汰广播 plugin.evicted)
- `plugin.py`: **ExpertPlugin 专家插件基类** — on_load / on_think /
  on_unload 生命周期钩子 (规范 §5), 内核按序调用; on_think 每次激活
  自动记录情景记忆。**PluginMemory 三级记忆存档**: working (FIFO) /
  episodic (带时间戳事件流水) / semantic (LRU 知识), consolidate()
  情景 → 语义蒸馏, archive()/restore() JSON 安全存档
- `event_bus.py`: **EventBus 事件总线** — pub/sub + `*` 通配订阅,
  生命周期广播 (plugin.loaded / plugin.unloaded / plugin.evicted /
  kernel.think), 插件输出发布到 plugin.<name>.output 供其他插件订阅
  (插件间通信), 订阅方异常隔离不中断发布方
- `pkg.py`: **.CuteMamen 包格式** (单个 tar.gz: manifest.json +
  weights/weights.npz + memory/{working,episodic,semantic}.json)。
  **解码器层集中兼容** (规范 §2.1): v1 → v2 自动适配
  (model_type→base_model / on_init→on_load / 删 legacy_mode /
  补 on_unload 桩), 未知字段一律保留; min_core_version 检查;
  base_model 原生插件注册表分发; .dfpkg (v0.7.0) 特例自动分流到
  FacePlugin
- `bridge.py`: **LoRA/Adapter 兼容桥接** (规范 §6) — LoRAAdapter
  (ΔW = (α/r)·B@A), LoRABridgePlugin (on_think 计算 ΔW·x 低秩贡献,
  存档/加载/导出为 LoRA), apply_lora 权重合并, lora_from_weight
  任意线性权重 SVD 低秩导出
- `face_bridge.py`: **FacePlugin 模态面思考插件** — v0.7.0 .dfpkg 模态面
  pkg 的通用化: 一个 CubeFace = 一个思考插件, on_think 驱动面皮层计算,
  权重序列化复用 face_pkg 逐位可复现格式
- `migrate.py`: **migrate-v1-to-v2 迁移工具** (规范 §2.2) — 随包安装的
  命令行入口 (`migrate-v1-to-v2`), 支持 --recursive/--dry-run/--strict/
  --backup/--output, 退出码 0/1/2 (全部成功/部分需人工/严重错误)

### 新增: 模型精简 CubeGPTKernel
- CubeGPTKernel (kernel.py): CubeGPT 的内核形态, 只保留**必要思考** —
  立方体棱路由 / KV 堆工作记忆 / 输出头 16 单元 / 节律调制;
  模态面皮层计算整体外移为 FacePlugin 思考插件
- API 与经典 CubeGPT 兼容: step / faces / ring / kv_stack /
  export_face / import_face / unload_face / register_face_pkg /
  load_face / list_faces / get_network_stats / reset_state;
  step() 沿用随用随载热加载 (输入用到已注册未加载模态时现场加载)
- `CubeGPT.to_kernel()`: 经典形态 → 内核形态零拷贝转换
  (同一面对象 / KV 堆 / 输出头, 权重与状态不复制)

### 变更
- .dfpkg manifest 补 CuteMamen 规范字段: standard_version "2.0.0" +
  base_model "cubegpt.face" (旧包兼容不受影响, 解码器适配)
- `dformer test` 自检套件新增 [模式6] CuteMamen 插件标准
- Rust 基准 (v0.6.0) 复测值更新到 README: 57.6% ± 10.3%
  (修复 CubeFeatureExtractor 可复现性后的 5 种子结果, 初版 54.4% ± 5.4%)

### 质量
- 新增 tests/test_cutemamen.py 39 项: 生命周期钩子顺序 / 路由 / 热加载 /
  三级记忆往返与容量淘汰 / 事件总线通配与异常隔离 / 包格式往返 /
  解码器 v1→v2 适配与未知字段保留 / 内存预算淘汰 (含保护语义) /
  LoRA 数学与桥接往返 / FacePlugin 状态逐位还原 / dfpkg 特例加载 /
  CubeGPTKernel 全 API / to_kernel 零拷贝 / 迁移工具全部参数与退出码
- 测试总计 94 项全绿 (55 → 94)

## v0.7.1 (2026-09-13)

终端聊天: `dformer chat` 与 CubeGPT 直接对话。

### 新增
- `distributedformer/demos/chat.py`: CubeGPT 终端聊天 REPL —
  用户输入喂进 text 面, 每回合跑 8 步 (环形棱把活动传导到
  numeric/timeseries/image 面), 由输出头部脉冲模式驱动回复
- `dformer chat [--depth 0|1|2]`: CLI 子命令
- 回复如实反映网络内部状态 (输出脉冲数 / 模式能量 / 主活跃面 /
  KV 用量); CubeGPT 目前是脉冲 reservoir, 尚无语言生成头,
  回复由活动强度分级模板生成, 不伪装成自然语言
- 交互命令: `/reset` (清空网络与 KV 记忆) `/stats` (各面脉冲 /
  KV 堆 / STDP 统计) `/help` `/exit`
- 对话期间 STDP 在线学习持续进行
- `tests/test_chat.py`: 回合逻辑 / 状态累积 / reset / stats 测试

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
