# 更新日志

## v0.13.0 (2026-09-18)

响应 OmniSpace 实测反馈（issue #2）：把安全盾接入内核主路径，并补齐
ActionTracer 落盘持久化，同时修复一个测试收集遗留 bug。

### 变更
- **安全监控接入内核主路径**：`CuteMamenKernel` 新增 `enable_security()` /
  `security_stats()`/`_security_gate()`——以"执行层订阅者"形态订阅动作主题
  `action.request`，候选动作经 `SecurityMonitor.gate()` 判定后：ALLOW →
  转发到执行主题 `action.execute`（真实执行层订阅），DENY/REVIEW → 拦截留痕。
  内核与插件本身不感知监控面，监控面只拦动作下发；默认不启用，向后兼容。
- **`ActionTracer` 可选落盘持久化**：新增 `persist_path` 参数（默认 `None`
  不落盘），每条审计记录以 JSONL（每行一条）追加写入；新增 `load()` 类方法
  从落盘文件读回全量审计（跨进程/重启溯源）。落盘/导出/复盘统一 `_trace_dict`
  格式。
- **修复测试收集 bug**：`src/tests/test_security_integration_units.py` 的
  `from tests.test_security_integration import ...` 是 v0.8.7 tests 收进 src
  前的旧路径，改为 `from src.tests.test_security_integration import ...`（全量
  collect 阶段不再 `ModuleNotFoundError`）。
- **测试**：新增内核接线测试（default off / enable_security 幂等 / ALLOW 转发
  / DENY 拦截 / REVIEW 挂起）与 ActionTracer 落盘测试（写 JSONL / load 读回 /
  默认关 / 缺文件幂等），合并计 10 项。
- **版本号**：0.12.0 → 0.13.0（pyproject / `__version__` /
  `pkg.CORE_VERSION` / `min_core_version`）

### 测试
- 全量测试通过（244 项，含新增 10 项）。

## v0.12.0 (2026-09-17)

新增**视频生成训练**（真实运动识别基准，实验 R4）、**Web 图形界面**
（`dformer ui`）与**全面训练**（`dformer benchmark-all`，三真实任务统一
基准），并翻新 README：实验记录准确率表拆分为"主模型直评 / 思考插件
推理"两张、训练材料 Rust 两节合二为一。

### 变更
- **视频生成训练（`VideoMakingPlugin` 补训练数据接口）**：新增
  `motion_descriptors()` 导出真实运镜曲线参数表（10 种真实镜头运动，
  from/to 8 维描述符）、`training_data()` 导出稠密采样训练材料
  `(X, y)`（与 `RustCodingPlugin.training_data()` 同构）。
- **`src/data/video_motion.py`（VideoMotionDataset）**：第三个真实训练
  材料——视频内核的真实运镜曲线按其真实插值进度稠密采样为描述符
  `(dx, dy, zoom, rot, progress)`，识别属于哪一种真实运动类别（10 分类，
  随机基线 10%），纯真实、无合成标签。
- **`src/experiments/video_motion_benchmark.py`（实验 R4）**：5 种子 × 5
  折分层 CV，冻结水库（numeric 单面）+ 线性读出层 L2 选优——真实运镜
  识别 **val_acc 61.7% ± 9.8%**（25/25 折全超随机基线 10%）。
- **全面训练 `dformer benchmark-all`（`src/experiments/benchmark_all.py`）**：
  一次跑齐三真实任务统一基准并汇总（Rust / Markdown / 视频）——Rust
  76.7% ± 3.7% / Markdown 58.3% ± 2.8% / 视频 61.7% ± 9.8%，25/25 折全超
  各自随机基线，见 `benchmark_all_report.md`。
- **Web 图形界面 `dformer ui`（`src/deployment/web_ui.py`）**：零第三方
  依赖的本地浏览器控制台（stdlib http.server 单页 HTML/JS），含内核/插件
  状态、思考控制台（rust/video/chat 在线 think）、三任务基准结果表。
- **README 翻新**：实验记录准确率表拆分为**主模型直评**与**思考插件推理**
  两张（思考插件 = RustCodingPlugin 知识迁移 / VideoMakingPlugin 训练）；
  训练材料章节把 `Rust coding 真实基准` 与 `RustCoding 思考插件` 两节
  合二为一，并新增 `视频生成训练` 小节；命令行文档新增 `benchmark-all` /
  `ui`。
- **测试**：新增 `src/tests/test_video_training.py`（3 项）。
- **版本号**：0.11.0 → 0.12.0（pyproject / `__version__` /
  `pkg.CORE_VERSION` / `min_core_version`）

### 测试
- 全量测试通过（234 项，含新增 video_training 3 项）。

## v0.11.0 (2026-09-17)

大幅落地两条路线图主线，并正式开展与 **OmniSpace** 的生态合作。

### 变更
- **64 比特分类式 token**（`src/codec/class_token.py`）：一个 token 占 64
  比特，纯文本场景下高 32 比特为 token 组、低 32 比特直接承载 utf8-mb4
  字符（Unicode codepoint，≤0x10FFFF 覆盖 4 字节/astral/emoji）。设硬性
  分组：`0x00000000` 保留为 utf8-mb4 字符 token 组，另有 subword / control /
  保留组常量。提供 `encode / decode / tokenize / char_token / group_of /
  low_of / embed_tokens` 等完整接口。
- **CubeGPT 端到端可学习**（`src/training/trainer.py` v5.4+）：新增可训练
  `TokenEmbedding` 嵌入层——64 比特 class-token 序列经 crc32 哈希映射到
  可训练矩阵一列，词袋求和 + L2 归一；`PersistentOutputHead.dfeature()`
  把损失对原始特征的梯度回传到嵌入层，使**学习信号真正回传到输入编码层**，
  编码与水库一起端到端可学习。`real_dataset.py` 的 `TrainingSample` 新增
  `token_seq` 字段；`markdown text` 任务（`src/data/md_text.py`）作为第二
  真实数据集接入。
- **`src/data/md_text.py`（MarkdownTextDataset）**：第二个真实任务——
  从仓库真实多字节 utf8-mb4 文档（README / CHANGELOG / 架构 / 兼容性 /
  贡献 / 发布说明）抽取 5 类内容类型（heading/code/list/table/paragraph），
  纯真实、无合成。
- **`src/experiments/multi_task_benchmark.py`（实验 R3）**：5 种子 × 5 折
  分层 CV 双任务基准，验证"≥2 真实任务显著超随机基线"：

| 任务 | val_acc (折均值±std) | 随机 | 多数类 | 超随机折数 |
|------|----------------------|------|--------|-----------|
| Rust 编译错误族 | **76.7%** ±3.7% | 20% | 20% | 25 / 25 |
| Markdown 内容类型 | **57.7%** ±2.8% | 20% | 41.0% | 25 / 25 |

- **CLI**：`dformer train` 新增 `--dataset {rust,md}`（md 走端到端 token
  嵌入路径）；新增 `benchmark-multi` 子命令（等价运行实验 R3）。
- **OmniSpace 生态合作（正式开展）**：详见 README「生态合作」章节。
- **版本号**：0.10.2 → 0.11.0（pyproject / `__version__` /
  `pkg.CORE_VERSION` / `min_core_version`）

### 测试
- 全量测试通过（231 项，含新增 class-token 12 项、md_dataset 6 项、
  trainer token 端到端 4 项）

## v0.10.2 (2026-09-17)

修复端到端训练后期漂移（v0.8.2 已知问题）：非平稳水库 + 在线持久输出头
在扩语料（502 段）复评时再次确认后期验证准确率回落。新增 LR 调度与水库
冻结两种机制，默认 cosine 调度消除回落并在最终 epoch 稳定/提升。

### 变更
- **`src/training/trainer.py` v5.4**：`DFTrainer` 新增 `lr_schedule`
  （默认 `cosine`，可 `constant`/`step`）、`min_lr_ratio`（默认 0.1）、
  `freeze_reservoir_epoch`、`step_drop_epoch`。`_apply_schedule()` 每 epoch
  计算当前 LR：cosine 后期将 w_in/思考层 LR 平滑衰减至 0.1×（缓和非平稳
  水库对特征分布的持续扰动），`constant` 行为与旧版完全一致
- **水库冻结**：`freeze_reservoir_epoch` 到达后 `_update_weights_supervised`
  直接返回，仅在线输出头更新——与离线读出范式（冻结 CubeGPT 水库）一致，
  彻底隔绝特征漂移
- **`src/cli.py`**：`dformer train` 新增 `--lr-schedule`、`--freeze-reservoir-epoch`、
  `--step-drop-epoch`
- **新增测试** `src/tests/test_trainer.py`：6 项（默认 cosine / cosine 单调
  衰减 / constant 不变 / step 掉落 / 冻结守卫 w_in 不更新 / 冻结标志生效）
- **版本号**：0.10.1 → 0.10.2（pyproject / `__version__` /
  `pkg.CORE_VERSION` / `min_core_version`）

### 端到端复评（502 段真实语料，12 epoch）
| 方案 | seed0 final / 后期曲率 | seed1 final / 后期曲率 |
|------|----------------------|----------------------|
| baseline（constant）| 55.2%（55/64/55 高频抖动） | **51.2%**（10–11 轮跌至 44–46%） |
| **cosine（默认）** | **69.6%**（末端回升 0.7） | **59.2%**（稳定 57–63%，无崩落） |
| freeze@8 | 61.6%（稳定 58–64%） | 57.6%（稳定 54–59%） |

baseline 两 seed 均出现"后期崩落"（最佳 54–59% 后跌至 44–55%）；cosine
完全消除崩落且最终 epoch 稳定/更高，freeze 消除崩落但最佳验证略降（早期
未充分塑形）。cosine 为默认与推荐。

**深度 2（完整 ~4,400 单元架构）单种子确认**：cosine 末期曲线持续上升
（末三轮 59%→70%→72%），**最终 epoch 72.0%。**（最佳 61.6%），在完整
架构上也消除了后期崩落。

### 测试
- 全量测试通过（209 项，含新增 6 项 trainer 测试）

## v0.10.1 (2026-09-17)

继续解决准确率瓶颈：Rust 基准脚本同步到 502 段 5 折分层 CV，并引入
读出层 L2 正则选优压小样本过拟合。

### 变更
- **`src/experiments/rust_benchmark.py` 升级**：从旧的 100 段单次 75/25
  划分（结果文件停在 57.6%，与 README 脱节）升级为 **502 段 × 5 折分层
  CV × 5 种子**（复用 `run_cross_validation`），每个样本恰好验证一次，
  消除脚本/结果/README 脱节
- **`src/training/readout.py`**：`run_cross_validation` 新增 `l2` 参数
  （默认 1e-3，与插件迁移一致，行为不变）；`rust_benchmark` 在读出层上
  做 L2 网格选优以缓解训练/验证过拟合 gap（train≈97% vs val≈75%）
- **版本号**：0.10.0 → 0.10.1（pyproject / `__version__` /
  `pkg.CORE_VERSION` / `min_core_version`）

### 准确率（v0.10.1 重跑，502 段 × 5 种子 × 5 折分层 CV）
- **val_acc = 76.7% ± 3.7%**（25 折，全超随机基线 20%）
- 种子均值 75.1%–79.1%，train_acc 均值 97.1%
- 读出层 L2 选优：1e-3 最优（1e-2→78.5%、1e-1→77.9%，更强正则略降）
- 与 v0.8.6 插件知识迁移 76.7% 逐折一致

### 测试
- 全量测试通过（203 项）

## v0.10.0 (2026-09-17)

分布式多节点落地 —— Redis KV 堆在多节点工作流中实际启用。

### 变更
- **`src/deployment/redis_kv.py` `RedisKVStack` 协议兼容内存 `KVStack`**：
  - 构造签名对齐 `KVStack(capacity, dim, retention_policy)` + Redis 连接参数
    （host/port/db/tenant_id），可直接替换工作流 / 智能体 / 主模型的内存 KV
  - 补齐主计算路径 `retrieve(query_signal, top_k, scan_limit)`：按插入序号
    截取最近 scan_limit 条 → 打分 → top-k 聚合 → tanh（空堆返回零向量）
  - 补齐 `_coerce_vec`（变长载荷定长化）、`query`（返回
    `[(entry_id, retrieved, normalized)]`）、`get_stats`（含 `total_access`
    / `backend` / `tenant_id`）、`save_to_disk` / `load_from_disk`（Redis
    AOF 自持久化，协议兼容 no-op）；LRU 淘汰与内存语义一致（满容量淘汰
    最久未访问）
  - 新增 `create_kv_stack(backend, **kw)` 后端工厂（memory / redis /
    auto 读 `KV_BACKEND`）；`MockRedis` 底层按 host:port:db 命名空间进程内
    共享，单进程内即可模拟真实 Redis server 的多节点共享语义
- **`src/workflow/engine.py`**：`SpikeWorkflowEngine(config, kv_backend=...)`
  按后端创建全局 KV（`KV_BACKEND` 环境变量 / 参数可选）
- **`src/agents/base_agent.py` `MemoryAgent`**：支持注入共享 `kv_stack`
  （复用引擎的 Redis 后端），不再强制自建内存 KV
- **CLI / 部署**：`dformer serve` 新增 `--kv-backend memory|redis|auto`；
  `StockMonitorWorkflow` 透传后端；docker-compose 两个脉冲节点
  `KV_BACKEND=redis` 且同租户 `shared_tenant`，经 Redis 共享记忆
  （改 `TENANT_ID` 即可切换为租户隔离）

### 新增
- `src/experiments/distributed_kv_demo.py`：多节点共享记忆演示——节点 A 写入
  KV，节点 B 检索可见；输出报告 `reports/distributed_kv_demo.md`
- `src/tests/test_redis_kv.py`（12 项）：协议兼容（retrieve 空堆/维数/tanh）、
  query 格式、租户隔离、同租户跨节点共享、LRU 淘汰、get_stats 字段、
  save/load no-op、后端工厂、引擎后端选择、MockRedis 共享命名空间

### 测试
- 全量测试 203 项通过（191 + 新增 12），无回归

### 版本
- 版本 0.9.2 → 0.10.0（pyproject / `__version__` / `pkg.CORE_VERSION` /
  `min_core_version`）

## v0.9.2 (2026-09-16)

安全AI（AI 运行时安全监控模块）P1 落地 —— Safety Shield 四个方向全部完成。

### 新增
- **`src/security_monitor/behavioral_fingerprint.py`**（P1 行为指纹与异常基线检测）：
  - `observe/learn` 在正常业务流量上累积每工具频率基线（速率 = 次数/时间窗）
  - `check()` 实时三路判定：
    - `frequency-burst`：窗内同工具调用次数超硬上限或基线速率 × 偏离系数
    - `high-entropy-params`：载荷 Shannon 熵超阈值（端口扫描式枚举调用）
    - `unrelated-low-level`：任务白名单外的底层系统指令（无任务上下文时保守告警）
  - `FingerprintResult` 返回 anomalous/flags/metrics 供审计留痕
- **`src/security_monitor/action_tracer.py`**（P1 全链路行为审计与溯源）：
  - 思考状态 / 决策依据 / 工具参数 / 执行结果四元组绑定存储，全局 `action_id` 主键
  - `replay(id)` 溯源复盘单条链路；`query()` 多维 AND 过滤回溯；
    `export()/iter_traces()` 结构化日志导出（强监管合规审计）
- **`SecurityMonitor` 接入 P1**：`gate()` 现并跑行为指纹判定、ALLOW 回填基线、
  每条动作写入全链路审计（决策依据含指纹指标）；`stats()` 新增
  `fingerprint` / `tracer` 统计组

### 测试与版本
- `test_security_monitor.py` 由 15 → 36 项（新增 21 项 P1 单测）；
  `test_security_integration.py` 统计断言随新 stats 结构更新；
  全量测试 191 项通过
- 版本 0.9.1 → 0.9.2（pyproject / `__version__` / `CORE_VERSION` /
  `min_core_version`），README 安全AI 四方向全部标记 ✅ 已完成

## v0.9.1 (2026-09-16)

OpenAI 兼容接口服务器（落地 · OpenCode 对接）：纯标准库 `http.server`
暴露 `/v1/models` 与 `/v1/chat/completions`（含 SSE 流式），OpenCode
配置 `baseURL` 即可把 CubeGPT 当编码模型后端。CubeGPT 是脉冲水库、
无语言生成头，故"生成"如实接地：通用对话走脉冲统计回复，代码 / Rust
意图路由到 RustCodingPlugin 的 502 段真实语料知识（分类命中哪类编译
错误 + 给出可编译改法建议与置信度/来源）。新增 `dformer serve-opencode`
子命令。

### 变更
- **新增 `src/deployment/openai_server.py`**（纯标准库，零依赖）：
  - `GET /v1/models` → 返回模型 `cubegpt`
  - `POST /v1/chat/completions` → OpenAI 报文 (非流式 + SSE 流式)
  - 三级流水线 (落地): **CubeGPT 前置感知 → 真 LLM 生成 → CubeGPT 后置审计**
    - 前置感知: 每请求先喂用户文本进水库取脉冲统计
    - 生成: 外挂 LLM 封装为**标准 CuteMamen 插件 `LLMProviderPlugin`
      (base_model=`llm.provider`, route=`llm`)** —— 生命周期/三级记忆/
      权重存档/注册发现/`.CuteMamen` 打包交付全支持, 已注册进
      `native_registry()`; 配置 `--lm-base` / `CUBEGPT_LM_BASE`
      (Ollama/llama.cpp) 即转发给真 LLM 做任意语言代码生成; 未配置则
      回退 CubeGPT 自身 Rust 知识/脉冲模板 (离线可用)
    - 后置审计: 生成结果再喂水库, 命中 Rust 代码复用 `RustCodingPlugin`
      真实语料分类并附改法 + 置信度
  - API 密钥: `Authorization: Bearer <key>`, 默认测试密钥
    `cubegpt-local-test-key`, `--no-auth` 关闭
- **`RustCodingPlugin.classify()`**：把 on_think 里的分类逻辑抽成公开
  方法供服务器等外部调用者复用（原推理路径不变：主模型迁移读出层 →
  语料原型回退）。
- **CLI**：新增 `serve-opencode [--host --port --depth --dim --api-key
  --no-auth --lm-base --lm-key --lm-model]` 子命令。
- **测试**：新增 `src/tests/test_openai_server.py`（17 项）——OpenAI
  报文格式、代码意图识别、Rust 帮改法、鉴权(401/200)、三级流水线
  (假 LLM 后端)、SSE 流式、真实 HTTP 层（models / chat 非流式 + 流式）。
- 版本 0.8.7 → 0.9.1（pyproject / `__version__` / `pkg.CORE_VERSION` /
  `min_core_version`）

## v0.8.7 (2026-09-16)

更整洁的项目 + 轻量视频生成内核插件：运行时产物统一到 `./cache`、
仓库内零 `__pycache__`、tests 与 experiments 收进 `src/`，
清理上个版本废弃路径与一次性临时文件，新增视频生成思考插件。

### 变更
- **统一运行时缓存根目录 `./cache`**（精简项目）：
  - 插件卸载/淘汰自动存档：根目录 `cutemamen_pkgs/` →
    `cache/cutemamen_pkgs`（`src/cutemamen/kernel.py` `CACHE_DIR`）
  - 模态面卸载自动存档：根目录 `face_pkgs/` → `cache/face_pkgs`
    （`kernel.py` 与 `core/distributedformer.py` 两处）
  - pytest 缓存：`.pytest_cache/` → `cache/pytest`
    （`pyproject.toml` `cache_dir`）
  - `.gitignore` 以单条 `cache/` 覆盖全部运行时缓存
- **仓库内零 `__pycache__`**：
  - 曾用 `sys.pycache_prefix` 统一到 `cache/pycache`，但该机制按
    源文件绝对路径镜像目录树（层级又长又深）且会把 stdlib 一起
    复制进仓库缓存，故弃用并删除 `cache/pycache`
  - 改为根 `conftest.py`（pytest 场景）与 `src/__init__.py`
    （直接运行场景）设置 `sys.dont_write_bytecode`，仓库内任何
    位置（含根目录与 `src/`）不再生成 `__pycache__`；
    conftest 自身的 .pyc 由 atexit 清理
  - 如需字节码缓存，运行前设 `PYTHONPYCACHEPREFIX` 指向仓库外目录
- **目录结构收拢**：`tests/` → `src/tests/`、`experiments/` →
  `src/experiments/`（仓库根只剩 `src/` / `plugin/` / `docs/` /
  `cache/` / `conftest.py` 等必要条目）；两子包补带架构注释的
  `__init__.py`；测试/实验脚本路径引导上溯 3 级到仓库根；
  `pyproject.toml` `testpaths` 同步，打包排除 tests/experiments
- **清理上个版本废弃路径与临时文件**：
  - 删除 `experiments/` 中 18 个未跟踪一次性临时脚本/文本
    （`fix_*.py` / `backend_*.txt` / `merge_msg.txt` 等，
    仅保留 8 个真实实验脚本与结果）
  - 删除根目录 `cutemamen_pkgs/` 与全部散落 `__pycache__/`
  - 修复 `dformer --version` 废弃导入路径
    （`distributedformer` → `src`）
  - 卸载残留的 `distributedformer 0.7.2` 旧版 editable 安装
    （finder 仍映射到已删除的 `distributedformer/` 旧目录树）
  - README 示例与目录树同步到新布局
- **轻量视频生成内核插件** `plugin/VideoMaking.CuteMamen`
  （`src/cutemamen/video_making.py`）：关键帧 + 镜头运动曲线 →
  缓动仿射帧序列 + 转场合成，纯 numpy 零 GPU，供宿主
  （OmniSpace 等）作轻量视频生成档位

## v0.8.6 (2026-09-15)

新阶段 · 分布式架构 + 更整洁的项目结构：主模型 Rust 知识迁移到
思考插件，代码扁平化到 `src/` 顶层包，文档整理到 `docs/`，
清理无作用的遗留文件。

### 变更
- **知识迁移** `src/cutemamen/rust_coding.py`：新增
  `migrate_from_main_model()` —— 迁移路径与主模型读出层验证协议
  逐位一致（真实语料 → 双模态注入 → 冻结 CubeGPT 水库 → 线性
  softmax 读出层），W/mean/scale 存为插件可存档权重；
  `on_think` 优先走读出层，原型路径保留为无权重回退。
  5 种子 × 5 折交叉验证：**插件路由 76.7%，与主模型直评逐折一致（无损）**
- **可复现性修复**：
  - `src/core/distributedformer.py` `reset_state()` 补重置
    `_last_input`（原跨样本 KV 注意力泄漏历史输入）
  - `src/training/readout.py` `CubeFeatureExtractor` 补 seed
    Python `random`（小世界布线 `random.sample` 不可复现）
- **大重构**：全部代码统一到 `src/` 顶层包
  （`src/core` / `src/cutemamen` / `src/data` / `src/training` /
  `src/agents` / `src/codec` / `src/workflow` / `src/deployment` /
  `src/security_monitor` / `src/demos`），全量导入改写为
  `from src... import ...`；`pyproject` 适配 src 布局，入口
  `dformer = "src.cli:main"`；每个子包 `__init__.py` 附架构注释
- **文档整理**：CHANGELOG / COMPATIBILITY / CONTRIBUTING /
  RELEASE_NOTES 移至 `docs/`
- **结构清理**：删除无作用且增加理解成本的遗留文件——
  `main.py` / `test_import.py` / `test_perf.py`（入口统一为
  `dformer` / `python -m src`，测试统一在 `tests/`）、
  `elevator_question_full_computation.py` 与 `computation_output.txt`
  （一次性实验脚本及输出）、`reports/` 与 `visualization/`
  （历史生成物）、过期消融实验（`ablation_*` / `run_single_exp.py` /
  `partial_E1_Baseline.json`，脚本内含失效的硬编码路径）、
  根目录重复 `training/` 与 `src/training/checkpoints/` 二进制权重
- `plugin/RustCoding.CuteMamen` 以 v0.8.6 manifest 重建，
  携带主模型迁移权重；`experiments/distributed_architecture.py`
  输出新阶段评估结果

### 用法
```python
from src.cutemamen import CuteMamenKernel, RustCodingPlugin
plugin = RustCodingPlugin("rust-coding").migrate_from_main_model()
kernel = CuteMamenKernel(dim=16)
kernel.mount(plugin)
kernel.think({"topic": "rust", "data": 'let x: i32 = "hello";'})
# → 读出层推理 → 分类 {"label": "type", ...}
```

## v0.8.5 (2026-09-15)

.CuteMamen 插件标准落地 `./plugin` 目录：思考插件以独立包文件
交付，CubeGPT 路由到插件包文件本身（随用随载），宿主代码无需
import 插件模块。

### 变更
- `cutemamen/kernel.py`：
  - 新增 `DEFAULT_PLUGIN_DIR = "plugin"` 与
    `discover_plugins(plugin_dir)`：扫描插件目录下全部
    `*.CuteMamen` / `*.dfpkg` 包并注册（只记路径不加载）
  - 新增路由主题索引 `route_index`（route → 插件名）：已注册
    未加载的包可按路由主题（而非插件名）热加载；
    `_resolve` 查找顺序：已挂载路由表 → route_index → 注册表
    按名回退；`unmount`/`register_pkg` 同步维护索引
- `plugin/RustCoding.CuteMamen`：首个按标准独立交付的思考插件
  文件（rust-coding，route="rust"，base_model="rust.coding"），
  内含 502 段真实语料蒸馏的 5 类原型权重与三级记忆
- `tests/test_cutemamen.py` 新增第 11 节：插件目录发现标准 +
  CubeGPT 路由到仓库交付的 `plugin/RustCoding.CuteMamen`

### 用法
```python
from distributedformer.cutemamen import CubeGPTKernel
gpt = CubeGPTKernel(depth=1, dim=16, modalities=["numeric"])
gpt.discover_plugins()  # 扫描 ./plugin → 注册 RustCoding.CuteMamen
gpt.think({"topic": "rust", "data": "let x: i32 = \"hello\";"})
# → topic "rust" 命中 route_index → 热加载 → 分类 {"label": "type", ...}
```

## v0.8.4 (2026-09-15)

P2 扩真实语料：Rust 编码基准语料 100 → 502 段（rustc 错误索引官方
样例 + 真实 crate 编译失败样本），验证集 25 → 125，评估方差从
±6–9% 降至 ±2% 以内——准确率瓶颈解决方案全部完成。

### 变更
- `data/rust_coding.py`（v1 → v2）：`RUST_SNIPPETS` 每类 20 → 100+
  段（总计 502：move 101 / borrow 100 / lifetime 100 / type 100 /
  ok 101）。新增 402 段全部为真实模式：
  - rustc 错误索引官方样例族（E0382/E0505/E0507/E0502/E0499/
    E0597/E0106/E0515/E0716/E0623/E0308/E0277/E0599/E0300）
  - 真实 crate 编译失败高频模式（mpsc channel send 后使用、
    thread::spawn move 闭包、容器 push/insert 后使用键值、
    entry().or_insert 冲突、迭代中 retain/push、serde_json 风格
    所有权转移等）
  - 合法代码对照扩充（Arc/Mutex、trait/泛型、闭包、迭代器链等
    惯用可编译模式）
- `data/real_dataset.py`：文档同步 500 段语料；数据通路无变更
  （分层划分/K 折/静态特征归一化自动适配）

### 真实数据评估结果
- 读出层（R1, depth=1, 5 种子 × 5 折 CV, 502 段语料）：
  **75.6%**（种子均值范围 74.7%–76.3%，即 **±0.8%**；折展平
  ±3.9%，折范围 67.6%–82.0%）——全部 25 折超随机基线（20%）
- 对照 v0.8.3（100 段语料）：63.8%（种子均值 57%–67%）——
  扩语料后准确率提升 11.8 个百分点，种子间评估方差从 ±5% 收窄
  至 ±0.8%，达到 P2 的 ±2% 目标
- 折级 ±3.9% 为每折 ~100 验证样本的二项统计下限（p=0.75 时
  抽样噪声 ~±4.3%），属统计噪声而非方法不稳定
- 读出层训练准确率 ~97%（此前 100 段语料恒为 100%），小语料
  过拟合迹象显著缓解

### 文档
- `README.md` 路线图新增"安全AI（AI 运行时安全监控模块）"独立列表
  （与准确率瓶颈解决方案同级的子类别排期），四个方向按优先级：
  - **P0 · 意图与决策探针**（Intent Probing & Decision Audit）：决策
    下发前扫描内部决策日志，轻量安全分类器拦截越权/高权限/边界探索
    等逃逸倾向
  - **P0 · 高危工具调用熔断机制**（Critical Action Circuit Breaker）：
    动态工具调用权限围栏，高危操作（改系统配置/核心数据库/未白名单
    下载）拦截并触发深度审核（可人工介入），合规前拒绝下发
  - **P1 · 行为指纹与异常基线检测**（Behavioral Fingerprinting &
    Baseline Deviation）：正常业务行为建立安全基线，实时监控调用
    频率/参数熵值/链路逻辑关联性，异常逃逸行为立即告警
  - **P1 · 全链路行为审计与溯源**（End-to-End Action Tracing）：
    思考状态/决策依据/工具参数/执行结果四元组绑定留存，支持安全
    事件完整复盘与强监管合规审计

### 版本
- 版本 0.8.3 → 0.8.4 (pyproject / `__version__`)

## v0.8.3 (2026-09-14)

P2 交叉验证评估：读出层验证协议由单次 75/25 划分改为 5 折分层
交叉验证（× 5 种子），评估结论不再依赖划分运气。

### 新增
- `stratified_kfold(samples, n_folds, seed)`（`data/rust_coding.py`）：
  分层 K 折划分，每类别轮流分配到各折，折间类别比例一致
  （100 段语料 / 5 折 → 每折 4×5 = 20 验证样本）
- `RustCodingTrainingDataset.kfold_datasets(n_folds, seed)`
  （`data/real_dataset.py`）：返回 [(train, val), ...] 折划分，
  训练集 = 其余折并集；`stratified_split` 原路径保留不受影响
- `run_cross_validation()`（`training/readout.py`）：单种子 5 折 CV
  验证；特征按 sample_id 缓存，每样本仅提取一次，折级训练/评估
  线性读出层

### 变更
- `experiments/readout_validation.py`：实验 R1 切换为 5 种子 × 5 折
  交叉验证协议（共 25 次折评估），报告改为折级明细表 + 折展平
  汇总；`run_validation`（单次划分路径）保留供对照

### 真实数据评估结果
- 读出层（R1, depth=1, 5 种子 × 5 折 CV）：**63.8%**（折展平
  ±10.0%，折范围 45%–85%，种子均值 57%–67%）——全部 25 折
  超随机基线（20%）
- 对照 v0.8.1 单次 75/25 划分：61.6%（48%–72%，5 种子）——
  交叉验证结论与单次划分一致，方法学结论不依赖划分运气

### 版本
- 版本 0.8.2 → 0.8.3 (pyproject / `__version__` / pkg.CORE_VERSION /
  manifest.min_core_version)

## v0.8.2 (2026-09-14)

P1 修端到端权重更新：持久输出头 + w_in 感受野向量化，端到端训练
从刚至基线（20%）跃升到 67.2%——准确化开端。

### 变更 (`training/trainer.py`, v5.2 → v5.3)
- **移除注入-恢复启发式**（`_inject_supervisory_signal` /
  `_restore_supervisory_signal`）：临时改 gain/threshold 再还原，
  学习信号无法累积，端到端训练长期停留在基线
- **新增 `PersistentOutputHead`**：持久可训练线性 softmax 输出头
  （在线 SGD 交叉熵 + running z-score 标准化 + L2 正则），权重跨
  样本/epoch 持久累积；分类学习由输出头承担，发放率/抑制/思考层
  对齐损失保留为网络塑形监控信号
- **w_in 向量化更新**：`_update_weights_supervised` 中
  `w_in += lr * error * mean(input)` 的均值池化使所有单元收到
  无差异更新，改为每单元感受野投影 `w_in += lr * error * (receptive @ input)`
- `train_step` / `evaluate`：多步前向（4 步，水库充分演化）提取
  [思考层状态幅值 | 输出模式 | 思考层聚合模式] 特征，评估同步切换
  双模态前向 + 持久头预测（此前评估为单模态）
- `save_weights` / `load_weights` 持久化输出头权重

### 真实数据评估结果
- 端到端监督训练（depth=2, ~4,400 单元, 30 epochs, 5 种子）：
  最佳验证准确率 **67.2%**（56%–80%），v0.8.0 为 20%（刚至基线）
  ——全部种子超随机基线（20%）3 倍以上，端到端训练首次稳定超过基线
- 最终 epoch 验证准确率 36%–60%（在线头 + 非平稳水库存在后期漂移，
  如实报告；最佳值由验证集早停选取）
- 对照：冻结水库 + 离线线性读出层为 61.6%（v0.8.1）——端到端路径
  已反超离线读出

### 版本
- 版本 0.8.1 → 0.8.2 (pyproject / `__version__` / pkg.CORE_VERSION /
  manifest.min_core_version)

## v0.8.1 (2026-09-14)

P1 结构感知编码：针对混淆源新增结构特征，读出层 57.6% → 61.6%。

### 新增
- `structure_metrics(code)`（`data/rust_coding.py`）：6 维结构感知特征
  ——`&mut` 数、返回引用 `-> &` 数（lifetime 强信号）、类型标注 `: ` 数、
  `println` 数（use-after-move 场景）、`let` 绑定数、防御调用数
  （clone/to_string）。每维对应 v0.7.5 诊断的混淆源：
  move↔lifetime 互混与 type→ok 误判
- 原 `static_metrics` 保持 10 维不变，RustCodingPlugin 原型分类不受影响

### 变更
- `data/real_dataset.py`：`static_signal` 扩为 10+6=16 维，恰好填满
  numeric 通路（此前 10 维留有 6 维空闲），归一化仍为语料内逐维最大值

### 真实数据评估结果
- 读出层（R1, depth=1, 5 种子）：**61.6%**（48%–72%），
  v0.8.0 为 57.6%（52%–64%）——提升 4pp，超随机基线（20%）3.1 倍
- 进展轨迹：36%（v0.7.5 词袋）→ 57.6%（v0.8.0 P0 双模态）→
  61.6%（本版 P1 结构感知）

### 质量
- 新增 tests/test_rust_data.py 2 项（结构特征语义 / 16 维信号形状与
  归一化），测试总计 106 项全绿（104 → 106）

## v0.8.0 (2026-09-14)

P0 双模态注入：消除词袋编码信息瓶颈，真实数据读出层 36% → 57.6%。

### 诊断（本次变更的依据）
- 特征通路对照（同一真实语料、同一线性读出、5 种子分层划分）：
  static_metrics 线性可分 71.2% vs 词袋 TF-IDF 40%（类间/类内距离比
  1.36 vs 0.68 负可分）——瓶颈在编码阶段而非网络
- 混淆集中在 move↔lifetime（语法位置差异被词袋丢失）与 type→ok
  （短代码词面几乎一致）

### 新增
- `TrainingSample.static_signal`：10 维 static_metrics 语法扫描特征
  （语料内逐维归一化 [0,1]，跨进程确定）
- `TrainingSample.multimodal_input()`：返回
  `{"numeric": static_signal, "text": 代码原文}` 双模态字典，供
  CubeGPT.step() / CubeFeatureExtractor.features() 直接消费

### 变更
- `training/readout.py` `run_validation`：从单模态
  PatternExtractor（词袋→DistributedFormer 水库）切换为
  CubeFeatureExtractor 双模态注入
- `training/trainer.py` `train_step`：端到端前向同步双模态；监督
  权重更新改用语法特征信号

### 真实数据评估结果
- 读出层（R1, depth=1, 5 种子）：**57.6%**（52%–64%），
  P0 前 36% ± 6——提升 21.6pp，超随机基线（20%）2.9 倍
- 端到端监督训练 5 epochs 达 20%（P0 前 16%），刚至基线未超过——
  学习规则缺陷（w_in 标量更新）待 P1 修复
- 路线图新增「36% 瓶颈提升方案」5 项（P0 双模态 / P1 结构感知编码 /
  P1 修端到端权重更新 / P2 扩语料 / P2 交叉验证），P0 已完成

### 质量
- 测试 104 项全绿，无回归

## v0.7.5 (2026-09-14)

训练数据真实化：彻底移除合成训练数据，训练/评估管道 100% 采用真实数据集基准。

### 移除 (激进真实化)
- **删除 `distributedformer/training/data_generator.py`**（合成股票训练数据集
  `StockTrainingDataset`）——框架内不再存在任何合成训练样本生成路径

### 新增
- `distributedformer/data/real_dataset.py`: **RustCodingTrainingDataset** —
  纯真实训练数据集。数据全部来自 v0.6.0 Rust coding 真实基准语料
  (100 段真实风格代码 × 5 类真实 rustc 错误族)；输入信号 =
  代码感知 TF-IDF 脉冲编码 (`encode_text`, crc32 确定性)，监督模式 =
  16 维类别位；分层 75/25 划分；携带随机基线 (20%) 与多数类基线
- 有意义的评估基线：训练入口 / CLI / 消融实验统一输出
  随机 20% 与多数类基线对照

### 变更
- `trainer.py`: `SpikeSupervisedLoss` / `DFTrainer` 由硬编码 4 类泛化为
  `n_classes` (默认 5: move/borrow/lifetime/type/ok)，思考层分组监督、
  损失、准确率、类别统计全部适配
- `training/train.py` / `training/readout.py` / `cli.py train` /
  `experiments/`（ablation_study / ablation_study_extended / run_single_exp /
  readout_validation）：全部切换至真实数据集；readout 验证基线 25% → 20%

### 真实数据评估结果
- reservoir 读出层 (R1, depth=1, 5 种子): 验证准确率 28%–44%（均值 36%），
  随机基线 20%，多数类基线 20%——全部种子超过基线，网络内部表征在
  真实 Rust 语料上携带类别信息
- 端到端注入式监督训练在 100 样本真实语料上仍低于基线，与 v5.2
  历史消融结论一致，如实报告（学习规则改进仍在路线图中）

### 质量
- 测试 104 项全绿，无回归

## v0.7.4 (2026-09-14)

KV 堆注意力检索向量化：打分与 top-k 全程 numpy 批量计算，消除逐条 Python 循环。

### 性能
- `KVStack.retrieve()`（主计算路径，每个 FractalLayer / CubeGPT 每步调用）:
  4096 条堆 54ms → 1.3ms（约 **42×**），20000 条堆（scan_limit 截取）61ms → 1.4ms
- `query()` 同步向量化（全堆扫描），`_evict_lru` 改 numpy argmin

### 变更 (`distributedformer/core/distributedformer.py`)
- `KVStack` 内部维护增量矩阵索引（`_keys/_vals/_ts/_seqs` 等，指数扩容，
  swap-remove 淘汰不产生碎片），条目对象持有矩阵行视图，单一数据源
- `entries` dict 仍为权威存储，外部直接读取 / `entries.clear()` 完全兼容
  （长度失配自动重建索引）
- scan_limit 语义保持：按插入序号（`_seqs`）截取最近写入的 N 条，
  淘汰搬移与重复 id 覆盖均不破坏插入序
- push 支持变长载荷（不足 dim 补零 / 超长截断），修复 rust 插件压入
  10 维 key 的潜在广播错误

### 质量
- 新增 tests/test_fixes.py 7 项向量化回归（与旧逐条打分参考实现数值等价 /
  scan_limit 截取 / LRU 淘汰 / 重复 id 覆盖 / 外部 clear 重建 / 存档往返），
  测试总计 104 项全绿（97 → 104）

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
