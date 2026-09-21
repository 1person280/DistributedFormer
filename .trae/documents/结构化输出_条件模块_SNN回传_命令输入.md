# 模型接口升级：结构化输出 · 条件模块 · SNN 脉冲回传 · 命令输入

## Context（为什么做）
用户提出三项体验/架构问题：
1. **命令行不支持输入命令**——控制台只能填 topic+data 再点按钮，不能「输入指令回车执行」。
2. **模块接口名字晦涩**——模型节点输出没有统一的「类别+内容」结构。用户期望模型卡片输出带类别：**脉冲(spike)、自然语言(text)、帧(frame)、音频(audio)**，且一次可输出多种类型（如帧输出 + 文本摘要）。
3. **缺乏消息转发回传机制**——缺两样：
   - **条件模块**：数据满足条件 A 就输出到 A，满足 B 就输出到 B，多条件同时满足则**同步扇出**到多路。
   - **SNN 脉冲回传/补做**：把某路脉冲送回去重跑/补算并取回结果。

预期结果：模型输出结构化、可分类展示；工作流支持条件路由与多路同步扇出；SNN 脉冲可回传补做；控制台支持命令回车执行。

## 现状（复用点）
- 引擎拓扑执行：`WorkflowEngine.run`（src/deployment/workflow_ui.py L104-155），Kahn 排序；`inbound[nid]` 仅存源节点 id。
- 节点执行：`_run_node`（L192-229）处理 input/model/output；model 用 `engine.kernel.request({"topic":route,"data":data})`。
- JSON 化：`_json_safe`（L51-61）递归转换 ndarray，`frames` 键被丢弃。
- 节点端口/渲染：`TYPES` 与 `widgetHTML/renderNodes`（L650-723）。
- 插件返回：rust=`{label,label_name,rustc,confidence}`、video=`{summary,plan,frames(ndarray)}`、chat=`{reply,provider,persona}`。
- 脉冲载体：`SpikeMessage`（src/core/distributedformer.py L44-80，含 source_agent_id/target_agents/hops_remaining）。
- 同步请求应答：`CuteMamenKernel.request`（src/cutemamen/kernel.py L255-272）、`PluginContext.ask`（src/cutemamen/plugin.py L193-207）。
- 事件总线：`EventBus.publish/subscribe`（src/cutemamen/event_bus.py）。
- REPL 命令模式：`run_chat`（src/demos/chat.py L120-152，/reset /stats /help /exit）。

## 实施方案

### 一、结构化模型输出（类别 + 内容）
- 在 workflow 层 `_run_node` 的 model 分支，把插件返回标准化为**输出列表**：
  `[{"category":"spike|text|frame|audio", "content": ...}, ...]`，写入 `results[nid]["output"]`（保持旧字段兼容，新增 `typed` 字段）。
- 分类规则（在 `_run_node` 内一个 `_typed_outputs(route, res)` 辅助函数）：
  - rust/md_text/java → `text`（label 等）+ `spike`（logits/向量，若存在）
  - video → `frame`（summary/plan，帧经 `_json_safe` 汇总）+ `text`
  - chat → `text`
  - 兜底 → 单一 `text` 包裹原返回
- 插件层新增可复用帮助函数 `typed(category, content)`（src/cutemamen/plugin.py），供未来插件直接产出结构化输出（不强制改现有插件）。
- 前端 `workflow_ui.js`：模型卡片把 `typed` 列表渲染成带徽标（脉冲/文本/帧/音频）+ 内容预览的卡片；模型节点输出端口改为 `spike/text/frame/audio` 可视多端口（纯 UI 元数据，不影响引擎执行，端口约定沿用 `to_port=="in"` + `from_port` 命名，见项目记忆）。

### 二、条件模块节点
- **引擎小改**：`inbound[nid]` 从 `List[str]` 改为携带端口 `List[Tuple[src, from_port, to_port]]`；`out` 增加按端口条目 `out[f"{nid}:{from_port}"]`；`_incoming_data` 优先读 `out[f"{src}:{from_port}"]`，无则回退 `out[src]`（保证旧边兼容）。改动集中在 workflow_ui.py L111-122、L226-229、L69-75。
- **新节点 type="condition"**：`params.conditions` = `[{label, match}]`，`match` 为轻量谓词表达式（如 `label == "syntax"`、`data contains "error"`、`score > 0.5`），对入边数据求值；满足的每个条件写入对应输出端口 `out[f"{nid}:{label}"]`（多条件满足 → 多端口同步写，即扇出）。
- `_run_node` 增加 condition 分支：无入边 → `no_output`；多入边 → 汇聚求值。
- 前端 `TYPES` 增加 `condition`，渲染 N 个输出端口（A/B/C…，可增删）与 conditions 编辑框；连线时边 `from_port` 选具体条件端口。

### 三、SNN 脉冲回传 / 补做
- 内核新增 `CuteMamenKernel.feedback(payload, target)`（src/cutemamen/kernel.py）：用 SpikeMessage 语义把脉冲**回传**给目标插件并**补做/重跑**，返回结果；内部复用 `request`/`ctx.ask`；发布 `kernel.feedback` 事件供总线追踪。
- 新增 workflow 节点 type="feedback"：输入接某模型节点的 `spike` 输出，参数 `target`（回传目标 route/节点），执行时调 `kernel.feedback` 并把结果作为本节点输出（供下游继续消费）。
- 前端 `TYPES` 增加 `feedback` 节点（输入端 + target 参数）。

### 四、命令输入回车执行
- 控制台 `_PAGE`（src/deployment/web_ui.py）：在现有 topic+data+think 旁新增统一**命令输入框**，输入后回车即执行：
  - `/help /reset /stats /exit` 走聊天式命令（复用 run_chat 的命令解析思路）
  - 其余文本默认路由 `chat`，支持 `topic: 数据` 形式指定 route
- 新增后端 `POST /api/command`（web_ui.py do_POST）：解析命令字符串 → 调 `self.ui.think`/命令处理 → 返回结果；前端回车后回显结果。

## 关键文件
- src/deployment/workflow_ui.py（引擎端口化、condition/feedback 节点、_typed_outputs、TYPES、render）
- src/deployment/workflow_ui.js（前端节点类型、多端口、卡片徽标、命令框）
- src/deployment/web_ui.py（/api/command、_PAGE 命令输入）
- src/cutemamen/kernel.py（feedback）
- src/cutemamen/plugin.py（typed 帮助函数）
- src/tests/test_workflow_ui.py（新增 condition/feedback/typed 用例）

## 验证
- `python -m src.selfcheck`（或 `python -m src.cli test`）跑通全量测试；新增针对 condition 扇出、feedback 回传、typed 输出的单元测试。
- 重启 `.\run.cmd -Restart`，浏览器硬刷新：
  - 控制台：输入「你好」回车 → 回显 chat 自然语言；`/stats` 显示统计。
  - 工作流：搭建 input → model(chat) → condition（条件 `reply contains 你` 与 `reply contains 好` 都满足）→ 两个 output 同时收到数据（同步扇出）。
  - 模型卡片显示带徽标的结构化输出（chat→文本；video→帧+文本）。
  - 用 feedback 节点接 model 的 spike 输出回传补做，观察结果与 `kernel.feedback` 事件。
