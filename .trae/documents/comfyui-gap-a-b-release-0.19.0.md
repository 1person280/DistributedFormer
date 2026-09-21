# 补齐 ComfyUI 差距 · A 服务端执行模型 + B 节点库（0.19.0）

## Context

`docs/发行现状/COMFYUI_GAP_ROADMAP.md` 列出通往 1.0.0 的差距。当前为 0.18.1，其中下载执行仍由**前端 JS 队列**驱动，服务端 `/api/workflow/run` 在请求线程同步跑完整个 DAG；取消仅是前端 `runAbort` 标志，服务端仍在跑；执行历史仅存于前端内存。本次按用户选定范围实现：

- **A. 服务端执行模型（最重要）**：真实服务端任务队列（提交即得 `task_id`，可查状态/结果/历史）、服务端中断/取消、输出历史持久化并可回看/复用。
- **B. 节点库**：可搜索/分类折叠的完整节点调色板、按用途分类的模板示例图库、ComfyUI 原生 JSON 导入/导出（复用 `src/blueprint.py` 的 `to_comfy`/`from_comfy`）。

不改动其它小节。版本号统一升 **0.18.1 → 0.19.0**。

---

## 一、服务端任务队列 + 中断 + 历史

### 新增 `src/deployment/task_queue.py`（新文件，<600 行，扁平语义命名）

`TaskManager` 持有 `WorkflowEngine`，用**单一后台 daemon worker 串行消费**队列（`kernel.request`/EventBus 非线程安全，必须串行；绝不同时跑两个 run）。

```python
TASK_DIR = os.path.join("cache", "tasks")      # 复用 ./cache 体系
class TaskManager:
    def __init__(self, engine): ...            # self._lock, self._tasks={}, self._queue=[], self._seq
                                               # self._stop = threading.Event(); daemon worker 线程启动
    def submit(self, workflow, name=None) -> dict  # 分配 id、入 queue、立即返回 {"ok":True,"task_id":id}
    def status(self) -> list[dict]             # 全部任务，排序 running > pending > 新→旧
    def get(self, tid) -> dict|None            # 单任务（含 results/log/error/workflow）
    def interrupt(self) -> dict                # 置 self._stop 事件 + 清待处理 queue；返回当前任务 aborted
    def load_workflow(self, tid) -> dict|None  # 复用：返回该任务存的工作流定义
    def _persist(self, tid)                    # 写 TASK_DIR/<id>.json（临时文件 + os.replace 原子写）
    @classmethod def _recover(cls, engine)     # 启动扫描 TASK_DIR 恢复已完成历史
```

任务记录字段：`id/name/state(pending|running|ok|err|aborted)/start/ms/results/log/error/workflow`。
worker 循环：取队首 → 为该任务建独立 `stop_event`（`self._stop` 触发时同步置位）→ `engine.run(workflow, stop_event=...)` → 按结果写 state → `_persist`。daemon 线程不阻塞退出。

### 改 `src/deployment/workflow_ui.py` 的 `WorkflowEngine.run`

给 `run()` 加可选参数，**保持旧同步语义与 5 个既有测试不变**：

```python
def run(self, workflow, stop_event: Optional[threading.Event] = None) -> dict:
```
Kahn `while queue:` 每轮开头：`if stop_event is not None and stop_event.is_set():` → `log.append({"node_id":"_interrupt","status":"aborted"})` + `break`。已产生的 `results/log` 照常返回，末尾附 `aborted=True`（`stop_event=None` 时无此键，行为与旧一致）。`workflow_ui.py` 顶部 import `threading`；模块 docstring 版本号改 0.19.0。

---

## 二、B 节点库

### 2.1 可搜索/分类节点调色板（前端 JS）

改造 `filterAdd(q)`（workflow_ui.py WORKFLOW_PAGE 内，L1312-1325）：

- 用 `MODELS[].capability/route` 对模型分组；`input/output/reroute` 各成一关注组。
- 按组渲染折叠标题（点击展开/收起），`#addOverlay` 内加组 `<details>`/标题行，组内仍是 `.add-item`。
- 过滤：`q` 非空时对 label+sub 做**大小写不敏感子串 + 简单打分**（命中 name>route>capability），匹配组自动展开并隐藏不匹配组。
- 维持 `addNode(it)`（L1328）不变；`MODELS` 已含 `capability/base_model/loaded`，无需新后端字段。

### 2.2 模板/示例图库（后端 PRESETS + 前端 Load 页签）

- `PRESETS` 字典增加"用途分类"字段：把每条 preset 标 `"category"`（如 `文本 / 视频 / 时序`）。新增 2-3 个真实模型预设（金额用现有内核真实插件 route，如 rust/video + 分类/时序插件）作为可一键加载示例。
- 前端 `renderLoadTab()`（L1457）把模板按 `category` 分组，`<div class="title">` 显示类别名；复用现成 `loadByName()`/`loadGraph` 一键加载。

### 2.3 ComfyUI 原生 JSON 兼容（后端复用 blueprint）

新增两个后端端点（web_ui.py do_GET/do_POST）：
- `POST /api/workflow/export_comfy` {workflow} → `to_comfy(workflow)` 的 `{nodes,links,groups,...}`（`src/blueprint.py` L83 起，纯 dict 不落盘）。
- `POST /api/workflow/import_comfy` {wf} → `from_comfy(wf)` → 内部 `{nodes,edges,groups}` → 供 `loadGraph` 直接消费。

前端：导出按钮 `exportJson` 改为可选调 `/export_comfy` 下载 ComfyUI 格式；导入侧在 `#fileIn`/画布 drop（L1053-1067）JSON 解析处判别 `links && !edges` 即 ComfyUI 格式 → 走 `/import_comfy` 再 `loadGraph(w)`（现有 `loadGraph` 复用）。

---

## 三、web_ui.py 接线 + 端点表

- `UIServer.__init__`（L55 附近）`self.tasks = TaskManager(self.workflow)`。
- `do_POST` 新增分发：
  | 路径 | 行为 |
  |---|---|
  | `/api/workflow/submit` {workflow,name} | `tasks.submit(...)` → `{ok,task_id}` 立即返回 |
  | `/api/workflow/interrupt` | `tasks.interrupt()` |
  | `/api/workflow/export_comfy` {workflow} | blueprint `to_comfy` |
  | `/api/workflow/import_comfy` {wf} | blueprint `from_comfy` |
- `do_GET` 新增：`/api/tasks`（正则 `^/api/tasks$`）→ `tasks.status()`；`/api/tasks/<id>`（正则）→ `tasks.get(id)`；若 id 不存在 404。**注意路由判断放默认 404 之前**。
- 旧 `/api/workflow/run`、`/save`、`/load` 原样保留（同步语义不动）。

---

## 四、前端：执行改为服务端队列 + 轮询 + 全局中断

改动 `WORKFLOW_PAGE` JS：

- `enqueue()`（L1386）改为 `POST /api/workflow/submit`，前端不再直接 `execWF`；把服务端 `task_id` 记入排队 item。
- `execWF(item)`（L1398）→ `pollTask(item)`：每 ~500ms `GET /api/tasks/<id>`，据返回 `log` 逐节点高亮（复用现有 `nodes.forEach` + `sleep` 动画与 `_runStatus/_result` 渲染）；`state==='ok/err/aborted'` 结束。
- 全局中断 `$('sbInterrupt').onclick`（L1484）改为 `POST /api/workflow/interrupt`（服务端真终止）；`runAbort` 仅作前端跳出动画的本地标志。
- 侧栏「最近运行」改为从 `GET /api/tasks` 服务端历史渲染（重启仍在）；点击条目可「回看输出 / 复用」（`/api/tasks/<id>` 拿 workflow 后 `loadGraph` + 展示输出）。
- 保持 Ctrl+Enter、`wfSerializeForRun`、快捷键不变。

---

## 五、新版（0.19.0）信息披露与文档

- 版本号统一（对照记忆约定）：
  - `src/__init__.py:130` `__version__ = "0.19.0"`
  - `src/cutemamen/pkg.py:43` `CORE_VERSION = "0.19.0"`
  - `src/cutemamen/plugin.py:315` `min_core_version = "0.19.0"`
  - `pyproject.toml:7` `version = "0.19.0"`
  - `src/deployment/web_ui.py:201` `server_version="DistributedFormerUI/0.19.0"`、`_UI_VERSION`(L35)、页面 badge
  - `src/deployment/workflow_ui.py:3` docstring、`_VERSION`(L44)
  - **`src/tests/test_interplugin.py:208` 断言 `min_core_version=="0.19.0"`**（否则测试崩）
- 更新 `docs/发行现状/COMFYUI_GAP_ROADMAP.md`：将 A 与 B 项移到「已完成」区块，标注 **0.19.0**。
- 更新 `docs/历史文档/CHANGELOG.md`（0.19.0 条目）。

---

## 六、测试

新增 `src/tests/test_task_queue.py`（用真实 `CuteMamenKernel` + 可控耗时假插件 `SleepPlugin.on_think=>time.sleep`，route="slow"；**每个测试独立 TaskManager/内核，避免共享竞态**）：
- `test_submit_returns_id_immediately`：submit 返回 `{ok, task_id:int}`。
- `test_state_transitions_pending_to_ok`：轮询/join 至 ok，带输出 results。
- `test_interrupt_marks_aborted_keeps_done_nodes`：运行 slow 时 interrupt → state=aborted、已执行节点仍在 results/log、`aborted` 标记存在。
- `test_interrupt_clears_queued`：入队 3 个再 interrupt → 后续排队清空。
- `test_history_persisted_to_disk`：运行后 `cache/tasks/<id>.json` 存在且字段齐；新 `TaskManager._recover` 后可 get 恢复。
- `test_sync_run_still_works`：直接 `engine.run(wf)` 语义不变（回归旧 5 测试 `test_workflow_ui.py`）。
- blueprint 兼容分支：`to_comfy`/`from_comfy` 往返一致（复用 `src/tests/test_blueprint.py` 断言模式）。

既有 `test_workflow_ui.py` 5 个测试保持全绿。

### 验证步骤
1. 跑 `python -m pytest src/tests/test_workflow_ui.py src/tests/test_task_queue.py src/tests/test_interplugin.py src/tests/test_blueprint.py`。
2. `run.ps1 -Stop` 清干净 8011（历史教训：旧进程霸占端口）→ 重启 → 硬刷新 `http://127.0.0.1:8011/workflow`。
3. 手工验收：提交流程 → 侧栏出现服务端任务 → 执行中节点高亮轮询 → 全局 Interrupt 终止 → 「最近运行」显示历史并可回看/复用 → Load 页签模板按类别分组加载 → 导出 ComfyUI JSON / 拖入 ComfyUI JSON 导入成功。
4. 核对版本号 5 处 a.e. 一致性。

---

## 风险与约束
- **kernel 非线程安全**：TaskManager 单 worker 串行 + 锁；绝不并行 run（旧 `/run` 同步端点在请求线程跑，与 worker 互斥需同一把锁——`/run` 交 `WorkflowEngine` 时也经 TaskManager 锁，最小改动：让 `/run` 也走 `task_queue` 的锁，或接受旧端点为一次性同步调用；选后者不新增锁复杂度，但提交说明勿并发 submit+旧run）。
- **页内 JS 不受 py_compile 校验**（历史教训）：改 JS 后必须真实开浏览器验证 submit/轮询/中断回归，否则静默失效。
- **持久化原子性**：`_persist` 用临时文件 + `os.replace`；`_recover` 忽略损坏 json。
- 每个 `.py` ≤600 行约束（本项目为 Rust 约束，Python 文件仍控制在 ~600 内，`task_queue.py` 独立文件满足扁平+语义命名）。