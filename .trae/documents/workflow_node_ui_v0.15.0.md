# ComfyUI 式节点图工作流 GUI (v0.15.0)

## Context
现有 Web 控制台（`dformer ui`，[web_ui.py](file:///c:/Users/yxhcf/Desktop/DistributedFormer/src/deployment/web_ui.py)）是单页思考控制台，只能对单个 topic 调用内核。用户需要一个 **ComfyUI 式新的 GUI 页面**：在画布上拖拽节点、连线端口，把真实模型（内核插件）串联成自定义工作流，支持输入节点、模型节点、输出节点，并支持保存/加载工作流与内置样例。

约束：零第三方依赖（前端纯手写 JS/HTML/CSS、后端 stdlib http.server）；纯真实数据；平面目录结构；版本号跨 4 处一致升级（0.14.6 → 0.15.0）。

用户已确认：采用 **ComfyUI 式节点图**交互 + **保存/加载到本地 JSON + 内置样例工作流**。

## 节点 / 边 JSON Schema
```json
workflow = {
  "name": "Rust 分类", "version": 1,
  "nodes": [
    {"id":"n1","type":"input","x":40,"y":80,"params":{"topic":"rust","data":"fn main(){}"}},
    {"id":"n2","type":"model","x":320,"y":80,"params":{"route":"rust","data":null}},
    {"id":"n3","type":"output","x":620,"y":80,"params":{}}
  ],
  "edges": [
    {"id":"e1","from":"n1","from_port":"out","to":"n2","to_port":"in"},
    {"id":"e2","from":"n2","from_port":"out","to":"n3","to_port":"in"}
  ]
}
```
端口模型（每节点固定端口）：
- `input`：唯一输出端口 `out`（提供 topic + data）
- `model`：入端口 `in`（承载上游事件，可选）+ 出端口 `out`（结果）
- `output`：唯一入端口 `in`（展示结果）

## 文件改动
### 1. 新增 `src/deployment/workflow_ui.py`（核心）
- `WorkflowEngine(kernel)`
  - `list_models()`：调色板 = `kernel.plugins`（已挂载 ui-rust/ui-video，`loaded:True`，`base_model` 用类属性 `type(p).BASE_MODEL`）+ `kernel.registry`（未加载 pkg，`loaded:False`）；字段 `{name, route, base_model, capability, loaded}`。
  - `run(workflow)`：Kahn 拓扑执行。输入节点产出事件 `{topic, data}`；模型节点从唯一 `in` 边的上游取事件，`topic` = `params["route"]`（缺省继承上游 topic），`data` = `params["data"]`（null 则复用上游 data），调 `self.kernel.request(event)`，结果并回事件形 `{topic, data:result}` 供链式互喂；`None` → log 记 `no_output` 继续。输出节点取上游 data。环/缺节点 → `{ok:false,error}`。返回 `{ok:true, results:{id:value}, log:[{node_id,status,ms}]}`；结果做 JSON 安全化（复用 web_ui `_json_default` 思路）。
  - `save(name, workflow)` / `load(name)` / `list_workflows()`：存 `./cache/workflows/<name>.json`（name 清洗非 `[-_.\w]` 字符防路径穿越）；内置只读预设；同名时读优先用户文件。
  - `PRESETS = {"Rust 代码分类": {...}, "Video 渲染": {...}}`（预设只含 input→model→output 节点图）。
- `WORKFLOW_PAGE`（HTML/JS/CSS 字符串）：暗色主题（沿用 `--bg:#0f1420 --acc:#5b8cff`），画布平移/缩放、节点拖拽、端口连线贝塞尔、调色板、编辑面板、Run/Save/Load/Clear、per-node 结果浮层；页顶 nav 切换「控制台 | 工作流」。

### 2. 修改 `src/deployment/web_ui.py`
- `UIServer.__init__` 加 `self.workflow = WorkflowEngine(self.kernel)`。
- `_Handler` 追加路由：
  - `GET /workflow` → `WORKFLOW_PAGE`
  - `GET /api/workflow/models` → `{ok, models}`
  - `POST /api/workflow/run` → `{ok, results, log}` 或 `{ok:false, error}`
  - `GET /api/workflow/list` → `{ok, workflows, presets}`
  - `POST /api/workflow/save` → `{name, workflow}` → `{ok}`
  - `POST /api/workflow/load` → `{name}` → `{ok, workflow, is_preset}`
- 现有 `/` 页面顶加 nav 切换到 `/workflow`（workflow 页同样有 nav 映回）。

### 3. 版本号 0.14.6 → 0.15.0（4 处核心 + 一致性）
- [pyproject.toml:7](file:///c:/Users/yxhcf/Desktop/DistributedFormer/pyproject.toml) `version`
- [src/__init__.py:130](file:///c:/Users/yxhcf/Desktop/DistributedFormer/src/__init__.py) `__version__`
- [src/cutemamen/pkg.py:43](file:///c:/Users/yxhcf/Desktop/DistributedFormer/src/cutemamen/pkg.py) `CORE_VERSION`
- [src/cutemamen/plugin.py:315](file:///c:/Users/yxhcf/Desktop/DistributedFormer/src/cutemamen/plugin.py) `"min_core_version"`
- 一致性：`web_ui.py` 的 `_UI_VERSION` 与 `server_version="DistributedFormerUI/0.12.0"`、`_PAGE` 标题里的 `v0.12.0` 同步为 0.15.0；`cli.py:161/268` 的 `cmd_ui`/`p_ui` help 里的 `v0.12.0` 同步。

### 4. 新增 `src/tests/test_workflow_ui.py`（pytest）
- 真实 `CuteMamenKernel` + mount `RustCodingPlugin`，跑 `input→rust→output`，断言结果含 `label`。
- 语义：input 无 route 的输出节点 → log 记 `no_output`；自环/双节点环 → `{ok:false,error}`。
- 断言 `PRESETS` 加载、`list_models()` 含 `rust`。

## 验证
- 启动 `dformer ui --port 8001 --depth 1 --dim 16`，开 `http://127.0.0.1:8001/workflow`。
- 手工：建 input（topic=rust, data=`fn main(){}`）+ model(route=rust) + output，连线 → Run → output 显示 `{"label":"ok",...}`。切「Rust 代码分类」预设验证内置样例；Save→Load 验证落盘回读。
- pytest：`python -m pytest src/tests/test_workflow_ui.py -q`；继而跑全量 `python -m pytest src/tests -q` 确认 251+ 用例不回归。
- （可选）README/CHANGELOG 追加 v0.15.0 条目（沿用过往先例）。