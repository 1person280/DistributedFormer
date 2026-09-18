# v0.12.0 图形化：全面训练 + Web 界面 + 发布 (GitHub Release)

## Context

v0.12.0 已含视频生成训练（实验 R4）。用户为 0.12.0 追加三项：
1. **图形化** —— 选型：**Web 图形界面**（stdlib http.server + 单页 HTML/JS，零新依赖，复用 `openai_server` 风格）。
2. **全面训练** —— 选型：**三任务统一基准 + 汇总**（新增 `dformer benchmark-all`，一次跑 Rust/MD/视频并出总报告）。
3. **发布** —— 选型：git 提交+推送 main、打 v0.12.0 tag、用 gh 建 GitHub Release 并附 sdist/whl 包。

项目约束：零第三方运行时依赖（仅 numpy）；纯真实数据；版本号 4 处已同步到 0.12.0（无需再 bump）。`gh` 已认证（1person280，repo 权限），remote = origin。

## A. 全面训练 — `dformer benchmark-all`（三任务统一基准）

新文件：`src/experiments/benchmark_all.py`
- 复用现有入口，不做新训练算法（"简单"）：
  - 调 `src.experiments.multi_task_benchmark.main()` → 产出 `multi_task_results.json`（Rust + Markdown）
  - 调 `src.experiments.video_motion_benchmark.main()` → 产出 `video_motion_results.json`（视频）
- 聚合两者 → `benchmark_all_results.json` + `benchmark_all_report.md`，含一张三任务汇总表（任务 | 真实材料 | 类别数 | 随机基线 | val_acc | 25/25 折超基线）。聚合时按各自结果文件的既有键提取（实现时先读两文件确认键名，缺键则跳过该任务并打印）。

CLI：`src/cli.py` `build_parser()` 新增 `p_bench_all = sub.add_parser("benchmark-all", help="全面训练: 三真实任务统一基准并汇总")`，`set_defaults(func=cmd_benchmark_all)`；新增 `cmd_benchmark_all(args)` 委托 `benchmark_all.main()`。

## B. Web 图形界面 — `dformer ui`

新文件：`src/deployment/web_ui.py`（stdlib 仅 `http.server` + `json` + `threading`）
- 复用 `openai_server.py` 的 `ThreadingHTTPServer`/`BaseHTTPRequestHandler` 模式。
- 路由：
  - `GET /` → 内嵌单页 HTML/JS 控制台（单字符串，无外部资源/框架）。
  - `GET /api/status` → `{version, dim, depth, plugins:[{name,route,base_model}], uptime}`（`CuteMamenKernel(dim)` 挂载 `RustCodingPlugin` + `VideoMakingPlugin`）。
  - `POST /api/think` → body `{topic, data}`，`kernel.think(...)`，返回结果 JSON。
  - `GET /api/benchmarks` → 读 `benchmark_all_results.json`（缺省回退读 `multi_task_results.json` / `video_motion_results.json`）渲染三任务表。
- 页面含：头部（版本徽章）、状态面板（内核/插件）、思考控制台（topic+data → 结果）、基准结果表（任务/基线/val_acc/折数超基线）。
- CLI：`dformer ui --host 127.0.0.1 --port 8001 --depth 1 --dim 16`。

CLI：`src/cli.py` 新增 `p_ui` 子命令 + `cmd_ui(args)` 委托 `web_ui.main(host, port, depth, dim)`。

## C. 发布 GitHub + Release + 包

1. **文档收尾**：README 在 v0.12.0 相关处补 "Web 图形界面 `dformer ui`" 与 "全面训练 `dformer benchmark-all`"（准确率表附近/CLI 文档/项目结构）；CHANGELOG v0.12.0 条目补 GUI + benchmark-all。
2. **全面训练执行**：`python -m src.experiments.benchmark_all` 产出总报告，用结果刷新 README 三任务准确率表（如有变化）。
3. **测试**：`python -m pytest src/tests/ -q`（现 234 项，确保通过）；可选为 benchmark_all / web_ui 各加 1 项冒烟测试。
4. **构建包**：`pip install build` → `python -m build`（产出 `dist/distributedformer-0.12.0-py3-none-any.whl` + `.tar.gz`）。
5. **提交推送**：`git add`（纳入 README/CHANGELOG/cli/web_ui/benchmark_all/video 相关；**排除** `.trae/`、`docs/distributed_kv_demo.md` 等无关未跟踪文件）→ commit → push main。
6. **打 tag + Release**：`git tag v0.12.0` + push tag；`gh release create v0.12.0 dist/* --title "v0.12.0 图形化" --notes "<更新日志摘要>"`。

## 验证

1. `dformer benchmark-all` → 生成 `benchmark_all_results.json`/`report.md`，三任务数字与既有报告一致。
2. `dformer ui --port 8001` → 本机 `curl http://127.0.0.1:8001/` 返回 HTML，`/api/status` 返回 JSON，POST `/api/think` 返回结果。
3. 全量测试通过（234+）。
4. `python -m build` 成功产出 dist 下两个包，`pip install dist/*.whl` 后 `dformer --version` 显示 0.12.0。
5. GitHub Release v0.12.0 可见，含构建包附件。
