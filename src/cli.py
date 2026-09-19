"""
DistributedFormer 命令行入口

子命令:
  demo          股票监控端到端演示 (模拟数据)
  serve         长驻监控服务 (容器/生产部署入口, 支持 --realtime 真实行情)
  serve-opencode 启动 OpenAI 兼容接口服务器, 让 OpenCode 把 CubeGPT 当模型后端
  train         真实数据训练 (Rust / Markdown / 时序异常检测, --dataset)
  benchmark-multi 多任务真实基准 (Rust + Markdown, 2 真实任务显著超随机)
  benchmark-anomaly 真实时序异常检测基准 (P0, 窗口级二元检测, R5)
  benchmark-stream 时序流在线持续学习 漂移/稳定性验证 (P1, R6)
  test          运行各模块自检
  chat          终端聊天: 与 CubeGPT 对话
"""

import argparse
import os
import sys
import time
from datetime import datetime


def cmd_demo(args):
    from src.demos.stock_monitor import StockMonitorWorkflow

    workflow = StockMonitorWorkflow(tickers=args.tickers)
    workflow.run_simulation(num_cycles=args.cycles, interval_sec=args.interval)
    report = workflow.generate_summary_report()
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"stock_monitor_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 总结报告已保存: {path}")


def cmd_serve(args):
    """长驻监控服务: 周期性执行感知→推理→动作循环, 可写入报告目录。

    这是容器部署的默认入口 (替代旧版不存在的 `python main.py server`)。
    """
    from src.demos.stock_monitor import StockMonitorWorkflow

    data_source = None
    if args.realtime:
        from src.demos.stock_monitor import YFinanceDataSource
        data_source = YFinanceDataSource(args.tickers)
        print(f"[serve] 数据源: yfinance (真实行情)")
    else:
        print("[serve] 数据源: 内置模拟器 (加 --realtime 切换真实行情)")

    workflow = StockMonitorWorkflow(tickers=args.tickers, data_source=data_source,
                                    kv_backend=args.kv_backend)
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    print(f"[serve] 监控标的: {', '.join(args.tickers)}")
    print(f"[serve] 轮询间隔: {args.interval}s  报告目录: {report_dir}")
    print(f"[serve] KV 后端: {args.kv_backend}"
          f" ({workflow.engine.global_kv.get_stats().get('backend', 'memory')})")
    print("[serve] Ctrl+C 停止")

    cycle = 0
    try:
        while True:
            cycle += 1
            started = time.time()
            try:
                workflow.run_cycle()
            except Exception as e:
                print(f"[serve] 周期 {cycle} 异常: {e}", file=sys.stderr)
            elapsed = time.time() - started
            print(f"[serve] 周期 {cycle} 完成, 耗时 {elapsed:.2f}s")
            time.sleep(max(0.0, args.interval - elapsed))
            # 每 12 个周期归档一次摘要报告
            if cycle % 12 == 0:
                path = os.path.join(report_dir, f"summary_{datetime.now():%Y%m%d_%H%M}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(workflow.generate_summary_report())
                print(f"[serve] 摘要报告已写入 {path}")
    except KeyboardInterrupt:
        print(f"\n[serve] 已停止, 共运行 {cycle} 个周期")


def cmd_serve_opencode(args):
    """OpenAI 兼容接口服务器 — 让 OpenCode 把 CubeGPT 当编码模型后端"""
    from src.deployment.openai_server import run_openai_server
    run_openai_server(host=args.host, port=args.port,
                      depth=args.depth, dim=args.dim,
                      api_key=None if args.no_auth else args.api_key,
                      lm_base=args.lm_base, lm_key=args.lm_key,
                      lm_model=args.lm_model)


def cmd_train(args):
    from src.data.real_dataset import RustCodingTrainingDataset
    from src.training.trainer import DFTrainer

    if args.dataset == "md":
        # Task 2: Markdown 内容类型 — 64 比特 class-token 序列接入可训练嵌入层
        from src.data.md_text import MarkdownTextDataset
        dataset = MarkdownTextDataset(dim=16)
        train, val = dataset.generate_dataset(train_ratio=0.75, seed=args.seed)
        use_token_input = True
    elif args.dataset == "ts":
        # Task 3: 真实时序异常检测 — 时间顺序前后划分 (在线持续学习, v0.13.1)
        from src.data.metrics_time_series import OperationalMetricsDataset
        dataset = OperationalMetricsDataset()
        train, val = dataset.split_stream(train_ratio=0.75)
        use_token_input = False
    else:
        # Task 1: Rust 编译错误族 (默认)
        dataset = RustCodingTrainingDataset(dim=16)
        train, val = dataset.generate_dataset(train_ratio=0.75, seed=args.seed)
        use_token_input = False
    trainer = DFTrainer(depth=args.depth, dim=16, n_classes=dataset.N_CLASSES,
                        learning_rate=args.lr,
                        freeze_reservoir_epoch=args.freeze_reservoir_epoch,
                        lr_schedule=args.lr_schedule,
                        step_drop_epoch=args.step_drop_epoch,
                        use_token_input=use_token_input)
    summary = trainer.train(train, val, epochs=args.epochs, save_dir=args.save_dir)
    report = trainer.generate_training_report()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "training_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n最佳验证准确率: {summary['best_val_accuracy']:.2%}")
    print(f"随机基线: {dataset.RANDOM_BASELINE:.0%}")


def cmd_benchmark_multi(args):
    """多任务真实基准: 2 真实任务 5 种子 × 5 折 CV, 验证显著超随机基线."""
    from src.experiments.multi_task_benchmark import main as run_benchmark
    run_benchmark()


def cmd_benchmark_all(args):
    """全面训练: 四真实任务统一基准 (Rust/Markdown/视频/时序异常) 并汇总"""
    from src.experiments.benchmark_all import main as run_benchmark_all
    run_benchmark_all()


def cmd_benchmark_anomaly(args):
    """真实时序异常检测基准 (P0): 5 种子 × 5 折窗口级二元检测 (v0.13.1)"""
    from src.experiments.anomaly_benchmark import main as run_anomaly
    run_anomaly()


def cmd_benchmark_stream(args):
    """时序流在线持续学习 漂移/稳定性验证 (P1, v0.13.1)"""
    from src.experiments.anomaly_stream_stability import main as run_stream
    run_stream()


def cmd_chat(args):
    """终端聊天: 与 CubeGPT 对话 (v0.7.1)"""
    from src.demos.chat import run_chat
    run_chat(depth=args.depth, dim=args.dim)


def cmd_ui(args):
    """Web 图形界面: 本地浏览器控制台 + 节点图工作流 (v0.16.0)"""
    from src.deployment.web_ui import main as run_ui
    run_ui(host=args.host, port=args.port, depth=args.depth, dim=args.dim)


def cmd_test(args):
    from src.selfcheck import run_full_test_suite
    run_full_test_suite()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="distributedformer",
        description="DistributedFormer — 事件驱动脉冲神经网络智能体框架",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__import__('src').__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="股票监控端到端演示 (模拟数据)")
    p_demo.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "NVDA"])
    p_demo.add_argument("--cycles", type=int, default=10)
    p_demo.add_argument("--interval", type=float, default=0.5)
    p_demo.add_argument("--output-dir", default=".")
    p_demo.set_defaults(func=cmd_demo)

    p_serve = sub.add_parser("serve", help="长驻监控服务 (容器部署入口)")
    p_serve.add_argument("--tickers", nargs="+", default=["AAPL", "TSLA", "NVDA"])
    p_serve.add_argument("--interval", type=float, default=300.0, help="轮询间隔秒数")
    p_serve.add_argument("--realtime", action="store_true", help="使用 yfinance 真实行情")
    p_serve.add_argument("--output-dir", default=".")
    p_serve.add_argument("--kv-backend", default="auto", choices=["memory", "redis", "auto"],
                         help="KV 后端: memory/redis/auto (auto 读 KV_BACKEND, 默认 memory)")
    p_serve.set_defaults(func=cmd_serve)

    p_opencode = sub.add_parser(
        "serve-opencode",
        help="启动 OpenAI 兼容接口服务器 (让 OpenCode 把 CubeGPT 当模型后端)")
    p_opencode.add_argument("--host", default="127.0.0.1")
    p_opencode.add_argument("--port", type=int, default=8000)
    p_opencode.add_argument("--depth", type=int, default=2, choices=[0, 1, 2])
    p_opencode.add_argument("--dim", type=int, default=16)
    p_opencode.add_argument(
        "--api-key", default=None,
        help=f"API 密钥 (请求头 Authorization: Bearer <key>; 默认 "
             f"cubegpt-local-test-key)")
    p_opencode.add_argument(
        "--no-auth", action="store_true",
        help="关闭 API 密钥校验 (仅推荐纯本地/无外网环境)")
    p_opencode.add_argument(
        "--lm-base", default=None,
        help="本地 OpenAI 兼容生成后端 baseURL (如 http://127.0.0.1:11434/"
             "v1); 缺省读环境变量 CUBEGPT_LM_BASE")
    p_opencode.add_argument(
        "--lm-key", default=None,
        help="后端 Bearer key (Ollama 默认为 ollama); 缺省读 CUBEGPT_LM_KEY")
    p_opencode.add_argument(
        "--lm-model", default=None,
        help="后端模型名 (如 qwen2.5-coder:7b); 缺省读 CUBEGPT_LM_MODEL")
    p_opencode.set_defaults(func=cmd_serve_opencode)

    p_train = sub.add_parser("train",
                             help="真实数据训练 (Rust / Markdown / 时序异常检测, --dataset)")
    p_train.add_argument("--depth", type=int, default=1, choices=[0, 1, 2])
    p_train.add_argument("--dataset", default="rust", choices=["rust", "md", "ts"],
                         help="真实数据集: rust=Rust 编译错误族, md=Markdown 内容类型"
                              " (经 64 比特 class-token 端到端嵌入), ts=真实时序异常检测"
                              " (NAB 真实运维指标窗口, 时间顺序前后划分)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=0.008)
    p_train.add_argument("--seed", type=int, default=42, help="分层划分种子")
    p_train.add_argument("--save-dir", default="src/training/checkpoints")
    p_train.add_argument("--freeze-reservoir-epoch", type=int, default=None,
                         help="冻结水库epoch (仅调输出头, 同离线读出范式)")
    p_train.add_argument("--lr-schedule", default="cosine",
                         choices=["constant", "cosine", "step"],
                         help="w_in 学习率调度 (cosine 后期降低, 缓解端到端后期漂移)")
    p_train.add_argument("--step-drop-epoch", type=int, default=None,
                         help="step 调度 LR 掉落点")
    p_train.set_defaults(func=cmd_train)

    p_bench_multi = sub.add_parser(
        "benchmark-multi",
        help="多任务真实基准 (2 真实任务 5 种子 × 5 折 CV, 显著超随机基线)")
    p_bench_multi.set_defaults(func=cmd_benchmark_multi)

    p_bench_anomaly = sub.add_parser(
        "benchmark-anomaly", help="真实时序异常检测基准 (P0, 窗口级二元检测, R5)")
    p_bench_anomaly.set_defaults(func=cmd_benchmark_anomaly)

    p_bench_stream = sub.add_parser(
        "benchmark-stream",
        help="时序流在线持续学习 漂移/稳定性验证 (P1, R6)")
    p_bench_stream.set_defaults(func=cmd_benchmark_stream)

    p_bench_all = sub.add_parser(
        "benchmark-all",
        help="全面训练: 四真实任务统一基准 (Rust/Markdown/视频/时序异常) 并汇总 (v0.13.1)")
    p_bench_all.set_defaults(func=cmd_benchmark_all)

    p_chat = sub.add_parser("chat", help="终端聊天: 与 CubeGPT 对话")
    p_chat.add_argument("--depth", type=int, default=2, choices=[0, 1, 2],
                        help="CubeGPT 深度 (默认 2)")
    p_chat.add_argument("--dim", type=int, default=16)
    p_chat.set_defaults(func=cmd_chat)

    p_ui = sub.add_parser(
        "ui", help="Web 图形界面: 控制台 + 节点图工作流 (v0.16.0)")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8011)
    p_ui.add_argument("--depth", type=int, default=1, choices=[0, 1, 2])
    p_ui.add_argument("--dim", type=int, default=16)
    p_ui.set_defaults(func=cmd_ui)

    p_test = sub.add_parser("test", help="运行模块自检")
    p_test.set_defaults(func=cmd_test)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
