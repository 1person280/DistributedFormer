"""
DistributedFormer 命令行入口

子命令:
  demo      股票监控端到端演示 (模拟数据)
  serve     长驻监控服务 (容器/生产部署入口, 支持 --realtime 真实行情)
  train     运行合成数据训练
  test      运行各模块自检
"""

import argparse
import os
import sys
import time
from datetime import datetime


def cmd_demo(args):
    from distributedformer.demos.stock_monitor import StockMonitorWorkflow

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
    from distributedformer.demos.stock_monitor import StockMonitorWorkflow

    data_source = None
    if args.realtime:
        from distributedformer.demos.stock_monitor import YFinanceDataSource
        data_source = YFinanceDataSource(args.tickers)
        print(f"[serve] 数据源: yfinance (真实行情)")
    else:
        print("[serve] 数据源: 内置模拟器 (加 --realtime 切换真实行情)")

    workflow = StockMonitorWorkflow(tickers=args.tickers, data_source=data_source)
    report_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    print(f"[serve] 监控标的: {', '.join(args.tickers)}")
    print(f"[serve] 轮询间隔: {args.interval}s  报告目录: {report_dir}")
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


def cmd_train(args):
    from distributedformer.training.data_generator import StockTrainingDataset
    from distributedformer.training.trainer import DFTrainer

    dataset = StockTrainingDataset(dim=16, seed=args.seed)
    train, val = dataset.generate_dataset(
        samples_per_class=args.samples_per_class, train_ratio=0.8
    )
    trainer = DFTrainer(depth=args.depth, dim=16, learning_rate=args.lr)
    summary = trainer.train(train, val, epochs=args.epochs, save_dir=args.save_dir)
    report = trainer.generate_training_report()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "training_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n最佳验证准确率: {summary['best_val_accuracy']:.2%}")


def cmd_test(args):
    from distributedformer.selfcheck import run_full_test_suite
    run_full_test_suite()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="distributedformer",
        description="DistributedFormer — 事件驱动脉冲神经网络智能体框架",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__import__('distributedformer').__version__}")
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
    p_serve.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="合成数据训练")
    p_train.add_argument("--depth", type=int, default=1, choices=[0, 1, 2])
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--samples-per-class", type=int, default=50)
    p_train.add_argument("--lr", type=float, default=0.008)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--save-dir", default="training/checkpoints")
    p_train.set_defaults(func=cmd_train)

    p_test = sub.add_parser("test", help="运行模块自检")
    p_test.set_defaults(func=cmd_test)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
