# -*- coding: utf-8 -*-
"""
实验汇总: 全面训练 · 三真实任务统一基准 (v0.12.0)

一次跑齐三个真实任务基准并汇总成一份总报告, 供发布前刷新 README 数字:
  1. 多任务基准 (multi_task_benchmark)  → Rust 编译错误族 + Markdown 内容类型
  2. 视频生成训练基准 (video_motion_benchmark) → 真实运镜运动识别

"简单": 复用既有基准入口, 不引入新训练算法。每个任务均为 5 种子 × 5 折
分层 CV (每折验证一次), 报告统一展示 任务 | 真实材料 | 类别数 | 随机基线
| val_acc | 25/25 折超基线。

运行: python -m src.experiments.benchmark_all  (或 dformer benchmark-all)
输出: src/experiments/benchmark_all_results.json
      src/experiments/benchmark_all_report.md
"""

import json
import os
import sys
import time

# src/experiments/<file> → 上溯 3 级到仓库根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _task_row(name, material, n_classes, base, acc, acc_std,
              folds, total, version, detection_rate=None):
    return {
        "task": name, "material": material, "n_classes": n_classes,
        "random_baseline": base, "val_acc_mean": acc,
        "val_acc_std": acc_std, "folds_beat_random": folds,
        "total_folds": total, "version": version,
        "detection_rate": detection_rate,
    }


def _run_or_load(name, run_fn, path, bake=True):
    """已有结果 JSON 则直接复用 (避免重复昂贵特征提取), 否则运行基准生成."""
    if os.path.exists(path) and bake:
        print(f"  复用已有 {os.path.basename(path)} (跳过重算)")
        return _load_json(path)
    print(f"  运行 {name} 基准 ...")
    run_fn()
    return _load_json(path)


def main():
    print("=" * 70)
    print("  全面训练 · 四真实任务统一基准 (v0.13.1)")
    print("  5 种子 × 5 折分层 CV × 4 真实任务 (含时序异常检测)")
    print("=" * 70)

    # 1) 多任务: Rust + Markdown
    from src.experiments.multi_task_benchmark import main as run_multi
    multi = _run_or_load(
        "多任务", run_multi,
        os.path.join(_OUT_DIR, "multi_task_results.json"))

    # 2) 视频运动识别
    from src.experiments.video_motion_benchmark import main as run_video
    video = _run_or_load(
        "视频", run_video,
        os.path.join(_OUT_DIR, "video_motion_results.json"))

    rows = []
    for t in multi.get("tasks", []):
        rows.append(_task_row(
            name=t.get("task", "?"),
            material="502 段真实 Rust 代码" if "Rust" in t.get("task", "")
                     else "仓库真实 Markdown 文档",
            n_classes=5,
            base=t.get("random_baseline", 0.2),
            acc=t.get("val_acc_mean"),
            acc_std=t.get("val_acc_std"),
            folds=t.get("folds_beat_random", 0),
            total=t.get("total_folds", 25),
            version=multi.get("version", "?"),
        ))
    rows.append(_task_row(
        name="视频运动识别",
        material="真实运镜曲线 × 10 类",
        n_classes=video.get("n_classes", 10),
        base=video.get("random_baseline", 0.1),
        acc=video.get("val_acc_mean"),
        acc_std=video.get("val_acc_std"),
        folds=video.get("folds_beat_random", 0),
        total=video.get("total_folds", 25),
        version=video.get("version", "?"),
    ))

    # 3) 真实时序异常检测 (P0 立项, v0.13.1)
    from src.experiments.anomaly_benchmark import main as run_anomaly
    anomaly = _run_or_load(
        "时序异常", run_anomaly,
        os.path.join(_OUT_DIR, "anomaly_results.json"))
    rows.append(_task_row(
        name="真实时序异常检测",
        material="NAB 真实运维指标 × 5 序列 (40 点时序窗口)",
        n_classes=2,
        base=0.5,
        acc=anomaly.get("val_acc_mean"),
        acc_std=anomaly.get("val_acc_std"),
        folds=anomaly.get("folds_beat_random", 0),
        total=anomaly.get("total_folds", 25),
        version=anomaly.get("version", "?"),
        detection_rate=anomaly.get("anomaly_detection_rate_mean"),
    ))

    summary = {
        "experiment": "benchmark_all",
        "version": "0.13.1",
        "date": time.strftime("%Y-%m-%d"),
        "protocol": "stratified_5fold_cv",
        "n_seeds": 5,
        "n_folds": 5,
        "tasks": rows,
    }

    with open(os.path.join(_OUT_DIR, "benchmark_all_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 全面训练 · 四真实任务统一基准报告 (v0.13.1)\n",
        f"日期: {summary['date']}  |  5 种子 × 5 折分层 CV × 4 真实任务"
        f"  (每任务 25 折, 每样本恰好验证一次)\n",
        "| 任务 | 真实训练材料 | 类别数 | 随机基线 | val_acc | 异常检出率 | 超基线折数 |",
        "|------|--------------|--------|----------|---------|-------------|-------------|",
    ]
    for r in rows:
        det = (f"{r['detection_rate']:.0%}" if r["detection_rate"] is not None
               else "—")
        lines.append(
            f"| {r['task']} | {r['material']} | {r['n_classes']} | "
            f"{r['random_baseline']:.0%} | "
            f"**{r['val_acc_mean']:.1%}** ± {r['val_acc_std']:.1%} | "
            f"{det} | "
            f"{r['folds_beat_random']}/{r['total_folds']} |")
    lines += [
        "\n## 说明",
        "- Rust / Markdown / 视频 三个分类任务在折级显著超过各自随机基线",
        "  (25/25 折), 验证真实训练材料数据通路与信号的可学性 (纯真实, 无合成)。",
        "- 时序异常检测为窗口级二元检测 (随机 50%): 冻结水库 + 线性读出 + 类别",
        "  加权下, 异常检出率 (recall) 从多数类基线 0% 抬升到 ~49%, 但折级",
        "  ACC 均值 48.5% 未稳定超随机 (9/25 折), 反映真实异常窗口在聚合",
        "  统计描述符下的可分性有限 (ACC 与检出率有固有权衡)。",
        "- 数字来源: multi_task_results.json (Rust+MD) / video_motion_results.json",
        "  (视频) / anomaly_results.json (时序); 由本脚本汇总。",
    ]
    with open(os.path.join(_OUT_DIR, "benchmark_all_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n四真实任务统一基准汇总:")
    for r in rows:
        det = (f", 检出 {r['detection_rate']:.1%}"
               if r["detection_rate"] is not None else "")
        print(f"  - {r['task']}: {r['val_acc_mean']:.1%} ± "
              f"{r['val_acc_std']:.1%}  (随机 {r['random_baseline']:.0%}, "
              f"{r['folds_beat_random']}/{r['total_folds']} 折超基线){det}")
    print(f"\n已写入 {_OUT_DIR}/benchmark_all_results.json 与 "
          f"benchmark_all_report.md")


if __name__ == "__main__":
    main()
