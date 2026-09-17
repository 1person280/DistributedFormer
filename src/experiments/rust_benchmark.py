# -*- coding: utf-8 -*-
"""
实验 R2: Rust Coding 真实需求基准 (v0.10.1 重跑)

真实需求: 静态识别一段 Rust 代码命中哪一类编译错误
(所有权移动 / 借用冲突 / 生命周期 / 类型不匹配 / 合法代码)。
这是 lints / IDE 提示 / 自动修复工具的基础能力。

数据: distributedformer/data/rust_coding.py — 502 段真实风格 Rust
代码片段 (v0.8.4 P2 扩充), 标注真实 rustc 错误码 (E0382/E0502/
E0597/E0308/ok), 输入模态 = [text: 代码原文, numeric: 静态特征]。

模型: CubeGPT (depth=1, numeric+text 双面) 冻结作特征提取器,
水库状态 + 线性 softmax 读出层。

协议 (v0.10.1): 5 折分层交叉验证 × 5 种子 (每个样本恰好验证一次),
并在读出层上做 L2 正则网格选优 (压小样本过拟合 gap)。

运行: python src/experiments/rust_benchmark.py
输出: src/experiments/rust_results.json, src/experiments/rust_report.md
"""

import json
import os
import sys
import time

import numpy as np

# src/experiments/<file> → 上溯 3 级到仓库根 (导入 src 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# 输出固定写到脚本所在目录 (与运行 cwd 解耦)
_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.data.rust_coding import LABELS, LABEL_NAMES
from src.training.readout import run_cross_validation


# 读出层 L2 正则网格 (v0.10.1: 缓解 train≈97% vs val≈75.6% 的过拟合 gap)
L2_GRID = [1e-3, 1e-2, 1e-1]


def select_best_l2(seed: int = 0, n_folds: int = 5) -> float:
    """在单种子 5 折上比较各 L2, 返回验证均值最优的 L2"""
    best_l2, best_acc = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        r = run_cross_validation(seed=seed, n_folds=n_folds, l2=l2,
                                 verbose=False)
        acc = r["val_accuracy"]
        print(f"    L2={l2:.0e} → 5 折 val_acc={acc:.1%}")
        if acc > best_acc:
            best_l2, best_acc = l2, acc
    return best_l2


def main():
    seeds = [0, 1, 2, 3, 4]
    n_folds = 5

    print(f"{'='*70}\n  实验 R2: Rust Coding 真实需求基准 (v0.10.1 重跑)\n"
          f"  {len(seeds)} 种子 × {n_folds} 折分层 CV, 502 段真实代码 × "
          f"{len(LABELS)} 类\n{'='*70}")

    # 1) 读出层 L2 正则选优 (单种子快速扫)
    print("\n[1/2] L2 正则选优 (seed=0):")
    best_l2 = select_best_l2(seed=0, n_folds=n_folds)

    # 2) 用最优 L2 跑全种子 5 折 CV
    print(f"\n[2/2] 全种子 × {n_folds} 折 CV, L2={best_l2:.0e}:")
    results = []
    for seed in seeds:
        r = run_cross_validation(seed=seed, n_folds=n_folds, l2=best_l2,
                                 verbose=True)
        results.append(r)

    # 种子 × 折 全展平
    all_fold_accs = [a for r in results for a in r["val_acc_per_fold"]]
    seed_means = [r["val_accuracy"] for r in results]
    random_baseline = results[0]["random_baseline"]
    summary = {
        "experiment": "R2_rust_coding_benchmark",
        "version": "0.10.1",
        "protocol": "stratified_5fold_cv",
        "date": time.strftime("%Y-%m-%d"),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        "n_samples": results[0]["train_samples"] + results[0]["val_samples"],
        "best_l2": best_l2,
        "labels": LABELS,
        "val_acc_mean": float(np.mean(all_fold_accs)),
        "val_acc_std": float(np.std(all_fold_accs)),
        "val_acc_min": min(all_fold_accs),
        "val_acc_max": max(all_fold_accs),
        "seed_means": seed_means,
        "seed_mean_min": min(seed_means),
        "seed_mean_max": max(seed_means),
        "train_acc_mean": float(np.mean([r["train_accuracy"] for r in results])),
        "random_baseline": random_baseline,
        "beats_random_all_seeds": all(r["beats_random"] for r in results),
        "folds_beat_random": sum(r["folds_beat_random"] for r in results),
        "total_folds": len(seeds) * n_folds,
        "results": [{
            "seed": r["seed"], "train_accuracy": r["train_accuracy"],
            "val_accuracy": r["val_accuracy"],
            "val_acc_per_fold": r["val_acc_per_fold"],
            "beats_random": r["beats_random"],
        } for r in results],
    }
    with open(os.path.join(_OUT_DIR, "rust_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R2: Rust Coding 真实需求基准报告 (v0.10.1)\n",
        f"日期: {summary['date']}  |  样本: {summary['n_samples']} 段真实 Rust "
        f"代码 × {len(LABELS)} 类  |  {len(seeds)} 种子 × {n_folds} 折分层 CV\n",
        "## 真实需求",
        "静态识别 Rust 代码命中的编译错误类别 (所有权移动 / 借用冲突 /",
        "生命周期 / 类型不匹配 / 合法), 对应真实 rustc 错误码, 是 lints /",
        "IDE 提示 / 自动修复工具的基础能力。\n",
        "## 协议 (v0.10.1)",
        "- 数据: 502 段真实代码 (v0.8.4 P2 扩充), 每类 ~100 段",
        "- 输入模态: text (代码原文) + numeric (static_metrics 语法特征)",
        "- CubeGPT depth=1 (numeric+text 双面) 冻结, 水库状态特征",
        "- 线性 softmax 读出层 (z-score, L2 网格选优): "
        f"best_l2 = {best_l2:.0e}",
        "- 5 折分层交叉验证 × 5 种子 (每个样本恰好作为一次验证样本)\n",
        "## 结果 (折级, 最优 L2)",
        "| seed | train_acc | val_acc (5 折均值) | val_acc 逐折 |",
        "|------|-----------|--------------------|-------------|",
    ]
    for r in summary["results"]:
        folds = " ".join(f"{a:.0%}" for a in r["val_acc_per_fold"])
        lines.append(f"| {r['seed']} | {r['train_accuracy']:.1%} | "
                     f"**{r['val_accuracy']:.1%}** | {folds} |")
    lines += [
        f"\n**汇总** (全部 {summary['total_folds']} 折): "
        f"val_acc = {summary['val_acc_mean']:.1%} ± {summary['val_acc_std']:.1%} "
        f"(min {summary['val_acc_min']:.1%}, max {summary['val_acc_max']:.1%})\n",
        f"**种子均值范围**: {summary['seed_mean_min']:.1%} – "
        f"{summary['seed_mean_max']:.1%}  |  train_acc 均值: "
        f"{summary['train_acc_mean']:.1%}\n",
        f"随机基线 {random_baseline:.0%} — 超基线折数: "
        f"{summary['folds_beat_random']}/{summary['total_folds']}\n",
        "## 说明",
        f"v0.10.1 将基准脚本从 100 段单次 75/25 升级为 502 段 5 折分层 CV,"
        "并引入读出层 L2 正则选优以缓解小样本过拟合 (train≈97% vs val≈75%)。"
        f"选优 L2 = {best_l2:.0e}。",
    ]
    with open(os.path.join(_OUT_DIR, "rust_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写入 {_OUT_DIR}/rust_results.json 与 rust_report.md")


if __name__ == "__main__":
    main()
