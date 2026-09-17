# -*- coding: utf-8 -*-
"""
实验 R3: 多任务真实基准 (CubeGPT 端到端可学习 · 学习规则改进)

目标: 在 ≥2 个真实任务上显著超过随机基线 (> 随机 +5pct), 并对比多数类基线。
Rust 任务的 token 特征经由 64 比特分类式 token 嵌入消化的路径由
DFTrainer 端到端承担; 本基准为稳定读出层证据通道。

两个真实任务 (纯真实, 无合成):
  Task 1 · Rust 编译错误族      (move/borrow/lifetime/type/ok, 5 类, 基线 20%)
  Task 2 · Markdown 内容类型    (heading/code/list/table/paragraph, 5 类, 基线 20%)
           —— 语料为仓库真实多字节 utf8-mb4 文档, 经 class-token 嵌入
           (MarkdownTextDataset.input_signal = embed_tokens(token_seq))

协议: 5 种子 × 5 折分层交叉验证, 读出层 L2 网格选优, 与实验 R2 逐位一致。

运行: python src/experiments/multi_task_benchmark.py
输出: src/experiments/multi_task_results.json, multi_task_report.md
"""

import json
import os
import sys
import time
from typing import Dict

import numpy as np

# src/experiments/<file> → 上溯 3 级到仓库根 (导入 src 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.data.md_text import MarkdownTextDataset
from src.data.rust_coding import LABELS as RUST_LABELS
from src.training.readout import (
    CubeFeatureExtractor, LinearReadout, run_cross_validation)


L2_GRID = [1e-3, 1e-2, 1e-1]


def run_md_cross_validation(seed: int = 0, n_folds: int = 5,
                            l2: float = 1e-3, depth: int = 1,
                            verbose: bool = False) -> Dict:
    """Task 2: Markdown 内容类型 · token 嵌入特征 + 线性读出 (5 种子×5 折)

    特征: CubeFeatureExtractor 消费 MarkdownTextDataset 的双模态输入
    (numeric = 64 比特 class-token 折叠脉冲, text = 原始行), 冻结水库
    表征 → 线性 softmax 读出层。逐样本缓存, 每样本仅提取一次。
    """
    dataset = MarkdownTextDataset(dim=16)
    splits = dataset.kfold_datasets(n_folds=n_folds, seed=seed)
    random_baseline = dataset.RANDOM_BASELINE
    majority_baseline = dataset.majority_baseline(splits[0][0])

    extractor = CubeFeatureExtractor(depth=depth, seed=seed)
    feat_cache: Dict[str, np.ndarray] = {}

    def _feat(s) -> np.ndarray:
        if s.sample_id not in feat_cache:
            feat_cache[s.sample_id] = extractor.features(s.multimodal_input())
        return feat_cache[s.sample_id]

    fold_results = []
    for k, (train, val) in enumerate(splits):
        X_train = np.stack([_feat(s) for s in train])
        X_val = np.stack([_feat(s) for s in val])
        y_train = np.array([s.category for s in train])
        y_val = np.array([s.category for s in val])
        readout = LinearReadout(n_features=X_train.shape[1], n_classes=5,
                                seed=seed, l2=l2).fit(X_train, y_train)
        val_acc = readout.accuracy(X_val, y_val)
        train_acc = readout.accuracy(X_train, y_train)
        fold_results.append({"fold": k,
                             "train_accuracy": train_acc,
                             "val_accuracy": val_acc,
                             "majority_baseline": majority_baseline})
        if verbose:
            print(f"    seed={seed} fold={k + 1}/{n_folds} "
                  f"train_acc={train_acc:.2%} val_acc={val_acc:.2%} "
                  f"(随机 {random_baseline:.0%})")

    accs = [f["val_accuracy"] for f in fold_results]
    mean_acc = float(np.mean(accs))
    return {
        "seed": seed, "n_folds": n_folds,
        "train_accuracy": float(np.mean([f["train_accuracy"]
                                         for f in fold_results])),
        "val_accuracy": mean_acc,
        "val_acc_per_fold": accs,
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "beats_random": mean_acc > random_baseline + 0.05,
        "beats_majority": mean_acc > majority_baseline,
        "folds_beat_random": int(sum(a > random_baseline + 0.05 for a in accs)),
        "fold_results": fold_results,
    }


def _select_best_l2(runner, seed, n_folds, depth):
    best_l2, best_acc = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        r = runner(seed=seed, n_folds=n_folds, l2=l2, depth=depth,
                   verbose=False)
        acc = r["val_accuracy"]
        print(f"      L2={l2:.0e} → 5 折 val_acc={acc:.1%}")
        if acc > best_acc:
            best_l2, best_acc = l2, acc
    return best_l2


def _run_task(name, runner, seeds, n_folds, depth):
    print(f"\n[{name}] L2 选优 (seed=0):")
    best_l2 = _select_best_l2(runner, 0, n_folds, depth)
    results = []
    print(f"[{name}] 全种子 × {n_folds} 折, L2={best_l2:.0e}:")
    for seed in seeds:
        r = runner(seed=seed, n_folds=n_folds, l2=best_l2, depth=depth,
                   verbose=True)
        results.append(r)
    all_fold = [a for r in results for a in r["val_acc_per_fold"]]
    seed_means = [r["val_accuracy"] for r in results]
    return {
        "task": name,
        "best_l2": best_l2,
        "random_baseline": results[0]["random_baseline"],
        "majority_baseline": results[0]["majority_baseline"],
        "val_acc_mean": float(np.mean(all_fold)),
        "val_acc_std": float(np.std(all_fold)),
        "val_acc_min": float(min(all_fold)),
        "val_acc_max": float(max(all_fold)),
        "seed_means": seed_means,
        "seed_mean_min": float(min(seed_means)),
        "seed_mean_max": float(max(seed_means)),
        "train_acc_mean": float(np.mean([r["train_accuracy"]
                                         for r in results])),
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


def main():
    seeds = [0, 1, 2, 3, 4]
    n_folds = 5
    depth = 1

    print("=" * 70)
    print("  实验 R3: 多任务真实基准 (2 真实任务显著超随机)")
    print(f"  {len(seeds)} 种子 × {n_folds} 折分层 CV × 2 任务, depth={depth}")
    print("=" * 70)

    # 任务 1: Rust (复用 readout.run_cross_validation)
    rust = _run_task("Rust 编译错误族", run_cross_validation,
                     seeds, n_folds, depth)
    # 任务 2: Markdown (token 嵌入特征)
    md_task = _run_task("Markdown 内容类型", run_md_cross_validation,
                        seeds, n_folds, depth)

    summary = {
        "experiment": "R3_multi_task_benchmark",
        "version": "0.11.0",
        "protocol": "stratified_5fold_cv",
        "date": time.strftime("%Y-%m-%d"),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        "tasks": [rust, md_task],
    }

    with open(os.path.join(_OUT_DIR, "multi_task_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R3: 多任务真实基准报告 (v0.11.0)\n",
        f"日期: {summary['date']}  |  {len(seeds)} 种子 × {n_folds} 折分层 CV "
        f"× {len(summary['tasks'])} 个真实任务  |  CubeGPT 端到端可学习 / "
        f"学习规则改进\n",
        "## 真实任务",
        "1. **Rust 编译错误族** (move/borrow/lifetime/type/ok, 502 段真实代码)",
        "2. **Markdown 内容类型** (heading/code/list/table/paragraph, "
        "仓库真实多字节 utf8-mb4 文档行, 经 64 比特 class-token 嵌入)\n",
        "目标: 两个任务均显著超过随机基线 (随机 +5pct)。\n",
        "## 结果 (折级)",
        "| 任务 | val_acc (折均值±std) | 区间[min,max] | 随机基线 | 多数类基线 | "
        "超随机折数 | train_acc |",
        "|------|----------------------|---------------|----------|------------|"
        "-----------|-----------|",
    ]
    for t in summary["tasks"]:
        lines.append(
            f"| **{t['task']}** | {t['val_acc_mean']:.1%} ± {t['val_acc_std']:.1%} "
            f"| {t['val_acc_min']:.1%}–{t['val_acc_max']:.1%} "
            f"| {t['random_baseline']:.0%} | {t['majority_baseline']:.0%} "
            f"| {t['folds_beat_random']}/{t['total_folds']} "
            f"| {t['train_acc_mean']:.1%} |")
    lines += [
        "",
        "种子均值范围 + L2 选优: 见 multi_task_results.json。",
    ]
    with open(os.path.join(_OUT_DIR, "multi_task_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写入 {_OUT_DIR}/multi_task_results.json 与 multi_task_report.md")


if __name__ == "__main__":
    main()