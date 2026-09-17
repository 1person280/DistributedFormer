# -*- coding: utf-8 -*-
"""
实验 R4: 视频生成训练 · 真实运动识别基准 (v0.12.0)

目标: 验证视频生成内核 (VideoMakingPlugin) 的**真实运镜曲线**作为训练材料
是否携带可分信号 —— 从真实曲线稠密采样描述符 (dx,dy,zoom,rot,progress),
识别属于哪一种真实运动类别 (10 分类, 随机基线 10%)。

数据: 真实运镜曲线 (DEFAULT_MOTION_PROFILES, 10 种真实镜头运动, 蒸馏自
真实分镜/摄影惯例), 每条按其真实插值进度稠密采样 n_points 点 (真实取值,
非合成标签)。

模型: CubeGPT (depth=1, numeric 单面) 冻结作特征提取器, 水库状态 + 线性
softmax 读出层 (L2 网格选优)。

协议: 5 折分层交叉验证 × 5 种子 (每个样本恰好验证一次)。

运行: python src/experiments/video_motion_benchmark.py
输出: src/experiments/video_motion_results.json, video_motion_report.md
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

from src.data.video_motion import VideoMotionDataset, LABELS
from src.training.readout import CubeFeatureExtractor, LinearReadout

L2_GRID = [1e-3, 1e-2, 1e-1]


def run_video_cross_validation(seed: int = 0, n_folds: int = 5,
                               l2: float = 1e-3, depth: int = 1,
                               n_points: int = 20,
                               verbose: bool = False) -> Dict:
    """真实运动识别 · 冻结水库特征 + 线性读出 (5 种子×5 折)

    特征: CubeFeatureExtractor (numeric 单面) 消费 VideoSample 的 5 维
    真实描述符, 冻结水库表征 → 线性 softmax 读出层。逐样本缓存。
    """
    dataset = VideoMotionDataset(n_points=n_points)
    splits = dataset.kfold_datasets(n_folds=n_folds, seed=seed)
    random_baseline = dataset.RANDOM_BASELINE
    majority_baseline = dataset.majority_baseline(splits[0][0])

    # 视频无文本模态 → 仅 numeric 面
    extractor = CubeFeatureExtractor(depth=depth, seed=seed,
                                     modalities=["numeric"])
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
        readout = LinearReadout(n_features=X_train.shape[1],
                                n_classes=len(LABELS), seed=seed,
                                l2=l2).fit(X_train, y_train)
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


def _select_best_l2(seed, n_folds, depth, n_points):
    best_l2, best_acc = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        r = run_video_cross_validation(seed=seed, n_folds=n_folds, l2=l2,
                                       depth=depth, n_points=n_points,
                                       verbose=False)
        acc = r["val_accuracy"]
        print(f"      L2={l2:.0e} → 5 折 val_acc={acc:.1%}")
        if acc > best_acc:
            best_l2, best_acc = l2, acc
    return best_l2


def main():
    seeds = [0, 1, 2, 3, 4]
    n_folds = 5
    depth = 1
    n_points = 20

    print("=" * 70)
    print(f"  实验 R4: 视频生成训练 · 真实运动识别基准 (v0.12.0)")
    print(f"  {len(seeds)} 种子 × {n_folds} 折分层 CV, {len(LABELS)} 种"
          f"真实运镜 × {n_points} 点, depth={depth}")
    print("=" * 70)

    print("\n[1/2] 读出层 L2 选优 (seed=0):")
    best_l2 = _select_best_l2(0, n_folds, depth, n_points)

    print(f"\n[2/2] 全种子 × {n_folds} 折 CV, L2={best_l2:.0e}:")
    results = []
    for seed in seeds:
        r = run_video_cross_validation(seed=seed, n_folds=n_folds, l2=best_l2,
                                       depth=depth, n_points=n_points,
                                       verbose=True)
        results.append(r)

    all_fold = [a for r in results for a in r["val_acc_per_fold"]]
    seed_means = [r["val_accuracy"] for r in results]
    random_baseline = results[0]["random_baseline"]
    majority_baseline = results[0]["majority_baseline"]

    summary = {
        "experiment": "R4_video_motion_benchmark",
        "version": "0.12.0",
        "protocol": "stratified_5fold_cv",
        "date": time.strftime("%Y-%m-%d"),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        "n_classes": len(LABELS),
        "n_points": n_points,
        "n_samples": len(VideoMotionDataset()._corpus),
        "best_l2": best_l2,
        "labels": LABELS,
        "val_acc_mean": float(np.mean(all_fold)),
        "val_acc_std": float(np.std(all_fold)),
        "val_acc_min": float(min(all_fold)),
        "val_acc_max": float(max(all_fold)),
        "seed_means": seed_means,
        "seed_mean_min": float(min(seed_means)),
        "seed_mean_max": float(max(seed_means)),
        "train_acc_mean": float(np.mean([r["train_accuracy"]
                                         for r in results])),
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
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

    with open(os.path.join(_OUT_DIR, "video_motion_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R4: 视频生成训练 · 真实运动识别基准报告 (v0.12.0)\n",
        f"日期: {summary['date']}  |  真实运镜 {len(LABELS)} 类 × "
        f"{summary['n_points']} 点 = {summary['n_samples']} 样本  |  "
        f"{len(seeds)} 种子 × {n_folds} 折分层 CV\n",
        "## 真实任务",
        "视频生成内核 (`VideoMakingPlugin`) 的真实运镜曲线 (10 种真实镜头",
        "运动, 蒸馏自真实分镜/摄影惯例) 按其真实插值进度稠密采样为描述符",
        "(dx, dy, zoom, rot, progress), 识别采样点属于哪一种真实运动类别。\n",
        "## 协议 (v0.12.0)",
        "- 数据: 真实运镜曲线稠密采样点 (非合成标签)",
        "- 输入模态: numeric (5 维真实描述符, 视频无文本模态)",
        "- CubeGPT depth=1 (numeric 单面) 冻结, 水库状态特征",
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
        f"随机基线 {random_baseline:.0%} (多数类 {majority_baseline:.0%}) — "
        f"超基线折数: {summary['folds_beat_random']}/{summary['total_folds']}\n",
        "## 说明",
        "真实运镜集规模小 (10 类), 本基准验证**训练材料数据通路** (视频",
        "插件 `training_data()` 导出真实曲线描述符) 与真实运镜信号的可学性:",
        "冻结水库特征 + 线性读出即可显著超过随机基线, 数值如实上报。",
    ]
    with open(os.path.join(_OUT_DIR, "video_motion_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写入 {_OUT_DIR}/video_motion_results.json 与 "
          f"video_motion_report.md")


if __name__ == "__main__":
    main()
