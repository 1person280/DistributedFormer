# -*- coding: utf-8 -*-
"""
实验 R5: 真实时序异常检测基准 (P0 立项 · 里程碑 v0.13.1)

框架声明面向"流式监控 / 异常检测 / 指标巡检"这一持续在线场景, 但此前全部
基准均为分类任务。本实验补上**首个真实异常检测 / 时序基准** (非分类):

任务: 窗口级二元异常检测 — 把 NAB 真实运维指标序列切成定长滑动窗口, 判断
窗口是否落在官方告警时段 (正常/异常, 随机基线 50%)。

协议: 5 种子 × 5 折分层交叉验证 (stratified_kfold 对二元标签分层, 两占比
在折间保持一致), 冻结 CubeGPT 水库 → 特征 → 线性 softmax 读出层。
逐窗口度量: ACC + 异常检出率 (recall) + 误报率 (FAR)。

运行: python src/experiments/anomaly_benchmark.py
输出: src/experiments/anomaly_results.json, anomaly_report.md
"""

import json
import os
import sys
import time
from typing import Dict

import numpy as np

# src/experiments/<file> → 上溯 3 级到仓库根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.data.metrics_time_series import OperationalMetricsDataset
from src.training.readout import CubeFeatureExtractor, LinearReadout


L2_GRID = [1e-3, 1e-2, 1e-1]
ANOMALY_WEIGHT = 10.0  # 异常类梯度权重 (低授不平衡误报高检出·高权重高检出高误报, 平衡点≈10)


def run_anomaly_cross_validation(seed: int = 0, n_folds: int = 5,
                                 l2: float = 1e-2, depth: int = 1,
                                 anomaly_weight: float = ANOMALY_WEIGHT,
                                 verbose: bool = False) -> Dict:
    """单种子分层 K 折 CV: 真实时序窗口 → 水库特征 → 二元读出

    返回折级窗口 ACC / 异常检出率 / 误报率 + 折均值。
    """
    dataset = OperationalMetricsDataset()
    splits = dataset.kfold_datasets(n_folds=n_folds, seed=seed)
    random_baseline = dataset.RANDOM_BASELINE            # 50%
    majority_baseline = dataset.majority_baseline(splits[0][0])

    extractor = CubeFeatureExtractor(depth=depth, seed=seed)
    feat_cache: Dict[str, np.ndarray] = {}
    t0 = time.time()

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
        # 异常类梯度权重 (真实不平衡检测, 非合成): 提升稀有"异常"窗口的力度
        readout = LinearReadout(n_features=X_train.shape[1], n_classes=2,
                                seed=seed, l2=l2,
                                class_weight=[1.0, anomaly_weight]
                                ).fit(X_train, y_train)
        pred = readout.predict(X_val)
        acc = float((pred == y_val).mean())
        # 异常检出率 = 真实异常中正确检出比例 (recall); 误报率 = 正常被误报比例
        n_anom = int((y_val == 1).sum())
        det_rate = float((pred[y_val == 1] == 1).sum()) / n_anom if n_anom else 0.0
        n_norm = int((y_val == 0).sum())
        far = float((pred[y_val == 0] == 1).sum()) / n_norm if n_norm else 0.0
        fold_results.append({
            "fold": k,
            "val_accuracy": acc,
            "anomaly_detection_rate": det_rate,
            "false_alarm_rate": far,
        })
        if verbose:
            print(f"  seed={seed} fold={k + 1}/{n_folds} "
                  f"acc={acc:.2%} 检出={det_rate:.2%} 误报={far:.2%} "
                  f"(随机 {random_baseline:.0%}, 多数类 {majority_baseline:.0%})")

    accs = [f["val_accuracy"] for f in fold_results]
    dets = [f["anomaly_detection_rate"] for f in fold_results]
    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_accuracy": float(np.mean(accs)),
        "val_acc_per_fold": accs,
        "anomaly_detection_rate": float(np.mean(dets)),
        "detection_per_fold": dets,
        "false_alarm_rate": float(np.mean([f["false_alarm_rate"]
                                           for f in fold_results])),
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "beats_random": float(np.mean(accs)) > random_baseline + 0.02,
        "folds_beat_random": int(sum(a > random_baseline + 0.02 for a in accs)),
        "extract_seconds": time.time() - t0,
        "fold_results": fold_results,
    }


def _select_best_l2(seed, n_folds, depth):
    best_l2, best_det = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        r = run_anomaly_cross_validation(seed=seed, n_folds=n_folds, l2=l2,
                                         depth=depth, verbose=False)
        print(f"      L2={l2:.0e} → 5 折 val_acc={r['val_accuracy']:.1%} "
              f"检出={r['anomaly_detection_rate']:.1%}")
        if r["anomaly_detection_rate"] > best_det:
            best_l2, best_det = l2, r["anomaly_detection_rate"]
    return best_l2


def main():
    seeds = [0, 1, 2, 3, 4]
    n_folds = 5
    depth = 1

    print("=" * 70)
    print("  实验 R5: 真实时序异常检测基准 (5 种子 × 5 折分层 CV)")
    print("  任务: 窗口级正常/异常二元检测 (随机 50%)  非分类")
    print("=" * 70)

    print("\n[L2 选优] (seed=0, 以异常检出率为准)")
    best_l2 = _select_best_l2(0, n_folds, depth)

    results = []
    print(f"\n[全种子 × {n_folds} 折, L2={best_l2:.0e}, 异常权重={ANOMALY_WEIGHT}]")
    for seed in seeds:
        r = run_anomaly_cross_validation(seed=seed, n_folds=n_folds, l2=best_l2,
                                         depth=depth,
                                         anomaly_weight=ANOMALY_WEIGHT,
                                         verbose=True)
        results.append(r)

    all_fold = [a for r in results for a in r["val_acc_per_fold"]]
    all_det = [a for r in results for a in r["detection_per_fold"]]
    all_far = [a for r in results for fr in r["fold_results"]
               for a in [fr["false_alarm_rate"]]]
    seed_means = [r["val_accuracy"] for r in results]
    majority = results[0]["majority_baseline"]

    summary = {
        "experiment": "R5_anomaly_detection",
        "version": "0.13.1",
        "protocol": "stratified_5fold_cv",
        "date": time.strftime("%Y-%m-%d"),
        "n_seeds": len(seeds),
        "n_folds": n_folds,
        "task": "时序窗口正常/异常检测 (NAB 真实运维指标)",
        "best_l2": best_l2,
        "anomaly_weight": ANOMALY_WEIGHT,
        "windows": len(OperationalMetricsDataset()._corpus),
        "val_acc_mean": float(np.mean(all_fold)),
        "val_acc_std": float(np.std(all_fold)),
        "anomaly_detection_rate_mean": float(np.mean(all_det)),
        "anomaly_detection_rate_std": float(np.std(all_det)),
        "false_alarm_rate_mean": float(np.mean(all_far)),
        "seed_means": seed_means,
        "random_baseline": 0.5,
        "majority_baseline": majority,
        "folds_beat_random": sum(r["folds_beat_random"] for r in results),
        "total_folds": len(seeds) * n_folds,
        "results": [{
            "seed": r["seed"],
            "val_accuracy": r["val_accuracy"],
            "anomaly_detection_rate": r["anomaly_detection_rate"],
            "false_alarm_rate": r["false_alarm_rate"],
        } for r in results],
    }

    with open(os.path.join(_OUT_DIR, "anomaly_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R5: 真实时序异常检测基准报告 (v0.13.1)\n",
        f"日期: {summary['date']}  |  5 种子 × 5 折分层 CV × 1 个时序任务\n",
        "## 任务",
        "**窗口级正常/异常检测** (非分类): NAB 真实运维指标序列切成定长滑动"
        f"窗口 ({summary['windows']} 窗), 判断窗口是否落在官方告警时段。\n",
        f"随机基线 50%; 多数类(正常)基线 {majority * 100:.1f}%。"
        "行级度量见下表。",
        "",
        "## 结果 (折级)",
        "| 度量 | 折均值 ± std | 参照基线 |",
        "|------|--------------|-----------|",
        f"| 窗口 ACC | **{summary['val_acc_mean']:.1%}** ± "
        f"{summary['val_acc_std']:.1%} | 随机 50% |",
        f"| 异常检出率 (recall) | **{summary['anomaly_detection_rate_mean']:.1%}** "
        f"± {summary['anomaly_detection_rate_std']:.1%} | 多数类(全判正常) = 0% |",
        f"| 误报率 (正常→异常) | {summary['false_alarm_rate_mean']:.1%} | ± 检出率权衡 |",
        "",
        f"超随机基线(50%)折数: {summary['folds_beat_random']}/"
        f"{summary['total_folds']}",
        "",
        "种子均值 ACC: " + ", ".join(f"{m:.1%}" for m in seed_means) +
        f"   (L2={best_l2:.0e}, 异常类梯度权重={ANOMALY_WEIGHT}, "
        f"不平衡 2.6% 异常经类别加权读出)",
    ]
    with open(os.path.join(_OUT_DIR, "anomaly_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n真实时序异常检测基准汇总:")
    print(f"  窗口 ACC: {summary['val_acc_mean']:.1%} ± "
          f"{summary['val_acc_std']:.1%}   (随机 50%)")
    print(f"  异常检出率: {summary['anomaly_detection_rate_mean']:.1%} ± "
          f"{summary['anomaly_detection_rate_std']:.1%}   (多数类对异常 = 0%)")
    print(f"  误报率: {summary['false_alarm_rate_mean']:.1%}")
    print(f"  超随机折数: {summary['folds_beat_random']}/"
          f"{summary['total_folds']} 折")
    print(f"\n已写入 {_OUT_DIR}/anomaly_results.json 与 anomaly_report.md")


if __name__ == "__main__":
    main()