# -*- coding: utf-8 -*-
"""
实验 R2: Rust Coding 真实需求基准 (v0.6.0)

真实需求: 静态识别一段 Rust 代码命中哪一类编译错误
(所有权移动 / 借用冲突 / 生命周期 / 类型不匹配 / 合法代码)。
这是 lints / IDE 提示 / 自动修复工具的基础能力。

数据: distributedformer/data/rust_coding.py — 100 段真实风格 Rust
代码片段, 标注真实 rustc 错误码 (E0382/E0502/E0597/E0308/ok),
输入模态 = [text: 代码原文, numeric: 真实静态扫描特征]。

模型: CubeGPT (深度1, numeric+text 双面) 冻结作特征提取器,
水库状态 + 线性 softmax 读出层 (与实验 R1 相同的已验证范式)。

运行: python experiments/rust_benchmark.py
输出: experiments/rust_results.json, experiments/rust_report.md
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.distributedformer import CubeGPT
from src.data.rust_coding import (
    LABELS, LABEL_NAMES, load_rust_coding, static_metrics, stratified_split
)
from src.training.readout import CubeFeatureExtractor, LinearReadout


def run_benchmark(seed: int, verbose: bool = False) -> dict:
    samples = load_rust_coding()
    train, val = stratified_split(samples, train_ratio=0.75, seed=seed)

    extractor = CubeFeatureExtractor(depth=1, seed=seed)
    t0 = time.time()
    X_train = np.stack([extractor.features({
        "numeric": static_metrics(s["code"]), "text": s["code"]})
        for s in train])
    X_val = np.stack([extractor.features({
        "numeric": static_metrics(s["code"]), "text": s["code"]})
        for s in val])
    y_train = np.array([s["label"] for s in train])
    y_val = np.array([s["label"] for s in val])
    extract_sec = time.time() - t0

    readout = LinearReadout(n_features=X_train.shape[1], n_classes=len(LABELS),
                            seed=seed).fit(X_train, y_train)
    val_acc = readout.accuracy(X_val, y_val)
    train_acc = readout.accuracy(X_train, y_train)
    preds = readout.predict(X_val)

    per_class = {}
    for c, name in enumerate(LABELS):
        mask = y_val == c
        if mask.sum():
            per_class[name] = float((preds[mask] == c).mean())

    if verbose:
        print(f"  seed={seed} train={train_acc:.2%} val={val_acc:.2%} "
              f"(随机基线 {1/len(LABELS):.0%})")

    return {
        "seed": seed,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "random_baseline": 1.0 / len(LABELS),
        "beats_random": val_acc > 1.0 / len(LABELS) + 0.05,
        "per_class_val_accuracy": per_class,
        "extract_seconds": extract_sec,
    }


def main():
    seeds = [0, 1, 2, 3, 4]
    print(f"{'='*66}\n  实验 R2: Rust Coding 真实需求基准 (CubeGPT 读出层)\n"
          f"  {len(seeds)} 种子, 100 段真实代码 × {len(LABELS)} 类\n{'='*66}")
    results = [run_benchmark(seed=s, verbose=True) for s in seeds]

    accs = [r["val_accuracy"] for r in results]
    summary = {
        "experiment": "R2_rust_coding_benchmark",
        "date": time.strftime("%Y-%m-%d"),
        "n_seeds": len(seeds),
        "n_samples": 100,
        "labels": LABELS,
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs)),
        "val_acc_min": min(accs),
        "val_acc_max": max(accs),
        "random_baseline": 0.2,
        "beats_random_all_seeds": all(r["beats_random"] for r in results),
        "results": results,
    }
    with open("experiments/rust_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R2: Rust Coding 真实需求基准报告 (v0.6.0)\n",
        f"日期: {summary['date']}  |  样本: 100 段真实 Rust 代码 × {len(LABELS)} 类 "
        f"  |  种子数: {len(seeds)}\n",
        "## 真实需求",
        "静态识别 Rust 代码命中的编译错误类别 (所有权移动 / 借用冲突 /",
        "生命周期 / 类型不匹配 / 合法), 对应真实 rustc 错误码, 是 lints /",
        "IDE 提示 / 自动修复工具的基础能力。\n",
        "## 协议",
        "- 输入模态: text (代码原文, 代码感知分词) + numeric (静态扫描特征)",
        "- CubeGPT depth=1 (numeric+text 双面, 550 单元) 冻结, 水库状态特征",
        "- 线性 softmax 读出层 (z-score, L2=1e-3), 分层 75/25 划分\n",
        "## 结果",
        "| seed | train_acc | val_acc | 超过随机基线(25%+5%) |",
        "|------|-----------|---------|---------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['seed']} | {r['train_accuracy']:.1%} | {r['val_accuracy']:.1%} "
            f"| {'✅' if r['beats_random'] else '❌'} |")
    lines.append(f"\n**汇总**: val_acc = {summary['val_acc_mean']:.1%} ± "
                 f"{summary['val_acc_std']:.1%} (随机基线 20%), "
                 f"全部种子超过基线: {'是' if summary['beats_random_all_seeds'] else '否'}\n")
    lines.append("## 各类别验证准确率 (种子均值)")
    lines.append("| 类别 | rustc 错误码 | 准确率 |")
    lines.append("|------|-------------|--------|")
    for name in LABELS:
        per = [r["per_class_val_accuracy"].get(name, 0.0) for r in results]
        codes = {"move": "E0382/E0505/E0507", "borrow": "E0502/E0499",
                 "lifetime": "E0597/E0106/E0515/E0716/E0623",
                 "type": "E0308/E0277/E0599/E0300", "ok": "-"}[name]
        lines.append(f"| {LABEL_NAMES[name]} | {codes} | {np.mean(per):.1%} |")
    with open("experiments/rust_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写入 experiments/rust_results.json 与 experiments/rust_report.md")


if __name__ == "__main__":
    main()
