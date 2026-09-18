# -*- coding: utf-8 -*-
"""
实验 R6: 真实时序流 · 在线持续学习漂移/稳定性验证 (P1 · 里程碑 v0.13.1)

把框架"7×24 持续在线"卖点落到首个**非分类**任务: 用 DFTrainer (在线持久
输出头 + 非平稳水库, 与 v0.10.2 端到端同一套) 在真实运维时序流的**时间顺序**
上持续学习 —— 训练 = 时序前 70% 真实窗口 (在线看的是历史), 验证 = 未来
30% 真实窗口 (真正未见), 跨 epoch 记录验证准确率轨迹, 检验是否出现
v0.10.2 曾诊断的**后期漂移** (验证从最佳滑落到 36–60%)。

对照两种调度 (对齐 v0.10.2 缓解手段):
  - constant: 恒定学习率 (漂移对照组)
  - cosine   : 余弦衰减 w_in/思考层 LR 至 0.1× (默认, 应缓解后期漂移)
  - freeze   : freeze_reservoir_epoch 冻结非平稳水库只调输出头 (同离线读出)

漂移指标: drift = final_val - best_val; drift 越接近 0 越稳定;
显著负值 (|drift| ≥ 10pct) 记为"后期漂移未缓解"。

运行: python src/experiments/anomaly_stream_stability.py
输出: src/experiments/anomaly_stream_stability.json / .md
"""

import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

# src/experiments/<file> → 上溯 3 级到仓库根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.data.metrics_time_series import OperationalMetricsDataset
from src.training.trainer import DFTrainer


EPOCHS = int(os.environ.get("ANOM_STREAM_EPOCHS", "4"))
FWD_STEPS = 2            # 与读出层特征提取一致, 平衡速度与精度


def _run_schedule(name, sched, freeze_epoch, train, val):
    """单调度在线训练 → 每 epoch 验证准确率轨迹 + 漂移指标"""
    df = DFTrainer(depth=1, dim=16, n_classes=2, lr_schedule=sched,
                   freeze_reservoir_epoch=freeze_epoch, fwd_steps=FWD_STEPS,
                   use_think_supervision=False)
    history = []
    for epoch in range(EPOCHS):
        df.epoch = epoch
        df._apply_schedule(epoch, EPOCHS)
        df.train_epoch(train)
        m = df.evaluate(val)
        history.append(float(m["accuracy"]))
    arr = np.array(history)
    best = float(arr.max())
    final = float(arr[-1])
    drift = final - best
    return {
        "schedule": name,
        "val_acc_history": [round(v, 4) for v in history],
        "best_val": best,
        "final_val": final,
        "drift": drift,
        "stable": drift >= -0.10,   # 后期未显著下滑 → 缓解
    }


def main():
    start = time.time()
    dataset = OperationalMetricsDataset()
    tr, va = dataset.split_stream(train_ratio=0.7)
    # 时序降采样 (奇数窗口) 控制在线训练开销, 保持时间顺序与未来验证集完整
    train_sub = tr[1::2]
    print("=" * 70)
    print("  实验 R6: 真实时序流在线持续学习 漂移/稳定性验证 (P1)")
    print("=" * 70)
    print(f"  时序划分: 训练(前70%, 降采样)={len(train_sub)} / "
          f"验证(未来30%)={len(va)}")
    print(f"  Epochs={EPOCHS}, fwd_steps={FWD_STEPS}, 异常占比 "
          f"(验证)={sum(1 for s in va if s.category == 1) / len(va):.1%}")
    print()

    configs = OrderedDict([
        ("constant", dict(sched="constant", freeze_epoch=None)),
        ("cosine",   dict(sched="cosine",   freeze_epoch=None)),
        ("freeze@2", dict(sched="cosine",   freeze_epoch=2)),
    ])
    rows = []
    for name, cfg in configs.items():
        t0 = time.time()
        r = _run_schedule(name, cfg["sched"], cfg["freeze_epoch"],
                          train_sub, va)
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  [{name:9s}] best={r['best_val']:.1%} final={r['final_val']:.1%} "
              f"drift={r['drift']:+.1%} {'稳定' if r['stable'] else '漂移!!'} "
              f"({r['seconds']}s)")
        print(f"        val轨迹: {', '.join(f'{v:.1%}' for v in r['val_acc_history'])}")

    summary = {
        "experiment": "R6_stream_stability",
        "version": "0.13.1",
        "date": time.strftime("%Y-%m-%d"),
        "task": "真实运维时序流·在线持续学习 (时间顺序前后划分)",
        "epochs": EPOCHS,
        "train_samples": len(train_sub),
        "val_samples": len(va),
        "drift_threshold": -0.10,
        "configs": rows,
        "elapsed_seconds": round(time.time() - start, 1),
    }
    with open(os.path.join(_OUT_DIR, "anomaly_stream_stability.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# 实验 R6: 真实时序流在线持续学习 漂移/稳定性报告 (v0.13.1)\n",
        f"日期: {summary['date']}  |  epochs={EPOCHS}, 时间顺序前后划分 "
        f"(训练=[前70%, 降采样] {len(train_sub)}, 验证=[未来] {len(va)})\n",
        "## 任务",
        "把 v0.10.2 在线持久输出头 + 非平稳水库放到真实运维时序流上持续学习, "
        "检验**后期漂移**是否缓解 (drift = final − best; drift ≥ −10pct 记稳定)。\n",
        "## 结果",
        "| 调度 | best_val | final_val | drift | 结论 |",
        "|------|----------|-----------|-------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['schedule']:9s} | {r['best_val']:.1%} | "
            f"{r['final_val']:.1%} | {r['drift']:+.1%} | "
            f"{'稳定✅' if r['stable'] else '漂移⚠️'} |")
    lines += ["", "## 验证 acc 轨迹", "| 调度 | 每 epoch val_acc |", "|------|------------------|"]
    for r in rows:
        lines.append(f"| {r['schedule']:9s} | "
                     f"{', '.join(f'{v:.1%}' for v in r['val_acc_history'])} |")
    lines += [
        "",
        "结论: cosine (默认) 与 freeze 应在后期验证保持平稳; 若 constant 出现",
        "明显下滑而 cosine/freeze 不下滑, 说明 v0.10.2 的后期漂移缓解手段在",
        "真实时序流上同样有效。",
    ]
    with open(os.path.join(_OUT_DIR, "anomaly_stream_stability.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写入 {_OUT_DIR}/anomaly_stream_stability.json 与 .md")


if __name__ == "__main__":
    main()