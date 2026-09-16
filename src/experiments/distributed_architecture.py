"""新阶段 · 分布式架构评估: 主模型知识迁移到思考插件后的端到端准确率

v0.9.0 知识迁移路径 (与 v0.8.4 主模型读出层协议对齐):

    主模型 (冻结 CubeGPT 水库 + 线性读出层)
        ──migrate_from_main_model(训练折)──▶ RustCodingPlugin 读出权重
        ──CuteMamenKernel.mount──▶ think(topic="rust") 逐样本推理

评估协议 (与 v0.8.4 的 75.6% 完全可比):
- 真实语料 502 段, 5 种子 × 5 折分层交叉验证
- 每折: 插件只从训练折迁移知识, 验证折经内核路由逐样本分类
- 对照: (a) 主模型直评 (training/readout.run_cross_validation)
        (b) 未迁移插件 (语料原型最近质心, 旧阶段基线)
"""

import json
import os
import sys
import time

import numpy as np

# src/experiments/<file> → 上溯 3 级到仓库根 (导入 src 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.cutemamen import CuteMamenKernel, RustCodingPlugin
from src.data.real_dataset import RustCodingTrainingDataset
from src.data.rust_coding import LABELS
from src.training.readout import run_cross_validation

SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "distributed_architecture_results.json")


def eval_plugin_routing(train, val, seed: int, migrated: bool) -> float:
    """一折: 插件 (可选迁移主模型知识) 挂载内核 → 路由推理验证折"""
    kernel = CuteMamenKernel(dim=16)
    plugin = RustCodingPlugin("rust-coding")
    if migrated:
        plugin.migrate_from_main_model(train, seed=seed)
    kernel.mount(plugin)
    correct = 0
    for s in val:
        out = kernel.think({"topic": "rust", "data": s.metadata["code"]})
        # s.category 是类别索引 (int), 插件返回标签名 (str)
        correct += bool(out) and out[0]["label"] == LABELS[s.category]
    return correct / len(val)


def main() -> None:
    dataset = RustCodingTrainingDataset(dim=16)
    t0 = time.time()
    results = {"phase": "distributed-architecture", "seeds": list(SEEDS),
               "n_folds": N_FOLDS, "corpus": 502,
               "plugin_routed": {}, "prototype_baseline": {},
               "main_model_direct": {}}

    for seed in SEEDS:
        splits = dataset.kfold_datasets(n_folds=N_FOLDS, seed=seed)
        accs_m, accs_p = [], []
        for train, val in splits:
            accs_m.append(eval_plugin_routing(train, val, seed, migrated=True))
            accs_p.append(eval_plugin_routing(train, val, seed, migrated=False))
        results["plugin_routed"][seed] = accs_m
        results["prototype_baseline"][seed] = accs_p
        m = float(np.mean(accs_m))
        p = float(np.mean(accs_p))
        print(f"seed={seed} 插件(迁移)={m:.3%} 折={np.round(accs_m, 3).tolist()}")
        print(f"        插件(原型)={p:.3%} 折={np.round(accs_p, 3).tolist()}")

    # 主模型直评对照 (同协议)
    for seed in SEEDS:
        r = run_cross_validation(seed=seed, n_folds=N_FOLDS)
        results["main_model_direct"][seed] = r["val_acc_per_fold"]
        print(f"seed={seed} 主模型直评={r['val_accuracy']:.3%}")

    def _agg(seed_accs):
        means = [float(np.mean(v)) for v in seed_accs.values()]
        folds = [a for v in seed_accs.values() for a in v]
        return {"mean": float(np.mean(means)),
                "seed_min": float(np.min(means)), "seed_max": float(np.max(means)),
                "fold_min": float(np.min(folds)), "fold_max": float(np.max(folds))}

    results["summary"] = {
        "plugin_routed": _agg(results["plugin_routed"]),
        "prototype_baseline": _agg(results["prototype_baseline"]),
        "main_model_direct": _agg(results["main_model_direct"]),
    }
    results["seconds"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    s = results["summary"]
    print("\n=== 新阶段 · 分布式架构结果 ===")
    print(f"插件(主模型迁移, 内核路由): {s['plugin_routed']['mean']:.1%} "
          f"(种子 {s['plugin_routed']['seed_min']:.1%}~"
          f"{s['plugin_routed']['seed_max']:.1%})")
    print(f"主模型直评对照:            {s['main_model_direct']['mean']:.1%}")
    print(f"插件原型基线 (旧阶段):     {s['prototype_baseline']['mean']:.1%}")
    print(f"结果已写入 {OUT_JSON} ({results['seconds']}s)")


if __name__ == "__main__":
    main()
