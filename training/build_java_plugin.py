"""构建 JavaCoding.CuteMamen 插件包 (v0.14.0)

流程 (与 RustCoding 插件构建同构):
    真实 Java 语料 → 双模态注入 (numeric=static+structure, text=代码原文)
    → 冻结 CubeGPT 水库特征 → 线性 softmax 读出层训练
    → JavaCodingPlugin.migrate_from_main_model() → save_pkg()

同时跑 5 种子 × 5 折分层交叉验证, 报告读出层与随机/多数类基线对比。
输出: plugin/JavaCoding.CuteMamen (携带主模型迁移知识)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cutemamen import JavaCodingPlugin, save_pkg
from src.data.real_dataset import JavaCodingTrainingDataset
from src.data.java_coding import LABELS
from src.training.readout import run_cross_validation as _run_cv  # 与插件同构协议

SEEDS = (0, 1, 2, 3, 4)
N_FOLDS = 5
PKG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "plugin", "JavaCoding.CuteMamen")


def eval_plugin_routing(train, val, seed: int, migrated: bool) -> float:
    """一折: 插件 (可选迁移主模型知识) 挂载内核 → 路由推理验证折"""
    from src.cutemamen import CuteMamenKernel
    kernel = CuteMamenKernel(dim=16)
    plugin = JavaCodingPlugin("java-coding")
    if migrated:
        plugin.migrate_from_main_model(train, seed=seed)
    kernel.mount(plugin)
    correct = 0
    for s in val:
        out = kernel.think({"topic": "java", "data": s.metadata["code"]})
        correct += bool(out) and out[0]["label"] == LABELS[s.category]
    return correct / len(val)


def main() -> None:
    t0 = time.time()
    dataset = JavaCodingTrainingDataset(dim=16)
    from collections import Counter
    dist = dict(Counter(s["label_name"] for s in dataset._corpus))
    print(f"真实 Java 语料: {len(dataset._corpus)} 段, 分布: {dist}")

    # 1) 5 种子 × 5 折交叉验证 (插件内核路由 vs 原型基线)
    print("\n=== 插件路由评估 (5 种子 × 5 折) ===")
    mean_m = np.mean([
        np.mean([eval_plugin_routing(tr, va, seed, migrated=True)
                 for tr, va in dataset.kfold_datasets(
                     n_folds=N_FOLDS, seed=seed)])
        for seed in SEEDS])
    mean_p = np.mean([
        np.mean([eval_plugin_routing(tr, va, seed, migrated=False)
                 for tr, va in dataset.kfold_datasets(
                     n_folds=N_FOLDS, seed=seed)])
        for seed in SEEDS])
    random_baseline = dataset.RANDOM_BASELINE
    majority = dataset.majority_baseline(dataset.generate_dataset(0.75)[0])
    print(f"插件(主模型迁移): {mean_m:.1%}")
    print(f"插件(原型基线):   {mean_p:.1%}")
    print(f"随机基线: {random_baseline:.1%}, 多数类: {majority:.1%}")

    # 2) 构建交付包: 全量语料迁移全部知识 → save_pkg
    print("\n=== 构建 JavaCoding.CuteMamen ===")
    plugin = JavaCodingPlugin("java-coding")
    plugin.migrate_from_main_model(depth=1, dim=16, seed=0)  # None → 全量 294 段
    manifest = save_pkg(plugin, PKG_PATH)
    print(f"manifest: knowledge_source={manifest['knowledge_source']}, "
          f"corpus_size={manifest['corpus_size']}, "
          f"labels={manifest['labels']}")
    print(f"readout: n_features={manifest['readout']['n_features']}, "
          f"n_classes={manifest['readout']['n_classes']}, "
          f"footprint_mb={manifest['memory_footprint_mb']}")
    print(f"已写入 {PKG_PATH}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()