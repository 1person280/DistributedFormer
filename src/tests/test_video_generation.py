# -*- coding: utf-8 -*-
"""视频生成插件精度测试 (v0.17.0)

覆盖两块新能力:
1. 规模扩大 —— 真实运动识别数据集扩到 500 样本 (10 类 × 50 点) ×
   5 种子 × 5 折; 水库多一层 16 单元嵌套 (depth=2) 特征维度显著增大。
2. 生成精度边界 —— 分辨率 × 时长 网格上的结构保真度 (SSIM 概率云) :
   真实关键帧 (真实运维时序) 经真实运镜渲染, 静态基准一致性有界 [0,1],
   且高分辨率/短时长应 ≥ 低分辨率/长时长 (精度边界单调性守卫)。
"""

import numpy as np
import pytest

from src.data.video_motion import VideoMotionDataset, LABELS
from src.training.readout import CubeFeatureExtractor
from src.experiments.video_generation_precision import (
    keyframe_from_telemetry, ssim, generation_accuracy_cell,
    run_precision_grid, _box_mean)


# ── 1. 规模扩大: 500 样本 + 多一层 16 单元嵌套 ───────────────────
def test_dataset_expanded_to_500_samples():
    """50 点 × 10 类 = 500 真实样本; 5 折每样本恰好验证一次"""
    ds = VideoMotionDataset(n_points=50)
    assert len(ds._corpus) == len(LABELS) * 50 == 500
    splits = ds.kfold_datasets(n_folds=5, seed=0)
    val_ids = [s.sample_id for _, val in splits for s in val]
    assert len(val_ids) == 500
    assert len(set(val_ids)) == 500          # 每个真实样本恰好验证一次


def test_one_more_16_unit_nesting_layer():
    """depth=2 (多一层 16 单元嵌套) 特征维度应显著大于 depth=1"""
    ex1 = CubeFeatureExtractor(depth=1, seed=1, modalities=["numeric"])
    ex2 = CubeFeatureExtractor(depth=2, seed=1, modalities=["numeric"])
    f1 = ex1.features({"numeric": np.zeros(5)})
    f2 = ex2.features({"numeric": np.zeros(5)})
    assert f2.size > f1.size                 # 4344 vs 288 量级
    assert f2.size >= 16 + 16 ** 2 + 16 ** 3 + 32  # 水库最外层 4368 单元


# ── 2. 生成精度边界: 分辨率 × 时长 概率云 ────────────────────────
def test_keyframe_from_real_telemetry():
    """真实运维时序 → 方形灰度帧, 值域 [0,1], 非恒定 (真实内容)"""
    kf = keyframe_from_telemetry(side=32)
    assert kf.ndim == 2 and kf.shape[0] == kf.shape[1] >= 8
    assert kf.min() >= 0.0 and kf.max() <= 1.0
    assert kf.std() > 1e-6                     # 真实时序非恒定


def test_ssim_self_identical_is_one():
    """恒等图 SSIM ≈ 1; 结构相似度有界 [0,1]"""
    rng = np.random.RandomState(0)
    a = keyframe_from_telemetry(side=32)
    assert abs(ssim(a, a) - 1.0) < 1e-6
    b = rng.rand(*a.shape)                     # 独立内容 → 低相似
    assert ssim(a, b) < 0.9


def test_precision_boundary_monotonic():
    """高分辨率/短时长生成精度 ≥ 低分辨率/长时长 (精度边界守卫)"""
    kf = keyframe_from_telemetry(side=32)
    lo = generation_accuracy_cell(64, 8.0, kf, max_ssim_frames=6)
    hi = generation_accuracy_cell(640, 0.5, kf, max_ssim_frames=6)
    assert 0.0 <= lo["accuracy_mean"] <= 1.0
    assert 0.0 <= hi["accuracy_mean"] <= 1.0
    # 主指标 (可分辨帧占比): 高分辨率 + 短时长 不应低于 低分辨率 + 长时长
    assert hi["accuracy_mean"] >= lo["accuracy_mean"] - 0.05
    # 次指标 (SSIM 结构保真度) 同向守卫
    assert hi["content_fidelity"] >= lo["content_fidelity"] - 0.15
    # 长时长 → 步长跌破 1px → 主精度应低于短时长同分辨率
    short = generation_accuracy_cell(128, 0.5, kf, max_ssim_frames=6)
    long_ = generation_accuracy_cell(128, 8.0, kf, max_ssim_frames=6)
    assert short["accuracy_mean"] >= long_["accuracy_mean"] - 0.05


def test_precision_grid_cloud_shape():
    """2×2 网格 → 每格有可分辨性分布与散点云"""
    kf = keyframe_from_telemetry(side=32)
    s = run_precision_grid(kf, resolutions=[64, 160], durations=[0.5, 2.0],
                           max_ssim_frames=4)
    assert len(s["cells"]) == 4
    # 云点数 = Σ 每格可分辨性测量数 (相邻帧对数 × 真实运镜数)
    expect = sum(len(c["saliency"]) for c in s["cells"])
    assert len(s["cloud"]) == expect
    for c in s["cells"]:
        assert 0.0 <= c["accuracy_mean"] <= 1.0
        assert 0.0 <= c["content_fidelity"] <= 1.0
        assert 0.0 <= c["retention_mean"] <= 1.0
        assert len(c["saliency"]) == c["n_steps"]