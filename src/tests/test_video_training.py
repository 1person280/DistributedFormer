# -*- coding: utf-8 -*-
"""视频生成训练测试 (v0.12.0)

覆盖 VideoMakingPlugin 的真实训练材料接口 (training_data) 与
VideoMotionDataset 的真实运动识别数据通路, 并验证冻结水库 + 线性
读出在真实运镜任务上显著超过随机基线 (视频生成训练)。
"""

import numpy as np
import pytest

from src.cutemamen import VideoMakingPlugin
from src.data.video_motion import LABELS, VideoMotionDataset
from src.experiments.video_motion_benchmark import run_video_cross_validation


def test_video_plugin_training_data_shape():
    """VideoMakingPlugin.training_data() 导出真实运镜稠密采样描述符"""
    plugin = VideoMakingPlugin()
    X, y = plugin.training_data()
    n_motions = len(LABELS)
    assert X.shape == (n_motions * 20, 5)   # (dx,dy,zoom,rot,progress)
    assert y.shape == (n_motions * 20,)
    assert set(np.unique(y)) <= set(range(n_motions))
    # 真实曲线描述符导出 (8 维 from/to) 与真实运动一致
    desc = plugin.motion_descriptors()
    assert set(desc.keys()) == set(LABELS)
    assert desc["static"].shape == (8,)


def test_video_motion_kfold_every_sample_validated_once():
    """5 折分层 CV: 每个真实采样点恰好作为一次验证样本"""
    ds = VideoMotionDataset(n_points=20)
    splits = ds.kfold_datasets(n_folds=5, seed=0)
    assert len(splits) == 5
    val_ids = [s.sample_id for _, val in splits for s in val]
    assert len(val_ids) == len(ds._corpus) == 200
    assert len(set(val_ids)) == 200          # 无重复验证
    # 每折 10 类齐全 (stratified)
    for train, val in splits:
        assert len({s.category for s in val}) == len(LABELS)


def test_video_readout_beats_random():
    """真实运镜识别: 冻结水库 + 线性读出应显著超随机基线 (≥ +5pct)"""
    ds = VideoMotionDataset()
    r = run_video_cross_validation(seed=0, n_folds=5, l2=1e-3,
                                   n_points=20, verbose=False)
    assert r["val_accuracy"] > ds.RANDOM_BASELINE + 0.05
    assert r["beats_random"] is True
    assert len(r["val_acc_per_fold"]) == 5
