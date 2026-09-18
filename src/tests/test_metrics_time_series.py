# -*- coding: utf-8 -*-
"""真实时序异常检测训练测试 (v0.13.1, P0 立项)

覆盖 OperationalMetricsDataset 的真实时序数据通路 (窗口化 / 二元标签 /
training_data 导出 / 分层 K 折 / 时序前条划分), 以及 LinearReadout 新增的
可选类别权重在真实不平衡异常任务上把检出率从 0 抬升的行为。
"""

import numpy as np
import pytest

from src.data.metrics_time_series import (
    LABELS, DIM, DEFAULT_WINDOW, OperationalMetricsDataset,
    window_features, METRIC_FILES,
)
from src.training.readout import LinearReadout


def test_metrics_corpus_real_and_binary():
    """真实运维序列 → 窗口语料: 二元标签, 统计描述符维度正确"""
    ds = OperationalMetricsDataset()
    assert len(ds._corpus) > 1000                 # 足量真实窗口
    labels = {s["label"] for s in ds._corpus}
    assert labels == {"normal", "anomaly"}
    n_anom = sum(1 for s in ds._corpus if s["label"] == "anomaly")
    assert 5 <= n_anom < len(ds._corpus) // 2      # 稀有但存在的真实异常


def test_window_features_dim():
    """窗口统计描述符为 DIM(16) 维, 纯实值"""
    rng = np.random.RandomState(0)
    x = rng.randn(DEFAULT_WINDOW) * 3 + 10
    f = window_features(x)
    assert f.shape == (DIM,)
    assert np.isfinite(f).all()


def test_training_data_export():
    """training_data() 对齐插件导出: X 特征矩阵 + y 二元标签"""
    X, y = OperationalMetricsDataset().training_data()
    assert X.shape[1] == DIM
    assert y.dtype.kind == "i"
    assert set(np.unique(y)) == {0, 1}


def test_kfold_both_classes_in_every_fold():
    """分层 K 折: 每折训练/验证都含正常与异常 (折间比例一致)"""
    ds = OperationalMetricsDataset()
    splits = ds.kfold_datasets(n_folds=5, seed=0)
    assert len(splits) == 5
    val_ids = [s.sample_id for _, val in splits for s in val]
    assert len(set(val_ids)) == len(ds._corpus)   # 每窗口恰好验证一次
    for train, val in splits:
        assert {s.category for s in val} == {0, 1}
        assert {s.category for s in train} == {0, 1}


def test_stream_split_temporal_order():
    """时序在线划分: 训练 = 时间前段, 验证 = 未来段, 标签二元"""
    ds = OperationalMetricsDataset()
    train, val = ds.split_stream(train_ratio=0.7)
    assert len(train) + len(val) == len(ds._corpus)
    assert {s.category for s in val} == {0, 1}
    # 每个样本监督输出模式为 DIM 长 (trainer 契约), 类别位在 0/1
    s = next(s for s in val if s.category == 1)
    assert s.target_pattern.shape == (DIM,)
    assert s.target_pattern[1] > 0.6 and s.target_pattern[0] < 0.2
    s0 = next(s for s in val if s.category == 0)
    assert s0.target_pattern[0] > 0.6


def test_linear_readout_class_weight_raises_detection():
    """类别权重: 不平衡二元异常检测上, 加权应让异常检出率 > 0 (非合成)"""
    ds = OperationalMetricsDataset()
    splits = ds.kfold_datasets(n_folds=2, seed=0)
    (tr, va) = splits[0]
    Xtr = np.stack([s.static_signal for s in tr])
    Xva = np.stack([s.static_signal for s in va])
    ytr = np.array([s.category for s in tr])
    yva = np.array([s.category for s in va])
    assert (yva == 1).sum() > 0
    # 无权重读出 (多数类主导) 检出率应为 0
    plain = LinearReadout(n_features=Xtr.shape[1], n_classes=2, seed=0,
                          l2=1e-2).fit(Xtr, ytr)
    p = plain.predict(Xva)
    det_plain = float((p[yva == 1] == 1).sum()) / (yva == 1).sum()
    assert det_plain < 0.1
    # 加权读出能检出部分真实异常
    w = LinearReadout(n_features=Xtr.shape[1], n_classes=2, seed=0,
                      l2=1e-2, class_weight=[1.0, 10.0]).fit(Xtr, ytr)
    pw = w.predict(Xva)
    det_w = float((pw[yva == 1] == 1).sum()) / (yva == 1).sum()
    assert det_w > det_plain


def test_benchmark_modules_import():
    """基准入口可导入, 关键任务元信息存在"""
    import src.experiments.anomaly_benchmark as ab
    import src.experiments.anomaly_stream_stability as st
    assert ab.OperationalMetricsDataset is OperationalMetricsDataset
    assert callable(ab.run_anomaly_cross_validation)
    assert callable(st._run_schedule)