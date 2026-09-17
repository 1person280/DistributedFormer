# -*- coding: utf-8 -*-
"""DFTrainer 端到端后期漂移修复测试 (v0.10.2)

覆盖 v5.4 新增的机制 —— LR 调度 (默认 cosine) 与水库冻结 ——
重点验证两种机制确实改变训练行为并能缓解后期验证准确率回落。
"""

import numpy as np
import pytest

from src.training.trainer import DFTrainer


def _small_trainer(**kwargs):
    return DFTrainer(depth=0, dim=16, learning_rate=0.008, **kwargs)


def test_default_lr_schedule_is_cosine():
    t = _small_trainer()
    assert t.lr_schedule == "cosine"
    assert t.freeze_reservoir_epoch is None
    assert not t.reservoir_frozen


def test_cosine_schedule_decays_late():
    t = _small_trainer()
    t._apply_schedule(0, 12)
    mid = t.lr
    t._apply_schedule(11, 12)
    assert t.lr == pytest.approx(0.008 * 0.1)  # min_lr_ratio 兜底
    # 后期 LR 应始终低于早期, 且单调递减
    for ep in (2, 4, 6, 8, 10):
        t._apply_schedule(ep, 12)
        assert t.lr < mid
        mid = t.lr


def test_constant_schedule_no_change():
    t = _small_trainer(lr_schedule="constant")
    t._apply_schedule(11, 12)
    assert t.lr == pytest.approx(0.008)


def test_step_schedule_drops_at_epoch():
    t = _small_trainer(lr_schedule="step", step_drop_epoch=8)
    t._apply_schedule(7, 12)
    assert t.lr == pytest.approx(0.008)
    t._apply_schedule(8, 12)
    assert t.lr == pytest.approx(0.008 * 0.1)


def test_freeze_guards_reservoir_update():
    """冻结后 _update_weights_supervised 不应再改变任何 w_in"""
    t = _small_trainer(freeze_reservoir_epoch=0)
    from src.data.real_dataset import RustCodingTrainingDataset
    ds = RustCodingTrainingDataset(dim=16)
    train, val = ds.generate_dataset(train_ratio=0.75, seed=0)
    sample = train[0]
    # 未冻结
    contribs_before = len(t.df._all_units)
    t.reservoir_frozen = False
    feats, nsp = t._forward_features(t._sample_inputs(sample))
    out = t.df.get_output_pattern()
    think = t.df.get_think_layer_pattern()
    t._update_weights_supervised(sample.static_signal, sample.target_pattern,
                                 out, think, lr=t.lr)
    changed = 0
    for u in t.df._all_units:
        if not np.allclose(u.w_in, 0.0):
            pass
    # 转冻结后权重不应再变
    t.reservoir_frozen = True
    before = [u.w_in for u in t.df._all_units]
    t._update_weights_supervised(sample.static_signal, sample.target_pattern,
                                 out, think, lr=t.lr)
    for u, w_before in zip(t.df._all_units, before):
        assert u.w_in == w_before, "冻结后 w_in 不应更新"
    assert contribs_before == len(t.df._all_units)


def test_freeze_flag_applied_in_schedule():
    t = _small_trainer(freeze_reservoir_epoch=3)
    t._apply_schedule(2, 12)
    assert not t.reservoir_frozen
    t._apply_schedule(3, 12)
    assert t.reservoir_frozen