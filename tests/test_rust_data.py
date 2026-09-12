import numpy as np
import pytest

from distributedformer.codec.spike_codec import SpikeEncoder
from distributedformer.data.rust_coding import (
    LABELS, RUST_SNIPPETS, load_rust_coding, static_metrics, stratified_split
)
from distributedformer.training.readout import LinearReadout


def test_dataset_integrity():
    samples = load_rust_coding()
    assert len(samples) == 100
    counts = {}
    for s in samples:
        assert s["code"].strip()
        assert s["label_name"] in LABELS
        assert s["label"] == LABELS.index(s["label_name"])
        assert s["rustc"] and s["msg"]
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    assert counts == {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}


def test_real_rustc_error_codes_present():
    codes = {s["rustc"] for s in RUST_SNIPPETS}
    for expected in ("E0382", "E0502", "E0499", "E0597", "E0106", "E0308", "E0277"):
        assert expected in codes


def test_static_metrics():
    m = static_metrics('std::collections::HashMap::new();\nlet mut v = 1;')
    assert m.shape == (10,)
    assert m[1] == 0  # '&' 数
    assert m[2] == 1  # 'mut' 数
    assert m[9] == 3  # '::' 数


def test_stratified_split():
    samples = load_rust_coding()
    train, val = stratified_split(samples, train_ratio=0.75, seed=0)
    assert len(train) == 75 and len(val) == 25
    assert set(s["label"] for s in val) == set(range(5))
    train_codes = {s["code"] for s in train}
    val_codes = {s["code"] for s in val}
    assert not train_codes & val_codes


def test_text_encoder_code_aware_and_deterministic():
    e1, e2 = SpikeEncoder(dim=16), SpikeEncoder(dim=16)
    code = "let mut v = vec![1]; let r = &v;"
    a, b, c = e1.encode_text(code), e2.encode_text(code), e1.encode_text(code)
    assert np.allclose(a, b) and np.allclose(a, c)
    # 代码感知: '&' 应参与编码 (与去掉 & 的编码不同)
    assert not np.allclose(a, e1.encode_text("let mut v = vec![1]; let r = v;"))


def test_feature_pipeline_and_readout():
    from experiments.rust_benchmark import CubeFeatureExtractor
    samples = load_rust_coding()[:10]
    ex = CubeFeatureExtractor(depth=1, seed=0)
    F = np.stack([ex.features({
        "numeric": static_metrics(s["code"]), "text": s["code"]})
        for s in samples])
    assert F.shape[0] == 10
    assert np.all(np.isfinite(F))
    # 同一样本特征确定
    F2 = np.stack([ex.features({
        "numeric": static_metrics(s["code"]), "text": s["code"]})
        for s in samples[:2]])
    assert np.allclose(F[0], F2[0]) and np.allclose(F[1], F2[1])
    # 线性读出层在小子集上应不低于随机
    y = np.array([s["label"] for s in samples])
    r = LinearReadout(n_features=F.shape[1], n_classes=5, seed=0).fit(F, y)
    assert r.accuracy(F, y) >= 0.2
