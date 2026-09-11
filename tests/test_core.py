import numpy as np
import pytest

from distributedformer.core.distributedformer import (
    DistributedFormer, KVStack, SpikingUnit, calculate_scale
)


def test_calculate_scale():
    info = calculate_scale(2)
    assert info["base_units"] == 4368
    assert info["total_params"] == 4368 * 16


def test_kvstack_query_and_persistence(tmp_path):
    kv = KVStack(capacity=100, dim=16)
    for i in range(20):
        kv.push(f"k{i}", np.random.randn(16), np.random.randn(16))
    results = kv.query(np.random.randn(16), top_k=3)
    assert len(results) == 3
    kv.save_to_disk(str(tmp_path / "kv.json"))
    kv2 = KVStack(capacity=100, dim=16)
    kv2.load_from_disk(str(tmp_path / "kv.json"))
    assert len(kv2.entries) == 20


def test_spiking_unit_fires_spikes():
    unit = SpikingUnit("u")
    strong = np.ones(16) * 5.0  # 强输入保证确定性发放
    spikes = sum(
        1 for _ in range(200)
        if unit.step(strong, np.zeros(16), 1.0)
    )
    assert 0 < spikes <= 200


def test_network_step():
    df = DistributedFormer(depth=1, dim=16)
    for _ in range(3):
        spikes = df.step({"numeric": np.random.randn(16) * 0.5})
        assert isinstance(spikes, list)
    stats = df.get_network_stats()
    assert stats["total_units"] > 0
