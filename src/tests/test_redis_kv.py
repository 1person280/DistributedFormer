"""
Redis 分布式 KV 堆测试 (v0.10.0 分布式多节点落地)

覆盖:
- RedisKVStack 与内存 KVStack 协议兼容 (push/query/retrieve/get_stats/淘汰)
- 多租户 key 前缀隔离
- 同租户跨节点 (多实例共享同一 MockRedis 存储) 记忆可见
- create_kv_stack 后端工厂 (memory / redis / auto)
- SpikeWorkflowEngine 按后端选择全局 KV
"""

import numpy as np
import pytest

from src.core.distributedformer import KVStack
from src.deployment.redis_kv import (
    RedisKVStack, MockRedis, create_kv_stack,
)


# 每个用例用独立 tenant, 避免共享 MockRedis 存储的跨用例污染
def _kv(tenant, **kw):
    return RedisKVStack(tenant_id=tenant, **kw)


def test_rediskv_retrieve_matches_memory_semantics():
    """retrieve 返回 dim 维 tanh 聚合向量, 空堆返回零向量 (与 KVStack 一致)"""
    rkv = _kv("t_ret")
    rkv.connect()
    # 空堆
    assert np.array_equal(rkv.retrieve(np.random.randn(16)), np.zeros(16))
    # 写入后
    for i in range(10):
        rkv.push(f"e{i}", np.random.randn(16), np.random.randn(16))
    vec = rkv.retrieve(np.random.randn(16), top_k=3)
    assert vec.shape == (16,)
    assert np.all(np.abs(vec) <= 1.0 + 1e-9)  # tanh ∈ [-1,1]


def test_rediskv_query_format():
    """query 返回 [(entry_id, value·value_w·normalized, normalized)]"""
    rkv = _kv("t_query")
    rkv.connect()
    rkv.push("e0", np.random.randn(16), np.random.randn(16))
    rkv.push("e1", np.random.randn(16), np.random.randn(16))
    results = rkv.query(np.random.randn(16), top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    for eid, retrieved, norm in results:
        assert eid in ("e0", "e1")
        assert retrieved.shape == (16,)
        assert 0.0 <= norm <= 1.0 + 1e-9


def test_rediskv_tenant_isolation():
    """不同租户 key 前缀隔离, 互不可见"""
    a = _kv("t_iso_a")
    b = _kv("t_iso_b")
    a.connect(); b.connect()
    a.push("shared", np.ones(16), np.ones(16))
    b.push("shared", np.ones(16), np.ones(16))
    # 每个租户只能查到自己的条目
    assert len(a.query(np.ones(16), top_k=1)) == 1
    assert len(b.query(np.ones(16), top_k=1)) == 1
    # A 的条目对 B 不可见 (隔离)
    assert len(b.query(np.ones(16), top_k=5)) == 1


def test_rediskv_cross_node_shared_memory():
    """同租户多实例共享同一存储: 节点 A 写入, 节点 B 能检索到 (跨节点记忆可见)"""
    node_a = _kv("t_share")  # 同 host/port/db → 共享 MockRedis 存储
    node_b = _kv("t_share")
    node_a.connect(); node_b.connect()
    # 节点 A 写入记忆
    key = np.random.randn(16)
    node_a.push("node_a_entry", key, np.ones(16))
    # 节点 B 能检索到 A 写入的条目 (跨节点共享)
    results = node_b.query(key, top_k=3)
    eids = [eid for eid, _, _ in results]
    assert "node_a_entry" in eids
    # 节点 B 主计算路径也能 retrieve 到
    vec = node_b.retrieve(key, top_k=3)
    assert not np.allclose(vec, 0.0)


def test_rediskv_lru_eviction():
    """满容量时 LRU 淘汰, 保持容量不超限"""
    rkv = _kv("t_lru", capacity=5, dim=16)
    rkv.connect()
    for i in range(10):
        rkv.push(f"e{i}", np.random.randn(16), np.random.randn(16))
    assert rkv.get_stats()["used"] <= 5


def test_rediskv_get_stats_fields():
    """get_stats 含 capacity/used/utilization/total_access/backend (协议对齐 KVStack)"""
    rkv = _kv("t_stats")
    rkv.connect()
    rkv.push("e0", np.random.randn(16), np.random.randn(16))
    rkv.query(np.random.randn(16), top_k=1)
    s = rkv.get_stats()
    for field in ("capacity", "used", "utilization", "total_access"):
        assert field in s
    assert s["used"] == 1
    assert s["total_access"] >= 1


def test_rediskv_save_load_noop():
    """save_to_disk / load_from_disk 为协议兼容 no-op (Redis 自持久化)"""
    rkv = _kv("t_persist")
    rkv.connect()
    rkv.push("e0", np.random.randn(16), np.random.randn(16))
    # 不应抛错
    rkv.save_to_disk("n/a")
    rkv.load_from_disk("n/a")
    assert rkv.get_stats()["used"] == 1


def test_create_kv_stack_memory():
    """工厂 memory → 内存 KVStack"""
    kv = create_kv_stack("memory", capacity=10, dim=16)
    assert isinstance(kv, KVStack)


def test_create_kv_stack_redis():
    """工厂 redis → RedisKVStack"""
    kv = create_kv_stack("redis", capacity=10, dim=16, tenant_id="t_factory")
    assert isinstance(kv, RedisKVStack)


def test_create_kv_stack_auto_env(monkeypatch):
    """工厂 auto 读环境变量 KV_BACKEND"""
    monkeypatch.setenv("KV_BACKEND", "memory")
    assert isinstance(create_kv_stack("auto", capacity=10, dim=16), KVStack)
    monkeypatch.setenv("KV_BACKEND", "redis")
    kv = create_kv_stack("auto", capacity=10, dim=16, tenant_id="t_auto")
    assert isinstance(kv, RedisKVStack)


def test_engine_redis_backend():
    """SpikeWorkflowEngine 按 kv_backend 选择全局 KV 后端"""
    from src.workflow.engine import SpikeWorkflowEngine
    mem_eng = SpikeWorkflowEngine(kv_backend="memory")
    assert isinstance(mem_eng.global_kv, KVStack)
    redis_eng = SpikeWorkflowEngine(kv_backend="redis")
    assert isinstance(redis_eng.global_kv, RedisKVStack)


def test_mockredis_shared_namespace():
    """MockRedis 按命名空间共享存储, 模拟真实 Redis server 语义"""
    m1 = MockRedis("mock://s:6379/0")
    m2 = MockRedis("mock://s:6379/0")
    m3 = MockRedis("mock://other:6379/0")
    m1.set("k", b"v")
    assert m2.get("k") == b"v"
    assert m3.get("k") is None
