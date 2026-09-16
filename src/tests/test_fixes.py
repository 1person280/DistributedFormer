import time

import numpy as np
import pytest

from src.core.distributedformer import KVStack, CubeGPT


# ── v0.7.4 KV 堆检索向量化: 行为保持回归 ──────────────────────

def _ref_retrieve(kv: KVStack, q: np.ndarray, top_k=3, scan_limit=4096):
    """旧版 (Python 循环) 参考实现, 用于向量化语义等价校验"""
    items = list(kv.entries.items())[-scan_limit:]
    qn = np.linalg.norm(q)
    scores = []
    for eid, e in items:
        kn = np.linalg.norm(e.key_signal)
        s = abs(float(q @ e.key_signal)) / (qn * kn) if qn > 0 and kn > 0 else 0.0
        scores.append((eid, e, s * np.exp(-0.001 * (time.time() - e.timestamp))))
    scores.sort(key=lambda x: x[2], reverse=True)
    top = scores[:top_k]
    total = sum(x[2] for x in top) + 1e-8
    agg = np.zeros(kv.dim)
    for _, e, s in top:
        agg += e.value_state * kv.value_w * (s / total)
    return np.tanh(agg), [x[0] for x in top]


def test_kv_retrieve_matches_reference():
    """向量化 retrieve 与旧版逐条打分结果数值一致"""
    kv = KVStack(capacity=100, dim=16)
    rng = np.random.RandomState(7)
    for i in range(40):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    q = rng.randn(16)
    ref, ref_ids = _ref_retrieve(kv, q, top_k=3)
    out = kv.retrieve(q, top_k=3)
    assert np.allclose(out, ref, atol=1e-6)


def test_kv_retrieve_scan_limit_matches_reference():
    """scan_limit 截取语义 (最近写入的 N 条) 与旧版一致"""
    kv = KVStack(capacity=100, dim=16)
    rng = np.random.RandomState(3)
    for i in range(50):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    q = rng.randn(16)
    ref, _ = _ref_retrieve(kv, q, top_k=3, scan_limit=10)
    out = kv.retrieve(q, top_k=3, scan_limit=10)
    assert np.allclose(out, ref, atol=1e-6)


def test_kv_query_matches_reference_and_normalizes():
    """向量化 query 与旧版 top-k/归一化语义一致"""
    kv = KVStack(capacity=100, dim=16)
    rng = np.random.RandomState(11)
    for i in range(30):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    q = rng.randn(16)
    ref, ref_ids = _ref_retrieve(kv, q, top_k=3, scan_limit=10**9)
    # 参考只给 agg, 这里重建 query 参照: 按 ref_ids 取归一化权重
    results = kv.query(q, top_k=3)
    assert [r[0] for r in results] == ref_ids
    assert abs(sum(r[2] for r in results) - 1.0) < 1e-3
    for (rid, vec, w) in results:
        e = kv.entries[rid]
        assert np.allclose(vec, e.value_state * kv.value_w * w, atol=1e-9)


def test_kv_lru_eviction_vectorized():
    """容量满时按 last_access 淘汰最久未访问条目 (swap-remove 索引正确)"""
    kv = KVStack(capacity=8, dim=16)
    rng = np.random.RandomState(5)
    for i in range(8):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
        time.sleep(0.002)  # 保证 last_access 时钟可区分 (Windows 精度)
    # 访问 k0 使其变新, 再压入 3 条 → 淘汰应为 k1,k2,k3
    kv.retrieve(kv.entries["k0"].key_signal, top_k=1)
    for i in range(8, 11):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    ids = set(kv.entries)
    assert "k0" in ids
    assert not ({"k1", "k2", "k3"} & ids)
    assert len(kv.entries) == 8
    assert kv._n == 8
    # 淘汰后检索仍正确
    out = kv.retrieve(rng.randn(16))
    assert out.shape == (16,)


def test_kv_duplicate_push_overwrites():
    """重复 id 覆盖: 数量不变, 向量更新, 插入序保留"""
    kv = KVStack(capacity=10, dim=16)
    kv.push("a", np.ones(16), np.ones(16))
    kv.push("b", np.ones(16), np.ones(16))
    kv.push("a", np.full(16, 2.0), np.full(16, 3.0))
    assert len(kv.entries) == 2
    assert kv._n == 2
    assert np.allclose(kv.entries["a"].key_signal, 2.0)
    assert np.allclose(kv.entries["a"].value_state, 3.0)


def test_kv_external_clear_resync():
    """外部 entries.clear() (chat 模块用法) 后索引自动重建"""
    kv = KVStack(capacity=10, dim=16)
    rng = np.random.RandomState(1)
    for i in range(5):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    kv.entries.clear()
    assert np.allclose(kv.retrieve(np.ones(16)), 0)
    kv.push("new", np.ones(16), rng.randn(16))
    out = kv.retrieve(np.ones(16))
    assert np.linalg.norm(out) > 0
    assert len(kv.entries) == 1


def test_kv_save_load_roundtrip_vectorized():
    """存档往返后矩阵索引重建, 检索结果与参考实现一致"""
    import tempfile, os
    kv = KVStack(capacity=100, dim=16)
    rng = np.random.RandomState(9)
    for i in range(20):
        kv.push(f"k{i}", rng.randn(16), rng.randn(16))
    q = rng.randn(16)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kv.json")
        kv.save_to_disk(p)
        kv2 = KVStack(capacity=100, dim=16)
        kv2.load_from_disk(p)
    ref, _ = _ref_retrieve(kv2, q, top_k=3)
    out = kv2.retrieve(q, top_k=3)
    assert np.allclose(out, ref, atol=1e-6)



def test_kv_retrieve_returns_dim_vector():
    kv = KVStack(capacity=100, dim=16)
    assert np.allclose(kv.retrieve(np.ones(16)), 0)  # 空堆 → 零向量
    for i in range(10):
        kv.push(f"k{i}", np.ones(16), np.random.RandomState(i).randn(16))
    v = kv.retrieve(np.ones(16))
    assert v.shape == (16,)
    assert np.linalg.norm(v) > 0


def test_kv_retrieve_scan_limit():
    kv = KVStack(capacity=100, dim=16)
    for i in range(50):
        kv.push(f"k{i}", np.ones(16), np.ones(16))
    v = kv.retrieve(np.ones(16), scan_limit=10)
    assert np.linalg.norm(v) > 0


def test_attention_changes_state():
    """KV 堆有记忆时, 状态应与空 KV 时不同 (注意力接入主路径)"""
    g = CubeGPT(depth=0, dim=16)
    g2 = CubeGPT(depth=0, dim=16)
    for i in range(20):
        g.kv_stack.push(f"m{i}", np.ones(16) * 0.5, np.random.RandomState(i).randn(16) * 3)
    for _ in range(3):
        g.step({"numeric": 1.0})
        g2.step({"numeric": 1.0})
    s1 = np.concatenate([u.state for u in g.output_module.units])
    s2 = np.concatenate([u.state for u in g2.output_module.units])
    assert not np.allclose(s1, s2)


def test_api_call_no_endpoint_simulated(monkeypatch):
    monkeypatch.delenv("DF_WEBHOOK_URL", raising=False)
    from src.agents.base_agent import ActionAgent
    a = ActionAgent("a", df_depth=0)
    r = a._do_api_call({"payload": {}})
    assert r["status"] == "simulated"
    assert r["reason"] == "no_endpoint"


def test_api_call_real_http(monkeypatch):
    """mock urlopen, 验证真实请求路径的封装逻辑"""
    import io as _io
    from src.agents import base_agent

    class FakeResp:
        status = 200
        def read(self, n): return b'{"ok": true}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    captured = {}
    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    a = base_agent.ActionAgent("a", df_depth=0)
    r = a._do_api_call({"endpoint": "http://example.com/hook", "payload": {"x": 1}})
    assert r["status"] == "success"
    assert r["http_status"] == 200
    assert captured["url"] == "http://example.com/hook"
    assert b'"x"' in captured["data"]


def test_notification_sim_mode(monkeypatch):
    monkeypatch.setenv("DF_NOTIFY_MODE", "sim")
    from src.agents.base_agent import ActionAgent
    a = ActionAgent("a", df_depth=0)
    r = a._do_notification({"title": "t", "message": "m"})
    assert r["status"] == "success"
    assert r["mode"] == "sim"
