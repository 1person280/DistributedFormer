import numpy as np
import pytest

from distributedformer.core.distributedformer import KVStack, CubeGPT


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
    from distributedformer.agents.base_agent import ActionAgent
    a = ActionAgent("a", df_depth=0)
    r = a._do_api_call({"payload": {}})
    assert r["status"] == "simulated"
    assert r["reason"] == "no_endpoint"


def test_api_call_real_http(monkeypatch):
    """mock urlopen, 验证真实请求路径的封装逻辑"""
    import io as _io
    from distributedformer.agents import base_agent

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
    from distributedformer.agents.base_agent import ActionAgent
    a = ActionAgent("a", df_depth=0)
    r = a._do_notification({"title": "t", "message": "m"})
    assert r["status"] == "success"
    assert r["mode"] == "sim"
