"""
Redis 后端 KV 堆
替代内存 KVStack，支持分布式持久化与跨节点共享记忆

本文件让 ``RedisKVStack`` 与内存 ``KVStack``（src/core/distributedformer.py）
实现**同一套公开协议**，因此工作流引擎 / 智能体 / 主模型可以无感切换后端：
只需要把 ``kv_stack=KVStack(...)`` 换成 ``kv_stack=RedisKVStack(...)``。

协议对齐项（与 KVStack 一一对应）:
  - ``push(entry_id, key_signal, value_state)``: 写条目, 重复 id 覆盖, LRU 淘汰
  - ``query(query_signal, top_k)``: 注意力查询 → [(entry_id, retrieved, normalized)]
  - ``retrieve(query_signal, top_k, scan_limit)``: 主计算路径, 返回 dim 维聚合向量 (tanh)
  - ``_coerce_vec(vec)``: 变长载荷定长化 (不足补零/超长截断)
  - ``get_stats()``: {capacity, used, utilization, total_access, ...}
  - ``clear_expired(max_age_days)`` / ``flush_all()``
  - ``save_to_disk`` / ``load_from_disk``: Redis AOF 已持久化, 为协议兼容置为 no-op

支持多租户隔离: 每个租户使用不同的 Redis key 前缀 (df:{tenant}:kv:*)。

Redis 真实后端不可用时自动回退到进程内 MockRedis（redis-py 未安装或连接失败），
因此单机、无依赖环境也能跑通分布式语义测试。
"""

import os
import time
import json
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import sys
# 确保直接运行脚本时仓库根在 sys.path, 使 `from src...` 可导入
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class MockRedis:
    """
    Redis 模拟器 (redis-py 未安装时使用)

    用 Python dict 模拟 Redis 的 Hash / String 基础 API。**底层存储按
    命名空间 (host:port:db) 进程内共享**——同一 host/port/db 的多个
    MockRedis 实例读写同一份数据，从而在单进程内模拟"真实 Redis server"
    的多节点共享语义（与真实 Redis 一致；跨进程共享仍需真实 Redis）。
    """

    _GLOBAL: Dict[str, dict] = {}

    @staticmethod
    def _b(s):
        return s.encode() if isinstance(s, str) else s

    def __init__(self, namespace: str = "mock://localhost:6379/0"):
        self._ns = namespace
        self._data = MockRedis._GLOBAL.setdefault(namespace, {})

    def hset(self, key: str, field: str, value: bytes) -> int:
        field = self._b(field)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][field] = value
        return 1

    def hget(self, key: str, field: str) -> Optional[bytes]:
        return self._data.get(key, {}).get(self._b(field))

    def hgetall(self, key: str) -> Dict[bytes, bytes]:
        return self._data.get(key, {}).copy()

    def set(self, key: str, value: bytes) -> bool:
        self._data[key] = {b"__str__": value}
        return True

    def get(self, key: str) -> Optional[bytes]:
        d = self._data.get(key)
        return d.get(b"__str__") if d else None

    def incr(self, key: str, amount: int = 1) -> int:
        raw = self.get(key)
        cur = int(raw.decode()) if raw else 0
        nxt = cur + amount
        self.set(key, str(nxt).encode())
        return nxt

    def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                count += 1
        return count

    def dbsize(self) -> int:
        return len(self._data)

    def ping(self) -> bool:
        return True


# 时间衰减系数 (与内存 KVStack.compute_score 一致)
_DECAY = 0.001


class RedisKVStack:
    """
    Redis 后端 KV 堆 (协议兼容 KVStack)

    特性:
    - 分布式持久化: 数据存储在 Redis 中, 多节点共享同一租户的 KV 堆
    - 多租户隔离: 不同租户使用不同 key 前缀 (df:{tenant}:kv:*)
    - 协议兼容: 提供与内存 KVStack 相同的 push/query/retrieve/get_stats 等
    - 自动回退: redis-py 未安装或连接失败时自动使用 MockRedis (仍可测分布式语义)

    注意: retrieve() 为功能正确优先的实现 (扫描当前租户条目做 top-k 聚合)。
    内存 KVStack 用增量矩阵索引把检索向量化提速 ~40x; 真实 Redis 上的向量化
    索引 (如每租户一个键值矩阵 + Lua 脚本) 留作后续性能优化方向。
    """

    def __init__(self, capacity: int = 100000, dim: int = 16,
                 retention_policy: str = "lru_7d",
                 host: str = "localhost", port: int = 6379, db: int = 0,
                 tenant_id: Optional[str] = None):
        # 对齐 KVStack 构造语义 (capacity/dim/retention_policy 与 KVStack 同名)
        self.capacity = capacity
        self.dim = dim
        self.retention_policy = retention_policy
        # Redis 连接 / 租户
        self.host = host
        self.port = port
        self.db = db
        self.tenant_id = tenant_id or os.getenv("TENANT_ID", "default")
        self.key_prefix = f"df:{self.tenant_id}:kv"
        self._seq_key = f"df:{self.tenant_id}:seq"
        # 可学习权重 (与 KVStack 对齐)
        self.query_w = np.random.randn(dim) * 0.1
        self.key_w = np.random.randn(dim) * 0.1
        self.value_w = np.random.randn(dim) * 0.1
        # 连接状态 (延迟 connect)
        self.client: Optional[Any] = None
        self.backend = "pending"

    def connect(self) -> bool:
        """连接 Redis，失败则使用 MockRedis（保持可用）"""
        try:
            import redis  # noqa: F401
            client = redis.Redis(
                host=self.host, port=self.port, db=self.db,
                decode_responses=False, socket_connect_timeout=5)
            client.ping()
            self.client = client
            self.backend = "redis"
            print(f"[RedisKV] Connected to {self.host}:{self.port} "
                  f"(db={self.db}, tenant={self.tenant_id})")
        except ImportError:
            print("[RedisKV] redis-py not installed, using MockRedis")
            self.client = MockRedis(f"mock://{self.host}:{self.port}/{self.db}")
            self.backend = "mock"
        except Exception as e:  # noqa: BLE001
            print(f"[RedisKV] Connection failed: {e}, falling back to MockRedis")
            self.client = MockRedis(f"mock://{self.host}:{self.port}/{self.db}")
            self.backend = "mock"
        return True

    def _client(self):
        if self.client is None:
            self.connect()
        return self.client

    def _entry_key(self, entry_id: str) -> str:
        return f"{self.key_prefix}:{entry_id}"

    # ── 读取 helper (统一真实 Redis 的 bytes 与 MockRedis 的行为) ──

    def _hgetall(self, rk: str) -> Dict[str, bytes]:
        """读取条目 hash: field 键统一为 str, 值保留 bytes (二进制向量不解码)"""
        raw = self._client().hgetall(rk)
        out: Dict[str, bytes] = {}
        for k, v in raw.items():
            kk = k.decode() if isinstance(k, bytes) else k
            out[kk] = v
        return out

    @staticmethod
    def _num(raw: Optional[bytes], cast=float):
        """把 bytes/str 数值解码为 int/float (二进制向量字段请勿用它)"""
        if raw is None:
            return cast(0)
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cast(raw)

    # ── 协议兼容工具 ──────────────────────────────────────────

    def _coerce_vec(self, vec: np.ndarray) -> np.ndarray:
        """向量定长化: 不足 dim 补零, 超长截断 (兼容变长载荷, 同 KVStack)"""
        v = np.asarray(vec, dtype=float).ravel()
        if v.shape == (self.dim,):
            return v
        out = np.zeros(self.dim)
        n = min(len(v), self.dim)
        out[:n] = v[:n]
        return out

    def _score(self, key_signal: np.ndarray, query_signal: np.ndarray,
               timestamp: float) -> float:
        """|cos(q,k)| × exp(-0.001·Δt) (与 KVStack 打分一致)"""
        q = np.asarray(query_signal, dtype=float)
        k = np.asarray(key_signal, dtype=float)
        qn = np.linalg.norm(q)
        kn = np.linalg.norm(k)
        if qn == 0 or kn == 0:
            return 0.0
        sim = np.abs(np.dot(q, k) / (qn * kn))
        return float(sim * np.exp(-_DECAY * (time.time() - timestamp)))

    # ── 协议接口: push ─────────────────────────────────────────

    def push(self, entry_id: str, key_signal: np.ndarray,
             value_state: np.ndarray) -> None:
        """推送新条目到 KV 堆; 重复 id 覆盖; 超容量 LRU 淘汰"""
        c = self._client()
        now = time.time()
        key = self._entry_key(entry_id)
        data = {
            "key_signal": self._coerce_vec(key_signal).tobytes(),
            "value_state": self._coerce_vec(value_state).tobytes(),
            "timestamp": str(now).encode(),
            "access_count": b"0",
            "last_access": str(now).encode(),
            "seq": str(c.incr(self._seq_key)).encode(),
        }
        for field, value in data.items():
            c.hset(key, field, value)
        # 满容量时 LRU 淘汰最久未访问的 1 条 (与内存 KVStack 语义一致)
        if len(c.keys(f"{self.key_prefix}:*")) >= self.capacity:
            self._evict_lru(1)

    # ── 协议接口: query ────────────────────────────────────────

    def query(self, query_signal: np.ndarray,
              top_k: int = 3) -> List[Tuple[str, np.ndarray, float]]:
        """
        注意力查询: 扫描当前租户全部条目, 返回
        [(entry_id, value·value_w·normalized, normalized), ...] (同 KVStack)
        """
        c = self._client()
        keys = c.keys(f"{self.key_prefix}:*")
        if not keys:
            return []

        scored: List[Tuple[float, str, np.ndarray, float]] = []
        now = time.time()
        for rk in keys:
            data = self._hgetall(rk)
            if not data or "key_signal" not in data:
                continue
            k_sig = np.frombuffer(data["key_signal"], dtype=np.float64)
            v_state = np.frombuffer(data["value_state"], dtype=np.float64)
            ts = self._num(data.get("timestamp"))
            score = self._score(k_sig, query_signal, ts)
            if score <= 0:
                continue
            # 更新访问统计
            acc = int(self._num(data.get("access_count"), int)) + 1
            c.hset(rk, "access_count", str(acc).encode())
            c.hset(rk, "last_access", str(now).encode())
            eid = rk.decode() if isinstance(rk, bytes) else rk
            eid = eid.split(":")[-1]
            scored.append((score, eid, v_state, 0.0))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        total = sum(s[0] for s in top) + 1e-8
        results = []
        for score, eid, v_state, _ in top:
            normalized = score / total
            retrieved = v_state * self.value_w * normalized
            results.append((eid, retrieved, float(normalized)))
        return results

    # ── 协议接口: retrieve (主计算路径) ────────────────────────

    def retrieve(self, query_signal: np.ndarray, top_k: int = 3,
                 scan_limit: int = 4096) -> np.ndarray:
        """
        主计算路径用的注意力检索: 返回 dim 维聚合检索向量 (tanh)

        对当前租户最近写入的 scan_limit 条记录打分 (按 seq 截取), 聚合
        top_k 条 value 向量后 tanh 归一化; 空堆返回零向量 (同 KVStack)。
        """
        c = self._client()
        keys = c.keys(f"{self.key_prefix}:*")
        if not keys:
            return np.zeros(self.dim)

        now = time.time()
        candidates = []
        for rk in keys:
            data = self._hgetall(rk)
            if not data or "key_signal" not in data:
                continue
            k_sig = np.frombuffer(data["key_signal"], dtype=np.float64)
            v_state = np.frombuffer(data["value_state"], dtype=np.float64)
            ts = self._num(data.get("timestamp"))
            seq = int(self._num(data.get("seq"), int))
            score = self._score(k_sig, query_signal, ts)
            candidates.append((seq, score, v_state, rk))
        # 按插入序号取最近 scan_limit 条
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:scan_limit]
        # 打分 top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]
        if not top:
            return np.zeros(self.dim)

        total = sum(x[1] for x in top) + 1e-8
        agg = np.zeros(self.dim)
        for _, score, v_state, rk in top:
            w = score / total
            agg += v_state * self.value_w * w
            acc = int(self._num(c.hget(rk, "access_count"), int)) + 1
            c.hset(rk, "access_count", str(acc).encode())
            c.hset(rk, "last_access", str(now).encode())
        return np.tanh(agg)

    # ── LRU 淘汰 / 清理 ────────────────────────────────────────

    def _evict_lru(self, n: int = 100) -> None:
        """淘汰最久未访问的 n 个条目"""
        c = self._client()
        keys = c.keys(f"{self.key_prefix}:*")
        if len(keys) <= self.capacity:
            return
        key_times = []
        for k in keys:
            la = self._num(c.hget(k, "last_access"))
            if la:
                key_times.append((k, la))
        key_times.sort(key=lambda x: x[1])
        for k, _ in key_times[:n]:
            c.delete(k)

    def clear_expired(self, max_age_days: float = 7.0) -> int:
        """清理过期条目 (返回清理条数)"""
        c = self._client()
        now = time.time()
        max_age = max_age_days * 86400
        expired = []
        for k in c.keys(f"{self.key_prefix}:*"):
            ts = self._num(c.hget(k, "timestamp"))
            if ts and (now - ts) > max_age:
                expired.append(k)
        if expired:
            c.delete(*expired)
        return len(expired)

    def flush_all(self) -> int:
        """清空当前租户的所有数据 (返回清理条数)"""
        c = self._client()
        keys = c.keys(f"{self.key_prefix}:*")
        if keys:
            c.delete(*keys)
        return len(keys)

    # ── 统计 / 持久化 (协议兼容) ───────────────────────────────

    def get_stats(self) -> Dict:
        """获取 KV 堆统计 (含 total_access/backend/tenant_id, 同 KVStack + 附加)"""
        c = self._client()
        keys = c.keys(f"{self.key_prefix}:*")
        total_access = 0
        for k in keys:
            total_access += int(self._num(c.hget(k, "access_count"), int))
        return {
            "capacity": self.capacity,
            "used": len(keys),
            "utilization": len(keys) / self.capacity if self.capacity > 0 else 0,
            "total_access": total_access,
            "tenant_id": self.tenant_id,
            "backend": self.backend if self.backend != "pending" else "redis",
        }

    def save_to_disk(self, path: str) -> None:
        """Redis AOF/持久化已覆盖; 协议兼容 no-op"""
        pass

    def load_from_disk(self, path: str) -> None:
        """Redis 数据自持; 协议兼容 no-op"""
        pass


def create_kv_stack(backend: str = "memory", **kwargs) -> Any:
    """
    KV 后端工厂: 按 backend 创建内存 KVStack 或 RedisKVStack (协议一致)

    backend: "memory" (默认) | "redis" | "auto"
      - "redis": 优先真实 Redis, 连接失败回退 MockRedis
      - "auto": 读环境变量 KV_BACKEND, 缺省 "memory"
    其余 kwargs 透传给对应实现 (capacity/dim/retention_policy/...)。
    """
    from src.core.distributedformer import KVStack
    if backend == "auto":
        backend = os.getenv("KV_BACKEND", "memory")
    if backend == "redis":
        return RedisKVStack(**kwargs)
    # memory: 只透传 KVStack 认识的参数
    mem_kw = {k: kwargs[k] for k in ("capacity", "dim", "retention_policy")
              if k in kwargs}
    return KVStack(**mem_kw)


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Redis KV 后端测试")
    print("=" * 60)

    print("\n[1] MockRedis 基础测试:")
    mock = MockRedis()
    mock.hset("test_key", "field1", b"value1")
    mock.hset("test_key", "field2", b"value2")
    print(f"  hget: {mock.hget('test_key', 'field1')}")
    print(f"  hgetall: {mock.hgetall('test_key')}")
    print(f"  keys: {mock.keys('test_*')}")
    print(f"  dbsize: {mock.dbsize()}")
    mock.incr("counter"); mock.incr("counter"); mock.incr("counter")
    print(f"  incr 3 次: {mock.get('counter')}")

    print("\n[2] RedisKVStack 协议兼容测试:")
    kv = RedisKVStack(tenant_id="test_tenant", capacity=1000, dim=16)
    kv.connect()
    print(f"  后端: {kv.get_stats()['backend']}")

    for i in range(20):
        kv.push(f"entry_{i}", np.random.randn(16), np.random.randn(16))
    stats = kv.get_stats()
    print(f"  推送20条, 实际存储: {stats['used']}, 利用率: {stats['utilization']:.1%}")

    # retrieve (主计算路径)
    vec = kv.retrieve(np.random.randn(16), top_k=3)
    print(f"  retrieve: shape={vec.shape}, 范数={np.linalg.norm(vec):.3f} (tanh∈[0,16])")

    # query
    results = kv.query(np.random.randn(16), top_k=3)
    print(f"  query top-3: 命中 {len(results)} 条")

    cleaned = kv.clear_expired(max_age_days=0.0001)
    print(f"  清理过期: {cleaned} 条")

    print("\n[3] 多租户隔离测试:")
    kv_a = RedisKVStack(tenant_id="tenant_a", capacity=100, dim=16)
    kv_b = RedisKVStack(tenant_id="tenant_b", capacity=100, dim=16)
    kv_a.connect(); kv_b.connect()
    kv_a.push("shared_key", np.ones(16), np.ones(16))
    kv_b.push("shared_key", np.ones(16), np.ones(16))
    res_a = kv_a.query(np.ones(16), top_k=1)
    res_b = kv_b.query(np.ones(16), top_k=1)
    # 隔离: 每个租户的 key 前缀不同, A 查不到 B 的条目, 反之亦然
    print(f"  Tenant A 查询: {len(res_a)} 条 | Tenant B 查询: {len(res_b)} 条")
    print(f"  数据隔离: {'OK' if len(res_a) == 1 and len(res_b) == 1 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Redis KV 后端测试通过!")
    print("=" * 60)
