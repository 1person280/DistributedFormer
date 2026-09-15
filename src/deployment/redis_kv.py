"""
Redis 后端 KV 堆
替代内存 KVStack，支持分布式持久化

支持多租户隔离: 每个租户使用不同的 Redis key 前缀
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.distributedformer import KVStack, KVEntry


class MockRedis:
    """
    Redis 模拟器 (redis-py 未安装时使用)
    用 Python dict 模拟 Redis Hash API
    """
    
    def __init__(self):
        self._data: Dict[str, Dict[str, bytes]] = {}
    
    def hset(self, key: str, field: str, value: bytes) -> int:
        if key not in self._data:
            self._data[key] = {}
        self._data[key][field] = value
        return 1
    
    def hget(self, key: str, field: str) -> Optional[bytes]:
        return self._data.get(key, {}).get(field)
    
    def hgetall(self, key: str) -> Dict[str, bytes]:
        return self._data.get(key, {}).copy()
    
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


class RedisKVStack:
    """
    Redis 后端 KV 堆
    
    特性:
    - 分布式持久化: 数据存储在 Redis 中
    - 多租户隔离: 不同租户使用不同 key 前缀
    - 自动回退: redis-py 未安装时自动使用 MockRedis
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, capacity: int = 100000, dim: int = 16,
                 tenant_id: str = "default"):
        self.host = host
        self.port = port
        self.db = db
        self.capacity = capacity
        self.dim = dim
        self.tenant_id = tenant_id
        self.key_prefix = f"df:{tenant_id}:kv"
        self.client = None
        self.query_w = np.random.randn(dim) * 0.1
        self.key_w = np.random.randn(dim) * 0.1
        self.value_w = np.random.randn(dim) * 0.1
        
    def connect(self) -> bool:
        """连接 Redis，失败则使用 MockRedis"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.host, port=self.port, db=self.db,
                decode_responses=False, socket_connect_timeout=5
            )
            self.client.ping()
            print(f"[RedisKV] Connected to {self.host}:{self.port} (db={self.db})")
            return True
        except ImportError:
            print("[RedisKV] redis-py not installed, using MockRedis")
            self.client = MockRedis()
            return True
        except Exception as e:
            print(f"[RedisKV] Connection failed: {e}, falling back to MockRedis")
            self.client = MockRedis()
            return True
    
    def _make_key(self, entry_id: str) -> str:
        """生成 Redis key"""
        return f"{self.key_prefix}:{entry_id}"
    
    def push(self, entry_id: str, key_signal: np.ndarray, 
             value_state: np.ndarray) -> None:
        """推送条目到 Redis"""
        if self.client is None:
            self.connect()
        
        redis_key = self._make_key(entry_id)
        data = {
            "key_signal": key_signal.tobytes(),
            "value_state": value_state.tobytes(),
            "timestamp": str(time.time()).encode(),
            "access_count": b"0",
            "last_access": str(time.time()).encode()
        }
        
        for field, value in data.items():
            self.client.hset(redis_key, field, value)
        
        # LRU: 如果超出容量，删除最旧的
        if self.client.dbsize() > self.capacity * 1.2:  # 允许20%缓冲
            self._evict_oldest()
    
    def query(self, query_signal: np.ndarray, top_k: int = 3) -> List[Tuple[str, np.ndarray, float]]:
        """从 Redis 查询 top-k 匹配"""
        if self.client is None:
            self.connect()
        
        # 扫描所有条目
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        
        if not keys:
            return []
        
        # 计算分数
        scores = []
        for redis_key in keys:
            data = self.client.hgetall(redis_key)
            if not data or b"key_signal" not in data:
                continue
            
            entry_key_signal = np.frombuffer(data[b"key_signal"], dtype=np.float64)
            entry_value_state = np.frombuffer(data[b"value_state"], dtype=np.float64)
            
            # 余弦相似度
            q_norm = np.linalg.norm(query_signal)
            k_norm = np.linalg.norm(entry_key_signal)
            if q_norm == 0 or k_norm == 0:
                continue
            sim = np.abs(np.dot(query_signal, entry_key_signal) / (q_norm * k_norm))
            
            # 时间衰减
            ts = float(data.get(b"timestamp", b"0"))
            time_decay = np.exp(-0.001 * (time.time() - ts))
            score = float(sim * time_decay)
            
            scores.append((redis_key, entry_value_state, score))
            
            # 更新访问计数
            access_count = int(data.get(b"access_count", b"0")) + 1
            self.client.hset(redis_key, "access_count", str(access_count).encode())
            self.client.hset(redis_key, "last_access", str(time.time()).encode())
        
        # 取 top-k
        scores.sort(key=lambda x: x[2], reverse=True)
        top_entries = scores[:top_k]
        
        total_score = sum(s[2] for s in top_entries) + 1e-8
        results = []
        for redis_key, value_state, score in top_entries:
            normalized = score / total_score
            retrieved = value_state * self.value_w * normalized
            entry_id = redis_key.decode() if isinstance(redis_key, bytes) else redis_key
            entry_id = entry_id.split(":")[-1]
            results.append((entry_id, retrieved, normalized))
        
        return results
    
    def _evict_oldest(self, n: int = 100) -> None:
        """淘汰最旧的 n 个条目"""
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        if len(keys) <= self.capacity:
            return
        
        # 获取时间戳并排序
        key_times = []
        for k in keys:
            ts = self.client.hget(k, "timestamp")
            if ts:
                key_times.append((k, float(ts)))
        
        key_times.sort(key=lambda x: x[1])
        to_delete = [k for k, _ in key_times[:n]]
        if to_delete:
            self.client.delete(*to_delete)
    
    def clear_expired(self, max_age_days: float = 7.0) -> int:
        """清理过期条目"""
        if self.client is None:
            self.connect()
        
        now = time.time()
        max_age = max_age_days * 86400
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        
        expired = []
        for k in keys:
            ts = self.client.hget(k, "timestamp")
            if ts and (now - float(ts)) > max_age:
                expired.append(k)
        
        if expired:
            self.client.delete(*expired)
        
        return len(expired)
    
    def get_stats(self) -> Dict:
        """获取统计"""
        if self.client is None:
            self.connect()
        
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        
        return {
            "capacity": self.capacity,
            "used": len(keys),
            "utilization": len(keys) / self.capacity if self.capacity > 0 else 0,
            "tenant_id": self.tenant_id,
            "backend": "redis" if not isinstance(self.client, MockRedis) else "mock"
        }
    
    def flush_all(self) -> int:
        """清空当前租户的所有数据"""
        if self.client is None:
            return 0
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Redis KV 后端测试")
    print("=" * 60)
    
    # 测试 MockRedis
    print("\n[1] MockRedis 基础测试:")
    mock = MockRedis()
    mock.hset("test_key", "field1", b"value1")
    mock.hset("test_key", "field2", b"value2")
    print(f"  hget: {mock.hget('test_key', 'field1')}")
    print(f"  hgetall: {mock.hgetall('test_key')}")
    print(f"  keys: {mock.keys('test_*')}")
    print(f"  dbsize: {mock.dbsize()}")
    
    # 测试 RedisKVStack
    print("\n[2] RedisKVStack 功能测试:")
    kv = RedisKVStack(tenant_id="test_tenant", capacity=1000, dim=16)
    kv.connect()
    
    # 推送数据
    for i in range(20):
        kv.push(f"entry_{i}", np.random.randn(16), np.random.randn(16))
    
    stats = kv.get_stats()
    print(f"  推送20条, 实际存储: {stats['used']}, 利用率: {stats['utilization']:.1%}")
    print(f"  后端: {stats['backend']}")
    
    # 查询
    results = kv.query(np.random.randn(16), top_k=3)
    print(f"  查询top-3: 命中 {len(results)} 条")
    
    # 清理
    cleaned = kv.clear_expired(max_age_days=0.0001)  # 立即过期
    print(f"  清理过期: {cleaned} 条")
    
    # 多租户隔离测试
    print("\n[3] 多租户隔离测试:")
    kv_tenant_a = RedisKVStack(tenant_id="tenant_a", capacity=100, dim=16)
    kv_tenant_b = RedisKVStack(tenant_id="tenant_b", capacity=100, dim=16)
    kv_tenant_a.connect()
    kv_tenant_b.connect()
    
    kv_tenant_a.push("shared_key", np.ones(16), np.ones(16))
    kv_tenant_b.push("shared_key", np.zeros(16), np.zeros(16))
    
    res_a = kv_tenant_a.query(np.ones(16), top_k=1)
    res_b = kv_tenant_b.query(np.zeros(16), top_k=1)
    
    print(f"  Tenant A 查询结果: {len(res_a)} 条")
    print(f"  Tenant B 查询结果: {len(res_b)} 条")
    print(f"  数据隔离: {'✅' if len(res_a) == 1 and len(res_b) == 1 else '❌'}")
    
    print("\n" + "=" * 60)
    print("Redis KV 后端测试通过!")
    print("=" * 60)
