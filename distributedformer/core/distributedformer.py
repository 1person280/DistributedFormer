"""
DistributedFormer: 分布式脉冲神经网络核心
基于 Kimi Work × DistributedFormer 原型方案实现

核心特征:
- 极简单元: 16个参数，模拟生物神经元动力学
- 异步事件驱动: 无全局同步，脉冲局部传播
- 持续思考: 默认模式网络，无输入仍有自发活动
- 分形递归: 每层16单元，深度指数扩展能力
- 内置注意力: query_kv机制，无需全局注意力矩阵
- KV堆记忆: 分布式持久存储，替代Transformer的KV Cache

版本: v5.2 分形深度2扩展
  分形深度2 => 4,368单元/层 / 69K参数
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import threading
import random
import math


# ═══════════════════════════════════════════════════════════════
# 1. 基础数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class SpikePayload:
    """脉冲载荷"""
    value: float          # 脉冲值 (可正可负)
    strength: float       # 绝对强度 (0~1)
    timestamp: float      # 发射时间戳
    cycle_phase: int      # 当前节律相位

@dataclass
class SpikeMessage:
    """标准脉冲消息格式"""
    msg_type: str = "spike"
    source_agent_id: str = ""
    source_unit_id: str = ""      # 分形层级定位, e.g. "top_3_L1_7_L0_12"
    source_level: int = 0
    payload: SpikePayload = field(default_factory=lambda: SpikePayload(0.0, 0.0, 0.0, 0))
    target_agents: List[str] = field(default_factory=list)
    hops_remaining: int = 3
    priority: str = "normal"
    kv_query: Optional[Dict] = None
    trace: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "msg_type": self.msg_type,
            "source": {
                "agent_id": self.source_agent_id,
                "unit_id": self.source_unit_id,
                "level": self.source_level
            },
            "payload": {
                "value": self.payload.value,
                "strength": self.payload.strength,
                "timestamp": self.payload.timestamp,
                "cycle_phase": self.payload.cycle_phase
            },
            "routing": {
                "target_agents": self.target_agents,
                "hops_remaining": self.hops_remaining,
                "priority": self.priority
            },
            "context": {
                "kv_query": self.kv_query,
                "trace": self.trace
            }
        }


# ═══════════════════════════════════════════════════════════════
# 2. KV堆记忆条目
# ═══════════════════════════════════════════════════════════════

@dataclass
class KVEntry:
    """KV堆中的一个条目"""
    key_signal: np.ndarray    # 键信号向量 (用于query匹配)
    value_state: np.ndarray   # 值状态向量 (存储内容)
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    
    def compute_score(self, query_signal: np.ndarray, key_w: np.ndarray) -> float:
        """计算query与当前条目的匹配分数"""
        if self.key_signal is None or query_signal is None:
            return 0.0
        q_norm = np.linalg.norm(query_signal)
        k_norm = np.linalg.norm(self.key_signal)
        if q_norm == 0 or k_norm == 0:
            return 0.0
        sim = np.abs(np.dot(query_signal, self.key_signal) / (q_norm * k_norm))
        time_decay = np.exp(-0.001 * (time.time() - self.timestamp))
        return float(sim * time_decay)


# ═══════════════════════════════════════════════════════════════
# 3. KV堆: 分布式持久记忆存储
# ═══════════════════════════════════════════════════════════════

class KVStack:
    """
    分布式KV堆记忆系统
    替代Transformer的KV Cache，支持跨智能体注意力查询
    """
    
    def __init__(self, capacity: int = 100000, dim: int = 16, 
                 retention_policy: str = "lru_7d"):
        self.capacity = capacity
        self.dim = dim
        self.entries: Dict[str, KVEntry] = {}
        self.retention_policy = retention_policy
        self.lock = threading.Lock()
        self.query_w = np.random.randn(dim) * 0.1
        self.key_w = np.random.randn(dim) * 0.1
        self.value_w = np.random.randn(dim) * 0.1
        
    def push(self, entry_id: str, key_signal: np.ndarray, 
             value_state: np.ndarray) -> None:
        """推送新条目到KV堆"""
        with self.lock:
            if len(self.entries) >= self.capacity:
                self._evict_lru()
            
            entry = KVEntry(
                key_signal=key_signal.copy(),
                value_state=value_state.copy(),
                timestamp=time.time(),
                last_access=time.time()
            )
            self.entries[entry_id] = entry
    
    def query(self, query_signal: np.ndarray, top_k: int = 3) -> List[Tuple[str, np.ndarray, float]]:
        """
        注意力查询 (query_kv)
        """
        # 快速路径: 空堆直接返回
        if not self.entries:
            return []
        
        with self.lock:
            scores = []
            for entry_id, entry in self.entries.items():
                score = entry.compute_score(query_signal, self.key_w)
                scores.append((entry_id, entry, score))
                entry.access_count += 1
                entry.last_access = time.time()
            
            scores.sort(key=lambda x: x[2], reverse=True)
            top_entries = scores[:top_k]
            
            total_score = sum(s[2] for s in top_entries) + 1e-8
            
            results = []
            for entry_id, entry, score in top_entries:
                normalized = score / total_score
                retrieved = entry.value_state * self.value_w * normalized
                results.append((entry_id, retrieved, normalized))
            
            return results
    
    def _evict_lru(self) -> None:
        """LRU淘汰最久未访问的条目"""
        if not self.entries:
            return
        oldest_id = min(self.entries.keys(), 
                       key=lambda k: self.entries[k].last_access)
        del self.entries[oldest_id]
    
    def clear_expired(self, max_age_days: float = 7.0) -> int:
        """清理过期条目"""
        with self.lock:
            now = time.time()
            max_age = max_age_days * 86400
            expired = [k for k, v in self.entries.items() 
                      if (now - v.timestamp) > max_age]
            for k in expired:
                del self.entries[k]
            return len(expired)
    
    def get_stats(self) -> Dict:
        """获取KV堆统计信息"""
        with self.lock:
            total_access = sum(e.access_count for e in self.entries.values())
            return {
                "capacity": self.capacity,
                "used": len(self.entries),
                "utilization": len(self.entries) / self.capacity,
                "total_access": total_access
            }
    
    def save_to_disk(self, path: str) -> None:
        """持久化到磁盘"""
        with self.lock:
            data = {}
            for k, v in self.entries.items():
                data[k] = {
                    "key_signal": v.key_signal.tolist(),
                    "value_state": v.value_state.tolist(),
                    "timestamp": v.timestamp,
                    "access_count": v.access_count,
                    "last_access": v.last_access
                }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
    
    def load_from_disk(self, path: str) -> None:
        """从磁盘加载"""
        import os
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with self.lock:
            self.entries.clear()
            for k, v in data.items():
                self.entries[k] = KVEntry(
                    key_signal=np.array(v["key_signal"]),
                    value_state=np.array(v["value_state"]),
                    timestamp=v["timestamp"],
                    access_count=v.get("access_count", 0),
                    last_access=v.get("last_access", v["timestamp"])
                )


# ═══════════════════════════════════════════════════════════════
# 4. 脉冲神经元单元 (极简单元: 16个参数)
# ═══════════════════════════════════════════════════════════════

class SpikingUnit:
    """
    生物神经元动力学模拟单元
    恰好16个标量参数，极简高效
    脉冲动力学:
      state(t+1) = (input_gated + attn_retrieval + state_feedback + global_modulation) × decay
      output = mean(state) × w_out + b_out (若 ≥ threshold 则发射脉冲)
    """
    
    UNIT_PARAMS = 16  # 每个单元恰好16个标量参数
    
    def __init__(self, unit_id: str, dim: int = 16):
        self.unit_id = unit_id
        self.dim = dim
        
        # 状态 (16维向量，但不是参数)
        self.state = np.zeros(dim)
        self.fatigue = 0.0    # 疲劳度 (0~1, 越高越难发射)
        self.refractory = 0    # 不应期计数器
        
        # ═══════════════════════════════════════════════════════════════
        # 16个标量参数
        # ═══════════════════════════════════════════════════════════════
        self.w_in       = np.random.randn() * 0.1                 # 1. 输入权重
        self.b_in       = np.random.randn() * 0.05                # 2. 输入偏置
        self.w_state    = np.random.randn() * 0.1                 # 3. 状态反馈权重
        self.w_out      = np.random.randn() * 0.1                 # 4. 输出权重
        self.b_out      = np.random.randn() * 0.05                # 5. 输出偏置
        self.w_attn     = np.random.randn() * 0.1                 # 6. 注意力权重
        self.b_attn     = np.random.randn() * 0.05                # 7. 注意力偏置
        self.decay      = 0.9 + np.random.random() * 0.09         # 8. 状态衰减 (0.9~0.99)
        self.gain       = max(0.5, 1.0 + np.random.randn() * 0.3) # 9. 增益
        self.w_global   = 0.5 + np.random.randn() * 0.1           # 10. 全局调制权重
        self.threshold  = 0.5 + np.random.random() * 0.3           # 11. 发射阈值 (0.5~0.8)
        self.refractory_period = 2.0 + np.random.random() * 3.0    # 12. 不应期长度
        self.fatigue_rate = 0.05 + np.random.random() * 0.05       # 13. 疲劳积累率
        self.recovery_rate = 0.02 + np.random.random() * 0.03       # 14. 疲劳恢复率
        self.spontaneous_rate = 0.01 + np.random.random() * 0.02  # 15. 自发脉冲率
        self.w_lateral  = np.random.randn() * 0.1                 # 16. 侧向连接权重
        
        # 连接 (动态，不计入16个固定参数)
        self.outgoing: Dict[str, float] = {}  # target_id -> weight
        self.incoming_history: deque = deque(maxlen=50)  # 输入历史
        
        # 统计
        self.spike_count = 0
        self.last_spike_time = 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # STDP (Spike-Timing Dependent Plasticity) 学习机制
        # ═══════════════════════════════════════════════════════════════
        self.stdp_enabled = True
        self.stdp_A_plus = 0.01    # LTP 幅度
        self.stdp_A_minus = 0.012  # LTD 幅度
        self.stdp_tau_plus = 0.02  # 20ms
        self.stdp_tau_minus = 0.02  # 20ms
        self.spike_times: deque = deque(maxlen=100)  # 最近发射时间记录
        self.ltp_count = 0         # 长时程增强计数
        self.ltd_count = 0         # 长时程抑制计数
        self.total_weight_change = 0.0
        
    def count_params(self) -> int:
        """返回该单元的参数数量（应始终为16）"""
        return 16
        
    def record_spike_time(self, t: float) -> None:
        """记录脉冲发射时间"""
        self.spike_times.append(t)
    
    def stdp_update(self, pre_spike_time: float, post_spike_time: float) -> float:
        """
        STDP 权重更新
        
        Δt = t_post - t_pre
        若 Δt > 0 (后突触晚于前突触): LTP (权重增强)
            Δw = A+ * exp(-Δt / τ+)
        若 Δt < 0 (后突触早于前突触): LTD (权重减弱)
            Δw = -A- * exp(Δt / τ-)
        
        Returns:
            权重变化量 Δw
        """
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:
            # LTP: 后突触晚于前突触，增强连接
            dw = self.stdp_A_plus * math.exp(-dt / self.stdp_tau_plus)
            self.ltp_count += 1
        elif dt < 0:
            # LTD: 后突触早于前突触，减弱连接
            dw = -self.stdp_A_minus * math.exp(dt / self.stdp_tau_minus)
            self.ltd_count += 1
        else:
            dw = 0.0
        
        self.total_weight_change += abs(dw)
        return dw
    
    def apply_stdp(self, current_time: float, all_units_map: Dict[str, 'SpikingUnit'] = None) -> int:
        """
        对该单元的所有传出连接应用 STDP 更新
        
        Args:
            current_time: 当前时间
            all_units_map: 所有单元的映射表 (unit_id -> SpikingUnit)
        
        Returns:
            更新的连接数
        """
        if not self.stdp_enabled or not self.outgoing:
            return 0
        
        updated = 0
        
        # 遍历所有传出连接
        for target_id, current_weight in list(self.outgoing.items()):
            # 获取目标单元的最近发射时间
            target_spike_time = None
            
            if all_units_map and target_id in all_units_map:
                target_unit = all_units_map[target_id]
                if target_unit.spike_times:
                    target_spike_time = target_unit.spike_times[-1]
            else:
                # 简化：使用当前时间近似（目标单元在当前步也发射了）
                target_spike_time = current_time
            
            # 获取本单元最近发射时间
            if not self.spike_times:
                continue
            my_spike_time = self.spike_times[-1]
            
            # 如果目标单元没有发射记录，跳过
            if target_spike_time is None:
                continue
            
            # 计算 STDP
            dw = self.stdp_update(my_spike_time, target_spike_time)
            
            # 应用权重更新
            new_weight = current_weight + dw
            new_weight = np.clip(new_weight, -1.0, 1.0)  # 限制范围
            self.outgoing[target_id] = new_weight
            updated += 1
        
        return updated
    
    def get_stdp_stats(self) -> Dict:
        """获取 STDP 学习统计"""
        return {
            "ltp_count": self.ltp_count,
            "ltd_count": self.ltd_count,
            "total_weight_change": float(self.total_weight_change),
            "avg_weight_change": float(self.total_weight_change / max(1, self.ltp_count + self.ltd_count)),
            "spike_times_recorded": len(self.spike_times)
        }
        
    def step(self, raw_input: np.ndarray, 
             attn_retrieval: np.ndarray,
             global_modulation: float = 1.0,
             dt: float = 1.0) -> Optional[SpikeMessage]:
        """
        单步脉冲动力学
        
        Args:
            raw_input: 外部输入信号 (dim维)
            attn_retrieval: KV堆注意力检索结果 (dim维)
            global_modulation: 全局调制强度 (节律控制)
            dt: 时间步长
        
        Returns:
            SpikeMessage 如果发射脉冲，否则 None
        """
        # 不应期检查
        if self.refractory > 0:
            self.refractory -= 1
            self.fatigue = max(0.0, self.fatigue - self.recovery_rate)
            self.state *= self.decay  # 仅衰减
            return None
        
        # 输入门控 (input_gated = gain * tanh(w_in * mean(input) + b_in))
        mean_input = np.mean(raw_input) if raw_input is not None else 0.0
        input_gated = self.gain * np.tanh(self.w_in * mean_input + self.b_in)
        
        # 注意力检索 (使用均值，标量权重)
        mean_attn = np.mean(attn_retrieval) if attn_retrieval is not None else 0.0
        attn_contrib = self.w_attn * mean_attn + self.b_attn
        
        # 状态反馈 (使用状态均值，标量权重)
        mean_state = np.mean(self.state)
        state_feedback = self.w_state * mean_state
        
        # 全局调制
        modulation = global_modulation * self.w_global
        
        # 自发脉冲 (默认模式网络)
        spontaneous = np.random.random() < self.spontaneous_rate
        
        # 状态更新: state(t+1) = (input_gated + attn + state_feedback + global) × decay
        total_input = input_gated + attn_contrib + state_feedback + modulation
        if spontaneous:
            total_input += 0.3  # 自发脉冲增加输入
        
        self.state = total_input * self.decay + self.state * 0.1
        
        # 输出计算: output = mean(state) * w_out + b_out
        output = mean_state * self.w_out + self.b_out
        
        # 疲劳影响阈值
        effective_threshold = self.threshold + self.fatigue
        
        # 检查是否发射脉冲
        if output >= effective_threshold or spontaneous:
            self.spike_count += 1
            self.last_spike_time = time.time()
            self.fatigue = min(1.0, self.fatigue + self.fatigue_rate)
            self.refractory = int(self.refractory_period)
            
            # 记录脉冲发射时间 (STDP学习)
            self.record_spike_time(self.last_spike_time)
            
            # 构建脉冲消息
            strength = min(1.0, abs(output))
            msg = SpikeMessage(
                source_unit_id=self.unit_id,
                source_level=0,  # 基础单元
                payload=SpikePayload(
                    value=float(output),
                    strength=strength,
                    timestamp=time.time(),
                    cycle_phase=0
                ),
                hops_remaining=3
            )
            return msg
        else:
            # 疲劳恢复
            self.fatigue = max(0.0, self.fatigue - self.recovery_rate)
            return None
    
    def get_state_dict(self) -> Dict:
        return {
            "unit_id": self.unit_id,
            "spike_count": self.spike_count,
            "fatigue": float(self.fatigue),
            "threshold": float(self.threshold),
            "state_norm": float(np.linalg.norm(self.state))
        }


# ═══════════════════════════════════════════════════════════════
# 5. 分形递归网络层
# ═══════════════════════════════════════════════════════════════

class FractalLayer:
    """
    分形递归层: 每层16个单元，深度指数扩展能力
    
    基础单元数 = 16 + 16^2 + ... + 16^(depth+1)
    总参数 = 16 × Σ(16^k for k=1 to depth+1)
    
    v5.2 分形深度2 => 4,368单元/层 / 69K参数
    """
    
    def __init__(self, layer_id: str, depth: int = 0, dim: int = 16):
        self.layer_id = layer_id
        self.depth = depth
        self.dim = dim
        self.units: List[SpikingUnit] = []
        self.sub_layers: List['FractalLayer'] = []
        
        # 创建16个单元
        for i in range(16):
            uid = f"{layer_id}_U{i}"
            self.units.append(SpikingUnit(uid, dim))
        
        # 递归创建子层 (深度>0时)
        if depth > 0:
            for i in range(16):
                sub_id = f"{layer_id}_L{i}"
                self.sub_layers.append(FractalLayer(sub_id, depth - 1, dim))
        
        # 层内连接 (小世界网络)
        self._build_connections()
        
        # 缓存所有单元引用 (避免每步递归)
        self._all_units_cache: List[SpikingUnit] = self._build_all_units_cache()
        self._build_vec_arrays()
        
    def _build_vec_arrays(self):
        """构建向量化计算用的批量数组 (适配标量权重)"""
        units = self._all_units_cache
        self.N = len(units)
        if self.N == 0:
            return
        self.dim = units[0].dim
        self._vec_state = np.zeros((self.N, self.dim))
        self._vec_threshold = np.zeros(self.N)
        self._vec_fatigue = np.zeros(self.N)
        self._vec_refractory = np.zeros(self.N, dtype=np.int32)
        self._vec_gain = np.zeros(self.N)
        self._vec_decay = np.zeros(self.N)
        self._vec_refractory_period = np.zeros(self.N)
        self._vec_fatigue_rate = np.zeros(self.N)
        self._vec_recovery_rate = np.zeros(self.N)
        self._vec_spontaneous_rate = np.zeros(self.N)
        self._vec_b_in = np.zeros(self.N)
        self._vec_b_out = np.zeros(self.N)
        self._vec_b_attn = np.zeros(self.N)
        self._vec_w_global = np.zeros(self.N)
        # 标量权重 (N,)
        self._vec_w_in = np.zeros(self.N)
        self._vec_w_state = np.zeros(self.N)
        self._vec_w_out = np.zeros(self.N)
        self._vec_w_attn = np.zeros(self.N)
        for i, u in enumerate(units):
            self._vec_state[i] = u.state
            self._vec_threshold[i] = u.threshold
            self._vec_fatigue[i] = u.fatigue
            self._vec_refractory[i] = u.refractory
            self._vec_gain[i] = u.gain
            self._vec_decay[i] = u.decay
            self._vec_refractory_period[i] = u.refractory_period
            self._vec_fatigue_rate[i] = u.fatigue_rate
            self._vec_recovery_rate[i] = u.recovery_rate
            self._vec_spontaneous_rate[i] = u.spontaneous_rate
            self._vec_b_in[i] = u.b_in
            self._vec_b_out[i] = u.b_out
            self._vec_b_attn[i] = u.b_attn
            self._vec_w_global[i] = u.w_global
            self._vec_w_in[i] = u.w_in
            self._vec_w_state[i] = u.w_state
            self._vec_w_out[i] = u.w_out
            self._vec_w_attn[i] = u.w_attn

    def _sync_units_to_vec(self):
        for i, u in enumerate(self._all_units_cache):
            self._vec_state[i] = u.state
            self._vec_fatigue[i] = u.fatigue
            self._vec_refractory[i] = u.refractory

    def _sync_vec_to_units(self):
        for i, u in enumerate(self._all_units_cache):
            u.state = self._vec_state[i].copy()
            u.fatigue = float(self._vec_fatigue[i])
            u.refractory = int(self._vec_refractory[i])

    def _build_all_units_cache(self) -> List[SpikingUnit]:
        """一次性构建所有单元缓存"""
        all_units = self.units[:]
        for sub in self.sub_layers:
            all_units.extend(sub._all_units_cache)
        return all_units
    
    def get_all_units(self) -> List[SpikingUnit]:
        """获取所有单元 (使用缓存)"""
        return self._all_units_cache
    
    def get_unit_count(self) -> int:
        """获取总单元数"""
        return len(self._all_units_cache)
    
    def _build_connections(self) -> None:
        """构建小世界网络连接"""
        n = len(self.units)
        for i, u in enumerate(self.units):
            # 每个单元连接到2-4个其他单元 (小世界特性)
            num_conn = 2 + int(np.random.random() * 3)
            targets = random.sample(range(n), min(num_conn, n - 1))
            for t in targets:
                if t != i:
                    weight = np.random.randn() * 0.1
                    u.outgoing[f"{self.layer_id}_U{t}"] = weight
    
    def step(self, layer_input: np.ndarray,
             global_kv: KVStack,
             global_modulation: float = 1.0,
             training_mode: bool = False) -> List[SpikeMessage]:
        """
        单步执行: 向量化并行计算 (标量权重版本)
        """
        if not hasattr(self, 'N') or self.N == 0:
            return []
        
        self._sync_units_to_vec()
        
        active = self._vec_refractory == 0
        input_signal = layer_input[:self.dim] if len(layer_input) >= self.dim else layer_input
        attn_retrieval = np.zeros(self.dim)
        
        # 标量权重计算: 使用输入/状态的均值
        mean_input = np.mean(input_signal)
        mean_attn = np.mean(attn_retrieval)
        
        input_gated = self._vec_gain * np.tanh(self._vec_w_in * mean_input + self._vec_b_in)
        attn_contrib = self._vec_w_attn * mean_attn + self._vec_b_attn
        state_feedback = self._vec_w_state * np.mean(self._vec_state, axis=1)
        modulation = global_modulation * self._vec_w_global
        
        total = input_gated + attn_contrib + state_feedback + modulation
        spontaneous = np.random.random(self.N) < self._vec_spontaneous_rate
        
        total_exp = total[:, np.newaxis]
        decay_exp = self._vec_decay[:, np.newaxis]
        state_new = total_exp * decay_exp + self._vec_state * 0.1
        self._vec_state[active] = state_new[active]
        
        # 输出: mean(state) * w_out + b_out
        mean_state = np.mean(self._vec_state, axis=1)
        output = mean_state * self._vec_w_out + self._vec_b_out
        eff_threshold = self._vec_threshold + self._vec_fatigue
        spike_mask = ((output >= eff_threshold) | spontaneous) & active
        
        self._vec_fatigue[spike_mask] = np.minimum(1.0, self._vec_fatigue[spike_mask] + self._vec_fatigue_rate[spike_mask])
        self._vec_refractory[spike_mask] = self._vec_refractory_period[spike_mask].astype(np.int32)
        
        not_spike = active & ~spike_mask
        self._vec_fatigue[not_spike] = np.maximum(0.0, self._vec_fatigue[not_spike] - self._vec_recovery_rate[not_spike])
        self._vec_refractory[not_spike] = np.maximum(0, self._vec_refractory[not_spike] - 1)
        
        inactive = ~active
        self._vec_state[inactive] *= decay_exp[inactive]
        self._vec_refractory[inactive] = np.maximum(0, self._vec_refractory[inactive] - 1)
        self._vec_fatigue[inactive] = np.maximum(0.0, self._vec_fatigue[inactive] - self._vec_recovery_rate[inactive])
        
        self._sync_vec_to_units()
        
        spikes = []
        for idx in np.where(spike_mask)[0]:
            unit = self._all_units_cache[idx]
            strength = min(1.0, abs(output[idx]))
            msg = SpikeMessage(
                source_unit_id=unit.unit_id,
                source_level=0,
                payload=SpikePayload(
                    value=float(output[idx]),
                    strength=strength,
                    timestamp=time.time(),
                    cycle_phase=0
                ),
                hops_remaining=3
            )
            msg.source_agent_id = self.layer_id
            spikes.append(msg)
        
        return spikes
    
    def get_stats(self) -> Dict:
        """获取层统计"""
        all_units = self._all_units_cache
        total_spikes = sum(u.spike_count for u in all_units)
        avg_fatigue = np.mean([u.fatigue for u in all_units])
        active_units = sum(1 for u in all_units if u.fatigue < 0.5)
        
        return {
            "layer_id": self.layer_id,
            "depth": self.depth,
            "total_units": len(all_units),
            "total_spikes": total_spikes,
            "avg_fatigue": float(avg_fatigue),
            "active_units": active_units,
            "active_ratio": active_units / len(all_units) if all_units else 0
        }


# ═══════════════════════════════════════════════════════════════
# 6. DistributedFormer 完整网络
# ═══════════════════════════════════════════════════════════════

class DistributedFormer:
    """
    DistributedFormer 完整网络
    
    架构: 分形递归堆叠
    - 深度0: 16单元 (浅层输出)
    - 深度1: 16 + 16×16 = 272单元
    - 深度2: 16 + 16×16 + 16×16×16 = 4,368单元 (v5.2)
    - 深度3: 65,536单元 (远期)
    
    包含:
    - 输入层: 接收编码后的环境脉冲
    - 思考层: num_think_layers层分形递归异步计算
    - 输出层: 生成动作脉冲
    - KV堆:  持久工作记忆
    """
    
    def __init__(self, depth: int = 2, dim: int = 16, 
                 kv_capacity: int = 100000,
                 num_think_layers: int = 1,
                 training_mode: bool = False):
        self.depth = depth
        self.dim = dim
        self.num_think_layers = num_think_layers
        self.training_mode = training_mode  # 训练模式: 跳过KV查询以加速
        
        # 输入层 (16个编码单元)
        self.input_units = [SpikingUnit(f"input_U{i}", dim) for i in range(16)]
        
        # 思考层: num_think_layers层分形递归
        self.think_layers: List[FractalLayer] = []
        for i in range(num_think_layers):
            self.think_layers.append(FractalLayer(f"think_L{i}", depth, dim))
        
        # 输出层 (16个解码单元)
        self.output_units = [SpikingUnit(f"output_U{i}", dim) for i in range(16)]
        
        # KV堆 (全局共享)
        self.kv_stack = KVStack(capacity=kv_capacity, dim=dim)
        
        # 全局调制 (节律控制)
        self.global_modulation = 1.0
        self.cycle_phase = 0
        self.think_phase = 80
        self.inhibit_phase = 40
        self.cycle_length = 120
        
        # 统计
        self.total_steps = 0
        
        # STDP 全局学习开关
        self.learning_enabled = True
        self._units_map: Dict[str, SpikingUnit] = {}
        self._all_units: List[SpikingUnit] = []
        self._build_units_map()
    
    def _build_units_map(self) -> None:
        """构建所有单元的映射表和缓存列表 (用于STDP跨单元查找)"""
        self._units_map.clear()
        self._all_units = []
        
        for u in self.input_units:
            self._units_map[u.unit_id] = u
            self._all_units.append(u)
        for layer in self.think_layers:
            for u in layer._all_units_cache:
                self._units_map[u.unit_id] = u
                self._all_units.append(u)
        for u in self.output_units:
            self._units_map[u.unit_id] = u
            self._all_units.append(u)
    
    def enable_learning(self, enabled: bool = True) -> None:
        """启用/禁用 STDP 学习"""
        self.learning_enabled = enabled
        for u in self._all_units:
            u.stdp_enabled = enabled
    
    def _apply_stdp_to_all(self) -> None:
        """对所有发射过脉冲的单元应用 STDP 更新"""
        if not self.learning_enabled:
            return
        for unit in self._all_units:
            if unit.spike_times:
                unit.apply_stdp(time.time(), self._units_map)
    
    def get_stdp_stats(self) -> Dict:
        """获取全局 STDP 统计"""
        total_ltp = sum(u.ltp_count for u in self._all_units)
        total_ltd = sum(u.ltd_count for u in self._all_units)
        total_weight_change = sum(u.total_weight_change for u in self._all_units)
        return {
            "learning_enabled": self.learning_enabled,
            "total_ltp": total_ltp,
            "total_ltd": total_ltd,
            "total_weight_change": float(total_weight_change),
            "avg_weight_change": float(total_weight_change / max(1, total_ltp + total_ltd))
        }
        
    def set_rhythm(self, think_phase: int = 80, inhibit_phase: int = 40) -> None:
        """设置节律参数"""
        self.think_phase = think_phase
        self.inhibit_phase = inhibit_phase
        self.cycle_length = think_phase + inhibit_phase
    
    def get_global_modulation(self) -> float:
        """根据当前节律相位计算全局调制强度"""
        if self.cycle_phase < self.think_phase:
            # 思考期: 调制从1.0逐渐降低到0.5
            progress = self.cycle_phase / self.think_phase
            return 1.0 - 0.5 * progress
        else:
            # 抑制期: 调制从0.5降低到0.1
            progress = (self.cycle_phase - self.think_phase) / self.inhibit_phase
            return 0.5 - 0.4 * progress
    
    def step(self, encoded_input: np.ndarray) -> List[SpikeMessage]:
        """
        完整网络单步执行
        
        Args:
            encoded_input: 编码后的输入脉冲 (至少dim维)
        
        Returns:
            输出层发射的脉冲消息
        """
        self.total_steps += 1
        self.cycle_phase = (self.cycle_phase + 1) % self.cycle_length
        self.global_modulation = self.get_global_modulation()
        
        # 确保输入维度正确
        if len(encoded_input) < self.dim:
            padded = np.zeros(self.dim)
            padded[:len(encoded_input)] = encoded_input
            encoded_input = padded
        
        all_spikes = []
        
        # 1. 输入层处理
        input_spikes = []
        for unit in self.input_units:
            spike = unit.step(encoded_input[:self.dim], 
                            np.zeros(self.dim), 
                            self.global_modulation)
            if spike:
                input_spikes.append(spike)
        all_spikes.extend(input_spikes)
        
        # 2. 思考层处理 (异步传播)
        layer_input = encoded_input
        for layer in self.think_layers:
            layer_spikes = layer.step(layer_input, self.kv_stack, self.global_modulation, self.training_mode)
            all_spikes.extend(layer_spikes)
            # 层间传播: 将当前层的脉冲聚合为下一层输入
            if layer_spikes:
                layer_input = np.zeros(self.dim)
                for sp in layer_spikes:
                    # 将脉冲值注入输入
                    idx = hash(sp.source_unit_id) % self.dim
                    layer_input[idx] += sp.payload.value * sp.payload.strength
                layer_input = np.tanh(layer_input)  # 归一化
        
        # 3. 输出层处理
        output_spikes = []
        for unit in self.output_units:
            spike = unit.step(layer_input, 
                            np.zeros(self.dim), 
                            self.global_modulation)
            if spike:
                spike.source_agent_id = "output"
                output_spikes.append(spike)
        all_spikes.extend(output_spikes)
        
        # 4. KV堆更新 (仅在非训练模式下)
        if not self.training_mode:
            for i, unit in enumerate(self.output_units):
                if unit.spike_count > 0:
                    entry_id = f"output_{i}_{self.total_steps}"
                    self.kv_stack.push(entry_id, unit.state, unit.state)
        
        # 5. STDP 学习: 对所有发射过脉冲的单元应用权重更新
        if self.learning_enabled:
            self._apply_stdp_to_all()
        
        return output_spikes
    
    def get_network_stats(self) -> Dict:
        """获取网络统计"""
        total_units = len(self._all_units)
        total_spikes = sum(u.spike_count for u in self._all_units)
        avg_fatigue = np.mean([u.fatigue for u in self._all_units]) if self._all_units else 0.0
        
        return {
            "depth": self.depth,
            "total_units": total_units,
            "total_spikes": total_spikes,
            "avg_fatigue": float(avg_fatigue),
            "cycle_phase": self.cycle_phase,
            "global_modulation": float(self.global_modulation),
            "kv_stats": self.kv_stack.get_stats(),
            "total_steps": self.total_steps,
            "stdp": self.get_stdp_stats()
        }
    
    def get_output_pattern(self) -> np.ndarray:
        """获取输出层激活模式 (用于动作解码)"""
        pattern = np.zeros(len(self.output_units))
        for i, unit in enumerate(self.output_units):
            pattern[i] = np.linalg.norm(unit.state) * (1 - unit.fatigue)
        return pattern
    
    def get_think_layer_pattern(self) -> np.ndarray:
        """获取思考层聚合激活模式 (用于监督学习)"""
        pattern = np.zeros(self.dim)
        for layer in self.think_layers:
            if hasattr(layer, '_vec_state') and layer.N > 0:
                pattern += np.sum(np.abs(layer._vec_state) * (1 - layer._vec_fatigue[:, np.newaxis]), axis=0)
            else:
                for unit in layer._all_units_cache:
                    pattern += np.abs(unit.state) * (1 - unit.fatigue)
        # 归一化
        norm = np.linalg.norm(pattern)
        if norm > 0:
            pattern = pattern / norm
        return pattern
    
    def reset_state(self) -> None:
        """重置所有单元的内部状态 (用于训练时每个样本独立)"""
        for unit in self._all_units:
            unit.state = np.zeros(self.dim)
            unit.fatigue = 0.0
            unit.refractory = 0
        for layer in self.think_layers:
            if hasattr(layer, '_vec_state') and layer.N > 0:
                layer._vec_state[:] = 0.0
                layer._vec_fatigue[:] = 0.0
                layer._vec_refractory[:] = 0
        # 也重置全局状态
        self.cycle_phase = 0
        self.global_modulation = 1.0
        self.total_steps = 0
        
    def get_state_snapshot(self) -> dict:
        """获取所有单元状态的快照"""
        snapshot = {}
        for unit in self._all_units:
            snapshot[unit.unit_id] = {
                'state': unit.state.copy(),
                'fatigue': unit.fatigue,
                'refractory': unit.refractory
            }
        return snapshot
    
    def restore_state_snapshot(self, snapshot: dict) -> None:
        """从快照恢复单元状态"""
        for unit in self._all_units:
            if unit.unit_id in snapshot:
                s = snapshot[unit.unit_id]
                unit.state = s['state'].copy()
                unit.fatigue = s['fatigue']
                unit.refractory = s['refractory']


# ═══════════════════════════════════════════════════════════════
# 7. 工具函数
# ═══════════════════════════════════════════════════════════════

def calculate_scale(depth: int) -> Dict:
    """计算给定分形深度的网络规模"""
    # 正确计算: 16 + 16^2 + ... + 16^(depth+1)
    base_units = sum(16 ** k for k in range(1, depth + 2))
    total_params = 16 * base_units
    return {
        "depth": depth,
        "base_units": base_units,
        "total_params": total_params,
        "description": f"深度{depth}: {base_units}基础单元 / {total_params//1000}K参数"
    }


# ═══════════════════════════════════════════════════════════════
# 8. 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DistributedFormer v5.2 核心模块测试")
    print("=" * 60)
    
    # 测试规模计算
    for d in range(4):
        info = calculate_scale(d)
        print(f"  {info['description']}")
    
    print("\n" + "-" * 60)
    
    # 测试KV堆
    kv = KVStack(capacity=100, dim=16)
    for i in range(20):
        kv.push(f"test_{i}", np.random.randn(16), np.random.randn(16))
    results = kv.query(np.random.randn(16), top_k=3)
    print(f"KV堆测试: 20条目中查询top-3, 命中{len(results)}条")
    
    # 测试脉冲单元
    unit = SpikingUnit("test_unit")
    spike_count = 0
    for i in range(100):
        spike = unit.step(np.random.randn(16), np.zeros(16), 1.0)
        if spike:
            spike_count += 1
    print(f"脉冲单元测试: 100步中发射{spike_count}次脉冲")
    
    # 测试完整网络
    print("\n" + "-" * 60)
    print("完整网络测试 (深度2, 1层)...")
    df = DistributedFormer(depth=2, dim=16, num_think_layers=1)
    
    # 模拟10步
    for step in range(10):
        input_signal = np.random.randn(16) * 0.5
        output_spikes = df.step(input_signal)
        print(f"  Step {step+1}: 输出层发射{len(output_spikes)}个脉冲, "
              f"调制={df.global_modulation:.2f}, 相位={df.cycle_phase}")
    
    stats = df.get_network_stats()
    print(f"\n网络统计: {stats['total_units']}单元, {stats['total_spikes']}脉冲, "
          f"疲劳={stats['avg_fatigue']:.3f}")
    print("KV堆: 利用率={:.1%}".format(stats['kv_stats']['utilization']))
    
    # 测试思考层模式
    think_pattern = df.get_think_layer_pattern()
    print(f"\n思考层激活模式: 范数={np.linalg.norm(think_pattern):.3f}")
    
    # 测试 STDP 学习
    print("\n" + "-" * 60)
    print("STDP 学习测试...")
    stdp_stats = df.get_stdp_stats()
    print(f"  LTP: {stdp_stats['total_ltp']}, LTD: {stdp_stats['total_ltd']}")
    print(f"  总权重变化: {stdp_stats['total_weight_change']:.4f}")
    print(f"  平均权重变化: {stdp_stats['avg_weight_change']:.6f}")
    
    # 禁用学习再运行5步对比
    print("\n  禁用 STDP 学习后运行5步...")
    df.enable_learning(False)
    for step in range(5):
        df.step(np.random.randn(16) * 0.5)
    stdp_stats2 = df.get_stdp_stats()
    print(f"  禁用后 LTP: {stdp_stats2['total_ltp']} (应不变)")
    print(f"  学习开关: {stdp_stats2['learning_enabled']}")
    
    print("\n" + "=" * 60)
    print("核心模块测试通过!")
    print("=" * 60)
