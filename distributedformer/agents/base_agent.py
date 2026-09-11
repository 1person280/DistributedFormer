"""
脉冲智能体基类
所有智能体继承此类，实现统一的脉冲通信接口
"""

import numpy as np
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import uuid
import json

# 导入核心模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.core.distributedformer import (
    DistributedFormer, SpikeMessage, SpikePayload, KVStack
)
from distributedformer.codec.spike_codec import MultiModalCodec


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    agent_type: str
    status: str = "idle"  # idle, running, thinking, inhibited, error
    cycle_count: int = 0
    total_spikes_received: int = 0
    total_spikes_sent: int = 0
    last_active: float = 0.0
    error_count: int = 0


class BaseSpikeAgent(ABC):
    """
    脉冲智能体基类
    
    每个智能体 = Kimi Work Agent + DistributedFormer 实例
    
    角色类型:
    - perception: 感知智能体 (编码环境输入)
    - reasoning: 推理智能体 (核心DF网络计算)
    - action: 动作智能体 (解码输出脉冲)
    - memory: 记忆智能体 (管理全局KV堆)
    - rhythm: 节律智能体 (控制全局抑制/思考周期)
    """
    
    def __init__(self, agent_id: str, agent_type: str, 
                 df_depth: int = 2, dim: int = 16,
                 kv_stack: Optional[KVStack] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.df_depth = df_depth
        self.dim = dim
        
        # DistributedFormer 实例 (推理智能体才有深度>0)
        self.df = None
        if df_depth >= 0:
            self.df = DistributedFormer(depth=df_depth, dim=dim, kv_capacity=10000)
        
        # 共享KV堆 (外部注入或自建)
        self.kv_stack = kv_stack or KVStack(capacity=10000, dim=dim)
        if self.df:
            self.df.kv_stack = self.kv_stack  # 确保网络使用共享KV
        
        # 编解码器
        self.codec = MultiModalCodec(dim=dim)
        
        # 状态
        self.state = AgentState(agent_id=agent_id, agent_type=agent_type)
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # 消息队列
        self.inbox: deque = deque(maxlen=1000)   # 接收到的脉冲
        self.outbox: deque = deque(maxlen=1000)   # 待发送的脉冲
        
        # 连接
        self.output_targets: List[str] = []  # 目标智能体ID
        self.lateral_connections: List[str] = []  # 侧向连接
        
        # 回调
        self.on_spike_received: Optional[Callable] = None
        self.on_spike_sent: Optional[Callable] = None
        
        # 特殊化标识 (如 "trend_analysis", "anomaly_detection")
        self.specialization = None
        
    def set_targets(self, targets: List[str]) -> None:
        """设置输出目标"""
        self.output_targets = targets
    
    def set_lateral(self, laterals: List[str]) -> None:
        """设置侧向连接"""
        self.lateral_connections = laterals
    
    def receive_spike(self, msg: SpikeMessage) -> None:
        """
        接收脉冲消息
        
        衰减机制: 每跳传递 strength × 0.7
        """
        # 衰减
        if msg.hops_remaining > 0:
            msg.payload.strength *= 0.7
            msg.hops_remaining -= 1
        else:
            return  # 跳数耗尽，丢弃
        
        # 记录
        self.state.total_spikes_received += 1
        self.state.last_active = time.time()
        msg.trace.append(self.agent_id)
        
        # 入队
        self.inbox.append(msg)
        
        # 回调
        if self.on_spike_received:
            self.on_spike_received(msg)
    
    def send_spike(self, msg: SpikeMessage) -> None:
        """发送脉冲消息 (由工作流引擎实际路由)"""
        self.state.total_spikes_sent += 1
        self.outbox.append(msg)
        
        if self.on_spike_sent:
            self.on_spike_sent(msg)
    
    def process_inbox(self) -> List[SpikeMessage]:
        """处理收件箱中的所有脉冲"""
        output_spikes = []
        while self.inbox:
            msg = self.inbox.popleft()
            spikes = self._process_spike(msg)
            output_spikes.extend(spikes)
        return output_spikes
    
    @abstractmethod
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        """子类实现具体的脉冲处理逻辑"""
        pass
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """
        单步执行
        
        Args:
            external_input: 外部输入信号 (如环境数据编码)
        
        Returns:
            输出脉冲列表
        """
        self.state.cycle_count += 1
        self.state.status = "thinking"
        
        # 1. 处理收件箱
        inbox_spikes = self.process_inbox()
        
        # 2. 聚合输入
        combined_input = np.zeros(self.dim)
        
        # 外部输入 (向量或 {模态: 数据} 字典)
        if external_input is not None and not isinstance(external_input, dict):
            if len(external_input) >= self.dim:
                combined_input += external_input[:self.dim]
            else:
                combined_input[:len(external_input)] += external_input
        
        # 收件箱脉冲聚合
        for msg in inbox_spikes:
            idx = hash(msg.source_unit_id) % self.dim
            combined_input[idx] += msg.payload.value * msg.payload.strength
        
        combined_input = np.tanh(combined_input)  # 归一化
        
        # 3. DF网络计算
        output_spikes = []
        if self.df:
            if isinstance(external_input, dict):
                df_inputs = dict(external_input)
                if inbox_spikes and "numeric" not in df_inputs:
                    df_inputs["numeric"] = combined_input
            else:
                df_inputs = {"numeric": combined_input}
            output_spikes = self.df.step(df_inputs)
            # 标记来源
            for sp in output_spikes:
                sp.source_agent_id = self.agent_id
                sp.target_agents = self.output_targets[:]
        
        # 4. 侧向传播 (将输出脉冲复制到侧向连接)
        lateral_spikes = []
        for sp in output_spikes:
            for lateral_id in self.lateral_connections:
                lateral_msg = SpikeMessage(
                    source_agent_id=self.agent_id,
                    source_unit_id=sp.source_unit_id,
                    source_level=sp.source_level,
                    payload=SpikePayload(
                        value=sp.payload.value,
                        strength=sp.payload.strength * 0.5,  # 侧向衰减
                        timestamp=time.time(),
                        cycle_phase=sp.payload.cycle_phase
                    ),
                    target_agents=[lateral_id],
                    hops_remaining=2,
                    trace=sp.trace + [self.agent_id]
                )
                lateral_spikes.append(lateral_msg)
        
        all_outputs = output_spikes + lateral_spikes
        
        # 5. 输出到outbox
        for sp in all_outputs:
            self.send_spike(sp)
        
        self.state.status = "idle"
        return all_outputs
    
    def get_stats(self) -> Dict:
        """获取智能体统计"""
        stats = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.state.status,
            "cycles": self.state.cycle_count,
            "spikes_received": self.state.total_spikes_received,
            "spikes_sent": self.state.total_spikes_sent,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "specialization": self.specialization
        }
        
        if self.df:
            stats["df_stats"] = self.df.get_network_stats()
        
        return stats
    
    def start(self) -> None:
        """启动智能体线程"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """停止智能体"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _run_loop(self) -> None:
        """运行循环 (由子类覆盖)"""
        while self.is_running:
            self.step()
            time.sleep(0.1)  # 100ms步长
    
    def save_state(self, path: str) -> None:
        """保存状态"""
        if self.df:
            self.df.save_state(f"{path}_{self.agent_id}")
    
    def load_state(self, path: str) -> None:
        """加载状态"""
        if self.df:
            self.df.load_state(f"{path}_{self.agent_id}")


class PerceptionAgent(BaseSpikeAgent):
    """
    感知智能体
    
    负责将 WebBridge/文件/数据源的输入编码为脉冲
    
    输入源:
    - webbridge://finance.yahoo.com/AAPL
    - native://academic/arxiv
    - file://data/stock.csv
    
    输出: 编码后的脉冲信号 -> reasoning智能体
    """
    
    def __init__(self, agent_id: str, df_depth: int = 2, dim: int = 16,
                 kv_stack: Optional[KVStack] = None):
        super().__init__(agent_id, "perception", df_depth, dim, kv_stack)
        
        # 输入源配置
        self.inputs: List[Dict] = []  # [{"source": "...", "encoding": "..."}]
        
        # 最近读取的数据缓存
        self.data_cache: Dict[str, Any] = {}
        
    def add_input(self, source: str, encoding: str = "numeric_scalar") -> None:
        """添加输入源"""
        self.inputs.append({"source": source, "encoding": encoding})
    
    def fetch_and_encode(self) -> np.ndarray:
        """
        从输入源获取数据并编码为脉冲
        
        简化版本: 模拟数据读取
        """
        if not self.inputs:
            return np.zeros(self.dim)
        
        combined_signal = np.zeros(self.dim)
        
        for inp in self.inputs:
            source = inp["source"]
            encoding = inp["encoding"]
            
            # 模拟数据获取 (实际应使用WebBridge/Native数据源)
            data = self._simulate_fetch(source, encoding)
            self.data_cache[source] = data
            
            # 编码
            signal = self.codec.encode(data, encoding)
            combined_signal += signal
        
        # 归一化
        norm = np.linalg.norm(combined_signal)
        if norm > 0:
            combined_signal = combined_signal / norm
        
        return combined_signal
    
    def _simulate_fetch(self, source: str, encoding: str) -> Any:
        """模拟数据获取 (实际生产环境应使用真实API)"""
        import random
        
        if "finance.yahoo.com" in source or "stock" in source:
            # 模拟股票数据
            ticker = source.split("/")[-1] if "/" in source else "STOCK"
            return {
                "numeric": random.uniform(-10, 10),  # 涨跌幅
                "text": f"{ticker} trading volume up",
                "timeseries": [random.uniform(100, 200) for _ in range(5)]
            }
        elif "academic" in source or "arxiv" in source:
            return "New paper on neural architecture search published"
        elif "numeric_scalar" in encoding:
            return random.uniform(-5, 5)
        else:
            return random.uniform(0, 1)
    
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        """感知智能体主要处理外部输入，不处理内部脉冲"""
        return []  # 感知智能体不转发脉冲
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """感知智能体步骤: 获取数据 -> 编码 -> 输出到目标"""
        # 获取并编码数据
        encoded = self.fetch_and_encode()
        
        # 注入DF网络并生成输出脉冲
        output_spikes = super().step(encoded)
        
        return output_spikes


class ReasoningAgent(BaseSpikeAgent):
    """
    推理智能体
    
    核心DistributedFormer网络，执行异步脉冲计算
    
    特性:
    - 内部可包含子推理智能体
    - 支持侧向连接与其他推理智能体协同
    - 同步到全局KV堆
    """
    
    def __init__(self, agent_id: str, df_depth: int = 2, dim: int = 16,
                 kv_stack: Optional[KVStack] = None):
        super().__init__(agent_id, "reasoning", df_depth, dim, kv_stack)
        
        # 子智能体
        self.sub_agents: List[BaseSpikeAgent] = []
        self.kv_stack_sync = "global"  # "global" | "local" | "none"
        
    def add_sub_agent(self, agent: BaseSpikeAgent) -> None:
        """添加子推理智能体"""
        self.sub_agents.append(agent)
    
    def set_specialization(self, specialization: str) -> None:
        """设置特殊化方向"""
        self.specialization = specialization
        
        # 根据特殊化调整网络参数
        if specialization == "trend_analysis":
            # 趋势分析: 降低阈值，增加敏感性
            if self.df:
                for layer in self.df.think_layers:
                    for unit in layer.get_all_units():
                        unit.threshold *= 0.8
        elif specialization == "anomaly_detection":
            # 异常检测: 提高阈值，减少误报
            if self.df:
                for layer in self.df.think_layers:
                    for unit in layer.get_all_units():
                        unit.threshold *= 1.2
    
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        """推理智能体处理输入脉冲，通过DF网络计算"""
        # 主要计算在 step() 中完成，这里处理特殊逻辑
        return []
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """推理步骤: 执行DF计算 + 子智能体协同"""
        # 主网络计算
        output_spikes = super().step(external_input)
        
        # 子智能体协同 (异步)
        for sub in self.sub_agents:
            sub.step(external_input)  # 子智能体也接收相同输入
        
        # KV堆同步
        if self.kv_stack_sync == "global" and self.df:
            # 将网络状态存入KV堆
            pattern = self.df.get_output_pattern()
            entry_id = f"{self.agent_id}_{self.state.cycle_count}"
            self.kv_stack.push(entry_id, pattern, pattern)
        
        return output_spikes


class ActionAgent(BaseSpikeAgent):
    """
    动作智能体
    
    将输出脉冲解码为动作指令
    
    支持动作:
    - file_write: 文件写入
    - desktop_notification: 桌面通知
    - api_call: API调用 (Slack等)
    - web_browse: 网页浏览
    - data_query: 数据查询
    - report_generate: 报告生成
    """
    
    def __init__(self, agent_id: str, df_depth: int = 0, dim: int = 16,
                 kv_stack: Optional[KVStack] = None):
        super().__init__(agent_id, "action", df_depth, dim, kv_stack)
        
        # 输出配置
        self.outputs: List[Dict] = []
        
        # 动作执行历史
        self.action_history: deque = deque(maxlen=100)
        
    def add_output(self, output_type: str, **kwargs) -> None:
        """添加输出配置"""
        self.outputs.append({"type": output_type, **kwargs})
    
    def decode_and_execute(self, output_pattern: np.ndarray) -> List[Dict]:
        """
        解码脉冲模式并执行动作
        
        返回执行结果
        """
        # 解码
        actions = self.codec.decode(output_pattern, threshold=0.3)
        
        # 过滤: 只执行配置中允许的动作
        allowed_types = [o["type"] for o in self.outputs]
        filtered_actions = [a for a in actions if a["type"] in allowed_types]
        
        # 执行
        results = []
        for action in filtered_actions:
            # 检查条件
            if action["strength"] > 0.3:  # 最小强度阈值
                result = self._execute_action(action)
                results.append(result)
                self.action_history.append({
                    "action": action,
                    "result": result,
                    "timestamp": time.time()
                })
        
        return results
    
    def _execute_action(self, action: Dict) -> Dict:
        """执行具体动作 (简化版本)"""
        action_type = action["type"]
        
        if action_type == "file_write":
            return self._do_file_write(action)
        elif action_type == "desktop_notification":
            return self._do_notification(action)
        elif action_type == "api_call":
            return self._do_api_call(action)
        elif action_type == "report_generate":
            return self._do_report_generate(action)
        else:
            return {"status": "unknown_action", "type": action_type}
    
    def _do_file_write(self, action: Dict) -> Dict:
        """执行文件写入"""
        path = action.get("path", "reports/pulse_output.md")
        
        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        content = f"""# 脉冲智能体报告

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
动作强度: {action.get('strength', 0):.2f}
激活维度: {action.get('active_dims', [])}

## 状态摘要
- 智能体ID: {self.agent_id}
- 执行周期: {self.state.cycle_count}
- 历史动作数: {len(self.action_history)}

## 原始动作数据
```json
{json.dumps(action, ensure_ascii=False, default=str, indent=2)}
```
"""
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": "success", "type": "file_write", "path": path}
        except Exception as e:
            return {"status": "error", "type": "file_write", "error": str(e)}
    
    def _do_notification(self, action: Dict) -> Dict:
        """执行桌面通知 (模拟)"""
        title = action.get("title", "脉冲通知")
        message = action.get("message", "检测到脉冲信号")
        urgency = action.get("urgency", "normal")
        
        # 实际生产环境应调用系统通知API
        print(f"\n[通知] {title} [{urgency.upper()}]")
        print(f"  {message}")
        print(f"  强度: {action.get('strength', 0):.2f}")
        
        return {
            "status": "success", 
            "type": "desktop_notification",
            "title": title,
            "message": message
        }
    
    def _do_api_call(self, action: Dict) -> Dict:
        """执行API调用 (模拟)"""
        endpoint = action.get("endpoint", "")
        payload = action.get("payload", {})
        
        print(f"\n[API调用] {endpoint}")
        print(f"  Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        return {
            "status": "simulated",
            "type": "api_call",
            "endpoint": endpoint
        }
    
    def _do_report_generate(self, action: Dict) -> Dict:
        """生成报告"""
        template = action.get("template", "daily_pulse")
        sections = action.get("sections", ["summary"])
        
        report_path = f"reports/{template}_{int(time.time())}.md"
        
        content = f"""# {template.replace('_', ' ').title()} Report

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 章节
"""
        for section in sections:
            content += f"\n### {section.title()}\n\n[自动生成内容占位]\n"
        
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": "success", "type": "report_generate", "path": report_path}
        except Exception as e:
            return {"status": "error", "type": "report_generate", "error": str(e)}
    
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        """动作智能体处理输入脉冲"""
        return []
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """动作步骤: 接收脉冲 -> 解码 -> 执行动作"""
        # 获取DF输出模式
        if self.df:
            super().step(external_input)
            pattern = self.df.get_output_pattern()
        else:
            # 浅层动作智能体直接处理输入
            pattern = external_input if external_input is not None else np.zeros(self.dim)
        
        # 解码并执行
        if pattern is not None and np.linalg.norm(pattern) > 0.1:
            results = self.decode_and_execute(pattern)
            
            # 将执行结果反馈到KV堆
            for result in results:
                if result["status"] == "success":
                    feedback_signal = np.ones(self.dim) * 0.5
                    self.kv_stack.push(
                        f"action_feedback_{self.agent_id}_{time.time()}",
                        feedback_signal,
                        feedback_signal
                    )
        
        return []


class MemoryAgent(BaseSpikeAgent):
    """
    记忆智能体
    
    管理全局KV堆，支持跨智能体注意力查询
    
    职责:
    - KV堆容量管理
    - LRU淘汰策略
    - 跨智能体查询路由
    - 持久化/恢复
    """
    
    def __init__(self, agent_id: str, kv_capacity: int = 100000, 
                 dim: int = 16, retention_policy: str = "lru_7d"):
        super().__init__(agent_id, "memory", df_depth=0, dim=dim)
        
        # 使用更大的KV堆
        self.kv_stack = KVStack(capacity=kv_capacity, dim=dim, 
                                 retention_policy=retention_policy)
        self.retention_policy = retention_policy
        
    def handle_query(self, query_signal: np.ndarray, top_k: int = 3) -> List[Tuple]:
        """处理KV查询请求"""
        return self.kv_stack.query(query_signal, top_k)
    
    def handle_push(self, entry_id: str, key_signal: np.ndarray, 
                  value_state: np.ndarray) -> None:
        """处理KV推送请求"""
        self.kv_stack.push(entry_id, key_signal, value_state)
    
    def cleanup(self, max_age_days: float = 7.0) -> int:
        """清理过期条目"""
        return self.kv_stack.clear_expired(max_age_days)
    
    def persist(self, path: str) -> None:
        """持久化到磁盘"""
        self.kv_stack.save_to_disk(path)
    
    def restore(self, path: str) -> None:
        """从磁盘恢复"""
        self.kv_stack.load_from_disk(path)
    
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        """处理KV相关的脉冲消息"""
        if msg.kv_query:
            # KV查询请求
            query = msg.kv_query
            query_signal = np.array(query.get("query_signal", np.zeros(self.dim)))
            top_k = query.get("top_k", 3)
            results = self.handle_query(query_signal, top_k)
            
            # 构建查询结果脉冲
            result_spikes = []
            for entry_id, retrieved, score in results:
                result_spikes.append(SpikeMessage(
                    source_agent_id=self.agent_id,
                    source_unit_id="memory_kv",
                    payload=SpikePayload(
                        value=float(score),
                        strength=float(score),
                        timestamp=time.time(),
                        cycle_phase=msg.payload.cycle_phase
                    ),
                    target_agents=[msg.source_agent_id],
                    trace=msg.trace + [self.agent_id]
                ))
            return result_spikes
        
        return []
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """记忆智能体步骤: 处理查询 + 清理"""
        # 处理收件箱中的查询
        output_spikes = self.process_inbox()
        
        # 定期清理 (每100周期)
        if self.state.cycle_count % 100 == 0:
            cleaned = self.cleanup()
            if cleaned > 0:
                print(f"[MemoryAgent {self.agent_id}] 清理 {cleaned} 条过期KV")
        
        self.state.cycle_count += 1
        return output_spikes


class RhythmAgent(BaseSpikeAgent):
    """
    节律智能体
    
    基于Cron引擎，控制全局抑制/思考周期
    
    节律参数:
    - cycle_length: 120步一个周期
    - think_phase: 80步思考期
    - inhibit_phase: 40步抑制期
    - global_inhibition_curve: "sinusoidal" | "linear" | "step"
    """
    
    def __init__(self, agent_id: str, cycle_length: int = 120,
                 think_phase: int = 80, inhibit_phase: int = 40,
                 dim: int = 16):
        super().__init__(agent_id, "rhythm", df_depth=0, dim=dim)
        
        self.cycle_length = cycle_length
        self.think_phase = think_phase
        self.inhibit_phase = inhibit_phase
        self.global_inhibition_curve = "sinusoidal"
        self.phase_sync = "broadcast"
        
        self.current_phase = 0  # 0 = think, 1 = inhibit
        self.cycle_count = 0
        self.step_in_cycle = 0
        
        # 受控智能体列表
        self.controlled_agents: List[str] = []
        
    def add_controlled(self, agent_id: str) -> None:
        """添加受控智能体"""
        self.controlled_agents.append(agent_id)
    
    def get_inhibition_strength(self) -> float:
        """
        计算当前抑制强度
        
        思考期: 抑制从0逐渐增加到0.5
        抑制期: 抑制从0.5增加到0.9
        """
        if self.step_in_cycle < self.think_phase:
            # 思考期
            progress = self.step_in_cycle / self.think_phase
            if self.global_inhibition_curve == "sinusoidal":
                return 0.5 * (1 - np.cos(progress * np.pi)) / 2
            else:
                return 0.5 * progress
        else:
            # 抑制期
            progress = (self.step_in_cycle - self.think_phase) / self.inhibit_phase
            if self.global_inhibition_curve == "sinusoidal":
                return 0.5 + 0.4 * (1 - np.cos(progress * np.pi)) / 2
            else:
                return 0.5 + 0.4 * progress
    
    def broadcast_rhythm(self) -> SpikeMessage:
        """广播节律信号"""
        inhibition = self.get_inhibition_strength()
        
        msg = SpikeMessage(
            source_agent_id=self.agent_id,
            source_unit_id="rhythm_controller",
            payload=SpikePayload(
                value=-inhibition,  # 负值表示抑制
                strength=inhibition,
                timestamp=time.time(),
                cycle_phase=self.step_in_cycle
            ),
            target_agents=self.controlled_agents[:],
            hops_remaining=1,
            priority="high"
        )
        
        return msg
    
    def step(self, external_input: Optional[np.ndarray] = None) -> List[SpikeMessage]:
        """节律步骤: 更新周期 -> 广播抑制信号"""
        self.step_in_cycle = (self.step_in_cycle + 1) % self.cycle_length
        
        if self.step_in_cycle == 0:
            self.cycle_count += 1
            self.state.cycle_count += 1
        
        # 确定当前相位
        if self.step_in_cycle < self.think_phase:
            self.current_phase = 0  # think
            self.state.status = "think"
        else:
            self.current_phase = 1  # inhibit
            self.state.status = "inhibit"
        
        # 广播节律信号
        rhythm_msg = self.broadcast_rhythm()
        self.send_spike(rhythm_msg)
        
        return [rhythm_msg]
    
    def _process_spike(self, msg: SpikeMessage) -> List[SpikeMessage]:
        return []
    
    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update({
            "cycle_length": self.cycle_length,
            "think_phase": self.think_phase,
            "inhibit_phase": self.inhibit_phase,
            "current_phase": "think" if self.current_phase == 0 else "inhibit",
            "step_in_cycle": self.step_in_cycle,
            "inhibition": float(self.get_inhibition_strength()),
            "controlled_agents": len(self.controlled_agents)
        })
        return stats
