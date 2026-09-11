"""
工作流引擎
管理脉冲智能体的生命周期、消息路由和协调

工作流定义 (YAML):
  rhythm: 节律控制参数
  agents: 智能体集群定义
  connections: 智能体间连接

执行生命周期:
  1. Cron触发
  2. 感知编码
  3. 脉冲注入
  4. 异步思考
  5. 侧向协同
  6. KV查询
  7. 输出生成
  8. 动作执行
  9. 反馈记录
  10. 周期结束
  11. 状态持久化
"""

import numpy as np
import time
import threading
import json
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore
except ImportError:
    yaml = None  # type: ignore

from typing import Dict, List, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.core.distributedformer import SpikeMessage, KVStack
from distributedformer.agents.base_agent import (
    BaseSpikeAgent, PerceptionAgent, ReasoningAgent, 
    ActionAgent, MemoryAgent, RhythmAgent
)


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str = "脉冲智能体工作流"
    version: str = "1.0"
    cycle_length: int = 120
    think_phase: int = 80
    inhibit_phase: int = 40
    cron_schedule: str = "*/5 * * * *"
    kv_capacity: int = 100000
    dim: int = 16


class MessageRouter:
    """
    脉冲消息路由器
    
    路由策略:
    - 直接路由: target_agents中指定目标
    - 内容寻址: 基于脉冲内容哈希选择目标
    - 拓扑哈希: 基于智能体拓扑结构路由
    - 广播: 发送给所有连接的智能体
    """
    
    def __init__(self):
        self.routes: Dict[str, List[str]] = {}  # agent_id -> [target_ids]
        self.agent_registry: Dict[str, BaseSpikeAgent] = {}
        
    def register(self, agent_id: str, agent: BaseSpikeAgent) -> None:
        """注册智能体"""
        self.agent_registry[agent_id] = agent
        self.routes[agent_id] = []
    
    def connect(self, source_id: str, target_id: str) -> None:
        """建立连接"""
        if source_id in self.routes:
            if target_id not in self.routes[source_id]:
                self.routes[source_id].append(target_id)
    
    def route(self, msg: SpikeMessage) -> List[str]:
        """
        路由脉冲消息到目标智能体
        
        返回成功投递的目标列表
        """
        delivered = []
        
        # 直接路由
        for target_id in msg.target_agents:
            if target_id in self.agent_registry:
                agent = self.agent_registry[target_id]
                agent.receive_spike(msg)
                delivered.append(target_id)
        
        # 如果未指定目标，广播给源的所有连接
        if not msg.target_agents and msg.source_agent_id in self.routes:
            for target_id in self.routes[msg.source_agent_id]:
                if target_id in self.agent_registry and target_id not in delivered:
                    agent = self.agent_registry[target_id]
                    agent.receive_spike(msg)
                    delivered.append(target_id)
        
        return delivered
    
    def broadcast(self, msg: SpikeMessage, exclude: Optional[List[str]] = None) -> List[str]:
        """广播给所有注册的智能体"""
        exclude = exclude or []
        delivered = []
        
        for agent_id, agent in self.agent_registry.items():
            if agent_id not in exclude and agent_id != msg.source_agent_id:
                agent.receive_spike(msg)
                delivered.append(agent_id)
        
        return delivered


class SpikeWorkflowEngine:
    """
    脉冲工作流引擎
    
    管理整个脉冲智能体网络的执行
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        
        # 智能体注册表
        self.agents: Dict[str, BaseSpikeAgent] = {}
        
        # 消息路由器
        self.router = MessageRouter()
        
        # 全局KV堆 (共享)
        self.global_kv = KVStack(
            capacity=self.config.kv_capacity, 
            dim=self.config.dim
        )
        
        # 节律智能体
        self.rhythm_agent: Optional[RhythmAgent] = None
        
        # 运行状态
        self.is_running = False
        self.current_cycle = 0
        self.step_count = 0
        self.thread: Optional[threading.Thread] = None
        
        # 统计
        self.stats_history: deque = deque(maxlen=1000)
        
        # 回调
        self.on_cycle_complete: Optional[Callable] = None
        self.on_spike_routed: Optional[Callable] = None
        
        # 全局消息队列 (用于跨智能体通信)
        self.global_queue: deque = deque(maxlen=10000)
        
    def create_workflow_from_yaml(self, yaml_path: str) -> None:
        """从YAML文件加载工作流配置"""
        if yaml is None:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 解析配置
        if 'rhythm' in data:
            r = data['rhythm']
            self.config.cycle_length = r.get('cycle_length', 120)
            self.config.think_phase = r.get('think_phase', 80)
            self.config.inhibit_phase = r.get('inhibit_phase', 40)
            self.config.cron_schedule = r.get('cron_schedule', '*/5 * * * *')
        
        # 创建智能体
        if 'agents' in data:
            for agent_def in data['agents']:
                self._create_agent_from_def(agent_def)
        
        # 建立连接
        if 'connections' in data:
            for conn in data['connections']:
                self.connect(conn['from'], conn['to'])
    
    def create_workflow_from_dict(self, data: Dict) -> None:
        """从字典加载工作流配置"""
        # 解析配置
        if 'rhythm' in data:
            r = data['rhythm']
            self.config.cycle_length = r.get('cycle_length', 120)
            self.config.think_phase = r.get('think_phase', 80)
            self.config.inhibit_phase = r.get('inhibit_phase', 40)
        
        # 创建智能体
        if 'agents' in data:
            for agent_def in data['agents']:
                self._create_agent_from_def(agent_def)
    
    def _create_agent_from_def(self, agent_def: Dict) -> BaseSpikeAgent:
        """根据定义创建智能体"""
        agent_id = agent_def['id']
        agent_type = agent_def['type']
        df_depth = agent_def.get('df_depth', 2)
        dim = self.config.dim
        
        agent = None
        
        if agent_type == 'perception':
            agent = PerceptionAgent(agent_id, df_depth, dim, self.global_kv)
            # 添加输入源
            for inp in agent_def.get('inputs', []):
                agent.add_input(inp['source'], inp.get('encoding', 'numeric_scalar'))
            # 设置输出目标
            agent.set_targets(agent_def.get('output_targets', []))
            
        elif agent_type == 'reasoning':
            agent = ReasoningAgent(agent_id, df_depth, dim, self.global_kv)
            agent.set_lateral(agent_def.get('lateral_connections', []))
            agent.kv_stack_sync = agent_def.get('kv_stack_sync', 'global')
            if 'specialization' in agent_def:
                agent.set_specialization(agent_def['specialization'])
            # 创建子智能体
            for sub_def in agent_def.get('sub_agents', []):
                sub_agent = self._create_agent_from_def(sub_def)
                agent.add_sub_agent(sub_agent)
                
        elif agent_type == 'action':
            agent = ActionAgent(agent_id, df_depth, dim, self.global_kv)
            for out in agent_def.get('outputs', []):
                agent.add_output(out['type'], **{k:v for k,v in out.items() if k != 'type'})
                
        elif agent_type == 'memory':
            kv_cap = agent_def.get('kv_capacity', 100000)
            retention = agent_def.get('retention_policy', 'lru_7d')
            agent = MemoryAgent(agent_id, kv_cap, dim, retention)
            
        elif agent_type == 'rhythm':
            agent = RhythmAgent(
                agent_id,
                self.config.cycle_length,
                self.config.think_phase,
                self.config.inhibit_phase,
                dim
            )
            self.rhythm_agent = agent
        
        if agent:
            self.register_agent(agent_id, agent)
        
        return agent
    
    def register_agent(self, agent_id: str, agent: BaseSpikeAgent) -> None:
        """注册智能体到引擎"""
        self.agents[agent_id] = agent
        self.router.register(agent_id, agent)
        
        # 如果是节律智能体，添加到控制列表
        if self.rhythm_agent and agent.agent_type != 'rhythm':
            self.rhythm_agent.add_controlled(agent_id)
    
    def connect(self, source_id: str, target_id: str) -> None:
        """连接两个智能体"""
        self.router.connect(source_id, target_id)
        
        # 同时设置智能体的输出目标
        if source_id in self.agents:
            agent = self.agents[source_id]
            if target_id not in agent.output_targets:
                agent.output_targets.append(target_id)
    
    def step(self) -> Dict:
        """
        执行一个完整的工作流步骤
        
        生命周期:
        1. 节律控制
        2. 感知编码
        3. 推理计算
        4. 记忆查询
        5. 动作执行
        6. 消息路由
        7. 统计记录
        """
        self.step_count += 1
        cycle_stats = {
            'step': self.step_count,
            'timestamp': time.time(),
            'agents': {},
            'messages_routed': 0,
            'actions_executed': 0
        }
        
        # 1. 节律控制
        if self.rhythm_agent:
            rhythm_msgs = self.rhythm_agent.step()
            for msg in rhythm_msgs:
                self.router.route(msg)
            cycle_stats['rhythm'] = self.rhythm_agent.get_stats()
        
        # 2. 感知智能体 (获取环境数据并编码)
        perception_outputs = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == 'perception':
                spikes = agent.step()
                perception_outputs.extend(spikes)
                for sp in spikes:
                    self.router.route(sp)
        
        # 3. 推理智能体 (核心计算)
        reasoning_outputs = []
        for agent_id, agent in self.agents.items():
            if agent.agent_type == 'reasoning':
                spikes = agent.step()
                reasoning_outputs.extend(spikes)
                for sp in spikes:
                    self.router.route(sp)
        
        # 4. 记忆智能体 (处理KV查询)
        for agent_id, agent in self.agents.items():
            if agent.agent_type == 'memory':
                spikes = agent.step()
                for sp in spikes:
                    self.router.route(sp)
        
        # 5. 动作智能体 (解码并执行)
        for agent_id, agent in self.agents.items():
            if agent.agent_type == 'action':
                agent.step()  # 动作智能体内部执行解码和动作
        
        # 6. 路由所有待发送消息
        total_routed = 0
        for agent_id, agent in self.agents.items():
            while agent.outbox:
                msg = agent.outbox.popleft()
                delivered = self.router.route(msg)
                total_routed += len(delivered)
        
        cycle_stats['messages_routed'] = total_routed
        
        # 7. 收集统计
        for agent_id, agent in self.agents.items():
            cycle_stats['agents'][agent_id] = agent.get_stats()
        
        # 8. 记录历史
        self.stats_history.append(cycle_stats)
        
        # 回调
        if self.on_cycle_complete:
            self.on_cycle_complete(cycle_stats)
        
        return cycle_stats
    
    def run_cycles(self, num_cycles: int = 1, sleep_ms: int = 100) -> List[Dict]:
        """运行多个周期"""
        results = []
        for i in range(num_cycles):
            result = self.step()
            results.append(result)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)
        return results
    
    def start_continuous(self, interval_ms: int = 100) -> None:
        """启动持续运行"""
        self.is_running = True
        
        def _run():
            while self.is_running:
                self.step()
                time.sleep(interval_ms / 1000.0)
        
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """停止运行"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def get_global_stats(self) -> Dict:
        """获取全局统计"""
        total_spikes = sum(
            a.state.total_spikes_sent for a in self.agents.values()
        )
        total_received = sum(
            a.state.total_spikes_received for a in self.agents.values()
        )
        
        return {
            'total_agents': len(self.agents),
            'total_steps': self.step_count,
            'total_spikes_sent': total_spikes,
            'total_spikes_received': total_received,
            'kv_utilization': self.global_kv.get_stats()['utilization'],
            'config': {
                'cycle_length': self.config.cycle_length,
                'think_phase': self.config.think_phase,
                'inhibit_phase': self.config.inhibit_phase
            }
        }
    
    def save_all_states(self, path_prefix: str) -> None:
        """保存所有智能体状态"""
        for agent_id, agent in self.agents.items():
            agent.save_state(f"{path_prefix}_{agent_id}")
        
        # 保存全局KV
        self.global_kv.save_to_disk(f"{path_prefix}_global_kv.json")
    
    def load_all_states(self, path_prefix: str) -> None:
        """加载所有智能体状态"""
        for agent_id, agent in self.agents.items():
            agent.load_state(f"{path_prefix}_{agent_id}")
        
        self.global_kv.load_from_disk(f"{path_prefix}_global_kv.json")
    
    def generate_report(self) -> str:
        """生成运行报告"""
        stats = self.get_global_stats()
        
        report = f"""# 脉冲智能体工作流报告

## 全局统计
- 总智能体数: {stats['total_agents']}
- 总执行步数: {stats['total_steps']}
- 总脉冲发送: {stats['total_spikes_sent']}
- 总脉冲接收: {stats['total_spikes_received']}
- KV堆利用率: {stats['kv_utilization']:.1%}

## 节律配置
- 周期长度: {stats['config']['cycle_length']}
- 思考期: {stats['config']['think_phase']}
- 抑制期: {stats['config']['inhibit_phase']}

## 智能体详情
"""
        for agent_id, agent in self.agents.items():
            a_stats = agent.get_stats()
            report += f"\n### {agent_id} ({a_stats['agent_type']})\n"
            report += f"- 状态: {a_stats['status']}\n"
            report += f"- 周期: {a_stats['cycles']}\n"
            report += f"- 发送脉冲: {a_stats['spikes_sent']}\n"
            report += f"- 接收脉冲: {a_stats['spikes_received']}\n"
        
        return report


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("脉冲工作流引擎测试")
    print("=" * 60)
    
    # 创建引擎
    engine = SpikeWorkflowEngine()
    
    # 创建智能体
    perception = PerceptionAgent("perception_0", df_depth=2, dim=16, kv_stack=engine.global_kv)
    perception.add_input("webbridge://finance.yahoo.com/AAPL", "numeric_scalar")
    perception.add_input("native://academic/arxiv", "text_sparse")
    perception.set_targets(["reasoning_0"])
    
    reasoning = ReasoningAgent("reasoning_0", df_depth=2, dim=16, kv_stack=engine.global_kv)
    reasoning.set_lateral(["reasoning_1"])
    reasoning.set_specialization("trend_analysis")
    
    reasoning1 = ReasoningAgent("reasoning_1", df_depth=1, dim=16, kv_stack=engine.global_kv)
    reasoning1.set_specialization("anomaly_detection")
    
    action = ActionAgent("action_0", df_depth=0, dim=16, kv_stack=engine.global_kv)
    action.add_output("file_write", path="reports/test_output.md")
    action.add_output("desktop_notification", condition="output_strength > 0.8")
    action.add_output("report_generate", template="daily_pulse")
    
    memory = MemoryAgent("memory_0", kv_capacity=10000, dim=16)
    
    rhythm = RhythmAgent("rhythm_0", cycle_length=120, think_phase=80, inhibit_phase=40)
    
    # 注册
    engine.register_agent("perception_0", perception)
    engine.register_agent("reasoning_0", reasoning)
    engine.register_agent("reasoning_1", reasoning1)
    engine.register_agent("action_0", action)
    engine.register_agent("memory_0", memory)
    engine.register_agent("rhythm_0", rhythm)
    
    # 连接
    engine.connect("perception_0", "reasoning_0")
    engine.connect("reasoning_0", "action_0")
    engine.connect("reasoning_0", "reasoning_1")
    engine.connect("reasoning_1", "action_0")
    engine.connect("action_0", "memory_0")
    
    # 运行5个周期
    print("\n运行 5 个周期...")
    for i in range(5):
        stats = engine.step()
        print(f"\n  周期 {i+1}:")
        print(f"    消息路由: {stats['messages_routed']} 条")
        print(f"    感知_0 脉冲: {stats['agents']['perception_0']['spikes_sent']}")
        print(f"    推理_0 脉冲: {stats['agents']['reasoning_0']['spikes_sent']}")
    
    # 全局统计
    print("\n" + "-" * 60)
    gstats = engine.get_global_stats()
    print(f"全局统计: {gstats['total_agents']} 智能体, "
          f"{gstats['total_spikes_sent']} 脉冲, "
          f"KV利用率: {gstats['kv_utilization']:.1%}")
    
    print("\n" + "=" * 60)
    print("工作流引擎测试通过!")
    print("=" * 60)
