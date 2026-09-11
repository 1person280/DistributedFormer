"""
脉冲神经网络实时监控面板
支持脉冲传播动画、KV堆热力图、智能体协作图、节律波形、统计仪表盘
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Any
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.core.distributedformer import DistributedFormer, KVStack, SpikingUnit
from distributedformer.workflow.engine import SpikeWorkflowEngine

try:
    import matplotlib
    matplotlib.use('Agg')  # 无界面后端
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Dashboard] matplotlib not available, using text-only mode")


class PulseDashboard:
    """
    脉冲神经网络实时监控面板
    
    组件:
    - 脉冲传播动画: 网络中脉冲的实时流动
    - KV堆热力图: 各单元历史激活模式
    - 智能体协作图: 消息传递网络拓扑
    - 节律波形图: 全局抑制强度时序
    - 统计仪表盘: 脉冲频率、活跃比例、疲劳分布
    """
    
    def __init__(self, engine: Optional[SpikeWorkflowEngine] = None,
                 network: Optional[DistributedFormer] = None,
                 refresh_interval: float = 0.5):
        self.engine = engine
        self.network = network
        self.refresh_interval = refresh_interval
        self.is_running = False
        self.data_history: deque = deque(maxlen=200)
        self.thread: Optional[threading.Thread] = None
        
        # 颜色映射: 蓝->绿->黄->红 (低->高强度)
        if MATPLOTLIB_AVAILABLE:
            self.cmap = LinearSegmentedColormap.from_list(
                'spike_cmap', ['#1a237e', '#00695c', '#fdd835', '#b71c1c']
            )
    
    # ═══════════════════════════════════════════════════════════════
    # 数据收集
    # ═══════════════════════════════════════════════════════════════
    
    def collect_snapshot(self) -> Dict:
        """收集当前网络状态快照"""
        snapshot = {
            "timestamp": time.time(),
            "pulse_data": [],
            "kv_data": [],
            "agent_data": [],
            "rhythm_data": [],
            "stats": {}
        }
        
        if self.network:
            stats = self.network.get_network_stats()
            snapshot["stats"] = stats
            
            # 收集所有单元状态
            all_units = (self.network.input_units + self.network.output_units)
            for layer in self.network.think_layers:
                all_units.extend(layer.get_all_units())
            
            for unit in all_units[:256]:  # 限制采样数量
                snapshot["pulse_data"].append({
                    "unit_id": unit.unit_id,
                    "state_norm": float(np.linalg.norm(unit.state)),
                    "fatigue": float(unit.fatigue),
                    "spike_count": unit.spike_count,
                    "threshold": float(unit.threshold)
                })
        
        if self.engine:
            for agent_id, agent in self.engine.agents.items():
                stats = agent.get_stats()
                snapshot["agent_data"].append({
                    "agent_id": agent_id,
                    "type": stats["agent_type"],
                    "spikes_sent": stats["spikes_sent"],
                    "spikes_received": stats["spikes_received"],
                    "status": stats["status"]
                })
        
        self.data_history.append(snapshot)
        return snapshot
    
    # ═══════════════════════════════════════════════════════════════
    # 可视化渲染
    # ═══════════════════════════════════════════════════════════════
    
    def render_pulse_propagation(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        渲染脉冲传播快照
        展示网络中各单元的发射状态
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._render_pulse_text()
        
        snapshot = self.collect_snapshot()
        pulse_data = snapshot["pulse_data"]
        
        if not pulse_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('🔥 脉冲神经网络实时监控', fontsize=16, fontweight='bold')
        
        # 1. 单元状态散点图 (左上)
        ax1 = axes[0, 0]
        states = [d["state_norm"] for d in pulse_data]
        fatigues = [d["fatigue"] for d in pulse_data]
        spikes = [d["spike_count"] for d in pulse_data]
        
        scatter = ax1.scatter(range(len(states)), states, 
                             c=spikes, cmap=self.cmap, s=50, alpha=0.7)
        ax1.set_xlabel('单元索引')
        ax1.set_ylabel('状态范数')
        ax1.set_title('脉冲传播状态')
        plt.colorbar(scatter, ax=ax1, label='脉冲计数')
        
        # 2. 疲劳度分布 (右上)
        ax2 = axes[0, 1]
        ax2.hist(fatigues, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.set_xlabel('疲劳度')
        ax2.set_ylabel('单元数')
        ax2.set_title('疲劳度分布')
        ax2.axvline(np.mean(fatigues), color='red', linestyle='--', label=f'均值={np.mean(fatigues):.2f}')
        ax2.legend()
        
        # 3. 时序脉冲频率 (左下)
        ax3 = axes[1, 0]
        if len(self.data_history) > 1:
            times = [i for i in range(len(self.data_history))]
            avg_states = [np.mean([d["state_norm"] for d in h["pulse_data"]]) 
                         if h["pulse_data"] else 0 for h in self.data_history]
            ax3.plot(times, avg_states, color='#d32f2f', linewidth=2)
            ax3.fill_between(times, avg_states, alpha=0.3, color='#d32f2f')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('平均状态范数')
        ax3.set_title('脉冲频率时序')
        
        # 4. 活跃单元比例 (右下)
        ax4 = axes[1, 1]
        active_ratio = sum(1 for f in fatigues if f < 0.5) / max(1, len(fatigues))
        inactive_ratio = 1 - active_ratio
        
        wedges, texts, autotexts = ax4.pie(
            [active_ratio, inactive_ratio],
            labels=['活跃', '疲劳'],
            colors=['#4caf50', '#f44336'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0)
        )
        ax4.set_title(f'活跃单元比例 (n={len(pulse_data)})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            path = f"reports/dashboard_{int(time.time())}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            return path
    
    def _render_pulse_text(self) -> str:
        """文本模式渲染 (无 matplotlib 时使用)"""
        snapshot = self.collect_snapshot()
        pulse_data = snapshot["pulse_data"]
        
        lines = []
        lines.append("=" * 60)
        lines.append("  脉冲网络监控 (文本模式)")
        lines.append("=" * 60)
        
        if pulse_data:
            states = [d["state_norm"] for d in pulse_data]
            fatigues = [d["fatigue"] for d in pulse_data]
            active = sum(1 for f in fatigues if f < 0.5)
            
            lines.append(f"\n  总单元: {len(pulse_data)}")
            lines.append(f"  活跃: {active} ({active/len(pulse_data)*100:.1f}%)")
            lines.append(f"  平均状态: {np.mean(states):.3f}")
            lines.append(f"  平均疲劳: {np.mean(fatigues):.3f}")
            
            # 可视化条
            lines.append("\n  单元状态 (| = 高活性):")
            for i, d in enumerate(pulse_data[:32]):
                bar_len = int(d["state_norm"] * 20)
                bar = "█" * min(bar_len, 20) + "░" * max(0, 20 - bar_len)
                fatigue_icon = "🔋" if d["fatigue"] < 0.5 else "😴"
                lines.append(f"    U{i:3d} {fatigue_icon} [{bar}] {d['state_norm']:.2f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def render_kv_heatmap(self, kv_stack: Optional[KVStack] = None,
                          save_path: Optional[str] = None) -> Optional[str]:
        """渲染 KV 堆热力图"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        kv = kv_stack or (self.network.kv_stack if self.network else None)
        if not kv or not kv.entries:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 构建热力图数据 (最近 50 条目的键信号)
        entries = list(kv.entries.items())[-50:]
        data = np.array([e[1].key_signal for e in entries])
        
        im = ax.imshow(data.T, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xlabel('KV条目 (最近50条)')
        ax.set_ylabel('信号维度')
        ax.set_title('KV堆键信号热力图')
        plt.colorbar(im, ax=ax, label='信号强度')
        
        plt.tight_layout()
        
        path = save_path or f"reports/kv_heatmap_{int(time.time())}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def render_agent_network(self, save_path: Optional[str] = None) -> Optional[str]:
        """渲染智能体协作网络图"""
        if not MATPLOTLIB_AVAILABLE or not self.engine:
            return None
        
        try:
            import networkx as nx
        except ImportError:
            return None
        
        G = nx.DiGraph()
        
        # 添加节点
        for agent_id, agent in self.engine.agents.items():
            stats = agent.get_stats()
            G.add_node(agent_id, 
                      type=agent.agent_type,
                      spikes=stats.get("spikes_sent", 0))
        
        # 添加边
        for agent_id, agent in self.engine.agents.items():
            for target in agent.output_targets:
                if target in self.engine.agents:
                    G.add_edge(agent_id, target, weight=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 节点颜色按类型
        type_colors = {
            'perception': '#4caf50',
            'reasoning': '#2196f3',
            'action': '#ff9800',
            'memory': '#9c27b0',
            'rhythm': '#f44336'
        }
        node_colors = [type_colors.get(G.nodes[n]['type'], '#757575') for n in G.nodes()]
        node_sizes = [200 + G.nodes[n]['spikes'] * 10 for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, 
                              arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # 图例
        legend_patches = [mpatches.Patch(color=c, label=t) 
                         for t, c in type_colors.items()]
        ax.legend(handles=legend_patches, loc='upper left')
        ax.set_title('智能体协作网络')
        ax.axis('off')
        
        plt.tight_layout()
        
        path = save_path or f"reports/agent_network_{int(time.time())}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    def render_rhythm_wave(self, save_path: Optional[str] = None) -> Optional[str]:
        """渲染节律波形图"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if len(self.data_history) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        times = list(range(len(self.data_history)))
        modulations = [h["stats"].get("global_modulation", 0.5) 
                      for h in self.data_history]
        
        ax.plot(times, modulations, color='#1565c0', linewidth=2)
        ax.fill_between(times, modulations, alpha=0.3, color='#1565c0')
        
        # 标注思考期/抑制期
        think_end = 80
        ax.axvline(think_end, color='green', linestyle='--', alpha=0.5, label='思考期结束')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('时间步')
        ax.set_ylabel('全局调制强度')
        ax.set_title('节律波形: 思考期→抑制期')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        path = save_path or f"reports/rhythm_wave_{int(time.time())}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path
    
    # ═══════════════════════════════════════════════════════════════
    # 实时模式
    # ═══════════════════════════════════════════════════════════════
    
    def start_realtime(self, num_frames: int = 50, save_dir: str = "reports/animation") -> List[str]:
        """
        启动实时监控模式
        生成一系列快照 PNG
        
        Args:
            num_frames: 生成帧数
            save_dir: 保存目录
        
        Returns:
            生成的文件路径列表
        """
        self.is_running = True
        os.makedirs(save_dir, exist_ok=True)
        paths = []
        
        for i in range(num_frames):
            if not self.is_running:
                break
            
            # 执行一步网络
            if self.network:
                self.network.step(np.random.randn(16) * 0.5)
            
            # 收集数据
            self.collect_snapshot()
            
            # 每5帧渲染一次
            if i % 5 == 0:
                path = self.render_pulse_propagation(
                    f"{save_dir}/frame_{i:04d}.png"
                )
                if path:
                    paths.append(path)
            
            time.sleep(self.refresh_interval)
        
        self.is_running = False
        return paths
    
    def stop(self) -> None:
        """停止实时监控"""
        self.is_running = False
    
    def generate_full_report(self, save_path: str = "reports/dashboard_full.png") -> str:
        """生成完整监控报告 (单张大图)"""
        if not MATPLOTLIB_AVAILABLE:
            return self._render_pulse_text()
        
        self.collect_snapshot()
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 脉冲传播 (大)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._draw_pulse_on_ax(ax1)
        
        # 2. 活跃比例 (小)
        ax2 = fig.add_subplot(gs[0, 2])
        self._draw_pie_on_ax(ax2)
        
        # 3. 疲劳分布 (小)
        ax3 = fig.add_subplot(gs[1, 2])
        self._draw_hist_on_ax(ax3)
        
        # 4. 时序 (底部)
        ax4 = fig.add_subplot(gs[2, :])
        self._draw_timeseries_on_ax(ax4)
        
        fig.suptitle('DistributedFormer 完整监控报告', fontsize=18, fontweight='bold')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def _draw_pulse_on_ax(self, ax):
        """在指定 ax 上绘制脉冲状态"""
        if not self.data_history:
            return
        data = self.data_history[-1]["pulse_data"]
        if not data:
            return
        states = [d["state_norm"] for d in data]
        colors = [d["spike_count"] for d in data]
        ax.scatter(range(len(states)), states, c=colors, cmap=self.cmap, s=30)
        ax.set_title('脉冲状态')
        ax.set_xlabel('单元')
        ax.set_ylabel('状态范数')
    
    def _draw_pie_on_ax(self, ax):
        """绘制活跃比例饼图"""
        if not self.data_history:
            return
        data = self.data_history[-1]["pulse_data"]
        if not data:
            return
        fatigues = [d["fatigue"] for d in data]
        active = sum(1 for f in fatigues if f < 0.5)
        ax.pie([active, len(fatigues)-active], labels=['活跃','疲劳'],
               colors=['#4caf50', '#f44336'], autopct='%1.0f%%')
        ax.set_title('活跃比例')
    
    def _draw_hist_on_ax(self, ax):
        """绘制疲劳度直方图"""
        if not self.data_history:
            return
        data = self.data_history[-1]["pulse_data"]
        if not data:
            return
        fatigues = [d["fatigue"] for d in data]
        ax.hist(fatigues, bins=15, color='steelblue', edgecolor='white')
        ax.set_title('疲劳分布')
        ax.set_xlabel('疲劳度')
    
    def _draw_timeseries_on_ax(self, ax):
        """绘制时序图"""
        if len(self.data_history) < 2:
            return
        times = range(len(self.data_history))
        avg_states = [np.mean([d["state_norm"] for d in h["pulse_data"]])
                     if h["pulse_data"] else 0 for h in self.data_history]
        ax.plot(times, avg_states, color='#d32f2f', linewidth=2)
        ax.fill_between(times, avg_states, alpha=0.3, color='#d32f2f')
        ax.set_title('脉冲频率时序')
        ax.set_xlabel('时间步')
        ax.set_ylabel('平均状态')


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("脉冲神经网络监控面板测试")
    print("=" * 60)
    
    # 创建网络
    df = DistributedFormer(depth=1, dim=16)
    
    # 运行一段时间收集数据
    print("\n运行网络收集数据...")
    for i in range(30):
        df.step(np.random.randn(16) * 0.5)
    
    # 创建面板
    dashboard = PulseDashboard(network=df)
    
    # 生成快照
    print("\n生成脉冲传播快照...")
    path1 = dashboard.render_pulse_propagation()
    if path1:
        print(f"  保存: {path1}")
    else:
        print(dashboard._render_pulse_text())
    
    # 生成 KV 热力图
    print("\n生成 KV 堆热力图...")
    path2 = dashboard.render_kv_heatmap()
    if path2:
        print(f"  保存: {path2}")
    
    # 生成完整报告
    print("\n生成完整监控报告...")
    path3 = dashboard.generate_full_report()
    print(f"  保存: {path3}")
    
    print("\n" + "=" * 60)
    print("监控面板测试通过!")
    print("=" * 60)
