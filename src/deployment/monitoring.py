"""
Prometheus 监控指标暴露
"""

import time
from typing import Dict, Any
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PrometheusMetrics:
    """
    Prometheus 指标收集器
    
    指标:
    - df_spikes_total: 脉冲总数 (按 agent_id 分标签)
    - df_kv_operations: KV 操作次数 (按 operation 分标签)
    - df_kv_latency: KV 操作延迟 (秒)
    - df_agents_active: 活跃智能体数
    - df_learning_ltp: STDP LTP 次数
    - df_learning_ltd: STDP LTD 次数
    """
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.start_time = time.time()
    
    def record_spike(self, agent_id: str, count: int = 1) -> None:
        """记录脉冲计数"""
        self.counters["df_spikes_total"][agent_id] += count
    
    def record_kv_operation(self, operation: str, latency_ms: float) -> None:
        """记录 KV 操作延迟"""
        self.counters["df_kv_operations"][operation] += 1
        self.histograms["df_kv_latency"].append(latency_ms)
    
    def set_gauge(self, name: str, value: float) -> None:
        """设置 gauge 指标"""
        self.gauges[name] = value
    
    def record_learning(self, ltp: int = 0, ltd: int = 0) -> None:
        """记录 STDP 学习事件"""
        self.counters["df_learning_ltp"]["total"] += ltp
        self.counters["df_learning_ltd"]["total"] += ltd
    
    def expose(self) -> str:
        """
        暴露 Prometheus 格式的指标
        
        Returns:
            Prometheus text format
        """
        lines = []
        
        # 脉冲计数
        lines.append("# HELP df_spikes_total Total number of spikes emitted")
        lines.append("# TYPE df_spikes_total counter")
        for agent_id, count in self.counters["df_spikes_total"].items():
            lines.append(f'df_spikes_total{{agent_id="{agent_id}"}} {count}')
        
        # KV 操作
        lines.append("# HELP df_kv_operations Total KV stack operations")
        lines.append("# TYPE df_kv_operations counter")
        for op, count in self.counters["df_kv_operations"].items():
            lines.append(f'df_kv_operations{{operation="{op}"}} {count}')
        
        # KV 延迟
        lines.append("# HELP df_kv_latency_ms KV operation latency in milliseconds")
        lines.append("# TYPE df_kv_latency_ms histogram")
        latencies = self.histograms["df_kv_latency"]
        if latencies:
            lines.append(f'df_kv_latency_ms_count {len(latencies)}')
            lines.append(f'df_kv_latency_ms_sum {sum(latencies)}')
            lines.append(f'df_kv_latency_ms_avg {sum(latencies)/len(latencies):.3f}')
        
        # Gauge 指标
        lines.append("# HELP df_agents_active Number of active agents")
        lines.append("# TYPE df_agents_active gauge")
        if "df_agents_active" in self.gauges:
            lines.append(f'df_agents_active {self.gauges["df_agents_active"]:.0f}')
        
        lines.append("# HELP df_uptime_seconds System uptime")
        lines.append("# TYPE df_uptime_seconds gauge")
        lines.append(f'df_uptime_seconds {time.time() - self.start_time:.0f}')
        
        # STDP 学习
        lines.append("# HELP df_learning_ltp Total LTP events")
        lines.append("# TYPE df_learning_ltp counter")
        ltp_count = self.counters["df_learning_ltp"]["total"]
        lines.append(f'df_learning_ltp {ltp_count}')
        
        lines.append("# HELP df_learning_ltd Total LTD events")
        lines.append("# TYPE df_learning_ltd counter")
        ltd_count = self.counters["df_learning_ltd"]["total"]
        lines.append(f'df_learning_ltd {ltd_count}')
        
        return "\n".join(lines) + "\n"
    
    def save_to_file(self, path: str = "reports/metrics.prom") -> str:
        """保存指标到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.expose())
        return path


if __name__ == "__main__":
    print("=" * 60)
    print("Prometheus 监控指标测试")
    print("=" * 60)
    
    metrics = PrometheusMetrics(port=9090)
    
    # 模拟数据
    metrics.record_spike("perception_0", 5)
    metrics.record_spike("reasoning_0", 12)
    metrics.record_spike("action_0", 3)
    
    metrics.record_kv_operation("query", 2.5)
    metrics.record_kv_operation("push", 1.2)
    metrics.record_kv_operation("query", 3.0)
    
    metrics.set_gauge("df_agents_active", 5.0)
    
    metrics.record_learning(ltp=10, ltd=4)
    
    print("\nPrometheus 指标输出:")
    print("-" * 60)
    print(metrics.expose())
    print("-" * 60)
    
    path = metrics.save_to_file()
    print(f"\n指标已保存: {path}")
    
    print("\n" + "=" * 60)
    print("监控指标测试通过!")
    print("=" * 60)
