import os
from typing import Dict, Any


class CloudConfig:
    """
    云端部署配置
    
    支持:
    - 多租户隔离 (不同租户使用不同 Redis key 前缀)
    - 环境变量配置 (12-factor app 规范)
    - 负载均衡和自动扩展参数
    - Prometheus 监控
    """
    
    def __init__(self):
        # 租户配置
        self.tenant_id = os.getenv("TENANT_ID", "default")
        self.workspace_id = os.getenv("WORKSPACE_ID", "ws_001")
        
        # Redis 配置
        self.redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", 6379)),
            "db": int(os.getenv("REDIS_DB", 0)),
            "password": os.getenv("REDIS_PASSWORD", None),
        }
        
        # 网络配置
        self.network_config = {
            "depth": int(os.getenv("DF_DEPTH", 2)),
            "dim": int(os.getenv("DF_DIM", 16)),
            "kv_capacity": int(os.getenv("KV_CAPACITY", 100000)),
            "cycle_length": int(os.getenv("DF_CYCLE_LENGTH", 120)),
            "think_phase": int(os.getenv("DF_THINK_PHASE", 80)),
            "inhibit_phase": int(os.getenv("DF_INHIBIT_PHASE", 40)),
        }
        
        # 监控配置
        self.monitoring = {
            "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            "metrics_port": int(os.getenv("METRICS_PORT", 9090)),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        
        # 扩展配置
        self.scaling = {
            "min_instances": int(os.getenv("MIN_INSTANCES", 1)),
            "max_instances": int(os.getenv("MAX_INSTANCES", 10)),
            "scale_up_threshold": float(os.getenv("SCALE_UP_THRESHOLD", 0.8)),
            "scale_down_threshold": float(os.getenv("SCALE_DOWN_THRESHOLD", 0.3)),
            "scale_cooldown_seconds": int(os.getenv("SCALE_COOLDOWN", 60)),
        }
        
        # 性能配置
        self.performance = {
            "max_agents_per_workspace": int(os.getenv("MAX_AGENTS", 10)),
            "max_pulse_queue_size": int(os.getenv("MAX_PULSE_QUEUE", 10000)),
            "kv_eviction_policy": os.getenv("KV_EVICTION", "lru_7d"),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "redis": self.redis_config,
            "network": self.network_config,
            "monitoring": self.monitoring,
            "scaling": self.scaling,
            "performance": self.performance,
        }
    
    def print_config(self) -> str:
        """打印配置摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("  DistributedFormer 云端配置")
        lines.append("=" * 60)
        lines.append(f"\n  租户: {self.tenant_id}")
        lines.append(f"  工作区: {self.workspace_id}")
        lines.append(f"\n  Redis:")
        lines.append(f"    Host: {self.redis_config['host']}:{self.redis_config['port']}")
        lines.append(f"    DB: {self.redis_config['db']}")
        lines.append(f"\n  网络:")
        lines.append(f"    深度: {self.network_config['depth']}")
        lines.append(f"    维度: {self.network_config['dim']}")
        lines.append(f"    KV容量: {self.network_config['kv_capacity']:,}")
        lines.append(f"\n  监控:")
        lines.append(f"    Prometheus: {'启用' if self.monitoring['prometheus_enabled'] else '禁用'}")
        lines.append(f"    端口: {self.monitoring['metrics_port']}")
        lines.append(f"\n  扩展:")
        lines.append(f"    实例范围: {self.scaling['min_instances']} - {self.scaling['max_instances']}")
        lines.append(f"    扩容阈值: {self.scaling['scale_up_threshold']}")
        lines.append(f"    缩容阈值: {self.scaling['scale_down_threshold']}")
        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    config = CloudConfig()
    print(config.print_config())
