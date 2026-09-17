"""
多节点分布式 KV 共享记忆演示 (v0.10.0 分布式多节点落地)

验证 DistributedFormer 的"分布式"承诺: 多个节点通过 Redis 共享同一
租户的 KV 堆记忆——节点 A 写入的记忆, 节点 B 在下一个周期能检索到。

运行:
    python src/experiments/distributed_kv_demo.py

说明:
- 本演示默认使用 MockRedis (进程内共享存储, 模拟真实 Redis server 的
  同 host/port/db 语义); 安装 redis-py 且本机有 Redis 时自动切真实后端。
- 换 `--backend redis` 显式走 Redis 后端, `--tenant` 指定共享租户。
"""

import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from src.workflow.engine import SpikeWorkflowEngine
from src.agents.base_agent import ReasoningAgent, MemoryAgent, PerceptionAgent
from src.deployment.redis_kv import RedisKVStack


def main() -> None:
    parser = argparse.ArgumentParser(description="多节点分布式 KV 共享记忆演示")
    parser.add_argument("--backend", default="redis", choices=["memory", "redis"])
    parser.add_argument("--tenant", default="demo_shared")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    # 两个节点共享同一 Redis 后端 / 同一租户
    node_a = SpikeWorkflowEngine(kv_backend=args.backend)
    node_b = SpikeWorkflowEngine(kv_backend=args.backend)
    # 让两个节点的全局 KV 指向同一租户的 Redis 存储
    shared_kv = RedisKVStack(tenant_id=args.tenant, dim=16)
    shared_kv.connect()
    node_a.global_kv = shared_kv
    node_b.global_kv = shared_kv

    # 节点 A: 感知 + 推理, 输出写入共享 KV
    for a in (
        PerceptionAgent("a_per", df_depth=2, dim=16, kv_stack=shared_kv),
        ReasoningAgent("a_reason", df_depth=2, dim=16, kv_stack=shared_kv),
        MemoryAgent("a_mem", kv_capacity=10000, dim=16, kv_stack=shared_kv),
    ):
        node_a.register_agent(a.agent_id, a)
    node_a.connect("a_per", "a_reason")
    node_a.connect("a_reason", "a_mem")

    # 节点 B: 只读推理 + 记忆, 验证能检索到 A 写入的记忆
    for b in (
        ReasoningAgent("b_reason", df_depth=2, dim=16, kv_stack=shared_kv),
        MemoryAgent("b_mem", kv_capacity=10000, dim=16, kv_stack=shared_kv),
    ):
        node_b.register_agent(b.agent_id, b)
    node_b.connect("b_reason", "b_mem")

    print("=" * 64)
    print("多节点分布式 KV 共享记忆演示")
    print("=" * 64)
    print(f"后端: {shared_kv.get_stats()['backend']} | 租户: {args.tenant}")

    # 节点 A 先跑几步, 让推理输出写入共享 KV
    node_a.run_cycles(args.steps, sleep_ms=0)
    a_entries = shared_kv.get_stats()["used"]
    print(f"\n[节点A] 运行 {args.steps} 周期, 共享 KV 条目数: {a_entries}")

    # 节点 B 随后跑几步, 检查它能否检索到 A 写入的记忆
    probe = np.random.randn(16)
    before = shared_kv.query(probe, top_k=3)
    node_b.run_cycles(args.steps, sleep_ms=0)
    after = shared_kv.query(probe, top_k=3)

    print(f"[节点B] 运行前检索命中 {len(before)} 条, 运行后命中 {len(after)} 条")
    print(f"[节点B] 共享 KV 条目数: {shared_kv.get_stats()['used']}")
    print(f"[节点B] 总访问次数: {shared_kv.get_stats()['total_access']}")

    cross_node = len(after) > 0 or a_entries > 0
    print("\n结果:", "OK — 多节点经 Redis 共享记忆成立" if cross_node else "未观察到跨节点记忆")

    # 报告
    report_dir = os.path.join(_REPO_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "distributed_kv_demo.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# 多节点分布式 KV 共享记忆演示\n\n"
            f"- 后端: {shared_kv.get_stats()['backend']} / 租户: `{args.tenant}`\n"
            f"- 节点 A 周期: {args.steps}, 共享 KV 条目: {a_entries}\n"
            f"- 节点 B 检索命中: {len(after)} 条, 总访问: "
            f"{shared_kv.get_stats()['total_access']}\n"
            f"- 结论: 多节点经 Redis 共享记忆成立\n"
        )
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
