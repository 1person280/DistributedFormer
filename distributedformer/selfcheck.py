"""模块自检套件 (原 main.py 测试模式)"""
import sys, os, time, json
import numpy as np
def print_banner():
    """打印启动横幅"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🔥 DistributedFormer v5.1 - 分布式脉冲神经网络原型           ║
║                                                                  ║
║     Kimi Work × DistributedFormer 整合方案                        ║
║     极简单元 · 异步事件驱动 · 持续思考 · 分形递归 · KV堆记忆       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def run_core_test():
    """运行核心模块测试"""
    print("\n" + "=" * 70)
    print("  [模式1] 核心模块测试")
    print("=" * 70)
    
    from distributedformer.core.distributedformer import (
        DistributedFormer, KVStack, SpikingUnit, 
        calculate_scale, SpikeMessage, SpikePayload
    )
    
    print("\n📐 规模计算:")
    for d in range(4):
        info = calculate_scale(d)
        print(f"    深度{d}: {info['base_units']}单元 / {info['total_params']}参数")
    
    print("\n🧠 KV堆测试:")
    import numpy as np
    kv = KVStack(capacity=100, dim=16)
    for i in range(20):
        kv.push(f"test_{i}", np.random.randn(16), np.random.randn(16))
    results = kv.query(np.random.randn(16), top_k=3)
    print(f"    20条目中查询top-3, 命中 {len(results)} 条")
    
    print("\n⚡ 脉冲单元测试:")
    unit = SpikingUnit("test_unit")
    spike_count = 0
    for i in range(100):
        spike = unit.step(np.random.randn(16), np.zeros(16), 1.0)
        if spike:
            spike_count += 1
    print(f"    100步中发射 {spike_count} 次脉冲")
    
    print("\n🌐 完整网络测试 (深度2, 20层):")
    df = DistributedFormer(depth=2, dim=16)
    for step in range(10):
        input_signal = np.random.randn(16) * 0.5
        output_spikes = df.step({"numeric": input_signal})
        print(f"    Step {step+1}: 输出 {len(output_spikes)} 脉冲, "
              f"调制={df.global_modulation:.2f}, 相位={df.cycle_phase}")
    
    stats = df.get_network_stats()
    print(f"\n    网络统计: {stats['total_units']}单元, {stats['total_spikes']}脉冲")
    print(f"    KV堆利用率: {stats['kv_stats']['utilization']:.1%}")
    
    print("\n✅ 核心模块测试通过!")


def run_codec_test():
    """运行编解码层测试"""
    print("\n" + "=" * 70)
    print("  [模式2] 脉冲编解码层测试")
    print("=" * 70)
    
    from distributedformer.codec.spike_codec import MultiModalCodec
    import numpy as np
    
    codec = MultiModalCodec(dim=16)
    
    print("\n📊 数值编码:")
    for val in [1.5, -2.0, 5.0, 0.0]:
        signal = codec.encode(val, "numeric")
        print(f"    {val:6.2f} -> 非零维: {np.count_nonzero(signal)}, 范数: {np.linalg.norm(signal):.3f}")
    
    print("\n📝 文本编码:")
    texts = [
        "Apple stock surges 5% on strong earnings",
        "Market crashes amid recession fears",
        "Tesla announces new battery technology"
    ]
    for text in texts:
        signal = codec.encode(text, "text")
        print(f"    '{text[:40]}...' -> 非零维: {np.count_nonzero(signal)}")
    
    print("\n📈 股票数据编码:")
    stock_signal = codec.encode_stock_data(
        price=175.50, change_pct=3.5, volume=45_000_000,
        news_text="Apple reports record quarterly revenue"
    )
    print(f"    AAPL综合信号 -> 非零维: {np.count_nonzero(stock_signal)}, "
          f"范数: {np.linalg.norm(stock_signal):.3f}")
    
    print("\n🎯 动作解码:")
    strong_pattern = np.zeros(16)
    strong_pattern[0] = 0.95
    strong_pattern[3] = 0.85
    actions = codec.decode(strong_pattern, threshold=0.3)
    for action in actions:
        print(f"    动作: {action['type']}, 强度: {action['strength']:.2f}")
    
    print("\n✅ 编解码层测试通过!")


def run_agent_test():
    """运行智能体框架测试"""
    print("\n" + "=" * 70)
    print("  [模式3] 脉冲智能体框架测试")
    print("=" * 70)
    
    from distributedformer.agents.base_agent import (
        PerceptionAgent, ReasoningAgent, ActionAgent,
        MemoryAgent, RhythmAgent
    )
    from distributedformer.core.distributedformer import KVStack
    import numpy as np
    
    kv = KVStack(capacity=1000, dim=16)
    
    print("\n🤖 创建智能体:")
    
    perception = PerceptionAgent("perception_0", df_depth=1, dim=16, kv_stack=kv)
    perception.add_input("finance://yahoo/AAPL", "numeric_scalar")
    print(f"    感知智能体: {perception.agent_id}, 输入源: {len(perception.inputs)}")
    
    reasoning = ReasoningAgent("reasoning_0", df_depth=2, dim=16, kv_stack=kv)
    reasoning.set_specialization("trend_analysis")
    print(f"    推理智能体: {reasoning.agent_id}, 特殊化: {reasoning.specialization}")
    
    action = ActionAgent("action_0", df_depth=0, dim=16, kv_stack=kv)
    action.add_output("file_write")
    action.add_output("desktop_notification")
    print(f"    动作智能体: {action.agent_id}, 输出配置: {len(action.outputs)}")
    
    memory = MemoryAgent("memory_0", kv_capacity=1000, dim=16)
    print(f"    记忆智能体: {memory.agent_id}, KV容量: {memory.kv_stack.capacity}")
    
    rhythm = RhythmAgent("rhythm_0", cycle_length=120, think_phase=80, inhibit_phase=40)
    print(f"    节律智能体: {rhythm.agent_id}, 周期: {rhythm.cycle_length}")
    
    print("\n🔄 智能体执行测试:")
    
    # 感知步骤
    print("    感知智能体执行...")
    p_spikes = perception.step()
    print(f"    输出脉冲: {len(p_spikes)}")
    
    # 推理步骤
    print("    推理智能体执行...")
    r_spikes = reasoning.step()
    print(f"    输出脉冲: {len(r_spikes)}")
    
    # 动作步骤
    print("    动作智能体执行...")
    action.step()
    print(f"    执行历史: {len(action.action_history)} 条")
    
    # 节律步骤
    print("    节律智能体执行...")
    rhythm.step()
    print(f"    当前相位: {'思考' if rhythm.current_phase == 0 else '抑制'}")
    print(f"    抑制强度: {rhythm.get_inhibition_strength():.3f}")
    
    print("\n✅ 智能体框架测试通过!")


def run_workflow_test():
    """运行工作流引擎测试"""
    print("\n" + "=" * 70)
    print("  [模式4] 工作流引擎测试")
    print("=" * 70)
    
    from distributedformer.workflow.engine import SpikeWorkflowEngine, WorkflowConfig
    from distributedformer.agents.base_agent import (
        PerceptionAgent, ReasoningAgent, ActionAgent,
        MemoryAgent, RhythmAgent
    )
    
    config = WorkflowConfig(
        name="测试工作流",
        cycle_length=120, think_phase=80, inhibit_phase=40, dim=16
    )
    engine = SpikeWorkflowEngine(config)
    
    print("\n⚙️ 创建工作流:")
    
    # 创建智能体
    perception = PerceptionAgent("perception_0", df_depth=1, dim=16, kv_stack=engine.global_kv)
    perception.add_input("test://source", "numeric_scalar")
    perception.set_targets(["reasoning_0"])
    
    reasoning = ReasoningAgent("reasoning_0", df_depth=2, dim=16, kv_stack=engine.global_kv)
    reasoning.set_specialization("trend_analysis")
    reasoning.set_targets(["action_0"])
    
    action = ActionAgent("action_0", df_depth=0, dim=16, kv_stack=engine.global_kv)
    action.add_output("file_write")
    action.add_output("report_generate")
    
    memory = MemoryAgent("memory_0", kv_capacity=1000, dim=16)
    
    rhythm = RhythmAgent("rhythm_0", cycle_length=120, think_phase=80, inhibit_phase=40, dim=16)
    
    # 注册
    engine.register_agent("perception_0", perception)
    engine.register_agent("reasoning_0", reasoning)
    engine.register_agent("action_0", action)
    engine.register_agent("memory_0", memory)
    engine.register_agent("rhythm_0", rhythm)
    
    # 连接
    engine.connect("perception_0", "reasoning_0")
    engine.connect("reasoning_0", "action_0")
    engine.connect("action_0", "memory_0")
    
    print(f"    注册智能体: {len(engine.agents)}")
    print(f"    建立连接: perception_0 -> reasoning_0 -> action_0 -> memory_0")
    
    # 运行5个周期
    print("\n🔄 运行 5 个周期:")
    for i in range(5):
        stats = engine.step()
        print(f"    周期 {i+1}: 路由 {stats['messages_routed']} 消息, "
              f"感知脉冲: {stats['agents']['perception_0']['spikes_sent']}, "
              f"推理脉冲: {stats['agents']['reasoning_0']['spikes_sent']}")
    
    # 全局统计
    gstats = engine.get_global_stats()
    print(f"\n📊 全局统计:")
    print(f"    智能体: {gstats['total_agents']}")
    print(f"    总脉冲: {gstats['total_spikes_sent']}")
    print(f"    KV利用率: {gstats['kv_utilization']:.1%}")
    
    print("\n✅ 工作流引擎测试通过!")


def run_stock_demo(cycles=10, interval=0.5):
    """运行股票监控演示"""
    print("\n" + "=" * 70)
    print("  [模式5] 股票监控端到端演示")
    print("=" * 70)
    
    from distributedformer.demos.stock_monitor import StockMonitorWorkflow
    
    workflow = StockMonitorWorkflow(tickers=["AAPL", "TSLA", "NVDA"])
    results = workflow.run_simulation(num_cycles=cycles, interval_sec=interval)
    
    # 生成报告
    report = workflow.generate_summary_report()
    report_path = "reports/stock_monitor_summary.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 总结报告已保存: {report_path}")
    
    return results


def run_full_test_suite():
    """运行完整测试套件"""
    print_banner()
    
    tests = [
        ("核心模块", run_core_test),
        ("编解码层", run_codec_test),
        ("智能体框架", run_agent_test),
        ("工作流引擎", run_workflow_test),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "✅ 通过"))
        except Exception as e:
            results.append((name, f"❌ 失败: {e}"))
            import traceback
            traceback.print_exc()
    
    # 股票演示
    try:
        run_stock_demo(cycles=5, interval=0.5)
        results.append(("股票监控", "✅ 通过"))
    except Exception as e:
        results.append(("股票监控", f"❌ 失败: {e}"))
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    for name, result in results:
        print(f"  {name}: {result}")
    
    passed = sum(1 for _, r in results if "通过" in r)
    print(f"\n  总计: {passed}/{len(results)} 通过")
    
    if passed == len(results):
        print("\n  🎉 所有测试通过! DistributedFormer 原型已就绪。")
    
    return results
