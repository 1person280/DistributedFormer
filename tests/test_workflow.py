from src.agents.base_agent import (
    PerceptionAgent, ReasoningAgent, ActionAgent, MemoryAgent, RhythmAgent
)
from src.workflow.engine import SpikeWorkflowEngine, WorkflowConfig


def _build_engine():
    config = WorkflowConfig(name="t", cycle_length=120, think_phase=80,
                            inhibit_phase=40, dim=16)
    engine = SpikeWorkflowEngine(config)
    p = PerceptionAgent("perception_0", df_depth=1, dim=16, kv_stack=engine.global_kv)
    p.add_input("test://source", "numeric_scalar")
    p.set_targets(["reasoning_0"])
    r = ReasoningAgent("reasoning_0", df_depth=1, dim=16, kv_stack=engine.global_kv)
    r.set_targets(["action_0"])
    a = ActionAgent("action_0", df_depth=0, dim=16, kv_stack=engine.global_kv)
    a.add_output("report_generate")
    m = MemoryAgent("memory_0", kv_capacity=1000, dim=16)
    rhythm = RhythmAgent("rhythm_0", cycle_length=120, think_phase=80, inhibit_phase=40, dim=16)
    for aid, agent in [("perception_0", p), ("reasoning_0", r),
                       ("action_0", a), ("memory_0", m), ("rhythm_0", rhythm)]:
        engine.register_agent(aid, agent)
    engine.connect("perception_0", "reasoning_0")
    engine.connect("reasoning_0", "action_0")
    return engine


def test_engine_steps():
    engine = _build_engine()
    for _ in range(3):
        stats = engine.step()
        assert "messages_routed" in stats
    gstats = engine.get_global_stats()
    assert gstats["total_agents"] == 5


def test_stock_workflow_smoke():
    from src.demos.stock_monitor import StockMonitorWorkflow
    wf = StockMonitorWorkflow(tickers=["AAPL"])
    wf.run_cycle()
    assert wf.engine.get_global_stats()["total_agents"] > 0
