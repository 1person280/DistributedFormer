"""运行时安全监控骨架测试 (RFC #1 / v0.9.0 Safety Shield)"""

from distributedformer.security_monitor import (
    CriticalActionCircuitBreaker, IntentProbe, RuleBasedSafetyCheck,
    SecurityMonitor, Verdict,
)


def test_rule_based_check_verdicts():
    checker = RuleBasedSafetyCheck()
    assert checker.check({"tool": "report_generate"}).verdict is Verdict.ALLOW
    assert checker.check({"tool": "db_write"}).verdict is Verdict.REVIEW
    assert checker.check({"tool": "port_scan"}).verdict is Verdict.DENY


def test_intent_probe_records_history():
    probe = IntentProbe()
    audited = probe.inspect({"tool": "port_scan"})
    assert audited.verdict.verdict is Verdict.DENY
    assert probe.stats()["denied"] == 1
    assert len(probe.history) == 1


def test_circuit_breaker_fail_closed():
    breaker = CriticalActionCircuitBreaker()  # 无审核者
    review = breaker.trip({"tool": "db_write"}, None)
    assert not breaker.resolve(review)  # fail-closed: 默认拒绝
    # 提供审核者: 批准
    ok = CriticalActionCircuitBreaker(reviewer=lambda r: True)
    assert ok.resolve(ok.trip({"tool": "db_write"}, None))


def test_security_monitor_gate():
    monitor = SecurityMonitor()
    assert monitor.gate({"tool": "report_generate"})[0] is True
    denied = monitor.gate({"tool": "sudo"})
    assert denied[0] is False and denied[2] is None          # DENY: 无审核单
    tripped = monitor.gate({"tool": "db_write"})
    assert tripped[0] is False and tripped[2] is not None    # REVIEW: 熔断挂起
    assert monitor.stats()["breaker"]["tripped"] == 1
