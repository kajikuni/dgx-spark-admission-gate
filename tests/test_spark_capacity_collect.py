import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("collector", Path(__file__).parents[1] / "spark_capacity_collect.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


GOOD = """MemTotal:=1048576
MemAvailable:=524288
psi_some_avg10=0.01
psi_full_avg10=0.00
oom_kill_total=2
uptime_s=100.5
boot_id=12345678-1234-1234-1234-123456789abc
"""


def test_parse_probe_normalizes_kib_to_gib():
    item = collector.parse_probe(GOOD, 9.0)
    assert item["mem_total_gib"] == 1.0
    assert item["mem_available_gib"] == 0.5
    assert item["oom_kill_total"] == 2


def test_parse_probe_rejects_missing_and_invalid_units():
    import pytest
    with pytest.raises(ValueError):
        collector.parse_probe(GOOD.replace("oom_kill_total=2\n", ""), 9.0)
    with pytest.raises(ValueError):
        collector.parse_probe(GOOD.replace("MemAvailable:=524288", "MemAvailable:=1048577"), 9.0)


def test_engine_failure_does_not_forge_zero(monkeypatch):
    monkeypatch.setattr(collector, "get_json", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(collector, "health_ok", lambda *args, **kwargs: True)
    engine = collector.collect_engine(10.0, 5.0, 10.0)
    assert engine["ok"] is False
    assert engine["running"] is None
    assert engine["cached_tokens"] is None


def test_remote_probe_shell_syntax():
    import subprocess
    subprocess.run(['/bin/sh','-n','-c',collector.REMOTE_PROBE],check=True)


def test_slow_health_probe_does_not_void_source_age(monkeypatch):
    """A loaded engine answers /health slowly; the source age must survive that.

    The estimate is anchored on the moment /get_load answered, so health latency no longer
    pushes ts_tic past it. A voided age used to make the gate discard the sample and close
    admission exactly when the cluster was busiest.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(collector.time, "time", lambda: clock["t"])

    def slow_health(*args, **kwargs):
        clock["t"] += 4.0  # /health blocks while the engine is saturated
        return True

    def load(*args, **kwargs):
        clock["t"] += 0.1
        # ts_tic is the head node's clock at the moment the engine answered.
        return [{"num_reqs": 3, "num_waiting_reqs": 0, "num_tokens": 10, "num_pending_tokens": 0,
                 "ts_tic": 504.1}]

    monkeypatch.setattr(collector, "health_ok", slow_health)
    monkeypatch.setattr(collector, "get_json", load)
    # The head node reported uptime 500.0 for a probe stamped at t=1000.0.
    engine = collector.collect_engine(1000.0, 500.0, 1000.0)
    assert engine["ok"] is True
    assert engine["source_age_s"] is not None and 0 <= engine["source_age_s"] <= 30


def test_source_age_is_voided_when_engine_clock_is_implausible(monkeypatch):
    monkeypatch.setattr(collector.time, "time", lambda: 1000.0)
    monkeypatch.setattr(collector, "health_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(collector, "get_json", lambda *args, **kwargs: [
        {"num_reqs": 1, "num_waiting_reqs": 0, "num_tokens": 1, "num_pending_tokens": 0,
         "ts_tic": 600.0}])  # a full minute ahead of the head node's uptime
    engine = collector.collect_engine(1000.0, 500.0, 1000.0)
    assert engine["ok"] is True and engine["source_age_s"] is None
