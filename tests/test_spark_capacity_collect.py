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
