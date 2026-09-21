#!/usr/bin/env python3
"""Read-only, bounded capacity sampler for a four-node DGX Spark cluster."""
import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_NODES = {
    "spark-a": "spark-a",
    "spark-b": "spark-b",
    "spark-c": "spark-c",
    "spark-d": "spark-d",
}
DEFAULT_OUTPUT = Path.home() / ".local/state/dgx-spark-admission-gate/capacity.json"
REMOTE_PROBE = (
    "LC_ALL=C; boot_id=$(cat /proc/sys/kernel/random/boot_id) || exit 11; "
    "awk '/^MemTotal:|^MemAvailable:/{print $1 \"=\" $2}' /proc/meminfo; "
    "awk '/^some /{for(i=1;i<=NF;i++) if($i ~ /^avg10=/) print \"psi_some_avg10=\" substr($i,7)} "
    "/^full /{for(i=1;i<=NF;i++) if($i ~ /^avg10=/) print \"psi_full_avg10=\" substr($i,7)}' /proc/pressure/memory; "
    "awk '/^oom_kill /{print \"oom_kill_total=\" $2}' /proc/vmstat; "
    "awk '{print \"uptime_s=\" $1}' /proc/uptime; printf 'boot_id=%s\\n' \"$boot_id\""
)


def fail_class(exc):
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(exc, subprocess.CalledProcessError):
        return "ssh_error"
    if isinstance(exc, ValueError):
        return "parse_error"
    return "error"


def parse_probe(text, ts_unix):
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            values[key] = value
    required = {"MemTotal:", "MemAvailable:", "psi_some_avg10", "psi_full_avg10", "oom_kill_total", "uptime_s", "boot_id"}
    if not required.issubset(values):
        raise ValueError("missing_required_field")
    total_kib = int(values["MemTotal:"])
    available_kib = int(values["MemAvailable:"])
    some = float(values["psi_some_avg10"])
    full = float(values["psi_full_avg10"])
    oom = int(values["oom_kill_total"])
    uptime = float(values["uptime_s"])
    boot_id = values["boot_id"]
    if (total_kib <= 0 or not 0 <= available_kib <= total_kib or min(some, full, oom, uptime) < 0
            or not all(math.isfinite(v) for v in (some, full, uptime))
            or not re.fullmatch(r"[0-9a-f-]{36}", boot_id)):
        raise ValueError("invalid_units")
    return {"ts_unix": ts_unix, "boot_id": boot_id, "mem_available_gib": available_kib / 1048576,
            "mem_total_gib": total_kib / 1048576, "psi_some_avg10": some, "psi_full_avg10": full,
            "oom_kill_total": oom, "uptime_s": uptime}


def node_command(target, jump_host=None):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3"]
    if jump_host:
        command.extend(["-J", jump_host])
    return [*command, target, REMOTE_PROBE]


def collect_node(node_id, target, now, jump_host=None):
    try:
        result = subprocess.run(node_command(target, jump_host), capture_output=True, text=True, timeout=5, check=True)
        return node_id, parse_probe(result.stdout, now), None
    except Exception as exc:
        return node_id, None, fail_class(exc)


def get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(url, response.status, "status", response.headers, None)
        return json.loads(response.read().decode("utf-8"))


def health_ok(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status == 200


def collect_engine(now, a_uptime, a_sample_ts, engine_url="http://127.0.0.1:8888"):
    blank = {"ok": False, "running": None, "queued": None, "cached_tokens": None, "pending_tokens": None,
             "ts_unix": now, "ts_tic": None, "source_age_s": None}
    try:
        base = engine_url.rstrip("/")
        health = health_ok(base + "/health", timeout=3)
        loads = get_json(base + "/get_load", timeout=3)
        if not isinstance(loads, list) or not loads:
            raise ValueError("invalid_load")
        fields = ("num_reqs", "num_waiting_reqs", "num_tokens", "num_pending_tokens", "ts_tic")
        if any(not isinstance(row, dict) or any(isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float))
               or not math.isfinite(row[k]) or row[k] < 0 for k in fields) for row in loads):
            raise ValueError("invalid_load_fields")
        ts_tic = max(row["ts_tic"] for row in loads)
        if any(row["ts_tic"] != ts_tic for row in loads):
            raise ValueError("inconsistent_load_time")
        source_age = None
        if (isinstance(a_uptime, (int, float)) and not isinstance(a_uptime, bool)
                and isinstance(a_sample_ts, (int, float)) and math.isfinite(a_uptime)
                and math.isfinite(a_sample_ts)):
            # /get_load is fetched after the node probe. Estimate the matching /proc/uptime
            # at the engine observation time; tolerate scheduling/transport skew up to one second.
            estimated_uptime = a_uptime + max(0.0, now - a_sample_ts)
            if ts_tic <= estimated_uptime + 1.0:
                source_age = max(0.0, estimated_uptime - ts_tic)
        return {"ok": health, "running": sum(row["num_reqs"] for row in loads),
                "queued": sum(row["num_waiting_reqs"] for row in loads), "cached_tokens": sum(row["num_tokens"] for row in loads),
                "pending_tokens": sum(row["num_pending_tokens"] for row in loads), "ts_unix": now,
                "ts_tic": ts_tic, "source_age_s": source_age}
    except Exception:
        return blank


def collect_snapshot(nodes=None, engine_url="http://127.0.0.1:8888", jump_host=None):
    nodes_config = nodes or DEFAULT_NODES
    now = time.time()
    nodes, failures = {}, {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = [pool.submit(collect_node, node_id, target, now, jump_host) for node_id, target in nodes_config.items()]
        done, pending = concurrent.futures.wait(futures, timeout=8)
        for future in done:
            node_id, node, failure = future.result()
            if node is None:
                failures[node_id] = failure
            else:
                nodes[node_id] = node
        for future in pending:
            future.cancel()
            node_id = next(key for key, value in zip(nodes_config, futures) if value is future)
            failures[node_id] = "overall_timeout"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    complete = len(nodes) == len(nodes_config)
    engine_now = time.time()
    a_node = nodes.get("spark-a", {})
    engine = collect_engine(engine_now, a_node.get("uptime_s"), a_node.get("ts_unix"), engine_url)
    for node in nodes.values():
        node.pop("uptime_s", None)
    return {"ts_unix": now, "complete": complete, "nodes": nodes, "node_failures": failures, "engine": engine}


def atomic_write(path, payload):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def acquire_lock(output):
    lock_path = output.with_suffix(output.suffix + ".lock")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = open(lock_path, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def run_once(output, nodes=None, engine_url="http://127.0.0.1:8888", jump_host=None):
    data = collect_snapshot(nodes, engine_url, jump_host)
    atomic_write(output, data)
    failures = ",".join(sorted(data["node_failures"])) or "-"
    print("complete=%s engine_ok=%s nodes=%d failures=%s" % (str(data["complete"]).lower(), str(data["engine"]["ok"]).lower(), len(data["nodes"]), failures))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Collect memory pressure and engine load for the admission gate")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--node", action="append", metavar="NAME=SSH_TARGET",
                        help="four required node mappings; defaults to spark-a=spark-a through spark-d=spark-d")
    parser.add_argument("--jump-host", help="optional OpenSSH ProxyJump target")
    parser.add_argument("--engine-url", default="http://127.0.0.1:8888")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    nodes = DEFAULT_NODES
    if args.node:
        try:
            nodes = dict(item.split("=", 1) for item in args.node)
        except ValueError:
            parser.error("--node must be NAME=SSH_TARGET")
        if set(nodes) != set(DEFAULT_NODES) or any(not target for target in nodes.values()):
            parser.error("--node must define spark-a, spark-b, spark-c, and spark-d exactly once")
    args.output = args.output.expanduser()
    lock = acquire_lock(args.output)
    if lock is None:
        print("complete=false engine_ok=false nodes=0 skipped=locked")
        return 0
    try:
        if args.once:
            return run_once(args.output, nodes, args.engine_url, args.jump_host)
        while True:
            started = time.monotonic()
            run_once(args.output, nodes, args.engine_url, args.jump_host)
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
