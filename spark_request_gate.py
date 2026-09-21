#!/usr/bin/env python3
"""FIFO admission proxy for an OpenAI-compatible multi-node inference endpoint."""
import argparse
import asyncio
import json
import signal
import time
import math
import uuid
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, web

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade"}
HEAVY_PATHS = {"/v1/chat/completions", "/v1/completions"}
LIGHT_PATH = "/light/v1/chat/completions"
MAX_HEAVY_BYTES = 1048576
MAX_LIGHT_BYTES = 32768
TOKEN_BUDGET_TOTAL = 262144
# Admission estimates are not the engine's context length (currently 1M).
# Large normal turns consume more of the same shared budget, reducing overlap.
REQUEST_TOKEN_LIMIT = 131072
COMPRESSION_REQUEST_TOKEN_LIMIT = 65536
EXPECTED_OUTPUT_RESERVE = 4096
COMPRESSION_OUTPUT_RESERVE = 8192
ESTIMATE_OVERHEAD = 1024
ESTIMATE_METHOD = "unicode_weighted_json_plus_expected_output"


class EstimatedTokenBudget:
    """Shared admission estimate, independent of individual context length."""
    def __init__(self, total=TOKEN_BUDGET_TOTAL): self.total, self.reserved, self.gates = total, 0, []
    def can_reserve(self, cost): return self.reserved + cost <= self.total
    def reserve(self, cost):
        if not self.can_reserve(cost): return False
        self.reserved += cost; return True
    def release(self, cost):
        if cost > self.reserved: raise RuntimeError("budget release exceeds reservation")
        self.reserved -= cost
    def pump_all(self):
        for gate in self.gates: gate._pump()


class Gate:
    def __init__(self, limit, max_queue, name, budget=None):
        self.limit, self.max_queue, self.name = limit, max_queue, name
        self.active, self.waiting, self.completed, self.failed, self.cancelled = 0, 0, 0, 0, 0
        self._queue = deque()  # normal queue; retained for legacy inspection
        self._compression_queue = deque()
        self._kind_active = {"normal": 0, "compression": 0}
        self._kind_waiting = {"normal": 0, "compression": 0}
        self._kind_completed = {"normal": 0, "compression": 0}
        self._next_kind = "compression"  # when both have work, first grant is compression
        self._upstream_failures = deque()
        self._response_starts = deque()
        self._header_wait = {}
        self.budget = budget
        if budget is not None: budget.gates.append(self)

    def set_limit(self, new_limit):
        """Change future admission only; running requests are deliberately drained."""
        if not isinstance(new_limit, int) or new_limit < 0:
            raise ValueError("limit must be a non-negative integer")
        self.limit = new_limit
        self._pump()

    def record_upstream_failure(self, now=None):
        self._upstream_failures.append(time.monotonic() if now is None else now)

    def recent_upstream_failures(self, now=None, window=30):
        now = time.monotonic() if now is None else now
        while self._upstream_failures and self._upstream_failures[0] < now - window:
            self._upstream_failures.popleft()
        return len(self._upstream_failures)

    def record_response_start(self, seconds, ticket=None, now=None, control_eligible=True):
        self._response_starts.append((time.monotonic() if now is None else now, seconds, control_eligible))
        if ticket is not None: self._header_wait.pop(ticket.id, None)

    def latency_status(self, now=None):
        now = time.monotonic() if now is None else now
        while self._response_starts and self._response_starts[0][0] < now - 60:
            self._response_starts.popleft()
        values = sorted(value for _, value, _ in self._response_starts)
        control_values = [value for _, value, eligible in self._response_starts if eligible]
        return {"max": max(values, default=0), "p95": values[min(len(values) - 1, int(len(values) * .95))] if values else 0,
                "control_max": max(control_values, default=0),
                "inflight_header_age": max((now - started for started in self._header_wait.values()), default=0)}

    def _pump(self):
        while self.active < self.limit and (self._queue or self._compression_queue):
            heads = {"normal": self._queue[0] if self._queue else None,
                     "compression": self._compression_queue[0] if self._compression_queue and self._kind_active["compression"] < 1 else None}
            eligible = [kind for kind, ticket in heads.items() if ticket is not None and (self.budget is None or self.budget.can_reserve(ticket.cost))]
            if not eligible: break
            kind = self._next_kind if self._next_kind in eligible else eligible[0]
            ticket = heads[kind]
            queue = self._queue if kind == "normal" else self._compression_queue
            if ticket.future.cancelled():
                queue.popleft(); self.waiting -= 1; self._kind_waiting[kind] -= 1
                continue
            if self.budget is not None: assert self.budget.reserve(ticket.cost)
            queue.popleft()
            self.active += 1
            self._kind_active[kind] += 1
            if ticket.control_eligible: self._header_wait[ticket.id] = time.monotonic()
            self.waiting -= 1; self._kind_waiting[kind] -= 1
            ticket.granted = True
            ticket.future.set_result(None)
            if len(eligible) == 2: self._next_kind = "normal" if kind == "compression" else "compression"

    async def acquire(self, wait_seconds, request=None, cost=0, control_eligible=True, kind="normal"):
        if kind not in {"normal", "compression"}: raise ValueError("unknown ticket kind")
        if kind == "compression" and self._kind_waiting[kind] >= 4:
            raise web.HTTPTooManyRequests(text=json.dumps({"error": {"message": "compression queue is full", "type": "server_error", "code": "compression_queue_full"}}), content_type="application/json")
        can_start = self.active < self.limit and not self._queue and not self._compression_queue and (kind != "compression" or self._kind_active[kind] < 1) and (self.budget is None or self.budget.can_reserve(cost))
        if not can_start and self.waiting >= self.max_queue:
            raise web.HTTPTooManyRequests(text=json.dumps({"error": {"message": "request queue is full", "type": "server_error", "code": "queue_full"}}), content_type="application/json")
        ticket = Ticket(asyncio.get_running_loop().create_future(), cost=cost, control_eligible=control_eligible, kind=kind)
        ticket.id = id(ticket)
        ticket.queued_at = time.monotonic()
        queue = self._queue if kind == "normal" else self._compression_queue
        queue.append(ticket)
        self.waiting += 1; self._kind_waiting[kind] += 1
        self._pump()
        deadline = time.monotonic() + wait_seconds
        try:
            while not ticket.granted:
                if request is not None and (request.transport is None or request.transport.is_closing()):
                    raise asyncio.CancelledError()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                try:
                    await asyncio.wait_for(asyncio.shield(ticket.future), min(.1, remaining))
                except asyncio.TimeoutError:
                    continue
        except asyncio.TimeoutError:
            if not ticket.granted:
                queue.remove(ticket); self.waiting -= 1; self._kind_waiting[kind] -= 1
            self.failed += 1
            raise web.HTTPGatewayTimeout(text=json.dumps({"error": {"message": "timed out waiting for capacity", "type": "server_error", "code": "queue_timeout"}}), content_type="application/json")
        except BaseException:
            if not ticket.granted:
                queue.remove(ticket); self.waiting -= 1; self._kind_waiting[kind] -= 1
            else:  # cancellation raced with assignment: return the slot.
                self.active -= 1
                self._kind_active[kind] -= 1
                self._header_wait.pop(ticket.id, None)
                if self.budget is not None: self.budget.release(ticket.cost)
                self._pump()
                if self.budget is not None: self.budget.pump_all()
            self.cancelled += 1
            raise

        return ticket

    def release(self, success=True, cost=0, ticket=None):
        self.active -= 1
        if ticket is not None:
            cost = ticket.cost
            self._header_wait.pop(ticket.id, None)
        elif self._header_wait:
            # Compatibility path for callers that predate admission tickets.
            self._header_wait.pop(next(iter(self._header_wait)))
        if self.budget is not None: self.budget.release(cost)
        if ticket is not None:
            self._kind_active[ticket.kind] -= 1
        if success:
            self.completed += 1
            if ticket is not None: self._kind_completed[ticket.kind] += 1
        else:
            self.failed += 1
        self._pump()
        if self.budget is not None: self.budget.pump_all()

    def status(self):
        now = time.monotonic()
        queues = {"normal": self._queue, "compression": self._compression_queue}
        return {"active": self.active, "waiting": self.waiting, "completed": self.completed,
                "failed": self.failed, "cancelled": self.cancelled, "limit": self.limit,
                "max_queue": self.max_queue, "response_start_latency": self.latency_status(),
                "kinds": {kind: {"active": self._kind_active[kind], "waiting": self._kind_waiting[kind], "completed": self._kind_completed[kind], "queue_wait_seconds": max(0, now - queues[kind][0].queued_at) if queues[kind] else 0} for kind in self._kind_active}}


@dataclass
class Ticket:
    future: asyncio.Future
    granted: bool = False
    cost: int = 0
    id: int = 0
    control_eligible: bool = True
    kind: str = "normal"
    queued_at: float = 0


@dataclass
class Config:
    upstream: str = "http://127.0.0.1:8888"
    heavy_limit: int = 2
    light_limit: int = 1
    max_queue: int = 64
    queue_wait: float = 1500
    upstream_timeout: float = 1800
    adaptive: bool = False
    adaptive_poll_seconds: float = 5
    capacity_file: str = str(Path.home() / ".local/state/dgx-spark-admission-gate/capacity.json")
    token_budget_total: int = TOKEN_BUDGET_TOTAL
    request_token_limit: int = REQUEST_TOKEN_LIMIT
    compression_request_token_limit: int = COMPRESSION_REQUEST_TOKEN_LIMIT
    expected_output_reserve: int = EXPECTED_OUTPUT_RESERVE
    compression_output_reserve: int = COMPRESSION_OUTPUT_RESERVE


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class CapacityFileProvider:
    """Read collector's atomically replaced all-node snapshot; malformed is fail-closed."""
    def __init__(self, path): self.path = Path(path).expanduser()

    async def __call__(self):
        try:
            raw = json.loads(self.path.read_text())
            now = time.time()
            if not isinstance(raw, dict) or raw.get("complete") is not True or not finite_number(raw.get("ts_unix")):
                return None
            nodes, engine = raw.get("nodes"), raw.get("engine")
            if isinstance(nodes, dict):
                nodes = [{**value, "id": key} for key, value in nodes.items() if isinstance(value, dict)]
            if not isinstance(nodes, list) or len(nodes) != 4 or {n.get("id") for n in nodes if isinstance(n, dict)} != {"spark-a", "spark-b", "spark-c", "spark-d"} or not isinstance(engine, dict):
                return None
            age = now - raw["ts_unix"]
            required_engine = ("ok", "running", "queued", "cached_tokens", "pending_tokens", "ts_unix", "ts_tic")
            if age < -1 or age > 30 or not all(k in engine for k in required_engine) or not all(finite_number(engine[k]) and engine[k] >= 0 for k in required_engine if k != "ok"):
                return None
            if engine["ok"] is not True or now - engine["ts_unix"] < -1 or now - engine["ts_unix"] > 30 or not finite_number(engine.get("source_age_s")) or not 0 <= engine["source_age_s"] <= 30:
                return None
            headrooms, some, full, ooms = [], [], [], []
            for node in nodes:
                required = ("ts_unix", "boot_id", "mem_available_gib", "mem_total_gib", "psi_some_avg10", "psi_full_avg10", "oom_kill_total")
                if not all(k in node for k in required) or not all(finite_number(node[k]) and node[k] >= 0 for k in required if k != "boot_id") or not isinstance(node["boot_id"], str) or now - node["ts_unix"] < -1 or now - node["ts_unix"] > 30 or abs(node["ts_unix"] - raw["ts_unix"]) > 30 or node["mem_total_gib"] < node["mem_available_gib"]:
                    return None
                headrooms.append(node["mem_available_gib"]); some.append(node["psi_some_avg10"]); full.append(node["psi_full_avg10"])
                ooms.append((node["id"], node["boot_id"], node["oom_kill_total"]))
            tokens = engine["cached_tokens"] + engine["pending_tokens"]
            return {"fresh": True, "engines_ok": True, "age": age, "min_headroom": min(headrooms), "psi": max(some), "psi_full": max(full), "kv_usage": tokens, "ooms": ooms, "running": engine["running"], "queued": engine["queued"]}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None


class AdaptiveController:
    """Fail-closed admission controller. Telemetry parsing is intentionally isolated.

    `telemetry_provider` returns a normalized dict or None. A normalized sample
    has `fresh`, `engines_ok`, `min_headroom`, `psi`, and `kv_usage` fields.
    Missing or unhealthy data closes both gates while existing active work drains.
    """
    def __init__(self, heavy, light, telemetry_provider, poll_seconds=5, config=None):
        self.heavy, self.light, self.telemetry_provider = heavy, light, telemetry_provider
        self.poll_seconds = poll_seconds
        self.mode, self.target, self.reason = "adaptive", 0, "starting"
        self.last_change = 0.0; self.healthy_since = None; self.cooldown_until = 0.0
        self.telemetry_age = None; self.min_headroom = self.psi = self.kv_usage = None
        self._last_logged_reason = None
        self._oom_by_boot = {}
        self.config = config or Config()
        self.heavy.set_limit(0)
        self.light.set_limit(0)

    def status(self):
        budget = self.heavy.budget
        return {"mode": self.mode, "effective_limit": {"heavy": self.heavy.limit, "light": self.light.limit},
                "target": self.target, "reason": self.reason, "telemetry_age": self.telemetry_age,
                "min_headroom": self.min_headroom, "psi": self.psi, "kv_usage": self.kv_usage,
                "last_change": self.last_change, "token_reservation": {
                    "total": budget.total if budget is not None else None,
                    "per_request_limit": self.config.request_token_limit,
                    "compression_request_limit": self.config.compression_request_token_limit,
                    "expected_output_cap": self.config.expected_output_reserve,
                    "compression_output_cap": self.config.compression_output_reserve,
                    "estimate_method": ESTIMATE_METHOD,
                    "note": "Unicode-aware estimate; declared output above the expected cap is not fully reserved.",
                },
                "thresholds": {"critical_headroom_gib": 6, "caution_headroom_gib": 10, "critical_cached_pending": 262144, "caution_cached_pending": 196608}}

    def _set(self, target, reason, now):
        target = max(0, min(4, int(target)))
        changed = target != self.target or self.heavy.limit != target or self.light.limit != (1 if target else 0)
        self.target, self.reason = target, reason
        self.heavy.set_limit(target); self.light.set_limit(1 if target else 0)
        if changed:
            self.last_change = now
        if reason != self._last_logged_reason:
            print(json.dumps({"event": "adaptive_decision", "reason": reason, "target": target}), flush=True)
            self._last_logged_reason = reason

    def evaluate(self, sample, now=None):
        """One pure-ish decision tick; the provider/schema can be tested separately."""
        now = time.monotonic() if now is None else now
        required = ("min_headroom", "psi", "psi_full", "kv_usage", "running", "queued")
        if not isinstance(sample, dict) or not sample.get("fresh") or not sample.get("engines_ok") or not all(finite_number(sample.get(k)) for k in required):
            self.telemetry_age = sample.get("age") if isinstance(sample, dict) else None
            self.healthy_since = None; self._set(0, "telemetry_unhealthy", now); return
        if now < self.cooldown_until:
            self.healthy_since = None; self._set(0, "cooldown", now); return
        self.telemetry_age = sample.get("age"); self.min_headroom = sample.get("min_headroom")
        self.psi, self.kv_usage = sample.get("psi"), sample.get("kv_usage")
        oom_increase = False
        for node_id, boot_id, count in sample.get("ooms", []):
            prior = self._oom_by_boot.get(node_id)
            if prior and prior[0] == boot_id and count > prior[1]: oom_increase = True
            self._oom_by_boot[node_id] = (boot_id, count)
        # cached_tokens can include evictable idle cache: only treat it as critical
        # while the engine is actively serving work; idle cache drains to limit 1.
        token_critical = self.kv_usage >= 262144 and sample.get("running", 0) > 0
        critical = (self.min_headroom < 6 or sample.get("psi_full", 0) >= 5 or oom_increase or token_critical)
        caution = (self.min_headroom < 10 or self.psi >= 1 or self.kv_usage >= 196608)
        errors = self.heavy.recent_upstream_failures(now) + self.light.recent_upstream_failures(now)
        slow = self.heavy.latency_status(now)["control_max"] > 20 or self.light.latency_status(now)["control_max"] > 20
        if critical:
            self.healthy_since = None; self.cooldown_until = now + 60; self._set(0, "critical_pressure", now); return
        if caution:
            self.healthy_since = None; self._set(1, "memory_pressure", now); return
        if errors:
            self.healthy_since = None; self.cooldown_until = now + 60
            self._set(max(0, self.target - 1), "upstream_failure", now); return
        if slow:
            self.healthy_since = None; self.cooldown_until = now + 60
            self._set(max(0, self.target - 1), "slow_response", now); return
        if self.healthy_since is None: self.healthy_since = now
        # Healthy capacity is advertised immediately.  The former 30-second,
        # one-slot ramp meant a burst could finish while the controller still
        # exposed only one of four configured contexts.  Pressure, OOM,
        # upstream errors and slow-response cooldowns above still reduce or
        # close admission; this branch is reached only for a healthy sample.
        self._set(4, "healthy", now)

    async def run(self):
        while True:
            try:
                sample = await self.telemetry_provider()
                self.evaluate(sample)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Details can contain remote payloads; do not log them.
                self.evaluate(None)
                print(json.dumps({"event": "adaptive_telemetry_error", "kind": type(exc).__name__}), flush=True)
            await asyncio.sleep(self.poll_seconds)


def safe_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS | {"host", "content-length"}}


def openai_error(status, message, code):
    return web.Response(status=status, content_type="application/json", text=json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": code}}))


def estimate_request_cost(body, payload, *, output_cap=EXPECTED_OUTPUT_RESERVE):
    """Estimate prompt + expected output without retaining request content.

    ASCII-heavy JSON is close to four characters per token, while Japanese and
    other non-ASCII text is commonly near one code point per token.  Reserving
    the full declared output (often 32K) caused one ordinary request to block
    three otherwise available engine slots, so admission reserves at most the
    observed operational output envelope.  The hard per-request context check
    and live memory/OOM controller remain independent safety limits.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = body.decode("utf-8", "replace")
    ascii_chars = sum(ord(ch) < 128 for ch in text)
    non_ascii_chars = len(text) - ascii_chars
    prompt = math.ceil(ascii_chars / 4 + non_ascii_chars) + ESTIMATE_OVERHEAD
    outputs = [payload[k] for k in ("max_tokens", "max_completion_tokens") if isinstance(payload.get(k), int) and not isinstance(payload.get(k), bool) and payload[k] > 0]
    declared_output = max(outputs) if outputs else output_cap
    output = min(declared_output, output_cap)
    return prompt + output


async def phase(stream, name, request_id, **extra):
    payload = {"phase": name, "request_id": request_id, **extra}
    await stream.write(f"event: gate.phase\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())


async def compression_proxy(request):
    """Opt-in SSE admission protocol for Hermes compression only."""
    cfg, gate = request.app["config"], request.app["heavy"]
    started = time.monotonic()
    try: body = await request.read()
    except web.HTTPRequestEntityTooLarge: return openai_error(400, "request bytes exceed limit", "request_bytes_exceeded")
    if len(body) > MAX_HEAVY_BYTES:
        print(json.dumps({"event": "compression_rejected", "code": "request_bytes_exceeded", "actual_bytes": len(body)}), flush=True)
        return openai_error(400, "request bytes exceed limit", "request_bytes_exceeded")
    try: payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError): payload = None
    if not isinstance(payload, dict) or payload.get("stream") is not True:
        return openai_error(400, "compression route requires stream=true", "compression_stream_required")
    if payload.get("n", 1) != 1 or isinstance(payload.get("n", 1), bool):
        return openai_error(400, "this trial permits one completion per request (n=1)", "invalid_request_error")
    budget = request.app.get("token_budget")
    if budget is None:
        return openai_error(503, "compression admission is unavailable", "compression_unavailable")
    cost = estimate_request_cost(body, payload, output_cap=cfg.compression_output_reserve)
    if cost > cfg.compression_request_token_limit:
        print(json.dumps({"event": "compression_rejected", "code": "estimated_context_exceeded", "actual_bytes": len(body), "estimated_cost": cost, "limit": cfg.compression_request_token_limit}), flush=True)
        return openai_error(400, "request estimate exceeds model context; compact or split it", "estimated_context_exceeded")
    request_id = uuid.uuid4().hex
    stream = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    await stream.prepare(request)
    try:
        await phase(stream, "queued", request_id)
        admission = await gate.acquire(300, request, cost=cost, control_eligible=True, kind="compression")
    except web.HTTPException as exc:
        await phase(stream, "failed", request_id, code="queue_timeout" if exc.status == 504 else "queue_rejected", error_status=exc.status)
        await stream.write_eof(); return stream
    except asyncio.CancelledError:
        raise
    queue_wait_ms = int((time.monotonic() - started) * 1000)
    ok = False
    try:
        if time.monotonic() - started >= 600:
            await phase(stream, "failed", request_id, code="total_timeout", error_status=504)
        else:
            # A failed client write after admission must still release the shared slot.
            await phase(stream, "admitted", request_id, queue_wait_ms=queue_wait_ms)
            timeout = ClientTimeout(total=max(1, min(cfg.upstream_timeout, 600 - (time.monotonic() - started))))
            upstream_started = time.monotonic()
            async with request.app["session"].request("POST", cfg.upstream + "/v1/chat/completions", data=body, headers=safe_headers(request.headers), timeout=timeout, allow_redirects=False) as upstream:
                gate.record_response_start(time.monotonic() - upstream_started, admission, control_eligible=True)
                if not 200 <= upstream.status < 300:
                    if upstream.status >= 500: gate.record_upstream_failure()
                    await phase(stream, "failed", request_id, code="upstream_error", error_status=upstream.status)
                else:
                    done = False
                    frame_tail = b""
                    async for chunk in upstream.content.iter_chunked(16384):
                        await stream.write(chunk)
                        # The TCP reader may split the marker anywhere. Examine
                        # complete SSE lines only, never arbitrary body substrings.
                        frame_tail += chunk
                        while b"\n" in frame_tail:
                            line, frame_tail = frame_tail.split(b"\n", 1)
                            if line.rstrip(b"\r") == b"data: [DONE]": done = True
                        if len(frame_tail) > 1024 * 1024:
                            frame_tail = b""  # malformed oversized line cannot be a marker
                    if done:
                        ok = True
                    else:
                        await phase(stream, "failed", request_id, code="incomplete_stream", error_status=502)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        gate.record_upstream_failure()
        await phase(stream, "failed", request_id, code="upstream_error", error_status=502)
    finally:
        gate.release(ok, ticket=admission)
    await stream.write_eof()
    return stream


async def proxy(request, gate, upstream_path):
    cfg = request.app["config"]
    try:
        body = await request.read()
    except web.HTTPRequestEntityTooLarge:
        return openai_error(400, "request is too large; compact or split it", "context_length_exceeded")
    if gate.name == "heavy" and len(body) > MAX_HEAVY_BYTES:
        return openai_error(400, "request is too large; compact or split it", "context_length_exceeded")
    if gate.name == "light" and len(body) > MAX_LIGHT_BYTES:
        return openai_error(413, "light request exceeds 32768 bytes", "request_too_large")
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        return openai_error(400, "request must be a JSON object", "invalid_request_error")
    n = parsed.get("n", 1)
    if isinstance(n, bool) or n != 1:
        return openai_error(400, "this trial permits one completion per request (n=1)", "invalid_request_error")
    if gate.name == "light":
        if len(body) > MAX_LIGHT_BYTES:
            return openai_error(413, "light request exceeds 32768 bytes", "request_too_large")
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                return openai_error(400, "light request must be a JSON object", "invalid_request_error")
            requested = payload.get("max_tokens", payload.get("max_completion_tokens"))
            limits = [payload[k] for k in ("max_tokens", "max_completion_tokens") if k in payload]
            if not limits or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 512 for v in limits):
                return openai_error(400, "light requests require max_tokens or max_completion_tokens <= 512", "invalid_request_error")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return openai_error(400, "light request must be valid JSON", "invalid_request_error")
    cost = estimate_request_cost(body, parsed, output_cap=cfg.expected_output_reserve) if cfg.adaptive else 0
    budget = request.app.get("token_budget")
    if budget is not None and cost > cfg.request_token_limit:
        print(json.dumps({"event": "normal_rejected", "code": "estimated_context_exceeded", "actual_bytes": len(body), "estimated_cost": cost, "limit": cfg.request_token_limit}), flush=True)
        return openai_error(400, "request is too large; compact or split it", "context_length_exceeded")
    # Non-stream heavy responses commonly send headers only after full generation;
    # retain their latency for observability but exclude it from control policy.
    control_eligible = gate.name == "light" or parsed.get("stream") is True
    admission = await gate.acquire(cfg.queue_wait, request, cost=cost, control_eligible=control_eligible)
    ok = False
    prepared = False
    try:
        upstream_started = time.monotonic()
        async with request.app["session"].request("POST", cfg.upstream + upstream_path, data=body,
                headers=safe_headers(request.headers), timeout=ClientTimeout(total=cfg.upstream_timeout), allow_redirects=False) as upstream:
            gate.record_response_start(time.monotonic() - upstream_started, admission, control_eligible=control_eligible)
            response_headers = safe_headers(upstream.headers)
            stream = web.StreamResponse(status=upstream.status, headers=response_headers)
            await stream.prepare(request)
            prepared = True
            async for chunk in upstream.content.iter_chunked(16384):
                await stream.write(chunk)
            await stream.write_eof()
            ok = upstream.status < 500
            if upstream.status >= 500:
                gate.record_upstream_failure()
            return stream
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        gate.record_upstream_failure()
        print(json.dumps({"event": "upstream_error", "kind": type(exc).__name__}), flush=True)
        if prepared:
            if request.transport is not None:
                request.transport.close()
            return stream
        return web.Response(status=502, content_type="application/json", text=json.dumps({"error": {"message": "upstream unavailable", "type": "server_error", "code": "upstream_error"}}))
    finally:
        gate.release(ok, ticket=admission)


async def handler(request):
    path = request.path
    if path == "/healthz":
        return web.json_response({"ok": True})
    if path == "/status":
        result = {"heavy": request.app["heavy"].status(), "light": request.app["light"].status()}
        budget = request.app.get("token_budget")
        if budget is not None:
            result["estimated_token_reserved"] = budget.reserved
            result["estimated_token_total"] = budget.total
            result["estimated_token_utilization_pct"] = round(100 * budget.reserved / budget.total, 1) if budget.total else None
            result["waiting_head_cost"] = {"heavy": request.app["heavy"]._queue[0].cost if request.app["heavy"]._queue else 0,
                                           "light": request.app["light"]._queue[0].cost if request.app["light"]._queue else 0}
            result["waiting_head_cost_by_kind"] = {
                name: {
                    "normal": gate._queue[0].cost if gate._queue else 0,
                    "compression": gate._compression_queue[0].cost if gate._compression_queue else 0,
                }
                for name, gate in (("heavy", request.app["heavy"]), ("light", request.app["light"]))
            }
        if request.app.get("controller") is not None:
            result["controller"] = request.app["controller"].status()
        return web.json_response(result)
    if request.method == "GET" and path in {"/v1/models", "/get_model_info"}:
        cfg = request.app["config"]
        try:
            async with request.app["session"].get(cfg.upstream + path, headers=safe_headers(request.headers), timeout=ClientTimeout(total=cfg.upstream_timeout), allow_redirects=False) as r:
                return web.Response(status=r.status, body=await r.read(), headers=safe_headers(r.headers))
        except Exception:
            return web.Response(status=502)
    if request.method == "POST" and path in HEAVY_PATHS:
        return await proxy(request, request.app["heavy"], path)
    if request.method == "POST" and path == "/aux/compression/v1/chat/completions":
        return await compression_proxy(request)
    if request.method == "POST" and path == LIGHT_PATH:
        return await proxy(request, request.app["light"], "/v1/chat/completions")
    raise web.HTTPNotFound()


async def on_cleanup(app):
    task = app.get("controller_task")
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if app.get("session") is not None:
        await app["session"].close()


async def on_startup(app):
    app["session"] = ClientSession(auto_decompress=False)
    cfg = app["config"]
    if cfg.adaptive:
        controller = AdaptiveController(app["heavy"], app["light"], app["telemetry_provider"], cfg.adaptive_poll_seconds, cfg)
        app["controller"] = controller
        app["controller_task"] = asyncio.create_task(controller.run())


async def unavailable_telemetry():
    """Safe default until the deployment-specific all-node schema is wired in."""
    return None


def create_app(config=None):
    config = config or Config()
    positive_budgets = (config.token_budget_total, config.request_token_limit,
                        config.compression_request_token_limit, config.expected_output_reserve,
                        config.compression_output_reserve)
    if config.heavy_limit < 1 or config.light_limit < 1 or config.max_queue < 0 or config.queue_wait <= 0 or config.upstream_timeout <= 0 or config.adaptive_poll_seconds <= 0 or any(value <= 0 for value in positive_budgets):
        raise ValueError("limits and timeouts must be positive (max_queue may be zero)")
    app = web.Application(client_max_size=MAX_HEAVY_BYTES + 1, handler_args={"handler_cancellation": True})
    app["config"] = config
    app["heavy"] = Gate(config.heavy_limit, config.max_queue, "heavy")
    app["light"] = Gate(config.light_limit, config.max_queue, "light")
    if config.adaptive:
        budget = EstimatedTokenBudget(config.token_budget_total)
        app["token_budget"] = budget
        app["heavy"].budget = budget; app["light"].budget = budget
        budget.gates.extend((app["heavy"], app["light"]))
    app["telemetry_provider"] = CapacityFileProvider(config.capacity_file)
    app.router.add_route("*", "/{tail:.*}", handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", action="append", default=None,
                   help="bind address; repeat to bind fixed additional addresses (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=18888)
    p.add_argument("--upstream", default=Config.upstream); p.add_argument("--heavy-limit", type=int, default=2)
    p.add_argument("--light-limit", type=int, default=1); p.add_argument("--max-queue", type=int, default=64)
    p.add_argument("--queue-wait", type=float, default=1500); p.add_argument("--upstream-timeout", type=float, default=1800)
    p.add_argument("--adaptive", action="store_true"); p.add_argument("--adaptive-poll-seconds", type=float, default=5)
    p.add_argument("--capacity-file", default=Config.capacity_file)
    p.add_argument("--token-budget-total", type=int, default=TOKEN_BUDGET_TOTAL)
    p.add_argument("--request-token-limit", type=int, default=REQUEST_TOKEN_LIMIT)
    p.add_argument("--compression-request-token-limit", type=int, default=COMPRESSION_REQUEST_TOKEN_LIMIT)
    p.add_argument("--expected-output-reserve", type=int, default=EXPECTED_OUTPUT_RESERVE)
    p.add_argument("--compression-output-reserve", type=int, default=COMPRESSION_OUTPUT_RESERVE)
    a = p.parse_args(); cfg = Config(a.upstream.rstrip("/"), a.heavy_limit, a.light_limit, a.max_queue, a.queue_wait, a.upstream_timeout, a.adaptive, a.adaptive_poll_seconds, a.capacity_file, a.token_budget_total, a.request_token_limit, a.compression_request_token_limit, a.expected_output_reserve, a.compression_output_reserve)
    hosts = a.host or ["127.0.0.1"]
    print(json.dumps({"event": "started", "hosts": hosts, "port": a.port, "heavy_limit": a.heavy_limit, "light_limit": a.light_limit}), flush=True)
    web.run_app(create_app(cfg), host=hosts, port=a.port, print=None, access_log=None, shutdown_timeout=10)


if __name__ == "__main__": main()
