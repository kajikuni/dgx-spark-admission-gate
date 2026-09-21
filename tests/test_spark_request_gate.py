import asyncio
import json
import sys
from pathlib import Path
import pytest
import pytest_asyncio
from aiohttp import web, ClientSession

sys.path.insert(0, str(Path(__file__).parents[1]))
from spark_request_gate import AdaptiveController, CapacityFileProvider, Config, EstimatedTokenBudget, Gate, create_app, estimate_request_cost
import spark_request_gate as gate_module


@pytest.mark.asyncio
async def test_compression_is_at_most_one_and_normal_can_use_second_slot():
    gate = Gate(2, 64, "heavy", EstimatedTokenBudget(100))
    first = await gate.acquire(1, cost=10, kind="compression")
    second = asyncio.create_task(gate.acquire(1, cost=10, kind="compression"))
    normal = await gate.acquire(1, cost=10, kind="normal")
    assert gate.active == 2 and gate.status()["kinds"]["compression"]["active"] == 1
    gate.release(ticket=first)
    second_admission = await second
    gate.release(ticket=normal); gate.release(ticket=second_admission)
    assert gate.active == 0 and gate.budget.reserved == 0


@pytest.mark.asyncio
async def test_cross_kind_first_contended_grant_prefers_compression_then_alternates():
    gate = Gate(1, 64, "heavy", EstimatedTokenBudget(100))
    held = await gate.acquire(1, cost=1)
    normal = asyncio.create_task(gate.acquire(1, cost=1, kind="normal"))
    compression = asyncio.create_task(gate.acquire(1, cost=1, kind="compression"))
    await asyncio.sleep(.01); gate.release(ticket=held)
    comp_ticket = await compression
    assert not normal.done()
    gate.release(ticket=comp_ticket)
    normal_ticket = await normal; gate.release(ticket=normal_ticket)


@pytest_asyncio.fixture
async def compression_servers(servers):
    _, proxy, starts, release = servers
    # Keep controller out of this focused HTTP fixture while exercising the
    # exact shared budget path used by adaptive deployment.
    app = proxy.app; app["config"].adaptive = True
    # The focused fixture keeps a deliberately tiny shared budget so the
    # second request exercises the queued path with the production estimator.
    budget = EstimatedTokenBudget(15000); app["token_budget"] = budget
    for gate in (app["heavy"], app["light"]):
        gate.budget = budget; budget.gates.append(gate)
    return proxy, starts, release


def phase_events(raw):
    events = []
    for block in raw.decode().split("\n\n"):
        if block.startswith("event: gate.phase\n"):
            events.append(json.loads(block.split("data: ", 1)[1]))
    return events


@pytest.mark.asyncio
async def test_compression_http_phase_order_and_upstream_error(compression_servers):
    proxy, _, _ = compression_servers
    async with ClientSession() as s:
        response = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), headers={"X-Mode": "500"}, json={"stream": True, "max_tokens": 1})
        assert response.status == 200
        phases = phase_events(await response.read())
        assert [item["phase"] for item in phases] == ["queued", "admitted", "failed"]
        assert phases[-1]["code"] == "upstream_error" and phases[-1]["error_status"] == 500
        status = await (await s.get(proxy.make_url("/status"))).json()
        assert status["heavy"]["active"] == 0 and status["estimated_token_reserved"] == 0


@pytest.mark.asyncio
async def test_compression_split_done_is_success_and_3xx_is_failed(compression_servers):
    proxy, _, _ = compression_servers
    async with ClientSession() as s:
        split = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), headers={"X-Mode": "sse-split"}, json={"stream": True, "max_tokens": 1})
        raw = await split.read(); assert b"data: [DONE]" in raw
        assert [p["phase"] for p in phase_events(raw)] == ["queued", "admitted"]
        redirect = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), headers={"X-Mode": "302"}, json={"stream": True, "max_tokens": 1})
        phases = phase_events(await redirect.read())
        assert phases[-1]["phase"] == "failed" and phases[-1]["error_status"] == 302


@pytest.mark.asyncio
async def test_admitted_write_failure_and_stream_disconnect_release(compression_servers, monkeypatch):
    proxy, _, release = compression_servers
    original_phase = gate_module.phase
    async def fail_once(stream, name, request_id, **extra):
        if name == "admitted": raise ConnectionResetError("test")
        return await original_phase(stream, name, request_id, **extra)
    monkeypatch.setattr(gate_module, "phase", fail_once)
    async with ClientSession() as s:
        response = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), json={"stream": True, "max_tokens": 1})
        await response.read()
        for _ in range(30):
            status = await (await s.get(proxy.make_url("/status"))).json()
            if status["heavy"]["active"] == 0: break
            await asyncio.sleep(.01)
        assert status["heavy"]["active"] == 0 and status["estimated_token_reserved"] == 0
    monkeypatch.setattr(gate_module, "phase", original_phase)
    async with ClientSession() as s:
        response = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), headers={"X-Mode": "sse"}, json={"stream": True, "max_tokens": 1})
        await response.content.readany(); await response.content.readany(); response.close()
        for _ in range(50):
            status = await (await s.get(proxy.make_url("/status"))).json()
            if status["heavy"]["active"] == 0: break
            await asyncio.sleep(.01)
        assert status["heavy"]["active"] == 0 and status["estimated_token_reserved"] == 0
        release.set()


@pytest.mark.asyncio
async def test_compression_queued_disconnect_and_shared_budget_http(compression_servers):
    proxy, starts, release = compression_servers
    async with ClientSession() as s:
        # A normal request reserves budget first; compression waits even though
        # another active slot is free, then admits after normal EOF releases it.
        normal = asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"), data=b"{}"))
        while not starts: await asyncio.sleep(.01)
        queued = await s.post(
            proxy.make_url("/aux/compression/v1/chat/completions"),
            headers={"X-Mode": "sse"},
            json={"stream": True, "max_tokens": 140000,
                  "messages": [{"role": "user", "content": "a" * 20000}]},
        )
        assert (await queued.content.readany()).startswith(b"event: gate.phase")
        status = await (await s.get(proxy.make_url("/status"))).json()
        assert status["heavy"]["kinds"]["compression"]["waiting"] == 1
        release.set(); await (await normal).read()
        phases = phase_events(await queued.read())
        assert [p["phase"] for p in phases] == ["admitted"]
        assert (await (await s.get(proxy.make_url("/status"))).json())["estimated_token_reserved"] == 0


@pytest.mark.asyncio
async def test_compression_disconnect_removes_queued_ticket(compression_servers):
    proxy, starts, release = compression_servers
    async with ClientSession() as s:
        blockers = [asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"), data=b"{}")) for _ in range(2)]
        while len(starts) < 2: await asyncio.sleep(.01)
        response = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), json={"stream": True, "max_tokens": 1})
        await response.content.readany(); response.close()
        for _ in range(50):
            status = await (await s.get(proxy.make_url("/status"))).json()
            if status["heavy"]["kinds"]["compression"]["waiting"] == 0: break
            await asyncio.sleep(.01)
        assert status["heavy"]["kinds"]["compression"]["waiting"] == 0
        release.set(); await asyncio.gather(*blockers)


@pytest.mark.asyncio
async def test_compression_http_fairness_prefers_first_contended_compression(compression_servers):
    proxy, starts, release = compression_servers
    proxy.app["heavy"].set_limit(1)
    async with ClientSession() as s:
        blocker = asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"), data=b"{}"))
        while len(starts) < 1: await asyncio.sleep(.01)
        normal = asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"), data=b"{}"))
        await asyncio.sleep(.01)
        compressed = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), headers={"X-Mode": "sse"}, json={"stream": True, "max_tokens": 1})
        await compressed.content.readany(); release.set()
        await (await blocker).read(); await (await normal).read(); await compressed.read()
        assert starts[:3] == ["hold", "sse", "hold"]


@pytest.mark.asyncio
async def test_compression_queued_timeout_emits_failed_phase(compression_servers, monkeypatch):
    proxy, starts, release = compression_servers
    gate = proxy.app["heavy"]; original = gate.acquire
    async def short_wait(wait_seconds, *args, **kwargs):
        return await original(.02, *args, **kwargs)
    monkeypatch.setattr(gate, "acquire", short_wait)
    async with ClientSession() as s:
        blockers = [asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"), data=b"{}")) for _ in range(2)]
        while len(starts) < 2: await asyncio.sleep(.01)
        response = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"), json={"stream": True, "max_tokens": 1})
        phases = phase_events(await response.read())
        assert [p["phase"] for p in phases] == ["queued", "failed"]
        assert phases[-1]["code"] == "queue_timeout"
        release.set(); await asyncio.gather(*blockers)


def test_adaptive_controller_fails_closed_and_drains_active_work():
    heavy, light = Gate(4, 4, "heavy"), Gate(1, 4, "light")
    heavy.active = 2  # Existing work is never cancelled when limit falls to zero.
    controller = AdaptiveController(heavy, light, None)
    controller.evaluate(None, now=10)
    assert heavy.limit == 0 and light.limit == 0 and heavy.active == 2
    assert controller.status()["reason"] == "telemetry_unhealthy"


def test_adaptive_controller_exposes_all_four_slots_immediately_when_healthy():
    heavy, light = Gate(1, 4, "heavy"), Gate(1, 4, "light")
    controller = AdaptiveController(heavy, light, None)
    sample = {"fresh": True, "engines_ok": True, "age": 1, "min_headroom": 13, "psi": 0, "psi_full": 0, "kv_usage": .1, "running": 0, "queued": 0}
    controller.evaluate(sample, now=10)
    assert heavy.limit == 4 and light.limit == 1
    assert controller.reason == "healthy"


def test_adaptive_controller_timeout_reduces_and_cools_down():
    heavy, light = Gate(2, 4, "heavy"), Gate(1, 4, "light")
    controller = AdaptiveController(heavy, light, None); controller.target = 2
    heavy.record_upstream_failure(20)
    sample = {"fresh": True, "engines_ok": True, "min_headroom": 13, "psi": 0, "psi_full": 0, "kv_usage": 0, "running": 0, "queued": 0}
    controller.evaluate(sample, now=20)
    assert heavy.limit == 1 and controller.cooldown_until == 80


def test_nonstream_heavy_latency_is_observed_but_not_controlled():
    heavy, light = Gate(1, 2, "heavy"), Gate(1, 2, "light")
    controller = AdaptiveController(heavy, light, None)
    heavy.record_response_start(300, now=10, control_eligible=False)
    sample = {"fresh": True, "engines_ok": True, "min_headroom": 13, "psi": 0, "psi_full": 0, "kv_usage": 0, "running": 0, "queued": 0}
    controller.evaluate(sample, now=10)
    assert heavy.limit == 4 and heavy.latency_status(now=10)["max"] == 300
    assert heavy.status()["response_start_latency"]["control_max"] == 0


def test_stream_slow_response_reduces_capacity():
    heavy, light = Gate(2, 2, "heavy"), Gate(1, 2, "light")
    controller = AdaptiveController(heavy, light, None); controller.target = 2; heavy.set_limit(2)
    heavy.record_response_start(21, now=10, control_eligible=True)
    sample = {"fresh": True, "engines_ok": True, "min_headroom": 13, "psi": 0, "psi_full": 0, "kv_usage": 0, "running": 0, "queued": 0}
    controller.evaluate(sample, now=10)
    assert heavy.limit == 1 and controller.reason == "slow_response"


@pytest.mark.asyncio
async def test_capacity_provider_accepts_collector_node_dict(tmp_path):
    now = __import__('time').time()
    node = {"ts_unix": now, "boot_id": "b", "mem_available_gib": 13, "mem_total_gib": 16, "psi_some_avg10": 0, "psi_full_avg10": 0, "oom_kill_total": 0}
    data = {"ts_unix": now, "complete": True, "nodes": {name: node for name in ("spark-a", "spark-b", "spark-c", "spark-d")}, "engine": {"ok": True, "running": 0, "queued": 0, "cached_tokens": 1, "pending_tokens": 0, "ts_unix": now, "ts_tic": now, "source_age_s": 0}}
    path = tmp_path / "capacity.json"; path.write_text(json.dumps(data))
    result = await CapacityFileProvider(path)()
    assert result["fresh"] and result["min_headroom"] == 13


@pytest.mark.asyncio
async def test_shared_budget_reserves_on_grant_and_releases():
    budget = EstimatedTokenBudget(10); heavy = Gate(1, 2, "heavy", budget); light = Gate(1, 2, "light", budget)
    admission = await heavy.acquire(1, cost=7)
    assert budget.reserved == 7
    waiter = asyncio.create_task(light.acquire(1, cost=7)); await asyncio.sleep(.01)
    assert light.waiting == 1
    heavy.release(ticket=admission); light_admission = await waiter; assert light_admission.cost == 7 and budget.reserved == 7
    light.release(ticket=light_admission); assert budget.reserved == 0


def test_estimated_cost_uses_unicode_and_caps_declared_output_reservation():
    small = {"messages": [{"role": "user", "content": "a" * 4000 + "日" * 1000}], "max_tokens": 4096}
    huge_declared = {**small, "max_tokens": 32768}
    encoded = json.dumps(small, ensure_ascii=False).encode()
    assert estimate_request_cost(encoded, huge_declared) == estimate_request_cost(encoded, small)
    # ASCII is weighted at roughly 4 chars/token; non-ASCII at one/token.
    assert 7000 < estimate_request_cost(encoded, small) < 7500


def test_four_ordinary_requests_fit_shared_four_context_budget():
    payload = {"messages": [{"role": "user", "content": "a" * 120000}], "max_tokens": 32768}
    cost = estimate_request_cost(json.dumps(payload).encode(), payload)
    budget = EstimatedTokenBudget()
    assert cost < 65536
    assert all(budget.reserve(cost) for _ in range(4))


@pytest_asyncio.fixture
async def servers(aiohttp_server):
    starts, release = [], asyncio.Event()
    async def upstream(request):
        payload = await request.read(); mode = request.headers.get("X-Mode", "hold")
        starts.append(mode)
        if mode == "500": return web.Response(status=500, text="bad")
        if mode == "302": return web.Response(status=302, text="redirect")
        if mode == "sse":
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await r.prepare(request)
            await r.write(b"data: one\n\n"); await release.wait(); await r.write(b"data: two\n\n"); await r.write(b"data: [DONE]\n\n"); return r
        if mode == "sse-split":
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await r.prepare(request)
            await r.write(b"data: [DO"); await r.write(b"NE]\n\n"); return r
        if mode == "light": return web.Response(text="light")
        await release.wait(); return web.Response(text=mode)
    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream)
    upstream_app.router.add_post("/v1/completions", upstream)
    up = await aiohttp_server(upstream_app)
    proxy = await aiohttp_server(create_app(Config(str(up.make_url("/")).rstrip("/"), queue_wait=5, upstream_timeout=5)))
    return up, proxy, starts, release

@pytest.mark.asyncio
async def test_fifo_and_independent_light(servers):
    _, proxy, starts, release = servers
    async with ClientSession() as s:
        url = proxy.make_url("/v1/chat/completions")
        tasks = [asyncio.create_task(s.post(url, headers={"X-Mode": str(i)}, data=b"{}")) for i in range(3)]
        for _ in range(50):
            if len(starts) == 2: break
            await asyncio.sleep(.01)
        assert starts == ["0", "1"]
        light = await s.post(proxy.make_url("/light/v1/chat/completions"), headers={"X-Mode": "light"}, data=b'{"max_tokens": 1}')
        assert await light.text() == "light"
        release.set()
        assert [await (await t).text() for t in tasks] == ["0", "1", "2"]
        assert starts == ["0", "1", "light", "2"]


@pytest.mark.asyncio
async def test_large_normal_request_admission_preserves_shared_budget(servers):
    _, proxy, starts, release = servers
    app = proxy.app
    app["config"].adaptive = True
    budget = EstimatedTokenBudget()
    app["token_budget"] = budget
    for gate in (app["heavy"], app["light"]):
        gate.budget = budget
        budget.gates.append(gate)
    app["heavy"].set_limit(4)
    # Three larger requests fit; a fourth must wait despite four free slots.
    payload = {"messages": [{"role": "user", "content": "日" * 64000}]}
    cost = estimate_request_cost(b"", payload)
    assert 65536 < cost < gate_module.REQUEST_TOKEN_LIMIT
    assert cost * 3 <= budget.total < cost * 4
    async with ClientSession() as s:
        tasks = [asyncio.create_task(s.post(proxy.make_url("/v1/chat/completions"),
                 json=payload, headers={"X-Mode": str(i)})) for i in range(4)]
        try:
            for _ in range(100):
                if len(starts) == 3 and app["heavy"].waiting == 1:
                    break
                await asyncio.sleep(.01)
            assert len(starts) == 3
            assert app["heavy"].waiting == 1
            assert budget.reserved == cost * 3
        finally:
            release.set()
            responses = await asyncio.gather(*tasks)
        assert [r.status for r in responses] == [200] * 4
        assert budget.reserved == 0
        compression = await s.post(proxy.make_url("/aux/compression/v1/chat/completions"),
                                   json={**payload, "stream": True})
        assert compression.status == 400
        assert (await compression.json())["error"]["code"] == "estimated_context_exceeded"
        assert len(starts) == 4


@pytest.mark.asyncio
async def test_normal_request_above_admission_limit_still_rejected(servers):
    _, proxy, starts, _ = servers
    proxy.app["config"].adaptive = True
    proxy.app["token_budget"] = EstimatedTokenBudget()
    payload = {"messages": [{"role": "user", "content": "日" * 130000}]}
    assert estimate_request_cost(b"", payload) > gate_module.REQUEST_TOKEN_LIMIT
    assert gate_module.COMPRESSION_REQUEST_TOKEN_LIMIT == 65536
    async with ClientSession() as s:
        # UTF-8 stays under the unchanged one-MiB body cap.
        response = await s.post(proxy.make_url("/v1/chat/completions"),
                                data=json.dumps(payload, ensure_ascii=False).encode())
        assert response.status == 400
        assert (await response.json())["error"]["code"] == "context_length_exceeded"
    assert starts == []

@pytest.mark.asyncio
async def test_cancel_queue_and_error_and_size(servers):
    _, proxy, starts, release = servers
    async with ClientSession() as s:
        url = proxy.make_url("/v1/chat/completions")
        a = [asyncio.create_task(s.post(url, data=b"{}")) for _ in range(2)]
        while len(starts) < 2: await asyncio.sleep(.01)
        queued = asyncio.create_task(s.post(url, data=b"{}")); await asyncio.sleep(.03); queued.cancel()
        with pytest.raises(asyncio.CancelledError): await queued
        for _ in range(30):
            status = await (await s.get(proxy.make_url('/status'))).json()
            if status['heavy']['waiting'] == 0: break
            await asyncio.sleep(.02)
        assert status['heavy']['waiting'] == 0
        release.set(); await asyncio.gather(*a)
        too_big = await s.post(url, data=b'x' * 1048577); assert too_big.status == 400; assert (await too_big.json())['error']['code'] == 'context_length_exceeded'
        bad = await s.post(url, headers={'X-Mode': '500'}, data=b'{}'); assert bad.status == 500; assert await bad.text() == 'bad'
        status = await (await s.get(proxy.make_url('/status'))).json(); assert status['heavy']['active'] == 0

@pytest.mark.asyncio
async def test_sse_holds_slot_until_eof(servers):
    _, proxy, starts, release = servers
    async with ClientSession() as s:
        response = await s.post(proxy.make_url('/v1/chat/completions'), headers={'X-Mode': 'sse'}, data=b'{}')
        assert b'one' in await response.content.readany()
        status = await (await s.get(proxy.make_url('/status'))).json(); assert status['heavy']['active'] == 1
        release.set(); assert b'two' in await response.read()
        for _ in range(20):
            status = await (await s.get(proxy.make_url('/status'))).json()
            if status['heavy']['active'] == 0: break
            await asyncio.sleep(.01)
        assert status['heavy']['active'] == 0

@pytest.mark.asyncio
async def test_gate_cancellation_and_timeout_do_not_leak():
    gate = Gate(1, 3, 'heavy')
    await gate.acquire(1)
    waiter = asyncio.create_task(gate.acquire(1)); await asyncio.sleep(.01); waiter.cancel()
    with pytest.raises(asyncio.CancelledError): await waiter
    assert gate.active == 1 and gate.waiting == 0
    timed = asyncio.create_task(gate.acquire(.02))
    with pytest.raises(web.HTTPGatewayTimeout): await timed
    assert gate.active == 1 and gate.waiting == 0
    gate.release()

@pytest.mark.asyncio
async def test_light_validation_and_active_disconnect(servers):
    _, proxy, starts, release = servers
    async with ClientSession() as s:
        url=proxy.make_url('/light/v1/chat/completions')
        for payload in [[], 1, None, {"max_tokens": True}, {"max_tokens": -1}, {"max_tokens": 513}, {"max_tokens": 1, "max_completion_tokens": 8192}, {"max_tokens": 1, "n": 2}]:
            r=await s.post(url,json=payload)
            assert r.status==400
        r=await s.post(url,data=b'x'*32769); assert r.status==413
        assert not starts
        task=asyncio.create_task(s.post(proxy.make_url('/v1/chat/completions'),data=b'{}'))
        for _ in range(100):
            if starts:break
            await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        for _ in range(100):
            status=await (await s.get(proxy.make_url('/status'))).json()
            if status['heavy']['active']==0:break
            await asyncio.sleep(.01)
        assert status['heavy']['active']==0
        assert status['heavy']['waiting']==0
        release.set()
