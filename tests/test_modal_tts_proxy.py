"""Real loopback Socket.IO bridge, with only the private GPU RPC replaced."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest
import socketio


SPEC = importlib.util.spec_from_file_location(
    "modal_tts_proxy_under_test", Path(__file__).resolve().parents[1] / "servers/modal-tts/proxy.py")
proxy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)


class Worker:
    """Model the primitive render boundary, rejecting incorrect pieces/inputs."""
    def __init__(self, *, ready_blocked=False):
        self.ready_calls = 0
        self.calls = []
        self.ready_gate = asyncio.Event()
        if not ready_blocked:
            self.ready_gate.set()
        self.render_gate = asyncio.Event()
        self.render_gate.set()
        self.started = asyncio.Queue()
        self.fail = False
        self.audio = b"RIFF-test-wave"
        self.ready = SimpleNamespace(remote=SimpleNamespace(aio=self._ready))
        self.render_piece = SimpleNamespace(remote=SimpleNamespace(aio=self._render))

    async def _ready(self):
        self.ready_calls += 1
        await self.ready_gate.wait()
        return {"protocol_version": 1,
                "health": {"ok": True, "device": "cuda", "error": "private-credential",
                           "ffmpeg": "/private/workspace/ffmpeg", "token": "private-credential"},
                "characters": {"characters": {"amika": {"name": "Amika", "language": "Japanese",
                    "emotions": ["happy2", "neutral1"], "default_emotion": "neutral1", "loaded": True,
                    "private_path": "/private/voice/assets"}}, "current": "amika"}}

    async def _render(self, payload, index):
        assert set(payload) <= {"text", "character_id", "language", "audio_format", "turn_id", *proxy.voice.OPTIONS}
        assert "task_id" not in payload
        assert payload.get("turn_id") in (None, "turn")
        job = proxy.voice.parse_job(payload)
        pieces = job.pieces()
        assert 0 <= index < len(pieces)
        self.calls.append((payload.copy(), index))
        await self.started.put((payload["text"], index))
        await self.render_gate.wait()
        if self.fail:
            raise RuntimeError("private-credential workspace=/private/cloud/context")
        fmt = job.audio_format
        return {"audio": self.audio, "audio_format": fmt, "sr": 32000, "duration": 1.234,
                "stages": [], "text": " ".join(s.text for s in pieces[index]),
                "emotions": [s.tag or "neutral1" for s in pieces[index]], "fallback": [],
                "lang": job.lang, "retried": 0, "character_id": job.character_id or "amika",
                "synth_s": 0.1, "encode_s": 0.001, "warning": ""}


@asynccontextmanager
async def running(worker, **options):
    bridge = proxy.Proxy(worker, **options)
    runner = web.AppRunner(bridge.create_app(), access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    clients = []

    async def client():
        sio = socketio.AsyncClient(reconnection=False)
        events = asyncio.Queue()
        for event in ("task_queued", "audio_ready", "error"):
            async def receive(data, name=event):
                await events.put((name, data))
            sio.on(event, receive)
        await sio.connect(f"http://127.0.0.1:{port}", transports=["websocket"])
        clients.append(sio)
        return sio, events

    try:
        yield bridge, f"http://127.0.0.1:{port}", client
    finally:
        for sio in clients:
            if sio.connected:
                await sio.disconnect()
        await runner.cleanup()


async def next_event(events, expected):
    event, data = await asyncio.wait_for(events.get(), 2)
    assert event == expected
    return data


def test_cached_readiness_binds_while_warming_and_never_polls_gpu():
    async def scenario():
        worker = Worker(ready_blocked=True)
        # Initial model readiness has its own budget, not the much shorter render budget.
        async with running(worker, rpc_timeout_s=0.0001, ready_timeout_s=2) as (bridge, url, client):
            async with aiohttp.ClientSession() as http:
                for _ in range(3):
                    async with http.get(url + "/ping") as response:
                        assert response.status == 503
                        assert await response.text() == "warming"
                    async with http.get(url + "/api/health") as response:
                        assert (await response.json())["ok"] is False
                assert worker.ready_calls == 1
                worker.ready_gate.set()
                await bridge._ready_task
                for _ in range(3):
                    async with http.get(url + "/ping") as response:
                        assert response.status == 200
                        assert await response.text() == "ok"
                    async with http.get(url + "/api/health") as response:
                        health = await response.json()
                        assert health["ok"] is True
                        assert health["transport"] == "modal"
                        assert health["device"] == "cuda"
                        assert "private" not in str(health)
                    async with http.get(url + "/api/characters") as response:
                        characters = await response.json()
                        assert characters["current"] == "amika"
                        assert characters["characters"]["amika"]["emotions"] == ["happy2", "neutral1"]
                        assert "private" not in str(characters)
                assert worker.ready_calls == 1
                assert worker.calls == []
                async with http.get(url + "/ping", headers={"Origin": url}) as response:
                    assert response.status == 403
    asyncio.run(scenario())


def test_fifo_requests_preserve_sentences_turn_mode_binary_chunks_and_target_sid():
    async def scenario():
        worker = Worker()
        worker.render_gate.clear()
        worker.audio = b"W" * (proxy.voice.CHUNK_BYTES + 7)
        async with running(worker) as (bridge, url, client):
            first, first_events = await client()
            second, second_events = await client()
            await first.emit("tts_request_gateway", {"task_id": "sentence", "text": "【happy2】Hello! Again?",
                              "character_id": "amika", "language": "English", "speed": 1.2, "seed": 3})
            queued = await next_event(first_events, "task_queued")
            assert queued["sentence_count"] == 2
            assert queued["queue_position"] == 0
            assert await asyncio.wait_for(worker.started.get(), 2) == ("【happy2】Hello! Again?", 0)
            await second.emit("tts_request_gateway", {"task_id": "whole", "turn_id": "private-turn-id",
                               "text": "【neutral1】こんにちは。元気？", "language": "Japanese", "audio_format": "opus"})
            queued = await next_event(second_events, "task_queued")
            assert queued["sentence_count"] == 1
            assert queued["queue_position"] == 1
            worker.render_gate.set()
            messages = [await next_event(first_events, "audio_ready") for _ in range(4)]
            assert [m["text"] for m in messages] == ["Hello!", "Hello!", "Again?", "Again?"]
            assert [m["is_final"] for m in messages] == [False, False, False, True]
            assert [m["chunk_index"] for m in messages] == [0, 1, 0, 1]
            assert [m["chunk_count"] for m in messages] == [2, 2, 2, 2]
            assert b"".join(m["audio"] for m in messages[:2]) == worker.audio
            assert {m["task_id"] for m in messages} == {"sentence"}
            assert messages[0]["sample_rate"] == 32000
            assert messages[0]["emotions"] == ["happy2"]
            turn = [await next_event(second_events, "audio_ready") for _ in range(2)]
            assert [m["is_final"] for m in turn] == [False, True]
            assert turn[0]["turn_id"] == "private-turn-id"
            assert turn[0]["sample_rate"] == 48000
            assert turn[0]["text"] == "こんにちは。元気？"
            assert [index for _, index in worker.calls] == [0, 1, 0]
            assert worker.calls[0][0]["speed"] == 1.2
            assert worker.calls[0][0]["seed"] == 3
            assert first_events.empty() and second_events.empty()
    asyncio.run(scenario())


def test_disconnected_queued_work_and_active_remaining_sentences_are_discarded():
    async def scenario():
        worker = Worker()
        worker.render_gate.clear()
        async with running(worker) as (bridge, url, client):
            first, events = await client()
            queued, queued_events = await client()
            survivor, survivor_events = await client()
            await first.emit("tts_request_gateway", {"task_id": "active", "text": "First. Never synthesize."})
            await next_event(events, "task_queued")
            await asyncio.wait_for(worker.started.get(), 2)
            await queued.emit("tts_request_gateway", {"task_id": "queued", "text": "Discard queued."})
            await next_event(queued_events, "task_queued")
            await queued.disconnect()
            await first.disconnect()
            await survivor.emit("tts_request_gateway", {"task_id": "survivor", "text": "Still here."})
            await next_event(survivor_events, "task_queued")
            worker.render_gate.set()
            message = await next_event(survivor_events, "audio_ready")
            assert message["task_id"] == "survivor" and message["is_final"] is True
            assert [(p["text"], i) for p, i in worker.calls] == [
                ("First. Never synthesize.", 0), ("Still here.", 0)]
            assert events.empty() and queued_events.empty()
    asyncio.run(scenario())


def test_remote_failure_is_sanitized_and_queue_recovers():
    async def scenario():
        worker = Worker()
        worker.fail = True
        async with running(worker) as (bridge, url, client):
            sio, events = await client()
            await sio.emit("tts_request_gateway", {"task_id": "bad", "turn_id": "turn-id", "text": "Fail."})
            await next_event(events, "task_queued")
            error = await next_event(events, "error")
            assert error == {"task_id": "bad", "turn_id": "turn-id", "error": "Voice generation failed; try again later."}
            worker.fail = False
            await sio.emit("tts_request_gateway", {"task_id": "good", "text": "Recover."})
            await next_event(events, "task_queued")
            assert (await next_event(events, "audio_ready"))["task_id"] == "good"
            assert worker.ready_calls == 1
    asyncio.run(scenario())


def test_browser_origin_is_rejected_for_socketio_without_gpu_call():
    async def scenario():
        worker = Worker()
        async with running(worker) as (bridge, url, client):
            async with aiohttp.ClientSession() as http:
                browser = socketio.AsyncClient(http_session=http, reconnection=False)
                with pytest.raises(socketio.exceptions.ConnectionError):
                    await browser.connect(url, headers={"Origin": "https://foreign.invalid"},
                                          transports=["websocket"])
                assert bridge.clients == set()
                assert worker.calls == []
    asyncio.run(scenario())


def test_startup_failure_is_cached_and_never_retried_by_health():
    async def scenario():
        worker = Worker()

        async def failed_ready():
            worker.ready_calls += 1
            raise RuntimeError("private-credential workspace=/private/cloud/context")

        worker.ready.remote.aio = failed_ready
        async with running(worker) as (bridge, url, client):
            await bridge._ready_task
            async with aiohttp.ClientSession() as http:
                for _ in range(3):
                    async with http.get(url + "/ping") as response:
                        assert response.status == 503
                        assert await response.text() == "error"
                    async with http.get(url + "/api/health") as response:
                        assert (await response.json())["error"] == "Voice service unavailable."
            assert worker.ready_calls == 1
            sio, events = await client()
            await sio.emit("tts_request_gateway", {"task_id": "unavailable", "text": "Do not render."})
            assert (await next_event(events, "error"))["task_id"] == "unavailable"
            assert worker.calls == []
    asyncio.run(scenario())


def test_queue_and_rpc_deadlines_are_bounded_and_timeout_fails_closed():
    async def scenario():
        worker = Worker()
        worker.render_gate.clear()
        async with running(worker, max_pending=1, rpc_timeout_s=0.15) as (bridge, url, client):
            sio, events = await client()
            await sio.emit("tts_request_gateway", {"task_id": "slow", "text": "Slow."})
            await next_event(events, "task_queued")
            await asyncio.wait_for(worker.started.get(), 2)
            await sio.emit("tts_request_gateway", {"task_id": "overflow", "text": "Full."})
            error = await next_event(events, "error")
            assert error["task_id"] == "overflow" and "full" in error["error"]
            error = await next_event(events, "error")
            assert error["task_id"] == "slow"
            await sio.emit("tts_request_gateway", {"task_id": "late", "text": "Do not overtake."})
            error = await next_event(events, "error")
            assert error["task_id"] == "late" and "unavailable" in error["error"]
            assert len(worker.calls) == 1
            assert bridge.health()["ok"] is False
    asyncio.run(scenario())


def test_slow_rpc_cancellation_cannot_hold_request_or_publish_late_audio():
    async def scenario():
        worker = Worker()
        worker.render_gate.clear()
        cancellation = asyncio.Event()
        release = asyncio.Event()

        async def slow_cancel(payload, index):
            try:
                return await worker._render(payload, index)
            except asyncio.CancelledError:
                cancellation.set()
                await release.wait()
                return {}  # late remote result must never reach Socket.IO

        worker.render_piece.remote.aio = slow_cancel
        async with running(worker, rpc_timeout_s=0.02, cancel_timeout_s=0.01) as (bridge, url, client):
            sio, events = await client()
            await sio.emit("tts_request_gateway", {"task_id": "timeout", "text": "Late."})
            await next_event(events, "task_queued")
            error = await next_event(events, "error")
            assert error["task_id"] == "timeout"
            assert cancellation.is_set()
            assert len(bridge._rpc_tasks) == 1
            await sio.emit("tts_request_gateway", {"task_id": "next", "text": "No overlap."})
            assert (await next_event(events, "error"))["task_id"] == "next"
            pending = tuple(bridge._rpc_tasks)
            release.set()
            await asyncio.gather(*pending)
            assert events.empty()
            assert len(worker.calls) == 1
    asyncio.run(scenario())


def test_cli_cannot_bind_non_loopback():
    assert proxy.parse_args([]).host == "127.0.0.1"
    with pytest.raises(SystemExit):
        proxy.parse_args(["--host", "0.0.0.0"])
