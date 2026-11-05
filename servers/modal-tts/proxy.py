#!/usr/bin/env python3
"""Loopback-only Socket.IO adapter for the private Modal Voice RPC.

No model, voice assets, dotenv files, or Modal credentials are loaded here. The
Modal SDK uses its official profile only when the startup readiness task runs.
HTTP probes read a cache and never keep the remote GPU awake.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import importlib.util
import logging
from pathlib import Path
import sys
import time
from typing import Any

from aiohttp import web
import socketio


LOG = logging.getLogger("modal-tts-proxy")
_SPEC = importlib.util.spec_from_file_location(
    "_openamika_modal_wire", Path(__file__).resolve().parents[2] / "vendor/tts-engine/main.py")
voice = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = voice  # dataclasses resolves annotations through sys.modules
_SPEC.loader.exec_module(voice)
# The reused sender logs task identifiers and remote warnings. Keep those out of
# bridge logs; this independent module instance is used only for wire helpers.
voice.LOG = logging.getLogger("modal-tts-proxy.wire")
voice.LOG.disabled = True

HEALTH_FIELDS = {
    "device", "gpu", "half", "model_version", "character", "load_s", "warmup_s",
    "requests", "opus", "speed_needs_ffmpeg", "text_split", "batch_size",
    "trim_silence", "ref_max_s", "dropout_guard", "dropout_retries", "retries",
    "vram_allocated_mb", "vram_reserved_mb", "vram_peak_mb", "gpt_sovits_commit",
}


class Proxy:
    def __init__(self, worker: Any = None, *, app_name: str = "openamika-tts",
                 class_name: str = "Voice", environment: str | None = None,
                 max_pending: int = 32, rpc_timeout_s: float = 180.0,
                 ready_timeout_s: float = 900.0,
                 queue_timeout_s: float = 300.0, cancel_timeout_s: float = 0.5) -> None:
        if max_pending < 1 or min(rpc_timeout_s, ready_timeout_s, queue_timeout_s, cancel_timeout_s) <= 0:
            raise ValueError("queue and timeout limits must be positive")
        self.worker = worker
        self.app_name, self.class_name, self.environment = app_name, class_name, environment
        self.max_pending, self.rpc_timeout_s = max_pending, rpc_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.queue_timeout_s, self.cancel_timeout_s = queue_timeout_s, cancel_timeout_s
        self.state = "warming"
        self.metadata: dict = {}
        self.character_metadata: dict = {"characters": {}, "current": None}
        self.clients: set[str] = set()
        self.queue: deque = deque()
        self.active = False
        self._wake = asyncio.Event()
        self._accept_lock = asyncio.Lock()
        self._ready_task: asyncio.Task | None = None
        self._consumer: asyncio.Task | None = None
        self._rpc_tasks: set[asyncio.Task] = set()
        self.sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins=None,
                                        max_http_buffer_size=1024 * 1024,
                                        logger=False, engineio_logger=False)
        self.sio.on("connect", self.connect)
        self.sio.on("disconnect", self.disconnect)
        self.sio.on("tts_request_gateway", self.handle)

    async def connect(self, sid: str, environ: dict, auth: Any = None) -> bool:
        # Only the internal Python gateway is a client, never a browser page.
        if environ.get("HTTP_ORIGIN"):
            return False
        self.clients.add(sid)
        return True

    async def disconnect(self, sid: str, *reason: Any) -> None:
        self.clients.discard(sid)
        self.queue = deque(entry for entry in self.queue if entry[0] != sid)
        # An active call drains to its bounded deadline to preserve GPU ordering.
        # Its result is discarded and remaining pieces are never submitted.

    async def _rpc(self, awaitable: Any, *, timeout_s: float | None = None) -> Any:
        task = asyncio.create_task(awaitable)
        self._rpc_tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._rpc_tasks.discard(done)
            if not done.cancelled():
                done.exception()  # consume exceptions even after cancellation/timeout

        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=self.rpc_timeout_s if timeout_s is None else timeout_s)
            if done:
                return task.result()
            # A timed-out call may still be running remotely. Fail closed rather
            # than submit another call that could overtake it on the single GPU.
            self.state = "error"
            raise TimeoutError("voice RPC deadline exceeded")
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=self.cancel_timeout_s)

    async def _warm(self) -> None:
        try:
            if self.worker is None:
                import modal  # optional in offline CI, never needed by HTTP probes
                kwargs = {"environment_name": self.environment} if self.environment else {}
                self.worker = modal.Cls.from_name(self.app_name, self.class_name, **kwargs)()
            result = await self._rpc(self.worker.ready.remote.aio(), timeout_s=self.ready_timeout_s)
            if (not isinstance(result, dict) or result.get("protocol_version") != 1
                    or not isinstance(result.get("health"), dict)
                    or not isinstance(result.get("characters"), dict)
                    or result["health"].get("ok") is not True):
                raise ValueError("unsupported voice readiness response")
            self.metadata = {k: v for k, v in result["health"].items()
                             if k in HEALTH_FIELDS and isinstance(v, (str, bool, int, float, type(None)))}
            characters = result["characters"]
            public = {}
            for cid, value in characters.get("characters", {}).items():
                if not isinstance(cid, str) or not isinstance(value, dict):
                    continue
                public[cid] = {k: v for k, v in value.items()
                               if k in {"name", "language", "default_emotion", "loaded"}
                               and isinstance(v, (str, bool))}
                for key in ("emotions", "no_clip"):
                    public[cid][key] = [s for s in value.get(key, []) if isinstance(s, str)]
            current = characters.get("current")
            self.character_metadata = {"characters": public, "current": current if current in public else None}
            self.state = "ready"
            LOG.info("private voice RPC ready; metadata cached")
        except Exception:  # remote exception messages can contain workspace/auth context
            self.state = "error"
            LOG.warning("private voice RPC startup failed; check deployment and official Modal profile")

    async def _error(self, sid: str, job: Any, message: str) -> None:
        if sid in self.clients:
            await self.sio.emit("error", {"task_id": job.task_id, "turn_id": job.turn_id,
                                           "error": message}, to=sid)

    async def handle(self, sid: str, data: Any) -> None:
        if sid not in self.clients:
            return
        try:
            job = voice.parse_job(data)
        except Exception:
            ids = data if isinstance(data, dict) else {}
            await self.sio.emit("error", {"task_id": str(ids.get("task_id") or ""),
                                           "turn_id": ids.get("turn_id"),
                                           "error": "Invalid voice request."}, to=sid)
            return
        # Whitelist rendering inputs; gateway task/turn/client identifiers never
        # leave this machine. Only truthiness of turn_id affects remote parsing.
        payload = {k: data[k] for k in ("text", "character_id", "language", "audio_format", *voice.OPTIONS)
                   if k in data}
        if job.turn_mode:
            payload["turn_id"] = "turn"
        async with self._accept_lock:
            if self.state == "error":
                await self._error(sid, job, "Voice service unavailable; restart the local bridge.")
                return
            pending = len(self.queue) + int(self.active)
            if pending >= self.max_pending:
                await self._error(sid, job, "Voice queue is full; try again later.")
                return
            await self.sio.emit("task_queued", {"task_id": job.task_id, "turn_id": job.turn_id,
                                                "sentence_count": len(job.pieces()),
                                                "queue_position": pending}, to=sid)
            if sid in self.clients:
                self.queue.append((sid, job, payload, time.monotonic()))
                self._wake.set()

    async def _consume(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self.queue:
                sid, job, payload, enqueued = self.queue.popleft()
                self.active = True
                try:
                    await asyncio.shield(self._ready_task)
                    if sid not in self.clients:
                        continue
                    if self.state != "ready":
                        await self._error(sid, job, "Voice service unavailable; restart the local bridge.")
                        continue
                    if time.monotonic() - enqueued > self.queue_timeout_s:
                        await self._error(sid, job, "Voice request expired in the queue.")
                        continue
                    pieces = job.pieces()
                    for index in range(len(pieces)):
                        if sid not in self.clients:
                            break
                        result = await self._rpc(self.worker.render_piece.remote.aio(payload, index))
                        if sid not in self.clients:
                            break
                        # Free-form worker warnings may include encoder paths or
                        # exception details. Never relay those to the frontend.
                        result = dict(result)
                        if result.get("warning"):
                            result["warning"] = "Voice generation reported a warning."
                        await voice.Server._send_audio(self, sid, job, result, index == len(pieces) - 1)
                except Exception:
                    LOG.warning("private voice rendering failed")
                    await self._error(sid, job, "Voice generation failed; try again later.")
                finally:
                    self.active = False

    def health(self) -> dict:
        return {**self.metadata, "ok": self.state == "ready", "status": self.state,
                "transport": "modal", "metadata_cached": bool(self.metadata),
                "pending": len(self.queue) + int(self.active),
                "error": "Voice service unavailable." if self.state == "error" else ""}

    async def _startup(self, app: web.Application) -> None:
        self._ready_task = asyncio.create_task(self._warm())
        self._consumer = asyncio.create_task(self._consume())

    async def _close(self, app: web.Application) -> None:
        self.clients.clear()
        self.queue.clear()
        tasks = {task for task in (self._ready_task, self._consumer, *self._rpc_tasks)
                 if task is not None and not task.done()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self.cancel_timeout_s * 2)
        await self.sio.shutdown()

    def create_app(self) -> web.Application:
        @web.middleware
        async def internal_only(request: web.Request, handler: Any) -> web.StreamResponse:
            if request.headers.get("Origin"):
                raise web.HTTPForbidden(text="Internal voice bridge only.")
            return await handler(request)

        async def ping(request: web.Request) -> web.Response:
            return web.Response(text="ok" if self.state == "ready" else self.state,
                                status=200 if self.state == "ready" else 503)

        async def characters(request: web.Request) -> web.Response:
            return web.json_response(self.character_metadata)

        async def health(request: web.Request) -> web.Response:
            return web.json_response(self.health())

        app = web.Application(middlewares=[internal_only])
        self.sio.attach(app)
        app.router.add_get("/ping", ping)
        app.router.add_get("/api/characters", characters)
        app.router.add_get("/api/health", health)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._close)
        return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--app-name", "--appname", default="openamika-tts")
    parser.add_argument("--class-name", "--classname", default="Voice")
    parser.add_argument("--environment", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    proxy = Proxy(app_name=args.app_name, class_name=args.class_name, environment=args.environment)
    web.run_app(proxy.create_app(), host=args.host, port=args.port,
                access_log=None, print=None, shutdown_timeout=1.0)


if __name__ == "__main__":
    main()
