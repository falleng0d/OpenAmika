"""Regression tests for the STT carry lifecycle in payload/api.py.

DialogSession is extracted from api.py by AST so the speech stack (torch,
funasr, FastAPI) is never imported. Adapted from the review harness.
"""
import ast
import asyncio
import base64
import logging
import re
import sys
import time
import unittest
from unittest.mock import patch
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "payload"
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parents[1] / "speech"   # repo layout
sys.path.insert(0, str(ROOT))
import stt_carry  # noqa: E402
import stt_openrouter  # noqa: E402

SRC = (ROOT / "api.py").read_text(encoding="utf-8")
_TREE = ast.parse(SRC)
_CHUNKS = []
for node in _TREE.body:
    if isinstance(node, ast.ClassDef) and node.name in {"HaneClient", "TTSClient", "DialogSession"}:
        _CHUNKS.append(ast.get_source_segment(SRC, node))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"_now_ms", "_is_trivial_utterance"}:
        _CHUNKS.append(ast.get_source_segment(SRC, node))
    elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("_WORD_RE", "_SFX_QUEUE") for t in node.targets):
        _CHUNKS.append(ast.get_source_segment(SRC, node))


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Torch:
    @staticmethod
    def no_grad():
        return _NoGrad()

    @staticmethod
    def from_numpy(x):
        return x


class _Prob:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _Vad:
    prob = 0.0

    def __call__(self, x, sr):
        return _Prob(self.prob)

    def reset_states(self):
        pass


VAD = _Vad()
ASR_CALLS = []


async def _asr_async(audio):
    n = len(audio) // 512
    ASR_CALLS.append(n)
    await asyncio.sleep(0)
    return f"spoken words across {n} frames", f"<|en|>spoken words across {n} frames"


class _Sio:
    def __init__(self, *a, **k):
        self.emitted = []
        self.call_impl = None

    def on(self, *a, **k):
        if len(a) >= 2:
            return None
        return lambda f: f

    def event(self, f):
        return f

    async def emit(self, evt, payload=None):
        self.emitted.append((evt, payload))

    async def call(self, evt, payload=None, timeout=None):
        self.emitted.append((evt, payload))
        if self.call_impl:
            return await self.call_impl()
        return {"ok": True}

    async def connect(self, *a, **k):
        pass

    async def disconnect(self):
        pass


class _SocketIO:
    AsyncClient = _Sio


class _Extractor:
    current_emotion_key = None

    def reset(self):
        pass

    def feed(self, t):
        return []

    def flush(self):
        return []


class _Splitter:
    def reset(self):
        pass

    def flush(self):
        return []

    def feed(self, t):
        return []


class _WS:
    client = None

    def __init__(self):
        self.sent = []

    async def send_json(self, o):
        self.sent.append(o)


NS = dict(asyncio=asyncio, time=time, uuid=uuid, base64=base64, re=re, np=np, deque=deque, datetime=datetime,
          List=List, Optional=Optional, Tuple=Tuple, torch=_Torch, vad_model=VAD, _asr_async=_asr_async,
          socketio=_SocketIO, SurfaceSpeechExtractor=_Extractor, DeepSpeechExtractor=_Extractor,
          StreamingSplitter=_Splitter, stt_carry=stt_carry, apply_wordmap=lambda t, w: t,
          logger=logging.getLogger("test_api_carry"), WebSocket=object, HANE_URL="x", TTS_URL="y",
          TTS_TRANSPORT="local", TTS_REQUEST_TIMEOUT_S=240, STT_BACKEND="local", stt_openrouter=stt_openrouter,
          TTS_CHARACTER_ID="c", TTS_AUDIO_FORMAT="wav", prefix_emotion=lambda *a, **k: "",
          parse_sfx_delta=lambda *a, **k: None, resolve_sfx=lambda *a, **k: None, pause_after=lambda *a, **k: 0.0,
          # vision hooks (speech/api.py vision section) are no-ops here
          vision_note_user_lang=lambda lang: None, vision_suffix_for=lambda text: "",
          vision_note_her_speech=lambda s: None)
exec(compile("\n\n".join(_CHUNKS), "api_extract", "exec"), NS)
DialogSession = NS["DialogSession"]
FRAME = np.zeros(512, dtype=np.float32)


def new_session():
    s = DialogSession(_WS())
    s.hane._connected = True
    s.tts._connected = False
    return s


async def frames(s, n, prob):
    VAD.prob = prob
    for _ in range(n):
        await s._on_frame(FRAME)
        await asyncio.sleep(0)


def msgs(s):
    return [(e, p.get("text"), p.get("turn_id")) for e, p in s.hane.sio.emitted if e in ("message", "inject")]


async def force_commit(s, n=938):
    await frames(s, n, 0.9)
    return s._committed_turn_id


def run(coro):
    return asyncio.run(coro)


class RemoteTranscriptionTests(unittest.TestCase):
    def test_remote_upload_waits_for_the_completed_utterance_and_skips_partials(self):
        async def main():
            calls = []

            async def recognize(audio):
                calls.append(audio.copy())
                return "This is the complete sentence.", "This is the complete sentence."

            with patch.dict(NS, STT_BACKEND="openrouter", _asr_async=recognize):
                s = new_session()
                await frames(s, 10, 0.1)       # idle silence never reaches recognition
                await frames(s, 10, 0.9)
                await frames(s, 3, 0.1)        # short pause, no remote partial
                await frames(s, 5, 0.9)
                self.assertEqual(calls, [])
                await frames(s, s.long_pause_frames, 0.1)
                task = s._pending_commit
                if task is not None:
                    await task
                self.assertEqual(len(calls), 1)
                # Six pre-roll frames, both speech sections, and both pauses are retained.
                self.assertEqual(len(calls[0]), (6 + 10 + 3 + 5 + s.long_pause_frames) * 512)
                self.assertEqual([m[1] for m in msgs(s)], ["This is the complete sentence."])
                self.assertFalse(any(m["type"] == "partial" for m in s.ws.sent))
                await frames(s, 50, 0.1)
                await s.force_stop()           # do not charge again for an already committed utterance
                self.assertEqual(len(calls), 1)
        run(main())

    def test_resumed_speech_cancels_pending_upload_and_discards_a_stale_result(self):
        async def main():
            entered = asyncio.Event()
            calls = []

            async def recognize(audio):
                calls.append(len(audio))
                if len(calls) == 1:
                    entered.set()
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        return "Stale result must not reach the brain.", "stale"
                return "The complete continued sentence.", "The complete continued sentence."

            with patch.dict(NS, STT_BACKEND="openrouter", _asr_async=recognize):
                s = new_session()
                await frames(s, 10, 0.9)
                await frames(s, s.long_pause_frames, 0.1)
                await entered.wait()
                pending = s._pending_commit
                # This must finish while recognition is still pending: VAD is not blocked by the network.
                await frames(s, 5, 0.9)
                await pending
                self.assertEqual(msgs(s), [])
                await frames(s, s.long_pause_frames, 0.1)
                pending = s._pending_commit
                if pending is not None:
                    await pending
                self.assertGreater(calls[1], calls[0])
                self.assertEqual([m[1] for m in msgs(s)], ["The complete continued sentence."])
        run(main())

    def test_remote_failure_is_visible_and_the_next_utterance_can_succeed(self):
        async def main():
            calls = []

            async def recognize(audio):
                calls.append(len(audio))
                if len(calls) == 1:
                    raise stt_openrouter.TranscriptionError("OpenRouter transcription failed (HTTP 402).")
                return "The second utterance succeeds.", "The second utterance succeeds."

            with patch.dict(NS, STT_BACKEND="openrouter", _asr_async=recognize):
                s = new_session()
                await frames(s, 5, 0.9)
                await frames(s, s.long_pause_frames, 0.1)
                task = s._pending_commit
                if task is not None:
                    await task
                self.assertEqual(s._state, "IDLE")
                self.assertEqual(msgs(s), [])
                self.assertIn({"type": "ai_error", "message": "OpenRouter transcription failed (HTTP 402)."}, s.ws.sent)
                await frames(s, 5, 0.9)
                await frames(s, s.long_pause_frames, 0.1)
                task = s._pending_commit
                if task is not None:
                    await task
                self.assertEqual([m[1] for m in msgs(s)], ["The second utterance succeeds."])
        run(main())

    def test_remote_force_cut_preserves_carry_and_explicit_stop_transcribes(self):
        async def main():
            with patch.dict(NS, STT_BACKEND="openrouter"):
                s = new_session()
                s.MAX_UTTERANCE_SEC = 0.2
                await frames(s, 7, 0.9)
                self.assertEqual(s._state, "IDLE")
                self.assertEqual(s._carry_text, "spoken words across 7 frames")
                await frames(s, 3, 0.9)
                await s.force_stop()
                self.assertEqual(msgs(s)[-1][1], "spoken words across 7 frames spoken words across 3 frames")
        run(main())

    def test_remote_commit_recovers_when_brain_disconnects_before_protect(self):
        async def main():
            calls = []

            async def recognize(audio):
                calls.append(audio.copy())
                return "A completed utterance.", "A completed utterance."

            with patch.dict(NS, STT_BACKEND="openrouter", _asr_async=recognize):
                s = new_session()
                await frames(s, 10, 0.9)
                s._speech_frames[0] = np.full(512, 0.4, dtype=np.float32)
                await frames(s, s.long_pause_frames, 0.1)
                task = s._pending_commit
                if task is not None:
                    await task
                self.assertEqual(s._state, "POST_COMMIT")
                s.hane._connected = False
                await frames(s, 1100, 0.1)       # >30 seconds without a protect acknowledgement
                self.assertEqual(s._state, "IDLE")
                self.assertEqual(len(s._speech_frames), 0)
                self.assertEqual(len(calls), 1)  # recovery must not retranscribe silence
                s.hane._connected = True
                await frames(s, 5, 0.9)
                await frames(s, s.long_pause_frames, 0.1)
                task = s._pending_commit
                if task is not None:
                    await task
                self.assertEqual(len(calls), 2)
                self.assertGreater(calls[0].max(), 0)
                self.assertEqual(calls[1].max(), 0)  # the old utterance is not submitted again
        run(main())

    def test_reset_cancels_pending_remote_work_without_publishing_it(self):
        async def main():
            entered = asyncio.Event()

            async def recognize(audio):
                entered.set()
                await asyncio.Future()

            with patch.dict(NS, STT_BACKEND="openrouter", _asr_async=recognize):
                s = new_session()
                await frames(s, 5, 0.9)
                await frames(s, s.long_pause_frames, 0.1)
                await entered.wait()
                pending = s._pending_commit
                s._reset_utterance()
                await asyncio.gather(pending, return_exceptions=True)
                self.assertTrue(pending.cancelled())
                self.assertEqual(msgs(s), [])
                self.assertIsNone(s._pending_commit)
                self.assertEqual(s._state, "IDLE")
        run(main())


class CarryLifecycleTests(unittest.TestCase):
    def setUp(self):
        ASR_CALLS.clear()

    def test_speaking_force_commit_runs_fresh_asr_over_all_frames(self):
        async def main():
            s = new_session()
            await frames(s, 625, 0.9)
            await frames(s, 1, 0.1)      # short pause -> partial over 626 frames
            await asyncio.sleep(0.01)
            self.assertEqual(s._state, "SHORT_PAUSE")
            await frames(s, 320, 0.9)    # resume; cross the 30 s cap while SPEAKING
            self.assertEqual(ASR_CALLS[-1], 938)
            sent = msgs(s)
            self.assertTrue(sent[-1][1].endswith("spoken words across 938 frames"))
            self.assertEqual(s._carry_text, "spoken words across 938 frames")
            self.assertIn(sent[-1][2], s._carry_turn_ids)
            # the continuation carries the text once
            await frames(s, 94, 0.9)
            await frames(s, 1, 0.1)
            await asyncio.sleep(0.01)
            last = msgs(s)[-1][1]
            self.assertTrue(last.startswith("spoken words across 938 frames spoken words across "), last)
            self.assertEqual(last.count("spoken words across 938 frames"), 1)
        run(main())

    def test_protect_of_carry_turn_clears_and_snapshots(self):
        async def main():
            s = new_session()
            x = await force_commit(s)
            self.assertTrue(s._carry_text)
            await s._protect_after_first_audio(x, delay=0)
            self.assertEqual(s._carry_text, "")
            self.assertEqual(s._carry_turn_ids, set())
            self.assertEqual(s._carry_snapshot[0], x)
            self.assertTrue(s._protected)
            await s._on_ai_event("end", {"turn_id": x})
            self.assertIsNone(s._carry_snapshot)
        run(main())

    def test_protect_not_acknowledged_keeps_carry(self):
        async def timeout():
            raise asyncio.TimeoutError("ack timeout")

        async def main():
            for label, setup in (("timeout", lambda s: setattr(s.hane.sio, "call_impl", timeout)),
                                 ("disconnected", lambda s: setattr(s.hane, "_connected", False)),
                                 ("nack", lambda s: setattr(s.hane.sio, "call_impl", (lambda: asyncio.sleep(0, result={"ok": False}))))):
                s = new_session()
                x = await force_commit(s)
                text = s._carry_text
                setup(s)
                await s._protect_after_first_audio(x, delay=0)
                self.assertEqual(s._carry_text, text, label)
                self.assertIn(x, s._carry_turn_ids, label)
                s.hane.sio.call_impl = None
                s.hane._connected = True
                await frames(s, 40, 0.9)
                await frames(s, 1, 0.1)
                await asyncio.sleep(0.01)
                self.assertTrue(msgs(s)[-1][1].startswith(text), label)
        run(main())

    def test_error_before_protect_keeps_carry(self):
        async def main():
            s = new_session()
            x = await force_commit(s)
            await s._on_ai_event("error", {"turn_id": x})
            await s._on_ai_event("end", {"turn_id": x})
            self.assertTrue(s._carry_text)
            self.assertIn(x, s._carry_error_turns)
        run(main())

    def test_error_after_protect_restores_snapshot(self):
        async def main():
            s = new_session()
            x = await force_commit(s)
            text = s._carry_text
            await s._protect_after_first_audio(x, delay=0)
            self.assertEqual(s._carry_text, "")
            await s._on_ai_event("error", {"turn_id": x})
            self.assertEqual(s._carry_text, text)
            self.assertIsNone(s._carry_snapshot)
        run(main())

    def test_typed_text_carries_and_inject_does_not(self):
        async def main():
            s = new_session()
            x = await force_commit(s)
            text = s._carry_text
            await s.on_text({"type": "inject", "text": "(operator note)"})
            self.assertEqual(msgs(s)[-1][1], "(operator note)")
            await s.on_text({"type": "text", "text": "typed follow-up"})
            self.assertEqual(msgs(s)[-1][1], text + " typed follow-up")
            t = s._committed_turn_id
            self.assertIn(t, s._carry_turn_ids)
            await s._protect_after_first_audio(t, delay=0)
            self.assertEqual(s._carry_text, "")
        run(main())

    def test_offline_force_commit_is_not_carried(self):
        async def main():
            s = new_session()
            s.hane._connected = False
            await force_commit(s)
            self.assertEqual(s._carry_text, "")
            self.assertEqual(s._carry_turn_ids, set())
        run(main())

    def test_force_commit_during_inflight_protect_is_not_recarried(self):
        async def main():
            s = new_session()
            await frames(s, 900, 0.9)
            await frames(s, 1, 0.1)
            await asyncio.sleep(0.01)
            # long pause commits X (POST_COMMIT); protect of X starts and stalls
            s.LONG_PAUSE_MS = 32
            await frames(s, 3, 0.1)
            await asyncio.sleep(0.01)
            x = s._committed_turn_id
            self.assertEqual(s._state, "POST_COMMIT")
            gate = asyncio.Event()

            async def slow():
                await gate.wait()
                return {"ok": True}
            s.hane.sio.call_impl = slow
            task = asyncio.create_task(s._protect_after_first_audio(x, delay=0))
            await asyncio.sleep(0.01)
            self.assertEqual(s._protect_inflight, x)
            await frames(s, 60, 0.1)     # silence pushes the utterance past the cap while protect is in flight
            self.assertEqual(s._carry_text, "")
            gate.set()
            await task
            self.assertTrue(s._protected)
            self.assertEqual(s._carry_text, "")
        run(main())

    def test_wipe_clears_carry_state(self):
        async def main():
            s = new_session()
            await force_commit(s)
            s._carry_snapshot = ("t", "x")
            s._reset_utterance()
            self.assertTrue(s._carry_text)  # utterance reset alone keeps the carry
            s._carry_text = ""
            s._carry_turn_ids.clear()
            s._carry_error_turns.clear()
            s._carry_snapshot = None
            self.assertEqual(s._carry_text, "")
        run(main())


if __name__ == "__main__":
    unittest.main()
