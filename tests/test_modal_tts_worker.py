"""Cloud adapter contracts, without Modal, network access, torch, or personal assets."""
import importlib.util
import io
from pathlib import Path
import sys
import threading
import traceback
from types import SimpleNamespace
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modal_tts_build_under_test", ROOT / "servers/modal-tts/build_engine.py")
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)
voice = cloud.engine_module()


@pytest.mark.parametrize("payload, index", [
    ({"text": "Hello", "model_path": "/private-assets/voice.pth"}, 0),
    ({"text": "Hello", "character_id": "../another-speaker"}, 0),
    ({"text": "Hello", "temperature": 0}, 0),
    ({"text": "Hello", "seed": {"path": "other"}}, 0),
    ({"text": "Hello", "language": ["English"]}, 0),
    ({"text": "Hello"}, -1),
    ({"text": "Hello"}, 1),
    ({"text": "Hello"}, True),
    ({"text": "Hello"}, 0.5),
    ({"text": " "}, 0),
    ({"text": "x" * 8001}, 0),
])
def test_rejects_unsupported_requests_and_piece_indices(payload, index):
    with pytest.raises(ValueError):
        cloud.validate_piece(voice, payload, index)


def test_piece_selection_preserves_original_turn_and_sentence_modes():
    payload = {"text": "【happy1】Hello! 【sad1】Goodbye!", "language": "English"}
    job, piece = cloud.validate_piece(voice, payload, 1)
    assert [(segment.tag, segment.text) for segment in piece] == [("sad1", "Goodbye!")]
    assert job.lang == "en"
    job, piece = cloud.validate_piece(voice, {**payload, "turn_id": "turn-1"}, 0)
    assert [(segment.tag, segment.text) for segment in piece] == [("happy1", "Hello!"), ("sad1", "Goodbye!")]
    with pytest.raises(ValueError):
        cloud.validate_piece(voice, {**payload, "turn_id": "turn-1"}, 1)


@pytest.fixture
def worker(monkeypatch):
    np = pytest.importorskip("numpy")
    calls = []

    class Pipeline:
        state = "starting"
        device = "cpu"
        output_sr = 32000
        requests = 0

        def load(self, character):
            calls.append(("load", threading.get_ident()))
            self.state = "loading"
            self.device = "cuda"

        def warmup(self):
            calls.append(("warmup", threading.get_ident()))

        def ensure_speaker(self, character):
            return SimpleNamespace(id=cloud.VOICE_ID)

        def synth(self, text, tag, lang, opts):
            calls.append(("synth", threading.get_ident()))
            return {"pcm": np.array([100, -100, 200], dtype=np.int16), "tag": tag or "neutral1",
                    "warning": "", "fallback": False, "lang": lang, "retried": 0, "stages": []}

    server = voice.Server.__new__(voice.Server)
    server.engine = Pipeline()
    server.pool = voice.SingleWorker("test-cloud-voice")
    server.opus_ok = False
    # Model health/speakers require real weights; the rendering implementation is real.
    server.health = lambda: {"ok": server.engine.state == "ready", "device": server.engine.device}
    server.characters = lambda: {"current": cloud.VOICE_ID}
    router = voice.StdoutRouter(sys.stdout)
    monkeypatch.setattr(voice, "_ROUTER", router)
    monkeypatch.setattr(sys, "stdout", router)
    adapter = cloud.EngineWorker(voice, server)
    try:
        yield adapter, server, calls
    finally:
        server.pool.shutdown(wait=True, cancel_futures=True)


def test_real_renderer_returns_encoded_audio_on_the_initialized_worker_thread(worker):
    adapter, server, calls = worker
    with pytest.raises(RuntimeError, match="not ready"):
        adapter.ready()
    adapter.initialize(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert adapter.ready() == {"health": {"ok": True, "device": "cuda"},
                               "characters": {"current": cloud.VOICE_ID}, "protocol_version": 1}
    result = adapter.render_piece({"text": "【happy1】Hello!", "turn_id": "turn-1", "audio_format": "wav"}, 0)
    with wave.open(io.BytesIO(result["audio"])) as wav:
        assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (32000, 2, 1)
        assert wav.readframes(3) == b"\x64\x00\x9c\xff\xc8\x00"
    assert result["text"] == "Hello!"
    assert result["emotions"] == ["happy1"]
    assert result["character_id"] == cloud.VOICE_ID
    assert result["duration"] == 3 / 32000
    assert [name for name, _ in calls] == ["load", "warmup", "synth"]
    assert len({thread for _, thread in calls}) == 1
    assert calls[0][1] != threading.get_ident()


def test_no_cuda_and_silent_cpu_fallback_cannot_become_ready(worker):
    adapter, server, calls = worker
    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.initialize(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert calls == []

    def load_on_cpu(character):
        server.engine.device = "cpu"
    server.engine.load = load_on_cpu
    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.initialize(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert server.engine.state != "ready"
    assert calls == []  # Not even warmup may run after CPU fallback.


def test_failed_warmup_does_not_advertise_ready(worker):
    adapter, server, _ = worker
    def failed_warmup():
        raise RuntimeError("warmup failed")
    server.engine.warmup = failed_warmup
    with pytest.raises(RuntimeError, match="warmup failed"):
        adapter.initialize(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    with pytest.raises(RuntimeError, match="not ready"):
        adapter.ready()


def test_upstream_transcript_is_absent_from_cloud_output_and_error(worker, capsys, monkeypatch):
    adapter, server, _ = worker
    # pytest installs the call-phase capture stream after fixture setup; restore the production router.
    monkeypatch.setattr(sys, "stdout", voice._ROUTER)
    adapter.initialize(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    private_text = "synthetic-private-transcript"
    def fail(job, piece):
        print(private_text)
        raise RuntimeError(private_text)
    server._render = fail
    with pytest.raises(RuntimeError, match="Cloud TTS synthesis failed") as caught:
        adapter.render_piece({"text": "Hello!"}, 0)
    assert private_text not in "".join(traceback.format_exception(caught.value))
    assert private_text not in capsys.readouterr().out
