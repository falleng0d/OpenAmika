"""Exercise the real remote adapter at an HTTP boundary; never contact a paid provider."""
import asyncio
import base64
import http.server
import json
import threading
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

import stt_openrouter


@pytest.fixture
def endpoint(monkeypatch):
    requests = []
    response = {"status": 200, "body": {"text": "  Please remember this sentence.  "}}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers.get("Authorization"),
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(response["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(json.dumps(response["body"]).encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(stt_openrouter, "API_URL", f"http://127.0.0.1:{server.server_port}/audio/transcriptions")
    try:
        yield requests, response
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("language", ["auto", "en"])
def test_remote_request_encodes_a_mono_wav_and_returns_plain_text(endpoint, language):
    requests, _ = endpoint
    audio = np.linspace(-0.75, 0.75, 16000, dtype=np.float32)
    result = asyncio.run(stt_openrouter.transcribe(
        audio, api_key="test-only-secret", model="microsoft/mai-transcribe-2", language=language,
    ))
    assert result == ("Please remember this sentence.", "Please remember this sentence.")
    assert len(requests) == 1
    path, authorization, body = requests[0]
    assert path == "/audio/transcriptions"
    assert authorization == "Bearer test-only-secret"
    assert body["model"] == "microsoft/mai-transcribe-2"
    assert body.get("language") == (None if language == "auto" else "en")
    assert body["input_audio"]["format"] == "wav"
    with sf.SoundFile(BytesIO(base64.b64decode(body["input_audio"]["data"]))) as wav:
        assert (wav.samplerate, wav.channels, wav.subtype) == (16000, 1, "PCM_16")
        np.testing.assert_allclose(wav.read(dtype="float32"), audio, atol=1 / 32768)


@pytest.mark.parametrize("status", [302, 401, 429, 502])
def test_failures_do_not_retry_follow_redirects_or_expose_response_secrets(endpoint, status):
    requests, response = endpoint
    response.update(status=status, body={"error": "test-only-secret"})
    with pytest.raises(stt_openrouter.TranscriptionError) as caught:
        asyncio.run(stt_openrouter.transcribe(np.zeros(100), api_key="test-only-secret", model="model"))
    assert str(caught.value) == f"OpenRouter transcription failed (HTTP {status})."
    assert "test-only-secret" not in str(caught.value)
    assert len(requests) == 1


def test_missing_key_empty_audio_and_malformed_transcript_do_not_become_chat_text(endpoint):
    requests, response = endpoint
    with pytest.raises(stt_openrouter.TranscriptionError, match="needs OPENROUTER_STT_API_KEY"):
        asyncio.run(stt_openrouter.transcribe(np.zeros(100), api_key="", model="model"))
    assert asyncio.run(stt_openrouter.transcribe(np.zeros(0), api_key="test", model="model")) == ("", "")
    assert requests == []
    response["body"] = {"text": None}
    with pytest.raises(stt_openrouter.TranscriptionError, match="no transcript"):
        asyncio.run(stt_openrouter.transcribe(np.zeros(100), api_key="test", model="model"))
