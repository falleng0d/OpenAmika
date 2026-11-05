"""The operational benchmark must not turn failed requests into reassuring timings."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_local", ROOT / "scripts/benchmark_local.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_word_error_rate_counts_substitution_deletion_and_insertion():
    assert benchmark.word_error_rate("Hello, WORLD!", "hello world") == 0
    assert benchmark.word_error_rate("one two three", "one four three five") == pytest.approx(2 / 3)
    assert benchmark.word_error_rate("one two three", "one three") == pytest.approx(1 / 3)
    assert benchmark.word_error_rate("one", "") == 1
    with pytest.raises(ValueError):
        benchmark.word_error_rate("", "unexpected")


def test_failed_trials_remain_visible_and_warmups_do_not_bias_summary():
    def operation(index):
        if index == 0:
            return {"wall_ms": 9000}
        if index == 2:
            raise TimeoutError("backend stalled")
        return {"wall_ms": index * 100}

    result = benchmark.measure("test", operation, warmups=1, repeats=3)
    assert result["summary"] == {"attempts": 3, "successes": 2, "errors": 1,
                                 "median_ms": 200, "p95_ms": 300}
    assert result["trials"][2]["error"] == "TimeoutError: backend stalled"
    failed = benchmark.summary([{"phase": "measured", "error": "failed"}])
    assert failed == {"attempts": 1, "successes": 0, "errors": 1, "median_ms": None, "p95_ms": None}


def test_changed_sample_is_rejected_before_audio_is_submitted(tmp_path):
    (tmp_path / "jfk.flac").write_bytes(b"not the public sample")
    with pytest.raises(ValueError, match="Sample hash mismatch"):
        benchmark.sample_files(tmp_path, download=False)


def test_report_counts_backend_failures_and_exits_unsuccessfully(tmp_path):
    class Embeddings(BaseHTTPRequestHandler):
        attempts = 0

        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/embed"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body == {"texts": ["What did we talk about yesterday?"], "task_type": "query"}
            Embeddings.attempts += 1
            # A wrong-dimension second result must be a failed trial, not a fast success.
            payload = {"embeddings": [[.6, .8]], "dims": 2 if self.attempts == 1 else 3, "ms": 4.2}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Embeddings)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    report = tmp_path / "measurements.json"
    try:
        result = benchmark.main(["--components", "embed", "--embed-url",
                                 f"http://127.0.0.1:{server.server_port}", "--warmups", "0",
                                 "--repeats", "2", "--output", str(report)])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert result == 1
    case, = json.loads(report.read_text("utf-8"))["cases"]
    assert case["summary"]["successes"] == 1
    assert case["summary"]["errors"] == 1
    assert case["trials"][0]["dims"] == 2
    assert case["trials"][0]["wall_ms"] > 0
    assert case["trials"][1]["error"] == "ValueError: Invalid embedding dimensions or values"
