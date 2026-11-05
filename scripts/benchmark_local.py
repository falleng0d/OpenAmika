#!/usr/bin/env python3
"""Measure the running local services without calling the LLM or changing settings.

Run with envs/speech/Scripts/python.exe. Samples and reports stay under ignored logs/.
Keep the companion idle while measuring; results include local request/queue overhead,
but not microphone capture, VAD endpointing, the cloud LLM, or browser playback.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import platform
import re
import statistics
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    {
        "name": "jfk.flac",
        "url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        "sha256": "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715",
        "reference": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country",
        "source": "OpenAI Whisper test clip; John F. Kennedy inaugural address",
    },
    {
        "name": "librispeech.wav",
        "url": "https://download.pytorch.org/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav",
        "sha256": "67dfc27041e572bea27936e75eedc542ccf6f3af5cc251a0ab4e1e947d787fc8",
        "reference": "I really was very much afraid of showing him how much shocked I was at some parts of what he said",
        "source": "LibriSpeech test-other 1688-142285-0007; PyTorch audio CTC tutorial",
    },
)
VOICE_CASES = (
    {"name": "en-neutral", "text": "【neutral1】.Welcome home. How was your day?", "language": "English"},
    {"name": "ja-happy", "text": "【happy2】.おかえりなさい。今日もおつかれさま。", "language": "Japanese"},
    {"name": "zh-caring", "text": "【caring2】.今天辛苦了，坐下来休息一下吧。", "language": "Chinese"},
)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Case/punctuation-insensitive English word edit distance / reference length."""
    expected = re.findall(r"\w+(?:'\w+)?", reference.lower())
    actual = re.findall(r"\w+(?:'\w+)?", hypothesis.lower())
    if not expected:
        raise ValueError("A nonempty reference transcript is required")
    previous = list(range(len(actual) + 1))
    for index, word in enumerate(expected, 1):
        row = [index]
        for column, observed in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[column] + 1,
                           previous[column - 1] + (word != observed)))
        previous = row
    return previous[-1] / len(expected)


def summary(trials: list[dict]) -> dict:
    measured = [trial for trial in trials if trial["phase"] == "measured"]
    good = [trial for trial in measured if "error" not in trial]
    times = sorted(trial["wall_ms"] for trial in good)
    return {
        "attempts": len(measured), "successes": len(good), "errors": len(measured) - len(good),
        "median_ms": statistics.median(times) if times else None,
        "p95_ms": times[math.ceil(len(times) * .95) - 1] if times else None,
    }


def sample_files(directory: Path, download: bool) -> list[tuple[dict, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for sample in SAMPLES:
        path = directory / sample["name"]
        if not path.exists() and download:
            with urllib.request.urlopen(sample["url"], timeout=60) as response:
                data = response.read(5 * 1024 * 1024 + 1)
            if hashlib.sha256(data).hexdigest() != sample["sha256"]:
                raise ValueError(f"Download hash mismatch: {sample['name']}")
            path.write_bytes(data)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; pass --download-samples")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Sample hash mismatch: {path}")
        result.append((sample, path))
    return result


def asr_request(url: str, sample: dict, path: Path) -> dict:
    import requests

    with path.open("rb") as audio:
        started = time.perf_counter()
        response = requests.post(url + "/api/v1/asr", files={"files": (path.name, audio)},
                                 data={"lang": "auto", "use_itn": "true"}, timeout=180)
        wall_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    data = response.json()
    transcript = " ".join(item["clean_text"] for item in data["result"]).strip()
    if not transcript:
        raise ValueError("ASR returned no speech for the spoken sample")
    duration = data["meta"]["audio_duration"]
    return {"wall_ms": wall_ms, "server_inference_ms": data["meta"]["inference_time"] * 1000,
            "audio_s": duration, "rtf": wall_ms / 1000 / duration,
            "transcript": transcript, "wer": word_error_rate(sample["reference"], transcript)}


async def tts_request(url: str, case: dict, output: Path) -> dict:
    import numpy as np
    import socketio
    import soundfile as sf

    client = socketio.AsyncClient(reconnection=False)
    complete = asyncio.Event()
    task_id = str(uuid.uuid4())
    chunks, metadata, errors, first_audio = [], {}, [], []
    started = 0.0

    @client.on("audio_ready")
    async def on_audio(data):
        if data.get("task_id") != task_id:
            return
        if not first_audio:
            first_audio.append((time.perf_counter() - started) * 1000)
        chunks.append(data["audio"])
        metadata.update({key: data[key] for key in ("warning", "elapsed_ms", "emotions") if key in data})
        if data.get("is_final"):
            complete.set()

    @client.on("error")
    async def on_error(data):
        if data.get("task_id") == task_id:
            errors.append(data.get("error", "TTS error"))
            complete.set()

    try:
        await client.connect(url, transports=["websocket"], wait_timeout=15)
        started = time.perf_counter()
        await client.emit("tts_request_gateway", {"task_id": task_id, "turn_id": task_id,
                    "text": case["text"], "character_id": "MekaHime_Amika",
                    "audio_format": "wav", "language": case["language"], "seed": 123})
        await asyncio.wait_for(complete.wait(), timeout=180)
        wall_ms = (time.perf_counter() - started) * 1000
        if errors:
            raise RuntimeError(errors[0])
        audio_bytes = b"".join(chunks)
        audio, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError("TTS returned empty, nonfinite, or silent audio")
        duration = len(audio) / rate
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio_bytes)
        return {"wall_ms": wall_ms, "first_audio_ms": first_audio[0], "audio_s": duration,
                "rtf": wall_ms / 1000 / duration, "peak": float(np.abs(audio).max()),
                "audio_path": str(output), **metadata}
    finally:
        if client.connected:
            await client.disconnect()


def embed_request(url: str) -> dict:
    import requests

    started = time.perf_counter()
    response = requests.post(url + "/embed", json={"texts": ["What did we talk about yesterday?"],
                                                   "task_type": "query"}, timeout=180)
    wall_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    data = response.json()
    vector, = data["embeddings"]
    if len(vector) != data["dims"] or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid embedding dimensions or values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1, abs_tol=.001):
        raise ValueError(f"Embedding is not unit normalized: {norm}")
    return {"wall_ms": wall_ms, "server_ms": data["ms"], "dims": len(vector)}


def measure(name: str, operation, warmups: int, repeats: int) -> dict:
    trials = []
    for index in range(warmups + repeats):
        phase = "warmup" if index < warmups else "measured"
        try:
            result = operation(index)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        trials.append({"phase": phase, **result})
        print(json.dumps({"case": name, "trial": index + 1, **trials[-1]}, ensure_ascii=False), flush=True)
    return {"name": name, "trials": trials, "summary": summary(trials)}


def local_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username:
        raise argparse.ArgumentTypeError("Use an http:// loopback URL without credentials")
    return value.rstrip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+", choices=("asr", "tts", "embed"), default=["asr", "tts", "embed"])
    parser.add_argument("--asr-url", type=local_url, default="http://127.0.0.1:53000")
    parser.add_argument("--tts-url", type=local_url, default="http://127.0.0.1:8002")
    parser.add_argument("--embed-url", type=local_url, default="http://127.0.0.1:8011")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--download-samples", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/performance/baseline.json")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups nonnegative")
    cases = []
    if "asr" in args.components:
        for sample, path in sample_files(ROOT / "logs/performance/samples", args.download_samples):
            case = measure("asr/" + sample["name"], lambda _, s=sample, p=path: asr_request(args.asr_url, s, p),
                           args.warmups, args.repeats)
            cases.append({**case, "sample": sample})
    if "tts" in args.components:
        for case in VOICE_CASES:
            cases.append(measure("tts/" + case["name"], lambda index, c=case: asyncio.run(tts_request(
                args.tts_url, c, args.output.parent / args.output.stem / f"{c['name']}-{index}.wav")),
                args.warmups, args.repeats))
    if "embed" in args.components:
        cases.append(measure("embed/query", lambda _: embed_request(args.embed_url), args.warmups, args.repeats))
    report = {"at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "scope": "Local service wall times, excluding VAD/LLM/playback; warmups excluded from summaries",
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}")
    return int(any("error" in trial for case in cases for trial in case["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
