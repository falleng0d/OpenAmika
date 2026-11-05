"""OpenAmika vision service — small local VLM describing the camera feed in real time.

Headless port of the Amika_Streaming_VLM program (same pipeline, no Gradio, no Ollama):
it subscribes to the speech gateway's /ws/vlm_feed relay (binary JPEG frames +
{"type":"prompt_update"} JSON), runs a pipelined describe loop, and serves the result:

    GET /api/latest  -> {"description": str, "fps": float, "ok": true}
    GET /api/health  -> {"ok": bool, "model": str, "model_loaded": bool, "running": bool}

Ships with small transformers-native models only (the original defaulted to a 30B via
Ollama; these run on one consumer GPU next to the rest of the stack):

    qwen3-vl-2b     Qwen/Qwen3-VL-2B-Instruct        (~4.3 GB bf16, zh/en/ja — default:
                    newest small Qwen, ~0.7 s/description on an RTX 5090)
    qwen3-vl-4b     Qwen/Qwen3-VL-4B-Instruct        (~9 GB bf16; sharper, but decoded
                    ~6 tok/s on torch 2.10+cu128 here — try on your own stack)
    qwen2.5-vl-3b   Qwen/Qwen2.5-VL-3B-Instruct      (~7.6 GB bf16, the previous default)
    smolvlm2-2.2b   HuggingFaceTB/SmolVLM2-2.2B-Instruct        (~5 GB bf16, EN-centric)
    smolvlm2-500m   HuggingFaceTB/SmolVLM2-500M-Video-Instruct  (~1.2 GB bf16, tiny)

Launch (normally done by launch.py from the `vision` block in settings.json):
    python vlm_service.py --port 8014 --feed-url ws://127.0.0.1:53000/ws/vlm_feed
    python vlm_service.py --download --model qwen3-vl-2b     # fetch the weights now (scripts/setup_core.py)
"""
from __future__ import annotations

import argparse
import gc
import json
import queue
import threading
import time
from collections import deque
from typing import Any, Optional

# Heavy imports happen in main()/load() so --check stays fast.

DEFAULT_PROMPT_ZH = (
    "请用一到两句简洁的话描述这张图片中的内容。"
    "请聚焦主要主体、其动作以及环境。只输出描述本身，不要开场白。"
)

MODELS = {
    "qwen3-vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", "qwen"),
    "qwen3-vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", "qwen"),
    "qwen2.5-vl-3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen"),
    "smolvlm2-2.2b": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "smol"),
    "smolvlm2-500m": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "smol"),
}


# ---------------------------------------------------------------------------
# Model backend (one class covers all three repos via Auto* APIs)
# ---------------------------------------------------------------------------

class SmallVLM:
    def __init__(self, key: str, device: str = "cuda", max_tokens: int = 60):
        self.key = key
        self.repo, self.family = MODELS[key]
        self.device = device
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None
        self._cached_prompt: Optional[str] = None
        self._cached_text: Optional[str] = None

    @staticmethod
    def _cached_snapshot(repo: str) -> bool:
        """Is a snapshot of `repo` in the local Hugging Face cache? A plain filesystem check, made before the
        Hub libraries are imported (their offline switch is read at import time)."""
        import os
        from pathlib import Path
        hub = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
        snapshots = Path(hub) / ("models--" + repo.replace("/", "--")) / "snapshots"
        return any((p / "config.json").is_file() for p in snapshots.glob("*")) if snapshots.is_dir() else False

    def load(self):
        import os
        import torch

        cached = self._cached_snapshot(self.repo)
        if cached:
            # The model is in the local cache: never ask the Hub. `local_files_only` alone was not enough —
            # transformers still sent a HEAD for processor_config.json, one reset connection closed the Hub
            # client, and the RuntimeError that followed killed the service (2026-10-31, 2026-11-02).
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoProcessor, AutoModelForImageTextToText

        kwargs = {}
        if self.family == "qwen":
            # same small pixel budget as the original service — keeps latency low
            kwargs = {"min_pixels": 28 * 28 * 4, "max_pixels": 28 * 28 * 128}
        try:
            self._processor = AutoProcessor.from_pretrained(self.repo, local_files_only=cached, **kwargs)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.repo, dtype=torch.bfloat16, local_files_only=cached
            )
        except Exception as error:
            if not cached:
                raise
            # the cache turned out incomplete: go online once (flip the switch the libraries read at import)
            print(f"[vlm] {self.repo} could not load from the local cache ({type(error).__name__}: {str(error)[:120]}); downloading", flush=True)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                os.environ.pop(key, None)
            try:
                import huggingface_hub.constants as hub_constants
                hub_constants.HF_HUB_OFFLINE = False
            except Exception:
                pass
            self._processor = AutoProcessor.from_pretrained(self.repo, **kwargs)
            self._model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=torch.bfloat16)
        self._model = self._model.to(self.device)
        self._model.eval()

    def unload(self):
        import torch
        self._model = self._processor = None
        self._cached_prompt = self._cached_text = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _template(self, prompt: str) -> str:
        if self._cached_prompt != prompt:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            self._cached_text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            self._cached_prompt = prompt
        return self._cached_text

    def preprocess(self, image, prompt: str) -> Any:
        return self._processor(
            text=[self._template(prompt)], images=[image], return_tensors="pt"
        )

    def generate(self, inputs: Any) -> str:
        import torch
        inputs = inputs.to(self.device)
        input_len = inputs.input_ids.shape[1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False
            )
        text = self._processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
        if "Assistant:" in text:                       # SmolVLM chat-template artifact
            text = text.split("Assistant:")[-1].strip()
        return text


class ModalVLM:
    """Same local feed contract, with inference on the private Modal voice worker."""

    def __init__(self, key: str):
        if key != "qwen3-vl-2b":
            raise ValueError("Modal vision supports only qwen3-vl-2b")
        self.key = key
        self.repo = MODELS[key][0]
        self.is_loaded = False
        self.worker = None

    def load(self):
        import modal
        self.worker = modal.Cls.from_name("openamika-tts", "Voice")()

    def preprocess(self, image, prompt: str):
        import io
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue(), prompt

    def generate(self, inputs) -> str:
        description = self.worker.describe.remote(*inputs)
        self.is_loaded = True
        return description


# ---------------------------------------------------------------------------
# Shared state + feed subscriber + inference loop (ported, trimmed)
# ---------------------------------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.description = ""
        self.fps = 0.0
        self.running = False
        self._timestamps: deque = deque(maxlen=120)
        self.prompt: Optional[str] = None            # live prompt from the gateway
        self.feed_connected = False
        self.frames_seen = 0

    def push(self, description: str):
        now = time.perf_counter()
        with self.lock:
            self.description = description
            self._timestamps.append(now)
            cutoff = now - 3.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            elapsed = now - self._timestamps[0] if len(self._timestamps) > 1 else 0.0
            self.fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def get_prompt(self, fallback: str) -> str:
        with self.lock:
            return self.prompt if self.prompt else fallback

    def set_prompt(self, prompt: str):
        with self.lock:
            self.prompt = prompt


class FeedSubscriber:
    """JPEG frames from the gateway relay WebSocket; reconnects forever."""

    def __init__(self, state: State, ws_url: str, width: int = 640, height: int = 480):
        self._state = state
        self._ws_url = ws_url
        self._width, self._height = width, height
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_rgb = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame_np(self):
        with self._lock:
            return self._latest_rgb

    def _main(self):
        import cv2
        import numpy as np
        import websocket

        delay = 0.5
        while self._running:
            ws = None
            try:
                ws = websocket.create_connection(self._ws_url, timeout=5)
            except Exception:
                self._state.feed_connected = False
                time.sleep(delay)
                delay = min(delay * 1.5, 5.0)
                continue
            delay = 0.5
            self._state.feed_connected = True
            print(f"[feed] connected to {self._ws_url}", flush=True)
            try:
                ws.settimeout(2.0)
                while self._running:
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception:
                        break
                    if isinstance(raw, str):
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            continue
                        if msg.get("type") == "prompt_update" and msg.get("prompt"):
                            self._state.set_prompt(msg["prompt"])
                        continue
                    if not raw:
                        continue
                    buf = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    if w > self._width or h > self._height:
                        scale = min(self._width / w, self._height / h)
                        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                                         interpolation=cv2.INTER_AREA)
                    with self._lock:
                        self._latest_rgb = rgb
                    self._state.frames_seen += 1
            finally:
                self._state.feed_connected = False
                try:
                    ws.close()
                except Exception:
                    pass
            if self._running:
                time.sleep(0.5)


class InferenceLoop:
    """Preprocess (CPU) and generate (GPU) run in parallel; stale frames are dropped.
    A small grayscale diff gates inference so a static scene costs nothing."""

    def __init__(self, model: SmallVLM, feed: FeedSubscriber, state: State,
                 prompt: str, max_dps: float, change_threshold: float):
        self._model = model
        self._feed = feed
        self._state = state
        self._prompt = prompt
        self.max_dps = max_dps
        self.change_threshold = change_threshold
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._last_sig = None

    def start(self):
        self._state.running = True
        threading.Thread(target=self._preprocess_loop, daemon=True).start()
        threading.Thread(target=self._generate_loop, daemon=True).start()

    def _signature(self, rgb):
        import cv2
        small = cv2.resize(rgb, (64, 48), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    def _frame_new(self, sig) -> bool:
        import numpy as np
        if self._last_sig is None:
            return True
        mad = float(np.mean(np.abs(sig.astype("int16") - self._last_sig.astype("int16"))))
        return mad >= self.change_threshold

    def _preprocess_loop(self):
        from PIL import Image
        while self._state.running:
            rgb = self._feed.get_frame_np()
            if rgb is None:
                time.sleep(0.05)
                continue
            sig = self._signature(rgb)
            if not self._frame_new(sig):
                time.sleep(0.05)
                continue
            prompt = self._state.get_prompt(self._prompt)
            try:
                inputs = self._model.preprocess(Image.fromarray(rgb), prompt)
            except Exception:
                continue
            self._last_sig = sig.copy()
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(inputs)

    def _generate_loop(self):
        next_allowed = 0.0
        while self._state.running:
            try:
                inputs = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._state.push(self._model.generate(inputs))
            except Exception as exc:
                print(f"[vlm] generate failed: {exc}", flush=True)
            if self.max_dps > 0:
                min_interval = 1.0 / self.max_dps
                now = time.perf_counter()
                if next_allowed > now:
                    time.sleep(next_allowed - now)
                    now = time.perf_counter()
                next_allowed = now + min_interval


# ---------------------------------------------------------------------------

def download(key: str) -> int:
    """Put a model into the Hugging Face cache now (scripts/setup_core.py), so the first start does not wait for it:
    the repository's files (no ONNX / GGUF variants), then its processor is loaded once from the cache as a check."""
    repo = MODELS[key][0]
    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, ignore_patterns=["onnx/*", "*.onnx", "*.onnx_data", "*.gguf", "*.msgpack", "*.h5",
                                                        "*.ot", "*.tflite"])
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(path)
    except Exception as error:  # noqa: BLE001  (reported in one line for the setup summary)
        print(f"[vlm] download failed: {type(error).__name__}: {str(error)[:300]}", flush=True)
        return 1
    print(f"[vlm] {repo} is in the Hugging Face cache ({path}, {time.time() - t0:.0f} s)", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--model", default="qwen3-vl-2b", choices=sorted(MODELS))
    parser.add_argument("--feed-url", default="ws://127.0.0.1:53000/ws/vlm_feed")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--max-dps", type=float, default=1.0,
                        help="max descriptions per second (GPU duty-cycle cap)")
    parser.add_argument("--change-threshold", type=float, default=5.0,
                        help="mean abs diff on a 64x48 gray signature; below = skip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true", help="verify imports only")
    parser.add_argument("--download", action="store_true",
                        help="download the model into the Hugging Face cache and check that its processor loads, then exit")
    args = parser.parse_args()

    if args.download:
        raise SystemExit(download(args.model))

    if args.check:
        import importlib.util
        names = (("modal", "cv2", "PIL", "numpy", "fastapi", "uvicorn", "websocket") if args.device == "modal"
                 else ("torch", "transformers", "cv2", "PIL", "numpy", "fastapi", "uvicorn", "websocket"))
        missing = [n for n in names if importlib.util.find_spec(n) is None]
        if missing:
            print(f"vision check: missing modules: {', '.join(missing)}")
            raise SystemExit(1)
        print(f"vision check: all importable ({', '.join(names)})")
        return

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    state = State()
    model = ModalVLM(args.model) if args.device == "modal" else SmallVLM(args.model, device=args.device, max_tokens=args.max_tokens)
    print(f"[vlm] loading {model.repo} …", flush=True)
    t0 = time.time()
    model.load()
    print(f"[vlm] model ready in {time.time() - t0:.1f}s", flush=True)

    feed = FeedSubscriber(state, args.feed_url)
    feed.start()
    InferenceLoop(model, feed, state, DEFAULT_PROMPT_ZH,
                  args.max_dps, args.change_threshold).start()

    app = FastAPI(title="OpenAmika Vision")

    @app.get("/api/latest")
    def api_latest():
        with state.lock:
            return JSONResponse({"description": state.description, "fps": state.fps, "ok": True})

    @app.get("/api/health")
    def api_health():
        return JSONResponse({
            "ok": model.is_loaded or isinstance(model, ModalVLM),
            "status": "ok" if model.is_loaded else ("idle" if isinstance(model, ModalVLM) else "loading"),
            "model": args.model,
            "model_loaded": model.is_loaded,
            "running": state.running,
            "feed_connected": state.feed_connected,
            "frames_seen": state.frames_seen,
        })

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
