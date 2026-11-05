"""Build the public engine and adapt its single worker, without importing Modal."""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "vendor" / "tts-engine"
VOICE_ID = "MekaHime_Amika"
PROTOCOL_VERSION = 1

# Individual public files only: never traverse the checkout or vendor directories.
SOURCE_FILES = (
    "servers/modal-tts/app.py",
    "servers/modal-tts/build_engine.py",
    "servers/modal-tts/vision_worker.py",
    "scripts/setup_tts.py",
    "scripts/MekaHime_Amika.manifest.json",
    "vendor/tts-engine/main.py",
    "vendor/tts-engine/requirements-tts.txt",
    "vendor/tts-engine/LICENSE",
    "vendor/tts-engine/THIRD_PARTY_NOTICES.md",
)


def load_module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve their defining module here
        spec.loader.exec_module(module)
    return sys.modules[name]


def engine_module():
    return load_module("openamika_cloud_voice_engine", ENGINE / "main.py")


def build_engine() -> None:
    """Install into THIS Python, retaining setup_tts's source and model integrity pins.

    Amika's personal assets are deliberately absent from this build path.
    """
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    setup = load_module("openamika_cloud_voice_setup", ROOT / "scripts" / "setup_tts.py")
    python = Path(sys.executable)
    constraints = setup.ensure_torch(python, "cuda", reinstall=False)
    setup.ensure_requirements(python, ENGINE, constraints, reinstall=False)
    verified = setup.Verified(ENGINE / ".openamika-verified.json")
    try:
        upstream = setup.ensure_checkout(ENGINE)
        setup.ensure_pretrained(upstream, verified)
        setup.ensure_g2pw(ENGINE, upstream, verified, keep=False)
        setup.ensure_nltk(ENGINE, verified)
        imports = setup.verify_imports(python)
        if any(name != "g2pk2" for name in imports["bad"]):
            raise RuntimeError("Voice engine dependency import check failed")
        setup.ensure_ja_dict(python, upstream)
    finally:
        verified.save()


def download_voice(assets: Path) -> None:
    """Download official, pinned voice files directly into the private runtime Volume."""
    env = {**os.environ, "HF_ENDPOINT": "https://huggingface.co"}
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_tts.py"),
         "--assets-only", "--skip-avatar", "--skip-paper-doll", "--accept-voice-license",
         "--engine-dir", str(assets)],
        env=env, check=True, timeout=720,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def validate_piece(module, payload: dict, piece_index: int):
    """Only the existing text-synthesis protocol is accepted, never paths or model overrides."""
    allowed = {"text", "task_id", "turn_id", "character_id", "language", "audio_format", *module.OPTIONS}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unsupported TTS payload")
    if not isinstance(payload.get("text"), str) or not payload["text"].strip() or len(payload["text"]) > 8000:
        raise ValueError("TTS text must contain between 1 and 8000 characters")
    for name in ("task_id", "turn_id", "character_id", "language", "audio_format"):
        value = payload.get(name)
        if value is not None and (not isinstance(value, str) or len(value) > 200):
            raise ValueError("Invalid TTS metadata")
    if payload.get("character_id") not in (None, "", VOICE_ID):
        raise ValueError("Only the configured personal voice is available")
    if isinstance(piece_index, bool) or not isinstance(piece_index, int):
        raise ValueError("piece_index must be an integer")
    try:
        job = module.parse_job(payload)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid TTS synthesis options") from None
    pieces = job.pieces()
    if not 0 <= piece_index < len(pieces):
        raise ValueError("piece_index is outside this request")
    return job, pieces[piece_index]


class EngineWorker:
    """All model startup, health reads and rendering stay on the original SingleWorker."""

    def __init__(self, module, server) -> None:
        self.module = module
        self.server = server

    def initialize(self, torch) -> None:
        def start():
            # Even startup imports/guard diagnostics can contain transcripts. Keep them off logs.
            with self.module._Quiet(False):
                if not torch.cuda.is_available():
                    raise RuntimeError("Cloud TTS requires CUDA; CPU fallback is disabled")
                engine = self.server.engine
                engine.load(VOICE_ID)
                if not engine.device.startswith("cuda"):
                    raise RuntimeError("Cloud TTS requires CUDA; CPU fallback is disabled")
                engine.state = "warming"
                engine.warmup()
                engine.state = "ready"

        self.server.pool.submit(start).result()

    def ready(self) -> dict:
        def status():
            if self.server.engine.state != "ready":
                raise RuntimeError("Cloud TTS is not ready")
            return {"health": self.server.health(), "characters": self.server.characters(),
                    "protocol_version": PROTOCOL_VERSION}

        return self.server.pool.submit(status).result()

    def render_piece(self, payload: dict, piece_index: int) -> dict:
        job, piece = validate_piece(self.module, payload, piece_index)

        def render():
            if self.server.engine.state != "ready":
                raise RuntimeError("Cloud TTS is not ready")
            try:
                with self.module._Quiet(False):
                    return self.server._render(job, piece)
            except Exception:
                # Do not serialize upstream exceptions containing text/transcripts or log their context.
                raise RuntimeError("Cloud TTS synthesis failed") from None

        return self.server.pool.submit(render).result()

    def close(self) -> None:
        self.server.pool.shutdown(wait=False, cancel_futures=True)


def start_worker(assets: Path) -> EngineWorker:
    # Container-wide policy, not a local application change. Upstream logs can echo reference transcripts.
    logging.disable(logging.CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")
    from loguru import logger
    logger.remove()
    module = engine_module()
    module.install_stdout_router()
    args = module.parse_args([
        "--device", "cuda", "--character", VOICE_ID,
        "--gpt-sovits-dir", str(ENGINE / "GPT-SoVITS"),
        "--speakers-dir", str(assets / "Speakers"),
        "--cache-dir", "/tmp/openamika-tts-cache",
    ])
    worker = EngineWorker(module, module.Server(args))
    try:
        # Import on the same worker thread as model initialization.
        def initialize():
            with module._Quiet(False):
                import torch
                return torch
        torch = worker.server.pool.submit(initialize).result()
        worker.initialize(torch)
    except Exception:
        worker.close()
        raise RuntimeError("Cloud TTS initialization failed; check CUDA and verified assets") from None
    return worker


if __name__ == "__main__":
    build_engine()
