"""Private OpenAmika TTS RPC. Deploy: modal deploy servers/modal-tts/app.py."""
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_engine import ROOT, SOURCE_FILES, download_voice, start_worker

REMOTE_ROOT = "/opt/openamika"
ASSETS_PATH = "/private-assets"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgomp1", "build-essential")
    .env({"HF_ENDPOINT": "https://huggingface.co", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
          "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": f"{REMOTE_ROOT}/servers/modal-tts"})
)
for relative in SOURCE_FILES:
    image = image.add_local_file(ROOT / relative, f"{REMOTE_ROOT}/{relative}", copy=True)
image = image.run_commands(f"python {REMOTE_ROOT}/servers/modal-tts/build_engine.py")
image = image.pip_install("torchvision==0.24.1", "qwen-vl-utils==0.0.14")

assets = modal.Volume.from_name("openamika-personal-voice", create_if_missing=True, version=2)
vision_cache = modal.Volume.from_name("openamika-vision-cache", create_if_missing=True, version=2)
app = modal.App("openamika-tts", include_source=False)


@app.cls(
    image=image, gpu="T4", cpu=2, memory=8192,
    volumes={ASSETS_PATH: assets, "/vision-cache": vision_cache},
    routing_region="eu-west",
    min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=300,
    timeout=180, startup_timeout=900, include_source=False,
)
class Voice:
    @modal.enter()
    def startup(self):
        import os
        os.environ["HF_HOME"] = "/vision-cache"
        download_voice(Path(ASSETS_PATH))
        assets.commit()
        self.worker = start_worker(Path(ASSETS_PATH))

    @modal.method()
    def ready(self) -> dict:
        return self.worker.ready()

    @modal.method()
    def render_piece(self, payload: dict, piece_index: int) -> dict:
        return self.worker.render_piece(payload, piece_index)

    @modal.method()
    def describe(self, jpeg: bytes, prompt: str) -> str:
        if not hasattr(self, "vision"):
            from vision_worker import VisionWorker
            self.vision = VisionWorker("/vision-cache")
            vision_cache.commit()
        return self.vision.describe(jpeg, prompt)

    @modal.exit()
    def shutdown(self):
        if hasattr(self, "worker"):
            self.worker.close()
