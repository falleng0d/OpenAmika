"""Private, lazy-loaded vision inference sharing the TTS worker's T4."""
import io
import os
from pathlib import Path
import subprocess
import sys


class VisionWorker:
    def __init__(self, cache_dir: str):
        os.environ["HF_HOME"] = cache_dir
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        repo = "Qwen/Qwen3-VL-2B-Instruct"
        hub_cache = str(Path(cache_dir) / "hub")
        marker = Path(cache_dir) / "qwen3-vl-2b.complete"
        if not marker.is_file():
            # The TTS engine forces Hugging Face offline in this process. Fetch the
            # public vision model in a separate process without changing that policy.
            env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
            code = ("from huggingface_hub import snapshot_download; "
                    "snapshot_download('Qwen/Qwen3-VL-2B-Instruct', cache_dir=" + repr(hub_cache) +
                    ", ignore_patterns=['onnx/*','*.onnx','*.onnx_data','*.gguf','*.msgpack','*.h5','*.ot','*.tflite'])")
            subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=720,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(repo, cache_dir=hub_cache, local_files_only=True,
                                                       min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 128)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, cache_dir=hub_cache, local_files_only=True,
                                                                torch_dtype=torch.float16).to("cuda").eval()
        marker.touch()

    def describe(self, jpeg: bytes, prompt: str) -> str:
        if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= 2 * 1024 * 1024:
            raise ValueError("Invalid image")
        if not isinstance(prompt, str) or not 0 < len(prompt) <= 4000:
            raise ValueError("Invalid prompt")
        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as image:
            image.thumbnail((640, 480))
            image = image.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        return self.processor.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
