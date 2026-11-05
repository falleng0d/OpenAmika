"""Exercise the gateway's health response without loading the speech models."""
import ast
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("device, expected_name", [
    ("xpu:0", "Intel Arc 140T"),
    ("cuda:1", "NVIDIA second GPU"),
    ("cpu", None),
])
def test_health_names_the_selected_accelerator(device, expected_name):
    path = Path(__file__).resolve().parents[1] / "speech/api.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "health")
    node.decorator_list = []

    def name(selected, required, value):
        assert selected == required
        return value

    # Both backends available: choosing the global first CUDA GPU would be wrong for XPU and cuda:1.
    torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_name=lambda value: name(value, "cuda:1", "NVIDIA second GPU")),
        xpu=SimpleNamespace(is_available=lambda: True,
                            get_device_name=lambda value: name(value, "xpu:0", "Intel Arc 140T")),
    )
    namespace = {"torch": torch, "DEVICE": device, "ASR_LANGUAGE": "en", "APP_NAME": "OpenAmika", "MODEL_DIR": "test-model",
                 "STT_BACKEND": "local"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    health = asyncio.run(namespace["health"]())
    assert health["device"] == device
    assert health["gpu_name"] == expected_name
    assert health["xpu_available"] is True
    assert health["language"] == "en"


def test_remote_health_reports_the_provider_model_without_a_local_gpu():
    path = Path(__file__).resolve().parents[1] / "speech/api.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "health")
    node.decorator_list = []
    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                 "DEVICE": "remote", "ASR_LANGUAGE": "en", "APP_NAME": "OpenAmika",
                 "STT_BACKEND": "openrouter", "STT_MODEL": "microsoft/mai-transcribe-2"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    health = asyncio.run(namespace["health"]())
    assert (health["backend"], health["model"], health["device"], health["gpu_name"]) == (
        "openrouter", "microsoft/mai-transcribe-2", "remote", None,
    )


@pytest.mark.parametrize("language", ["en", "auto"])
def test_conversation_inference_uses_configured_recognition_language(language):
    path = Path(__file__).resolve().parents[1] / "speech/api.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "_asr_sync")
    audio = np.zeros(16000, dtype=np.float32)

    def inference(**options):
        assert options["language"] == language
        assert options["data_in"][0] is audio
        assert options["fs"] == 16000
        return [[{"text": "<|en|>Hello there."}]]

    namespace = {"np": np, "Tuple": tuple, "re": re, "regex": r"<\|.*?\|>", "ASR_LANGUAGE": language,
                 "torch": SimpleNamespace(from_numpy=lambda value: value), "m": SimpleNamespace(inference=inference),
                 "kwargs": {}, "TARGET_FS": 16000}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_asr_sync"](audio) == ("Hello there.", "<|en|>Hello there.")
