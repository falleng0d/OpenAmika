"""Backend selection is tested without loading proprietary voice assets or torch."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "vendor/tts-engine/main.py"
SPEC = importlib.util.spec_from_file_location("tts_devices_under_test", PATH)
voice = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = voice
SPEC.loader.exec_module(voice)


def select(monkeypatch, requested, *, xpu=False, cuda=False, mps=False, half=None):
    backend = lambda available: SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=backend(cuda), xpu=backend(xpu), backends=SimpleNamespace(mps=backend(mps))))
    engine = voice.VoiceEngine(SimpleNamespace(device=requested, half=half), {}, None)
    return engine._pick_device()


def test_intel_requires_explicit_selection_and_fp32(monkeypatch):
    assert select(monkeypatch, "auto", xpu=True) == ("cpu", False)
    assert select(monkeypatch, "xpu:0", xpu=True) == ("xpu:0", False)
    # Prevent the measured Unsupported dtype Half failure inside speaker verification.
    assert select(monkeypatch, "xpu", xpu=True, half=True) == ("xpu", False)


def test_missing_intel_backend_falls_back_to_cpu(monkeypatch):
    assert select(monkeypatch, "xpu", half=True) == ("cpu", False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace()))
    engine = voice.VoiceEngine(SimpleNamespace(device="xpu", half=None), {}, None)
    assert engine._pick_device() == ("cpu", False)


@pytest.mark.parametrize("options, expected", [
    ({"cuda": True, "xpu": True}, ("cuda", True)),
    ({"mps": True, "xpu": True}, ("mps", False)),
    ({}, ("cpu", False)),
])
def test_auto_preserves_existing_backend_priority(monkeypatch, options, expected):
    assert select(monkeypatch, "auto", **options) == expected
