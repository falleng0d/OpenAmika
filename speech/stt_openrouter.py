"""Recorded-utterance transcription through OpenRouter; VAD stays on this machine."""
import asyncio
import base64
from io import BytesIO

import aiohttp
import numpy as np
import soundfile as sf


API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"


class TranscriptionError(RuntimeError):
    """Safe to show to the user: never includes credentials or upstream response bodies."""


async def transcribe(audio: np.ndarray, *, api_key: str, model: str,
                     language: str = "auto", timeout_s: float = 30) -> tuple[str, str]:
    if not api_key.strip():
        raise TranscriptionError("OpenRouter transcription needs OPENROUTER_STT_API_KEY (or OPENROUTER_API_KEY).")
    if audio.size == 0:
        return "", ""
    wav = BytesIO()
    sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")
    payload = {
        "model": model,
        "input_audio": {"data": base64.b64encode(wav.getvalue()).decode("ascii"), "format": "wav"},
    }
    if language not in ("auto", "nospeech", ""):
        payload["language"] = language
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as client:
            async with client.post(API_URL, json=payload,
                                   headers={"Authorization": f"Bearer {api_key}"},
                                   allow_redirects=False) as response:
                if response.status != 200:
                    raise TranscriptionError(f"OpenRouter transcription failed (HTTP {response.status}).")
                result = await response.json()
    except asyncio.TimeoutError:
        raise TranscriptionError("OpenRouter transcription timed out. Please try again.") from None
    except (aiohttp.ClientError, ValueError):
        raise TranscriptionError("OpenRouter transcription returned an invalid response or could not be reached.") from None
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise TranscriptionError("OpenRouter transcription returned no transcript.")
    text = result["text"].strip()
    # Remote models do not supply SenseVoice emotion/language/event tags.
    return text, text
