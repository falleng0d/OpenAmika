# TTS: the contract, the voice server, and the voice adapter

**Her voice is one command away.** `envs\brain\Scripts\python.exe scripts\setup_tts.py` (after the core setup,
`powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`; any Python 3.11 runs it) installs Amika's voice
and her Live2D model from Hugging Face
(personal-use license, 18+; the script shows it and asks you to accept it first) and the OpenAmika voice server
(`vendor/tts-engine`, a MekaHime build of GPT-SoVITS, MIT), and `settings.example.json` already has the launcher
start that server on port 8002. Without a voice server she still works in text: her replies appear as live
subtitles and the Live2D avatar still acts them out, but her voice is an essential part, so the launcher's
`TTS (voice)` row and the health light show it red with the command that installs it.

## The contract the speech layer speaks

OpenAmika's speech layer is a **Socket.IO client** to the TTS server (`settings.json → tts_url`).
With the default settings (voice adapter ON → old-vocabulary voice ids), the wire looks like:

```jsonc
// emit("tts_request_gateway", …)
{
  "task_id":      "uuid",
  "turn_id":      "<brain turn>-<idx>",
  "text":         "【happy2】.今日もおつかれさま。",   // emotion tag first, then the sentence
  "character_id": "MekaHime_Amika",                      // settings.json → tts_character (adapter may swap it)
  "audio_format": "wav",                                  // browsers decode containers, not raw opus
  "language":     "Japanese"                              // optional: Chinese | English | Japanese
}

// on("audio_ready")   — one per request (filter by task_id)
{ "task_id", "turn_id", "audio": <bytes>, "audio_format", "audio_duration", "text", "is_final": true }
// plus "task_queued" (accepted) and "error" { task_id, turn_id, error }
```

- One request per sentence; `turn_id` lets a server keep one turn's sentences in order.
- `GET /ping` answering `ok` is what the launcher and the Setup page poll for health.
- `audio` is the encoded file (WAV by default); `audio_duration` is in seconds.

## The OpenAmika voice server

OpenAmika ships its own: the **OpenAmika voice server** (`vendor/tts-engine/main.py`, a MekaHime build of
GPT-SoVITS v2ProPlus, MIT), derived from the OpenSakura voice server. One command installs it with Amika's voice:

```text
envs\brain\Scripts\python.exe scripts\setup_tts.py     type yes to accept her license; --cpu without an NVIDIA GPU
```

- First Amika's voice, her Live2D model and the paper doll that stands in for her helpers, from the Hugging Face
  repository [MekaHime/MekaHime_Amika](https://huggingface.co/MekaHime/MekaHime_Amika) (one license for all three),
  file by file with the standard library and checked against `scripts/MekaHime_Amika.manifest.json` (size and
  SHA-256 of all 250 files at the pinned revision, and of the paper doll's 70 at its own): the voice into
  `vendor/tts-engine/Speakers/MekaHime_Amika/`, the model into `avatar/models/Amika/` (`--skip-avatar` leaves it
  out), the paper doll into `avatar/models/Host/` (`--skip-paper-doll`). Then it builds the rest of `vendor/tts-engine/` from public
  sources: a Python 3.11 venv with CUDA 12.8 torch, upstream GPT-SoVITS at a pinned commit with its pretrained
  models. Every download is pinned by size and SHA-256; a re-run verifies and skips what is already there, and
  `--verify` checks everything without downloading.
- `settings.example.json` lists it under `services.tts`, so the launcher starts it on port 8002 with the rest and
  waits for `GET /ping` (models plus four warm-up lines, about 20 s on a GPU). `tts_url` points at it and
  `tts_character` (`MekaHime_Amika`) names the voice.
- It needs about 2.5 GB of VRAM; `--cpu` installs a CPU build for machines without an NVIDIA GPU (slower).
- Flags, the request contract in full, how it uses upstream and measurements: `vendor/tts-engine/README.md`.

**One voice ships:** `MekaHime_Amika`, an old-vocabulary voice (the 67 V1 ids plus `happy4` and `sad4`, 69 clips).
Any other voice is yours to bring: a speaker folder in `vendor/tts-engine/Speakers/` (`character.json`, the
`.ckpt` / `.pth` weights, `references/emotions.json`), or any other server that speaks the contract above: point
`tts_url` at it and describe it under `services.tts` (command, cwd, health URL) so the launcher starts it with
the rest.

## Private Modal voice backend

See [`servers/modal-tts/README.md`](../servers/modal-tts/README.md) for deploying the private GPU worker and
loopback Socket.IO bridge. Set `tts_transport` to `modal` only when `services.tts` launches that bridge.
The browser still talks only to localhost; credentials stay in the bridge's Modal SDK profile, not in settings
or browser messages. Transcription, memory, avatar rendering and the OpenRouter brain remain unchanged.

The gateway waits for the bridge's actual GPU readiness before opening the cloud-backed dialog. Passive local
health requests do not keep a GPU alive. After an idle scale-down, synthesis can incur a new cold start; Modal
requests therefore use a separate bounded `tts_timeout_s` budget (default 240 seconds, outside the bridge's
180-second RPC budget). Initial asset/model readiness has a separate 900-second limit. The local backend's
original 2-second first-audio and 5-second in-flight timers are unchanged. Cloud mode does not invoke the local
first-audio fallback that clears the queue; it protects the turn when real audio arrives. An expired cloud
request reports `tts_error`, and late results still cannot replace a newer turn or sentence.

Model files are fetched from their official source directly into a private cloud volume. Personal assets and
credentials are not uploaded from this checkout or baked into the container image. Modal's RPC retention
policy applies to spoken-text inputs and generated-audio outputs; use this only for your personal voice backend.

## Emotion vocabularies: internal vs. old-style voices

- **Internal (V2)**: 69 `register_emotion` keys (`casual_happy`, `whisper_caring`, …). This is the
  contract the brain's templates, both streaming parsers and the Live2D banks are built on; it never
  changes regardless of which voice speaks.
- **Old-style voice ids (V1)**: 67 flat ids with intensity numbers (`happy2`, `tsun3`, `whisper1`, …),
  the vocabulary of MekaHime's GPT-SoVITS voices, Amika's included.

## The voice adapter (default: ON)

`speech/voice_adapter.py` bridges the internal vocabulary to V1-style voice ids and is **enabled by
default** (defaults in code; `config/voice_map.json` is written when you change it on the Setup page):

- the brain **always** emits the 69-key tag; the adapter rewrites **only the leading tag** of the
  outgoing TTS text into an old id via a validated 69→67 map, and swaps `character_id` to the configured
  voice;
- the `emotion` reported to the browser/avatar **stays the new key**, so face and voice never disagree;
- idempotent (an already-old tag passes through; untagged text gets `【neutral1】`);
- the map is hot-reloaded (mtime-gated, last-good on parse errors) and validated cell-by-cell on read; a
  stale or hand-edited file degrades safely to the defaults;
- Setup → Voice gives you the per-cell grid (changed cells highlight), enable/disable, and reset.

**Your own voice or server:** if it understands the 69 internal keys directly (or ignores tags), switch the
adapter off on Setup → Voice and set `tts_character` in `settings.json` to your voice id (that value is used
whenever the adapter is off). If it uses flat ids like `happy2`, keep the adapter on, set its voice on Setup →
Voice (the adapter sends its own voice id, default `MekaHime_Amika`) and tune the grid by ear; remapping is a
voice-acting decision per voice.
