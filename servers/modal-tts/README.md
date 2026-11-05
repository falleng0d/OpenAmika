# Private Modal TTS

Offloads GPT-SoVITS synthesis and, optionally, Qwen3-VL-2B vision to the same T4 container. STT, embeddings,
avatars, and the existing OpenRouter LLM stay local.
The local loopback Socket.IO proxy invokes the private `openamika-tts` / `Voice` class through Modal's Python SDK.
There is **no cloud HTTP endpoint** and no browser receives Modal credentials.

## Deploy (explicitly starts billable resources when invoked)

Requires Python 3.11+ and `modal==1.6.3` in the deployment environment. From the repository root, after the user
has authenticated Modal locally and accepted the adult/personal-use asset license:

Modal requires a **valid payment method on file for GPU use**, including L4/T4. Image building can succeed even
when GPU deployment is refused by billing; add the payment method in the Modal dashboard, then rerun deployment
to reuse the cached image. Never provide payment-card details to the CLI or put them in this repository.

```powershell
envs/modal-proxy/Scripts/python.exe -X utf8 -m modal deploy servers/modal-tts/app.py
```

The cloud image build installs the pinned torch/torchaudio 2.9.1 CUDA 12.8 pair and engine requirements into the
image's actual Python interpreter, not an unused venv. It reuses `scripts/setup_tts.py`'s pinned upstream checkout,
pretrained-model downloads, hashes, NLTK data, and Japanese dictionary setup. No local engine installation or GPU
is needed to deploy. Building downloads several GB and can take several minutes; subsequent image layers cache.

The remote methods are:

```python
Voice = modal.Cls.from_name("openamika-tts", "Voice")
worker = Voice()
status = await worker.ready.remote.aio()
# status = {"health": {...}, "characters": {...}, "protocol_version": 1}
audio = await worker.render_piece.remote.aio(payload, piece_index)
description = await worker.describe.remote.aio(jpeg_bytes, prompt)
```

`render_piece` takes the original Socket.IO request dictionary and a zero-based `parse_job(payload).pieces()` index.
It returns the original `Server._render` dictionary, with encoded `audio: bytes`, not a NumPy array or model object.
It validates metadata/options and allows only the configured personal voice; callers cannot select model paths.
No extra class parameters or runtime `.with_options()` variants are needed.

Run the local proxy separately as documented by `proxy.py --help`. The launcher should start that loopback service
instead of the CPU engine; keep `tts_url=http://127.0.0.1:8002` and `tts_character=MekaHime_Amika`.
For Windows, create an isolated environment and install the matching transport dependencies:

```powershell
uv venv --python 3.11 envs/modal-proxy
uv pip install --python envs/modal-proxy/Scripts/python.exe modal==1.6.3 aiohttp==3.14.5 python-socketio==5.17.0 opencv-python-headless pillow fastapi uvicorn websocket-client
envs/modal-proxy/Scripts/python.exe -X utf8 servers/modal-tts/proxy.py --port 8002
```

`-X utf8` prevents Windows console encoding failures when cloud build tools print non-ASCII text.
After deployment has passed real inference checks, adapt the ignored `settings.json`:

```json
{
  "tts_url": "http://127.0.0.1:8002",
  "tts_transport": "modal",
  "tts_timeout_s": 240,
  "services": {
    "tts": {
      "enabled": true,
      "command": ["envs\\modal-proxy\\Scripts\\python.exe", "-X", "utf8", "servers\\modal-tts\\proxy.py", "--port", "8002"],
      "cwd": ".",
      "health": "http://127.0.0.1:8002/ping",
      "timeout_s": 900
    }
  }
}
```

Merge just these fields; preserve the other settings/services. Back up the local file first. To roll back,
stop OpenAmika, restore the prior local voice-service command and `tts_transport: "local"`, then restart.
Do not delete the CPU engine or its environments. The bridge reads Modal's official local authentication profile;
it does not need a repository env file or a token in the command above.

To enable Modal vision, set `vision.enabled` to `true`, `vision.python` to the modal-proxy Python above,
`vision.device` to `modal`, and `vision.model` to `qwen3-vl-2b`. The local vision service forwards camera frames
to the private `Voice.describe` RPC. The remote model loads lazily on the first frame, uses FP16 on the T4, and
caches public model files in the `openamika-vision-cache` Volume. Voice and vision share the same serial worker;
inference can delay voice requests. A stopped camera sends no new frames. The browser requires HTTPS for camera
access from other devices; the included Tailscale Serve script provides tailnet-only HTTPS.

To stop cloud compute explicitly, run `modal app stop --yes openamika-tts` (or use its App ID).
Stopping retains the private model Volume and cached image. To resume after an explicit stop, rerun the deploy
command above **before** `Start_OpenAmika.bat`; the first request will initialize/warm a GPU again. Simply
starting the local bridge does not redeploy a stopped App. Passive health probes do not keep a GPU alive.

The speech integration must warm the cloud worker before connecting chat and use bounded **Modal-only** deadlines.
The proxy allows 900 seconds for initial readiness and 180 seconds per render RPC; configure the gateway's
`tts_timeout_s` to 240 seconds so it can receive bridge failures before its own outer deadline.
The original local 2-second first-audio / 5-second inflight watchdogs remain local defaults. A GPU is not a guarantee
of sub-2-second network-plus-inference latency.

## Assets and credentials

Only the individually named public code/manifest/license files in `build_engine.SOURCE_FILES` are uploaded.
Automatic source inclusion is disabled. Never add a recursive repository/vendor mount: ignored local files include
private voice weights, recordings, environments, credentials, and personal configuration.

The image contains **no Amika assets**. At container startup the standard-library installer downloads only the voice
directly from the official Hugging Face distribution at its pinned revision into a private v2 Modal Volume:

```text
setup_tts.py --assets-only --skip-avatar --skip-paper-doll --accept-voice-license --engine-dir /private-assets
```

The official Hugging Face origin is forced, every voice file retains its pinned size/SHA-256 verification, and the
Volume is committed before loading. The image's public pretrained models use the existing installer integrity pins.
No avatars/paper dolls, local asset uploads, generated WAV files, public asset routes, or model extraction methods
are created. Model/reference caches are ephemeral container files; no memory snapshots are enabled.

The user's accepted license is personal-use/18+ and prohibits redistribution/rehosting. It does not explicitly discuss
private rented-cloud inference; this setup follows the user's personal-inference request, not an explicit additional
license grant. Never share the Volume/image snapshots/assets or offer this worker as a service for other people.
See the [official license](https://huggingface.co/MekaHime/MekaHime_Amika/blob/main/LICENSE).

Keep `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` only in the local backend's protected environment (or Modal's local
configuration). Environment values override `.modal.toml`. Do not put them in settings, CLI arguments, source,
image variables, frontend payloads, or logs. Inject credentials only into the proxy process where practical.
The GPU worker is not given workspace token credentials. Public official downloads do not require `HF_TOKEN`;
private/gated access, if needed, requires an explicitly configured runtime Modal Secret, never a build-time token.

## Privacy, latency, and costs

- Modal documents retention of **RPC Function inputs/results for up to seven days**. This includes camera frames,
  their prompts and descriptions, and synthesized text/audio. RPC is authenticated/private, not zero-retention.
  Container logs have plan-dependent retention.
  The adapter suppresses upstream text/transcript diagnostics and returns sanitized synthesis errors.
- T4, two physical CPU cores and 8 GiB RAM, automatic compute placement (no `region` constraint),
  and Dublin RPC routing (`routing_region="eu-west"`),
  one serial container maximum. Automatic placement avoids the regional compute pricing multiplier and can run
  outside Europe. Dublin routing does not restrict compute placement; regional routing supports `.remote()`/
  `.remote.aio()` (not `.spawn()`). RPC payloads over 2 MiB may still transit `us-east` object storage;
  this is not an EU data-residency guarantee.
- At the prices checked on 2026-11-06, the requested compute costs approximately **$0.749 per running hour**
  before credits, versus $1.210 for the previous EU L4/four-core configuration (about 38% lower).
  CPU and RAM bill at the greater of requested or actual usage, so this is a floor, not a spending cap.
  Startup, synthesis and warm idle time all count. An uninterrupted 300-second idle tail costs about $0.062.
  See [the measured T4 results](../../docs/PERFORMANCE.md#modal-t4-cost-tuning--2026-11-06).
- `min_containers=0`, `buffer_containers=0`, idle scale-down after up to 300 seconds. **Warm GPU idle time is billed.**
  No scheduled keepalive, always-warm minimum, or remote health-poll loop is installed. Volume storage persists.
- Startup budget is 900 seconds (including first official asset download/load/multilingual warmup); each method
  has a 180-second execution budget. Containers can disappear on idle/preemption; readiness does not guarantee
  that the next request cannot cold-start. The local proxy/integration must handle this with bounded deadlines.
- Initialization and synthesis use the original `SingleWorker`; no Modal concurrent-input decorator is enabled.
  CUDA availability and the selected device are checked; CPU fallback fails startup rather than silently slowing down.
- Long input is limited to 8,000 characters; jobs are still serialized. Benchmark warm p50/p95 latency, cold readiness,
  languages/emotions, long audio, interruption, idle recovery, and failures before relying on interactive deadlines.
- `max_containers=1` is a pool bound, not a universal account spending cap. Other Apps/configuration variants can
  incur charges. Rolling redeployments can wait for capacity; use `--strategy=recreate` only when intentional
  interruption is acceptable. Deployment does not modify local services or choose a different LLM.

The Nemotron example's reusable patterns are persistent Volume caching, nearby compute/routing, and completed
inference warmup before readiness. Its B200, LLM, concurrency, and unauthenticated HTTP exposure are not used here.

Sources: [pricing](https://modal.com/pricing), [resource billing](https://modal.com/docs/guide/resources),
[Modal 1.6.3](https://modal.com/docs/sdk/py/changelog),
[private lookups](https://modal.com/docs/guide/trigger-deployed-functions),
[regions](https://modal.com/docs/guide/region-selection),
[retention](https://modal.com/docs/guide/security),
[cold starts](https://modal.com/docs/guide/cold-start),
[Nemotron example](https://modal.com/docs/examples/nemotron_inference).
