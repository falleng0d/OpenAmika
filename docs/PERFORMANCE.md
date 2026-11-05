# Measuring local performance

`scripts/benchmark_local.py` exercises the running speech-recognition, voice and memory services directly.
It does not call the language model, read credentials, change settings, or add turns to the companion's memory.
Keep the companion idle and avoid other inference workloads during a run.

```powershell
envs\speech\Scripts\python.exe scripts\benchmark_local.py --download-samples --repeats 5
```

The first run downloads two small public speech clips, checked against SHA-256 hashes in the script:

- John F. Kennedy's inaugural-address excerpt from [OpenAI Whisper's tests](https://github.com/openai/whisper/blob/main/tests/test_transcribe.py).
- LibriSpeech test-other `1688-142285-0007`, with the reference transcript from the
  [PyTorch audio CTC tutorial](https://github.com/pytorch/audio/blob/main/examples/tutorials/asr_inference_with_ctc_decoder_tutorial.py).

Downloaded inputs, generated voice clips and JSON reports stay in git-ignored `logs/performance/`.
Do not publish Amika's generated/downloaded assets as part of a benchmark bundle.

To measure just one component, or an experimental server on a different port:

```powershell
envs\speech\Scripts\python.exe scripts\benchmark_local.py --components asr --repeats 20 --output logs\performance\asr.json
envs\speech\Scripts\python.exe scripts\benchmark_local.py --components tts --tts-url http://127.0.0.1:18002 --repeats 5 --output logs\performance\tts-experiment.json
```

Only HTTP loopback service addresses are accepted. The existing speech environment supplies the dependencies;
the voice client uses Socket.IO's async WebSocket transport, like the application.

## What is measured

- **ASR:** local upload/decode/resample/inference round-trip; the server's inference time is reported separately.
  Both samples use automatic language detection and inverse text normalization, like the conversation path.
  Case/punctuation-insensitive word error rate (WER) is checked against the reference transcript.
- **TTS:** request-to-completed-audio latency, first returned audio, decoded audio duration and real-time factor
  (RTF = elapsed time / audio duration). English, Japanese and Chinese phrases exercise three emotion references.
  Requests have a fixed seed, but backend implementations and warm-up/cache state can still change output.
  WAVs are decoded and checked for nonempty, finite, non-silent audio, then saved for listening.
- **Memory:** single-query embedding round-trip and server time; returned vectors must have the declared
  dimensions, finite values and unit norm.

Each case gets one excluded warm-up by default. Reports retain every trial and any failure; the process exits
nonzero if any trial fails. Summaries include measured successes/errors, median and nearest-rank p95. With only
three measurements, p95 is simply the maximum, not a reliable tail-latency estimate. Do not hide failed trials
by quoting only successful timings.

These numbers **exclude** microphone capture, silence/VAD endpointing, cloud LLM generation and browser playback.
They are not end-to-end voice-chat latency. Service queueing and competing workloads can increase them.
Two English samples do not establish recognition quality for other languages, accents or background noise.
Finite audio is not proof of intelligible or natural speech: listen to the saved clips before enabling a backend.

## Initial CPU baseline — 2026-11-05

Windows 11, Core Ultra 9 285H, Intel Arc 140T, approximately 31 GiB system RAM. All local models on CPU;
vision disabled, cloud language model unchanged. Core PyTorch `2.10.0+cpu`; voice PyTorch `2.9.1+cpu`.
The installed stack remained running. One warm-up and three measured requests per case, sequential;
ASR/embedding and TTS were collected in separate runs.

| Case | Median request time | Maximum (3 trials) | Quality / output |
|---|---:|---:|---|
| JFK, 11.00 s speech | 570 ms | 616 ms | 0% WER |
| LibriSpeech, 7.06 s speech | 404 ms | 453 ms | 4.76% WER; `part` instead of `parts` |
| bge-m3 query | 59 ms | 69 ms | 768-dimensional normalized vector |
| English, neutral | 2.316 s | 2.343 s | 2.62 s audio; median RTF 0.884 |
| Japanese, happy | 3.430 s | 4.823 s | 2.28 s measured audio; median RTF 1.504 |
| Chinese, caring | 5.507 s | 5.513 s | 3.71 s measured audio; median RTF 1.484 |

All 18 measured component requests in the combined baseline succeeded. These are a small operational baseline,
not a statistical performance guarantee. Raw machine-local report: `logs/performance/cpu-combined-baseline.json`.

## Evaluating Intel acceleration

Keep the CPU environments intact. Use separate experimental environments and ports, and stop the normal stack
when needed to avoid memory pressure or benchmark interference. Do not replace the model or reduce recognition
quality merely to obtain a faster number.

An accelerator must execute real inference, not merely report availability or load weights. Synchronize its
device around direct model timings. Compare recognition output/embedding quality against CPU, repeat multiple
phrases and emotion references, check errors and audio quality, and test a fresh service restart. Report initial
compilation/loading separately from warmed requests. Enable only services that actually pass these checks;
keep a documented CPU rollback for failed or slower paths.

### Arc 140T results and selected configuration

Official XPU wheels executed real inference on this machine, not just device detection. Separate environments
used PyTorch/torchaudio `2.10.0+xpu` for core experiments and `2.9.1+xpu` for voice experiments. Existing CPU
environments and downloaded model/voice assets were retained unchanged.

| Service | Measured result | Selection |
|---|---|---|
| SenseVoiceSmall, fp32 | JFK 307 ms median / 321 ms p95; LibriSpeech 114 / 251 ms (12 trials each) | **XPU** |
| bge-m3, fp16 | Short-query HTTP 82 ms median versus CPU 59 ms | CPU/fp32 |
| GPT-SoVITS, fp32 | English/Japanese/Chinese medians 4.36 / 4.21 / 4.88 s; no consistent speed win | CPU/fp32 |

An independent ASR service restart gave 308 ms JFK and 272 ms LibriSpeech medians (three trials each).
Do not promise the best 114 ms short-sample result on every run. Thirty measured HTTP recognitions and sixteen
direct XPU recognitions completed without errors or transcript changes. WER matched CPU on both samples.
The first recognition after a cold load took about 3.3 seconds; report this separately from warmed timings.
VAD remains on CPU. This is repeatability evidence on a small corpus, not a long-duration stability guarantee.
The local installation subsequently fixes conversation recognition to English with `"speech_language": "en"`
in `settings.json`. The default remains `auto` for other installations. Restart after changing this setting;
`/health` reports the selected recognition language. An explicit `lang` on the standalone ASR upload API can
still override its default (the benchmark sends `auto` to preserve comparability with the original baseline).

Embedding batches accelerated, but the actual interactive short-query endpoint did not; a batch speedup alone
does not justify switching that service. XPU/fp32 embedding cosine similarity to CPU exceeded 0.999999999998.

Voice experiments completed 45 successful syntheses across fp32 and experimental mixed precision. Unmodified
fp16 failed during speaker verification (`torch.fft.rfft`: `Unsupported dtype Half`). Explicit XPU voice selection
therefore forces fp32; automatic CUDA/MPS/CPU selection is unchanged. A mixed-precision prototype worked but was
not shipped. The CPU HTTP baseline and XPU direct synthesis differed in request scope, seed and generated lengths;
the voice numbers are **not paired per-utterance speedup ratios**. No human listening/MOS evaluation was performed.

The tested machine-local settings select only the accelerated speech interpreter:

```json
{
  "speech_python": "envs\\xpu-probe\\Scripts\\python.exe",
  "speech_device": "xpu"
}
```

`envs/xpu-probe` contains additional dependencies used in the three-model investigation. The setup scripts do not
automatically install XPU. To reproduce that tested experimental environment (requires `uv`):

```powershell
envs\brain\Scripts\python.exe -m venv envs\xpu-probe
uv pip install --python envs\xpu-probe\Scripts\python.exe torch==2.10.0+xpu torchaudio==2.10.0+xpu --index-url https://download.pytorch.org/whl/xpu
uv pip install --python envs\xpu-probe\Scripts\python.exe -r vendor/tts-engine/requirements-tts.txt -r speech/requirements.txt -r brain/requirements-embedder.txt
```

Keep embeddings' interpreter/device and the TTS service command unchanged. Back up `settings.json` before editing;
the local backup for this rollout is under `logs/performance/settings-before-xpu-*.json`.

**Rollback:** stop OpenAmika, restore `speech_python` to `envs\speech\Scripts\python.exe` and `speech_device` to
`cpu` (or restore the settings backup), then run `launch.py --check` and restart. No reinstall is needed. Check
`http://localhost:53000/health` after either configuration: `device` reports the actually loaded backend, not just
the requested one; an accelerated run reports `xpu:0`, `xpu_available: true` and the Intel GPU name.

Detailed probe methodology and local evidence files: `logs/performance/intel-xpu-findings.md`. Generated clips and
full per-trial reports remain ignored; only the benchmark code and summarized results belong in the branch.

### Full-stack follow-up

With XPU speech plus CPU voice/embeddings running together, five measured trials per case gave ASR medians of
349 ms (JFK) and 274 ms (LibriSpeech), with the same transcripts/WER as CPU. All thirty measured requests
succeeded. This run still used automatic ASR language selection for a comparable baseline.

CPU voice medians were slower than the initial baseline: English 5.23 s, Japanese 5.83 s, Chinese 9.57 s;
one Chinese request took 41.24 s. The embedding median was 174 ms. These results do **not** establish a voice
performance improvement, or prove that resident XPU speech caused the slowdown. Machine load, shared-memory
pressure, power/thermal state and service queueing were not experimentally controlled across the two runs.
Retain the outlier in the report; investigate this full-stack variability before claiming an end-to-end win.
Raw report: `logs/performance/mixed-xpu-rollout.json`.

## Modal L4 TTS — 2026-11-05

Private class RPC in Europe, NVIDIA L4, existing GPT-SoVITS v2ProPlus engine and Amika voice, fp16.
One warm-up and five measured requests per language through the actual loopback Socket.IO bridge, including
network/RPC and audio encoding/transfer. No local TTS engine process is needed for this backend.

| Case | CPU median (initial baseline) | Modal median / p95 | Modal generated audio |
|---|---:|---:|---:|
| English, neutral | 2.316 s | 0.486 / 0.584 s | 1.88 s |
| Japanese, happy | 3.430 s | 0.609 / 0.640 s | 2.61 s |
| Chinese, caring | 5.507 s | 0.740 / 0.768 s | 3.11 s |

All 15 measured requests succeeded with finite, non-silent WAV output. The same input phrases/seed were used,
but CUDA and CPU produced different audio lengths, so these are request-latency comparisons, not paired
per-utterance rendering speedup ratios. First per-case requests took approximately 0.89 / 1.14 / 1.23 s.
The very first cloud readiness call took **120.39 s**, including the initial private official voice download;
model load was 18.29 s and multilingual warm-up 21.54 s. Do not confuse warm timings with cold-start latency.

A separate operational probe used the exact production `DialogSession` queue and real bridge/worker to synthesize
the user-reported long English reply. All **11 sentences** reached the browser audio boundary in index order,
with no errors: first audio after 0.98 s; all audio generated after 8.38 s. This verifies gateway delivery, not
actual browser speaker playback or human-perceived voice quality. The probe substituted only an in-memory brain
and capturing browser boundary, so no extra LLM calls or conversation-memory writes were made.

Reports and private generated audio stay in ignored `logs/performance/`: `modal-cold-ready.json`,
`modal-tts-baseline.json`, and `modal-full-reply.json`. Deployment/setup, cost bounds, retention, and CPU rollback
are documented in [`servers/modal-tts/README.md`](../servers/modal-tts/README.md).

After switching the real stack, its health endpoints confirmed cloud CUDA/L4 voice and local XPU/English
recognition. All 11 saved reply clips produced nonempty recognition results in an additional English ASR sanity
check. Several words differed (especially isolated short words); this does not distinguish recognizer errors
from TTS pronunciation defects and is not evidence of perceptual equivalence to CPU. Human listening is still
needed. The complete private results are in `logs/performance/modal-english-quality.json`.

## Modal T4 cost tuning — 2026-11-06

The selected deployment now requests **T4, two physical CPU cores, 8 GiB RAM**, with automatic compute placement
(no `region` constraint). Dublin RPC routing is retained; it does not constrain GPU placement to Europe.
The same pinned GPT-SoVITS v2ProPlus engine, Amika weights, fp16, serial worker and private Socket.IO bridge are used.
No model conversion, quantization, snapshots or local settings changes were made.

One excluded warm-up and three measured requests per case, with seed 123 and WAV output:

| Case | Median request time | Maximum (3 trials) | Median generated audio |
|---|---:|---:|---:|
| English, neutral | 0.729 s | 0.777 s | 2.33 s |
| Japanese, happy | 0.737 s | 0.745 s | 2.66 s |
| Chinese, caring | 0.872 s | 0.934 s | 3.11 s |
| Long English, neutral | 2.373 s | 2.381 s | 18.69 s |
| Long English, happy | 2.179 s | 2.253 s | 19.39 s |

All 20 syntheses (15 measured plus five warm-ups) returned finite, non-silent audio, without errors, warnings or
dropout retries. These are completed-audio latencies, not browser playback or end-to-end conversation times.
The long cases submit an entire multi-sentence English passage as one turn-mode request; the normal gateway
submits sentences separately. Three repetitions do not establish tail-latency reliability or perceptual quality.
No new human listening evaluation was performed.

Cold readiness took **45.12 s**, including allocation/container startup, asset verification, imports and warmup.
The worker reported 13.87 s model load and 17.48 s multilingual warmup. After the requests, PyTorch reported
1,830 MiB allocated, 2,574 MiB reserved and 2,033 MiB peak allocated VRAM. These are allocator statistics,
not total device memory usage; no host RAM/CPU utilization measurements were obtained.

The contemporaneous automatic-placement L4/four-core and L4/two-core attempts both failed readiness at the
150-second benchmark deadline, before audio tests. Earlier Europe-constrained L4 attempts also encountered
capacity waits, including a 900-second readiness timeout. The successful historical L4 timings above remain
useful context, but different placement, availability and audio lengths prevent a controlled speed/cost comparison.
The T4 selection is based on demonstrated fit and interactive warm latency at a lower requested running rate,
not a claim that it is as fast as L4 or has a measured lower cost per generated audio second.

At Modal prices checked that day, the requested running-rate floor falls from approximately $1.210/hour
(EU L4, four cores, 8 GiB) to $0.749/hour (automatic T4, two cores, 8 GiB), about **38% lower** before credits.
Actual CPU/RAM usage can increase the bill. Retain `min_containers=0`, `buffer_containers=0`, `max_containers=1`
and the 300-second idle window: with a measured 45-second cold start, aggressively shortening retention could
interrupt conversation. RAM was not reduced without utilization evidence. No one-core configuration completed
inference testing, so it was not selected.

Raw reports and private generated clips remain ignored under `logs/performance/modal-cost-20261106/`:
`auto-t4-cpu2-audio.json`, its `-resources.json` companion and `comparison.json`. Earlier resource-only/failed
wrapper attempts are not synthesis performance evidence. The app was explicitly stopped after testing and then
verified to have zero running containers. The T4/two-core source defaults take effect on the next deployment;
the successful cloud test used equivalent runtime resource overrides on the automatic-placement deployment.
