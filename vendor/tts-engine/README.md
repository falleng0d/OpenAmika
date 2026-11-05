# vendor/tts-engine: the OpenAmika voice server

Amika speaks through `main.py`, the **OpenAmika voice server, a MekaHime build of GPT-SoVITS (v2ProPlus)**,
derived from the OpenSakura voice server ([zeekk0/OpenSakura](https://github.com/zeekk0/OpenSakura), MIT). It is
a small MIT-licensed server on top of an unmodified upstream [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
checkout. Everything else in this folder is assembled from public sources by `scripts/setup_tts.py`, and Amika's
voice comes from a Hugging Face repository. Only `main.py`, `requirements-tts.txt`, this README, `LICENSE` and
`THIRD_PARTY_NOTICES.md` are in git.

## Install

```powershell
envs\brain\Scripts\python.exe scripts\setup_tts.py          # Windows 10/11 with an NVIDIA GPU (CUDA 12.8 torch): tested
envs\brain\Scripts\python.exe scripts\setup_tts.py --cpu    # no GPU: CPU torch (works, slower)
envs\brain\Scripts\python.exe scripts\setup_tts.py --verify # check every file; download and install nothing
```

(`envs\brain` comes from the core setup, `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`; any
Python 3.11 can run the script, for example `py -3.11 scripts\setup_tts.py`.)

Then start OpenAmika (double-click `Start_OpenAmika.bat`, or `.\Start_OpenAmika.bat` in PowerShell);
`<brain python> launch.py --check` shows the **TTS (voice)** row.

**Linux and macOS are untested.** The script has Linux (NVIDIA GPU or `--cpu`) and Apple Silicon (CPU torch
with MPS) paths, but neither has been run yet; see [Platforms](#platforms).

Needs Python 3.11 or 3.12 (3.11 is the tested one: the script takes the first 3.11 it finds, itself,
`py -3.11` or `python3.11`, and falls back to 3.12; 3.10 and 3.13 cannot install the pinned packages; the brain
environment's Python 3.11 works too), about 9 GB of disk, about 2.5 GB of free VRAM, and git (optional: without
it the pinned commit is fetched as a zip). On Linux, glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL /
Rocky 8+); on Ubuntu 22.04, whose `python3` is 3.10, install `python3.11` and `python3.11-venv` first. On a Mac,
macOS 14 or newer on Apple Silicon with a native arm64 Python (not one running under Rosetta). The script checks
all of this before installing anything. RTX 50-series cards need the CUDA 12.8 build, which is the default. On
Windows, keep the folder path short (under about 100 characters): even with long paths enabled, Windows cannot load
some DLLs deep in the venv from a longer folder (`The filename or extension is too long`).

**ffmpeg** is needed only for Opus output (`audio_format: "opus"`; OpenAmika asks for WAV). The script downloads
a build on Windows and Linux x86_64 (on macOS: `brew install ffmpeg`); without it, Opus requests get WAV and a
`warning`. WAV output and speed changes do not need it.

**Amika's voice** is the Hugging Face repository
[MekaHime/MekaHime_Amika](https://huggingface.co/MekaHime/MekaHime_Amika). No account or token is needed once
the repository is public; before that the script needs a read token in `HF_TOKEN` (or a saved `hf auth login`),
which it sends to huggingface.co only, never to the CDN it redirects to nor to a mirror, and never prints or stores.
A refusal says what to do (`Hugging Face refused the download (HTTP 401): MekaHime/MekaHime_Amika is not public yet.
…`). `HF_ENDPOINT` picks a mirror, as in huggingface_hub (every file is still checked). It is under MekaHime's
personal-use license (`LICENSE` in that repository: personal, hobbyist use, 18+, no training on it, no re-hosting
or resale, visible credit): the script shows it in short before anything else and downloads the voice only after
you type `yes`; `--accept-voice-license` accepts it up front for unattended installs. Without it the script
finishes everything else and says how to get the voice. The voice is downloaded first, before the venv and
torch, with the standard library only, at a pinned revision of that repository (`a3934e5a`, 69 emotion tags), and
every one of its files is checked by size and SHA-256 against `scripts/MekaHime_Amika.manifest.json` (the full list
of the repository at that revision, 250 files); an interrupted download resumes from its `.part` file. The same
repository, under the same license, holds **her Live2D model** (`live2d/Amika`), downloaded and checked the same way: the
script puts it into `avatar/models/Amika/`, where the Use page finds it (`--skip-avatar` leaves it out and the
stage then says "No Live2D model yet"; `--update-avatar` downloads it again and keeps a `banks.json` you edited in
Setup, while an unedited one follows a newer pin), and **the paper doll** that stands in for her helpers
(`live2d/Host`, pinned at its own revision, `PAPER_DOLL_REV`), which goes to `avatar/models/Host/`
(`--skip-paper-doll` leaves it out). After the downloads the script checks that every tag OpenAmika's
voice adapter can send (`V1_IDS` in
`speech/voice_adapter.py`, 67 tags) has a clip, and whether `settings.json` starts this server with that voice; it
never edits `settings.json`.

Re-running is safe: finished steps are verified (SHA-256) and skipped; a re-run of a complete install takes about
35 s. A venv an earlier run left unfinished (no pip, or an interpreter that no longer starts) is created again;
an existing GPT-SoVITS checkout must still have the pinned `TTS.py`, and other locally edited upstream files are
listed as a warning. A step that fails is reported and the steps that do not need it still run; the summary at
the end lists every step (`OK`, `--` skipped or optional, `!!` failed) and the exit code is non-zero if any
failed. Useful flags: `--verify` (check every file of her voice, her Live2D model and the server; download and
install nothing), `--assets-only` (her voice and Live2D model only), `--from-dir <folder>` (both from a local copy
laid out like the repository, still checked file by file), `--skip-voice`, `--voice-from <folder>` (copy a voice you
already have), `--update-voice`, `--voice-revision <rev>` (another revision of the voice repository; its files are
then not checked against the pins), `--skip-avatar`, `--update-avatar`, `--avatar-dir <path>`, `--skip-ffmpeg`,
`--reinstall`, `--engine-dir <path>`, `--python <interpreter>`, `--self-test` (starts the server once and speaks
a line the way OpenAmika asks for one).

What it builds:

```
vendor/tts-engine/
  main.py                    the server (this repository, MIT)
  requirements-tts.txt       its Python packages (inference subset of upstream's list, see below)
  .venv/                     Python venv: torch + torchaudio 2.9.1 (+cu128 / +cpu) and requirements-tts.txt
  GPT-SoVITS/                RVC-Boss/GPT-SoVITS @ b4570b876f5c (2026-10-19), MIT, unmodified
    GPT_SoVITS/pretrained_models/chinese-hubert-base/            HuBERT speech features   (lj1995/GPT-SoVITS)
    GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/  BERT text features       (lj1995/GPT-SoVITS)
    GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt  speaker embedding for v2Pro models
    GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.bin     fastText language ID
    GPT_SoVITS/text/G2PWModel/                                   Chinese polyphone model  (XXXXRT/GPT-SoVITS-Pretrained)
    GPT_SoVITS/text/ja_userdic/user.dict                         Japanese user dictionary, built by upstream's own code
  nltk_data/                 English POS tagger + CMUdict
  ffmpeg/                    ffmpeg + ffprobe (Windows: gyan.dev 8.0.1; Linux: BtbN static), only for Opus output
  Speakers/MekaHime_Amika/   Amika's voice: MekaHime_Amika.ckpt (GPT), MekaHime_Amika.pth (SoVITS),
                             character.json, references/emotions.json + one reference clip per emotion
  cache/                     derived files the server writes (padded clips, prompt config, re-marked weights)
avatar/models/Amika/         her Live2D model (from the same repository; outside this folder)
```

Every model and data file is pinned by repository revision, size and SHA-256 in `scripts/setup_tts.py`, and every
file of her voice and Live2D model in `scripts/MekaHime_Amika.manifest.json`. GPT-SoVITS is pinned by commit. Without git, the zip GitHub serves for that
commit is checked before unpacking against a pinned SHA-256 over all 282 of its files, and the unpacked
`GPT_SoVITS/TTS_infer_pack/TTS.py` against its own pinned SHA-256 (again on every re-run, also for a git
checkout). A download that does not match its pinned size or SHA-256 is deleted and stops that step at once
instead of being retried; network errors and timeouts are retried.

## Run

OpenAmika's launcher starts it from `settings.json` → `services.tts`; `settings.example.json` has the block:

```json
"tts": {
  "enabled": true,
  "command": ["vendor\\tts-engine\\.venv\\Scripts\\python.exe", "main.py", "--port", "8002",
              "--host", "127.0.0.1", "--character", "MekaHime_Amika"],
  "cwd": "vendor\\tts-engine",
  "health": "http://127.0.0.1:8002/ping",
  "timeout_s": 240
}
```

`tts_url` (`http://127.0.0.1:8002`) must name the same port, and `tts_character` the voice (`MekaHime_Amika`).
On Linux / macOS the interpreter is `vendor/tts-engine/.venv/bin/python`. By hand:

```bash
vendor/tts-engine/.venv/Scripts/python vendor/tts-engine/main.py --port 8002     # Windows
vendor/tts-engine/.venv/bin/python vendor/tts-engine/main.py --port 8002         # Linux / macOS
```

| flag | default | |
|---|---|---|
| `--port` / `--host` | `8002` / `127.0.0.1` | bind address |
| `--device` | `auto` | `cuda` (or `cuda:N`) when available, then Apple `mps`, then `cpu`; Intel `xpu` / `xpu:N` requires explicit selection and an XPU torch build |
| `--half` / `--no-half` | fp16 on CUDA | CPU and XPU use fp32; XPU's pinned speaker verifier cannot run its FFT in fp16 |
| `--character` | first folder in `Speakers/` | speaker loaded at start; requests for another speaker switch to it |
| `--ref-max-s` | `15` | longest reference clip used whole, with its transcript (upstream's own limit is 10 s; see below) |
| `--text-split` | `sentence` | `sentence`: every sentence of a run is its own GPT fragment (see [How it uses upstream](#how-it-uses-upstream)); or one of upstream's split methods for the whole run: `cut0` none, `cut1` every 4 phrases, `cut5` every punctuation mark |
| `--batch-size` | `4` | sentences the GPT stage decodes together (CUDA graphs are captured for 1, 2 and 4; each slot holds ~50 MB of VRAM) |
| `--top-k` `--top-p` `--temperature` `--repetition-penalty` `--speed` `--seed` | `15` `1.0` `1.0` `1.35` `1.0` `-1` | sampling (a request may also send `top_k`, `top_p`, `temperature`, `speed`, `seed`; ranges under [Contract](#contract-any-server-that-speaks-this-works)) |
| `--fragment-interval` | `0.25` | pause between sentences (fragments), seconds |
| `--trim-silence` / `--no-trim-silence` | on | cut the silence before and after the speech to 0.12 s (see below) |
| `--dropout-guard` / `--no-dropout-guard` | on | speak a sentence again when its audio is far too short for its text (see below) |
| `--dropout-retries` | `2` | at most this many extra takes of such a sentence |
| `--no-cuda-graph` | off | disable upstream's CUDA-graph acceleration of the GPT stage |
| `--no-warmup` | off | skip the start-up synthesis that makes the first real request fast |
| `--verbose` | off | show upstream's console output |
| `--cors` | off | let browser pages on other origins connect (local testing) |

Start-up loads the models and runs four short warm-up lines (English, Chinese, Japanese, and English with a kana
word, which loads the language splitter); `/ping` answers `503 loading` until then and `ok` afterwards, which is
what the launcher waits for. The console logs one line per request with the audio length, synthesis time,
real-time factor and the language mode used (`lang=en>auto` when an English run went through auto mode, see
below); `/api/health` also lists the settings in use (`text_split`, `batch_size`, `ref_max_s`, `eos_floor`, ...)
and counts the dropout guard's extra takes under `retries`. Ctrl-C stops the server within about a second, also
while a client is connected or a synthesis is running.

### Intel XPU: experimental, not automatically selected

Windows 11 / Core Ultra 9 285H / Arc 140T was tested with official PyTorch
`torch==2.9.1+xpu` and `torchaudio==2.9.1+xpu` from
`https://download.pytorch.org/whl/xpu`, in a separate environment. Explicit
`--device xpu --no-half` synthesized repeated English, Japanese and Chinese clips
with finite, non-silent audio and reproducible fixed-seed output, without changing
the downloaded GPT-SoVITS source or voice assets. An unavailable XPU falls back to CPU
at device selection; synthesis failures are reported, not silently retried on another backend.

This is **compatibility evidence, not a TTS speed recommendation**. CPU remained the
recommended voice backend on that machine; XPU was not consistently faster across
languages, and first-use kernel compilation was expensive. CUDA graphs and NVIDIA
FlashAttention are not available on XPU. `--half` is downgraded to fp32 with a warning:
the pinned v2ProPlus speaker verifier otherwise raises `Unsupported dtype Half` in
its FFT. A mixed-precision runtime prototype was tested but is deliberately not shipped.
The installer and `auto` device priority remain unchanged; do not replace a working
CPU environment merely because XPU is detected.

## Contract (any server that speaks this works)

```
emit  tts_request_gateway  {task_id, turn_id, text: "【happy2】.Hello!", character_id, language, audio_format}
on    task_queued          {task_id, turn_id, sentence_count, queue_position}
on    audio_ready          {task_id, turn_id, audio: <bytes>, audio_format, audio_duration, sample_rate,
                            is_final, text, character_id, language, emotions, elapsed_ms}
on    error                {task_id, turn_id, error}
GET   /ping  -> ok         GET /api/characters -> {characters: {...}, current}      GET /api/health -> JSON
```

- **What OpenAmika sends** (`speech/api.py`, `TTSClient.request`): one request per sentence, text
  `【<tag>】.<sentence>` (the `.` is OpenAmika's own start anchor), `character_id` from the voice adapter (or
  `tts_character`), `audio_format: "wav"`, a non-empty `turn_id` (`<turn>-<index>`), and `language` `Chinese`,
  `English` or `Japanese` when the sentence has one (a mixed sentence goes without it: the server then reads it as
  English, in auto mode as soon as it holds kana or Chinese characters, see below). Events go back to the client
  that asked; OpenAmika matches them by `task_id`.
- **Emotion tags** are `【<name><intensity>】` (`happy1-4`, `sad1-4`, `angry1-4`, `shy1-3`, `tsun1-3`,
  `whisper1-3`, ...): the keys of `Speakers/<id>/references/emotions.json`, each naming a reference clip and its
  (Chinese) transcript. Amika's `emotions.json` has 69 tags, each with its clip; `speech/voice_adapter.py` maps
  her 69 internal `register_emotion` keys onto 45 of them by default (the Setup page can pick any of the 67 V1
  tags per key). A tag without a recording falls back to `neutral1`, and the `audio_ready` then carries
  `fallback_from`. A text may switch tags mid-way (`【neutral1】Well. 【happy1】Hi!`): each run is spoken with its
  own reference and the pieces are joined. A run of only punctuation or symbols (`【shy1】……`, `【happy1】~`) is
  left out, since upstream would answer it with an error or with a second of 16 kHz silence. A text with nothing
  to speak at all (only tags, punctuation or emoji) gets 0.3 s of silence and a `warning` instead of an error; a
  blank `text` is `bad request: empty text`.
- **`/api/characters`** lists, per speaker folder: `name`, `language` (of the reference transcripts), the
  `emotions` that have a clip, `no_clip` (listed tags without one), `default_emotion` and `loaded`. No other
  text from a voice pack's `character.json` is served.
- **language**: `English`, `Chinese`, `Japanese`, `Korean`, `Cantonese` or `auto` (also `en`, `zh`, `ja`, ...).
  The reference transcripts are Chinese; GPT-SoVITS speaks the target language in her voice. Upstream's English
  front-end reads Latin letters only and silently drops a word written in kana, Hangul or Chinese characters, so
  an English run that contains any of those characters is spoken in upstream's `auto` mode, which splits it by
  language and reads each part with its own front-end (Latin text stays English); runs without them use the
  English front-end. The other languages are used as asked: in `Chinese` mode kana are still dropped, and auto
  mode is not used there because it can read Chinese characters next to kana as Japanese (send `auto` for mixed
  Chinese and Japanese).
- **turn mode** (a non-empty `turn_id`, what OpenAmika sends): one `audio_ready` for the whole text.
  Without `turn_id` every sentence is sent as soon as it is ready and the last one has `is_final: true`.
  Requests are spoken one at a time in arrival order, so one request's sentences are never interleaved with
  another's. When a client disconnects, its requests that have not started, and the sentences of a request not
  yet spoken, are dropped (a synthesis already running finishes).
- **audio_format**: `wav` (16-bit PCM mono at the model's 32 kHz, the default, plays in every browser) or
  `opus` (Ogg/Opus 48 kHz through ffmpeg). Without ffmpeg an `opus` request gets WAV, `audio_format: "wav"`
  and a `warning` field; any other value also gets WAV and a `warning`. Audio larger than 3 MiB (about 49 s of
  WAV) arrives as consecutive chunks of one file (`chunk_index` / `chunk_count`, `is_final` on the last).
  OpenAmika's client takes one `audio_ready` per request and does not join chunks; its requests are single
  sentences, far below that size.
- **sampling options** (optional numbers): `speed` 0.5-2.0, `top_k` a whole number 1-100, `top_p` above 0 and
  at most 1, `temperature` above 0 and at most 2, `seed` a whole number from 0 to 4294967295 or -1 (random). A
  value outside its range, or one that is not a number, gets an `error` event such as
  `bad request: speed must be between 0.5 and 2.0 (got 3)`; `null` counts as not sent.
- **warning** (optional on `audio_ready`): set when the audio is not quite what was asked for: WAV sent instead
  of Opus, a text with nothing to speak, or speech that came out too short for its text in every take (see the
  dropout guard below).

## How it uses upstream

`main.py` imports `GPT_SoVITS/TTS_infer_pack/TTS.py` (class `TTS` with a `TTS_Config`, the same library
upstream's `api_v2.py` uses) and adds what the OpenSakura voice server adds. The measurements in this section
were made in OpenSakura with Sakura's voice (same engine, same model family) unless they name Amika:

- speaker folders, emotion tags, tag fallback and mid-text tag switches;
- one prompt cache per reference clip, so switching emotions does not recompute the reference features;
- **reference clips longer than 10 s** (new in this build): upstream refuses a reference outside 3-10 s, and the
  OpenSakura server trims a longer clip and then speaks it without its transcript (the transcript no longer
  matches). 23 of Amika's 69 clips are 10-14.4 s long, among them the clips of common emotions (`sad2`,
  `caring2`, `playful2`, `relieved1`, `jealous1`, `whisper2`), and the voice server she was made with used every
  clip whole. So at start-up `main.py` compiles upstream's own `TTS._set_prompt_semantic` source with the upper
  bound raised to `--ref-max-s` (15 s) and puts it in place of the method (the files on disk are not changed;
  the same technique as the EOS floor below). Clips up to that length are used whole, with their transcripts;
  a longer clip is trimmed and spoken without its transcript as before. If that check is not in upstream's
  source (another upstream version) it logs a warning and keeps upstream's 10 s. `/api/health` shows
  `ref_max_s`. Checked on Amika's voice: requests on the 11.1 s, 11.8 s, 13.5 s and 14.4 s references answered in
  0.55-1.06 s, and speech recognition read every one back word for word;
- reference clips shorter than 3 s (upstream rejects them) are padded with silence into `cache/`;
- **SoVITS weights header.** Upstream saves v2Pro / v2ProPlus fine-tunes with the first two bytes of the torch
  zip (`PK`) replaced by a version marker (`05` / `06`) and reads the model version back from that marker.
  Such a file loads as is (Amika's `.pth` carries `06`, v2ProPlus). A file whose marker was turned back into `PK`
  is guessed as a plain v2 model from its size: upstream drops its speaker-embedding layers (`sv_emb`,
  `ge_to512`, `prelu`) and synthesis then fails with a channel mismatch. For that case `main.py` writes a copy
  with the marker restored to `cache/speakers/<id>/` (byte-identical to the file upstream saved) and loads that;
- small compatibility shims: `jieba_fast` (no wheels) is served by `jieba`, and with torchaudio 2.9 (which
  decodes through torchcodec + FFmpeg DLLs) the WAV reference clips are read with `soundfile` instead;
- **silence trimming**: the model sometimes opens a line with up to 1.3 s of silence (whispered lines most often)
  and upstream ends every run with a 0.25 s pause. Each run keeps 0.12 s of silence before and after its speech.
  On 10 ms frames, silence is what lies more than 45 dB under the loudest frame of the run, or under -80 dBFS:
  the model's own silence sits at -87 dBFS and below, while the softest onsets and word endings of whispers stay
  within 45 dB of their loudest frame. Over 96 recorded takes, nothing removed was louder than 46.7 dB under the
  loudest frame, and speech recognition gave the same words before and after trimming for 93 of them; in each of
  the other 3 one word differed although only silence (under -75 dBFS) had been removed. Each tagged run is
  trimmed on its own before the runs are joined, so a two-tag text has 0.24 s of silence at the joint (0.12 +
  0.12) whatever the natural pause was. `audio_duration` is the length after trimming (for Opus, of the PCM
  before encoding: the Ogg file adds its pre-skip and padding);
- every run comes back at the model's sample rate: when upstream finds nothing to read in a run it returns a
  second of silence at 16 kHz, which is dropped, and any other audio at another rate would be resampled;
- **one GPT fragment per sentence**: upstream's text-to-semantic (GPT) stage reads each fragment after the
  reference clip's transcript and sometimes starts at the fragment's second sentence, as if the reference clip had
  already said the first one. With a whole reply as one fragment (upstream's `cut1`) speech recognition found a
  phrase missing in 18 of 125 takes of 25 replies (2-4 sentences; ~10 s of speech each), 17 of them the opening
  sentence. So each tagged run is cut into sentences (at `。！？!?…` and `.`; a sentence under 4 syllables such as
  `okay!` joins the next one), every sentence gets a leading `.` (a short pause, as upstream gives its own short
  first sentences; not before a 1-3 letter head with a comma such as `oh,` / `嗯，`, nor before a one-word
  sentence; OpenAmika's own leading `.` counts as that pause), and the sentences go to upstream in one request,
  one fragment per line (`cut0`), decoded `--batch-size` (4) at a time. On the same 125 takes no phrase was
  missing, and the time per reply halved. Known limits: a short exclamation that joins the sentence before it can
  still be left out, and with quiet or sobbing references a very short line can come out without words (Amika's
  `sad4`, a 2.9 s sob, spoke a two-syllable `どうして……` as a sob that speech recognition could not read);
- **EOS floor**: upstream's plain GPT loop does not let a fragment end before 11 tokens (0.44 s), but its
  CUDA-graph loop (`GPT_SoVITS/Accel/PyTorch/AR/runner.py`, `ModelRunner.decode_step`) only forbids ending on the
  first token, and a sentence on its own then sometimes ended after 1-2 tokens, with no speech. At start-up on
  CUDA `main.py` compiles upstream's own `decode_step` source with that one comparison changed (`== 0` to `< 11`)
  and puts it in place of the method; the files on disk are not changed. If that line is not in upstream's source
  (another upstream version) it logs a warning and leaves upstream as it is. `/api/health` shows `eos_floor`:
  `11 tokens`, `built in` (CPU and MPS use the plain loop) or `not installed (...)`;
- **dropout guard**: the GPT stage can still end a sentence early. The expected length of a sentence is its
  syllable count (vowel groups per English word, one per Chinese character; Japanese kana and Korean Hangul count
  0.5 and 0.6, estimated) at 4.4 syllables per second. If its speech (first to last frame within 40 dB of the
  loudest) lasts under 55 % of that, that sentence alone is synthesized again with another seed (at the same
  batch size, so upstream does not rebuild its GPT decoder), up to `--dropout-retries` (2) more times while it
  stays short, and the take with the most speech is kept. A sentence under 6 syllables (where the estimate is too
  rough) counts as short only when it comes out nearly silent, with under 0.2 s of speech. The log has a warning,
  `/api/health` counts the extra takes under `retries`, and if every take of a sentence is short `audio_ready`
  carries a `warning`. It cannot loop (at most `--dropout-retries` extra takes per sentence);
- the Socket.IO + HTTP server, WAV / Opus encoding, queueing (one request at a time on one daemon worker
  thread, so Ctrl-C does not wait for a synthesis in flight; connected clients are disconnected on shutdown),
  warm-up and per-request timing logs. Upstream's console output is captured per request on the worker
  thread (for the timing split and error reports) without swapping `sys.stdout` while running;
  `--verbose` shows it instead.

## Python packages: what differs from upstream's requirements.txt

Upstream's list also serves training, dataset tools and the Gradio WebUI. `requirements-tts.txt` keeps the
inference subset and prefers packages that ship wheels, so no compiler is needed on Windows:

| upstream | here | why |
|---|---|---|
| `gradio<5` | left out | WebUI only |
| `tensorboard` | left out | training only |
| `funasr`, `faster-whisper` (extra-req.txt), `ctranslate2`, `modelscope` | left out | speech recognition / dataset preparation and model downloads; not imported for inference |
| `fastapi[standard]`, `pydantic<=2.10.6` | left out | upstream's `api_v2.py` HTTP server; `main.py` serves with aiohttp + python-socketio |
| `av`, `sentencepiece`, `chardet`, `rotary_embedding_torch` | left out | not imported by the inference path at this commit |
| `jieba_fast` | `jieba` | jieba_fast is a source-only C extension; jieba has the same API and dictionary (`main.py` aliases it) |
| `pyopenjtalk` | `pyopenjtalk-plus` | pyopenjtalk is source-only (cmake + C++); the -plus fork has wheels and bundles the Open JTalk dictionary |
| `onnxruntime-gpu` (x86_64) | `onnxruntime` | only the Chinese G2PW model uses it; the CPU build avoids a second CUDA/cuDNN stack |
| `--no-binary=opencc` | OpenCC wheel | the wheel works; building OpenCC needs cmake + a C++ compiler |
| `torchaudio` (unpinned) + `torchcodec` | torch/torchaudio 2.9.1 pinned, no torchcodec | torchcodec needs FFmpeg shared libraries; the references are WAV |
| (only transitive upstream) | `resampy`, `matplotlib`, `rich`, `loguru`, `einops`, `soundfile`, `huggingface_hub` | imported by the inference code; upstream gets them through the packages left out above |
| | `python-socketio`, `aiohttp` | the server |

Korean (`g2pk2`) and Cantonese (`ToJyutping`) are installed but not exercised by OpenAmika's tests.

## Performance

**Amika's voice on an RTX 5080 16 GB (Windows 11, fp16)**, measured on 2026-11-03 while the rest of a running
OpenAmika stack held about 11-13 GB of the same card:

| | |
|---|---|
| install from scratch | 375 s, including the 2.9 GB CUDA torch wheel (the other packages came from the pip cache); a re-run of a complete install 35 s |
| start-up | 20-21 s to `/ping ok` (models 9.1 s, four warm-up lines 10.8-11.9 s) |
| one OpenAmika sentence (11 requests: Japanese, Chinese, English and mixed; 1.1-4.0 s of audio) | 0.34-1.06 s each (real-time factor 0.13-0.39), the four on 11-14.4 s references included |
| the very first request after a fresh install | 22.8 s once (GPT stage 20.2 s), not seen again on the next start |
| VRAM | 1.8 GB allocated after warm-up, 2.1-2.2 GB reserved, peak 2.06 GB allocated |

**Sakura's voice in OpenSakura on an RTX 5070 12 GB** (the same server): start-up 25-33 s; a reply of 2-4
sentences (median 10.2 s of audio) in 0.75 s median, 90 % within 0.98 s; 84 s of text in one request in 4.4 s;
first use of an emotion tag +0.1-0.4 s once; VRAM ~2.2 GB idle; `--device cpu` on that machine spoke 3.5 s of
audio in about 1.8 s.

Each request logs its split (`ref + text + gpt + sovits` seconds); the GPT (text-to-semantic) stage
dominates.

## Platforms

Tested: Windows 11 with Python 3.11 and the CUDA 12.8 driver on an RTX 5080 16 GB (this build, Amika's voice) and
an RTX 5070 12 GB (the OpenSakura server, also with `--device cpu`).

Untested:

- **Linux**: the script takes the same steps (CUDA 12.8 wheels with an NVIDIA GPU, CPU wheels with `--cpu`;
  BtbN's static ffmpeg or the system one), but none of it has been run yet. It needs glibc 2.28 or newer
  (Ubuntu 20.04+, Debian 10+, RHEL / Rocky 8+): torch 2.9.1, onnxruntime 1.31.0 and soundfile 0.14.0 ship
  `manylinux_2_28` wheels only, onnxruntime has no source package, and on an older system pip would try to
  build pyopenjtalk-plus from source (cmake + a C++ compiler). The script stops with that reason first.
- **macOS**: Apple Silicon gets the PyPI torch build (CPU, and `--device mps` for the GPU) and needs
  `brew install ffmpeg` for Opus; none of it has been run yet. It needs macOS 14 or newer (onnxruntime 1.31.0
  has a single macOS wheel, `macosx_14_0_arm64`, and no source package) and a native arm64 Python: an x86_64
  Python running under Rosetta gets x86_64 wheels, which do not exist for these packages, and the script says
  so. Intel Macs have no torch 2.9 wheels.

The pins in `requirements-tts.txt` were resolved and tested on Windows only (numpy 1.26.4, transformers 4.57.6,
onnxruntime 1.31.0, ...). Upstream needs numpy < 2. If pip cannot resolve the set on Linux, the first thing to try
is relaxing a pin to upstream's own range (noted next to each pin).

## Licenses

`main.py` is MIT, © 2026 MekaHime Inc. (`LICENSE`), derived from the OpenSakura voice server (MIT, MekaHime
Inc.). GPT-SoVITS is MIT (RVC-Boss); the pretrained models, nltk data and ffmpeg builds keep their own licenses
(`THIRD_PARTY_NOTICES.md`). Amika's voice model, fine-tuned weights, reference audio and Live2D model are MekaHime
Inc. assets for personal, hobbyist use only (no distillation, training, extraction, resale or commercial use);
they are not covered by the MIT license.
