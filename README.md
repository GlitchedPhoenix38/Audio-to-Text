# FoziScribe — Offline AI Audio Transcription

A local, fully offline audio transcription web app that works like FoziScribe.
Powered by **faster-whisper** (CTranslate2) + a clean **Gradio** web UI.

No cloud, no OpenAI API, no internet required after the models are downloaded
once. Everything runs on your own machine.

![laptop](https://img.shields.io/badge/Arch%20Linux-RTX%204050%20Laptop%206GB-blue)

---

## Features

- **faster-whisper** engine (CTranslate2) — much faster and lighter on VRAM than
  the original openai-whisper.
- **Two modes:**
  - **Fast** → `small` model (~0.5 GB VRAM @ fp16)
  - **Accurate** → `large-v3-turbo` (default, ~2.5 GB VRAM @ fp16) — near
    large-v3 accuracy but ~8x faster and small enough for your 6 GB GPU.
- **Segment-level timestamps**, exactly like:
  ```
  [0:00] The temperature is minus 30 degrees.
  [0:02] The food ran out three weeks ago.
  [0:04] The man next to you just died.
  ```
- Drag & drop / file upload (MP3, WAV, M4A, FLAC, OGG, WebM, OPUS, MP4, …).
- Live progress bar with ETA while transcribing.
- Detected language + probability shown after transcription.
- **Copy to clipboard**, **Download .txt**, **Download .srt**.
- Optional **word-level timestamps**.
- **Session history** — the last 10 transcripts can be reloaded in the UI.
- **Auto-fallback**: CUDA + float16 → CUDA + int8 → CPU + int8, so the app
  never crashes if CUDA is unavailable.
- VAD (voice activity detection) to skip silence, `beam_size=5`, auto language.
- Handles long audio (1–2+ hours) — faster-whisper processes it in chunks.
- Dark modern theme, auto-opens your browser on launch.

---

## Requirements

- **OS:** Arch Linux (this project is tuned for your setup)
- **GPU (optional but recommended):** NVIDIA RTX 4050 Laptop, 6 GB VRAM
- **RAM:** 16 GB
- **Python:** 3.10, 3.11 or 3.12 (Arch ships 3.13 — that works too)
- Disk space for models: `small` ≈ 0.5 GB, `large-v3-turbo` ≈ 1.6 GB

---

## Installation (Arch Linux)

### 1. Make sure the NVIDIA driver is working

```bash
nvidia-smi
```

If the command is missing, install the driver (RTX 40-series uses the open
kernel modules):

```bash
sudo pacman -S nvidia-open nvidia-utils
sudo reboot
```

You should see your RTX 4050 listed with a CUDA version in `nvidia-smi`.

### 2. Install CUDA + cuDNN (for GPU acceleration)

```bash
sudo pacman -S cuda cudnn
```

That gives you cuBLAS (from `cuda`) and cuDNN 9 (from `cudnn`). Make sure the
runtime loader can find the libraries (add to `~/.bashrc` or `~/.zshrc`):

```bash
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH}
```

> **Note:** ctranslate2 loads `libcublas.so.12` / `libcudnn.so.9` by name. Arch's
> current `cuda` package ships only version 13, so `requirements.txt` also
> installs the CUDA-12 pip libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`)
> and `run.sh` adds them to `LD_LIBRARY_PATH` automatically. If the app still
> can't find CUDA it falls back to CPU (int8) — slower, but it works.

### 3. Set up the project and install dependencies

```bash
cd /home/abdullah/Projects/foziscribe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run it

```bash
./run.sh
```

or directly:

```bash
python app.py
```

Your browser opens automatically at **http://127.0.0.1:7860**.

---

## First run — model download (one time)

On the very first transcription the models are downloaded automatically from
Hugging Face (only once; afterwards they are cached and the app is fully
offline):

| Model             | Cache size | Notes                                        |
|-------------------|-----------|----------------------------------------------|
| `small`           | ~0.5 GB   | Fast mode                                    |
| `large-v3-turbo`  | ~1.6 GB   | Accurate mode (default)                      |

If you're offline and the model isn't cached yet, the app shows a friendly
error explaining that a one-time download is required.

---

## Usage

1. Drag & drop an audio file into the box (or click to browse).
2. Choose **Fast** or **Accurate**.
3. (Optional) tick **Word-level timestamps**.
4. Click **Transcribe** — watch the progress bar and ETA.
5. Copy the result, download `.txt` and/or `.srt`, or re-open a past transcript
   from **Session history**.

### Customizing the models

- Want an English-only ultra-fast mode? Edit `MODELS` in `app.py`:

  ```python
  MODELS = {
      "fast": "distil-small.en",   # English only, even faster
      "accurate": "large-v3-turbo",
  }
  ```

- Prefer the maximum-accuracy `large-v3` instead of turbo? Change the accurate
  entry to `"large-v3"` (≈3.1 GB VRAM @ fp16 — still fits, just slower).

---

## VRAM safety check (RTX 4050, 6 GB)

From published faster-whisper benchmarks on a 6 GB GPU:

| Model             | Precision | Peak VRAM | Within 6 GB? |
|-------------------|-----------|-----------|--------------|
| `small`           | float16   | ~0.5 GB   | Yes ✓        |
| `large-v3-turbo`  | float16   | ~2.5 GB   | Yes ✓        |
| `large-v3-turbo`  | int8      | ~1.5 GB   | Yes ✓        |
| `large-v3`        | float16   | ~3.1 GB   | Yes ✓        |

If float16 ever runs out of memory the app **automatically retries with int8**
(the runtime fallback chain is `cuda/float16 → cuda/int8 → cpu/int8`). Long
files are processed in fixed-size chunks, so peak VRAM stays flat even for
2-hour recordings.

---

## Troubleshooting

**`Failed to load library: libcublas.so.12 / libcudnn.so.9` (or CUDA errors)**
- cuBLAS/cuDNN are missing. Make sure `sudo pacman -S cuda cudnn` was run and
  `/opt/cuda/lib64` is on `LD_LIBRARY_PATH` (see step 2), then restart the app.
- Verify with: `.venv/bin/python -c "import ctranslate2; print('ct2 ok')"`
- If you can't fix CUDA right now, no problem — the app auto-falls back to CPU.

**Out-of-memory / CUDA out of memory during transcription**
- Switch to **Fast** mode, or force int8 by editing the fallback order in
  `DEVICE_PRECISION_PREFERENCE` in `app.py`.

**First run stuck on "Loading model"**
- The model is downloading; wait a few minutes. On a slow connection, run once
  with internet access to pre-cache the models.

**Port already in use / browser doesn't open**
- Change `server_port` in `app.py` (e.g. `7861`) and run again.

**Slow transcription**
- Make sure it says `GPU (CUDA)` in the Status box. CPU mode is much slower.
- Close apps using the GPU (game overlays, OBS, other AI tools).
- Check your GPU isn't stuck in a low power state: `nvidia-smi -q -d POWER`.

---

## Project structure

```
foziscribe/
├── app.py           # Main Gradio app (UI + transcription engine)
├── utils.py         # Timestamp/SRT formatting helpers
├── requirements.txt # Python dependencies
├── run.sh           # One-command launcher (use ./run.sh)
├── .gitignore
└── output/          # Generated .txt/.srt files (created automatically)
```

---

## How it works (short version)

1. You upload audio → the app decodes it with PyAV (bundled ffmpeg — no system
   ffmpeg needed).
2. faster-whisper loads a CTranslate2-optimized Whisper model on your GPU.
3. VAD trims silence, `beam_size=5` decoding runs with language auto-detection.
4. Segments are formatted with timestamps → shown in the UI, saved as `.txt`
   and `.srt` in `output/`.

License: MIT. Models are OpenAI Whisper / distil-whisper (MIT) converted to
CTranslate2 format — they run locally and never leave your machine.
