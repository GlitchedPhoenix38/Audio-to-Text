# FizzScribe

Independent, privacy-first local transcription workspace inspired by a polished transcription workflow. Built with Gradio and faster-whisper; it does not connect to FoziScribe's private backend or require a paid API.

## Features
- Audio and video uploads: MP3, WAV, M4A, FLAC, OGG, WebM, MP4, MOV, MKV and more.
- Fast (small), Balanced (large-v3-turbo), and Accuracy (large-v3, default) presets.
- Automatic or manual language selection, including English, Hindi, Urdu, Arabic, Marathi, Bengali, Tamil and Telugu.
- Original-language transcription or English translation.
- Editable timestamped transcript, optional word timestamps, adjustable beam size and optional VAD.
- TXT, SRT and WebVTT exports; persistent local history; search inside transcript; save edited TXT.
- CUDA-first with precision fallbacks and CPU fallback. No account, API key or cloud service.

## Accuracy first
The default is full Whisper large-v3, not turbo. Turbo is the faster alternative. Accuracy depends on audio quality, background noise, overlapping speakers, language and accent; no model guarantees every word. This app does not claim speaker labels or calibrated word-level confidence.

For best results choose Accuracy, select the spoken language if auto-detection is wrong, keep original-language transcription, use beam size 5–8, and leave VAD off when quiet words matter.

## Hardware
Tuned for RTX 4050 Laptop GPU (6 GB VRAM) and 16 GB RAM. The app tries CUDA float16, CUDA int8-float16, CUDA int8, then CPU int8. Use Balanced if GPU memory is limited.

## Windows 11
Install Python 3.11 or 3.12 (64-bit), then PowerShell:
    git clone https://github.com/GlitchedPhoenix38/FizzScribe.git
    cd FizzScribe
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:7860.

## Arch Linux
    sudo pacman -S --needed python python-pip python-virtualenv
    git clone https://github.com/GlitchedPhoenix38/FizzScribe.git
    cd FizzScribe
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ./run.sh

## First launch and offline use
The first use of each model downloads its weights from Hugging Face. This is a one-time download per model. Weights are cached outside the repository; do not commit model files. After download, that model works offline.

## Local data
Transcripts, exports and history are in output/. Model weights are in your user-level Hugging Face cache. The server binds to 127.0.0.1 only. Delete output/ to remove saved transcripts and history.

This is an independent local application and is not affiliated with FoziScribe.
