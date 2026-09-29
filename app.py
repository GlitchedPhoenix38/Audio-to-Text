"""FoziScribe — local, offline AI audio transcription.

A Gradio web app powered by faster-whisper + CTranslate2 that transcribes audio
files entirely on your own machine. No cloud, no API keys, no internet needed
after the models have been downloaded once.

Designed and tuned for:
    ASUS TUF A15 (2023) · AMD Ryzen 7 7435HS · RTX 4050 Laptop (6 GB VRAM) · 16 GB RAM

Modes:
    Fast     -> "small"           (~0.5 GB VRAM @ fp16)
    Accurate -> "large-v3-turbo"  (~2.5 GB VRAM @ fp16)  <- default

Run with:
    python app.py
"""

import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from utils import (
    build_srt,
    format_duration,
    format_segment,
    format_speed_ratio,
    format_timestamp,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model used per mode. "large-v3-turbo" is preferred over "large-v3" because it
# is ~8x faster and about 1.6 GB smaller while keeping near-large-v3 accuracy,
# so it fits comfortably in the RTX 4050's 6 GB VRAM at float16.
MODELS = {
    "fast": "small",          # or "distil-small.en" for English-only speed
    "accurate": "large-v3-turbo",
}

# (device, compute_type) attempts, in order of preference. CUDA + float16 first,
# then CUDA + int8 (if float16 runs out of VRAM), and finally CPU + int8 so the
# app always works even without a working CUDA installation.
DEVICE_PRECISION_PREFERENCE = [
    ("cuda", "float16"),
    ("cuda", "int8"),
    ("cpu", "int8"),
]

# Audio formats accepted by the upload box (anything PyAV/ffmpeg can decode).
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".opus", ".aac",
    ".wma", ".mp4", ".m4b", ".aiff", ".amr", ".oga", ".wv",
}

# How many past transcripts to keep in the session history list.
HISTORY_LIMIT = 10

# --------------------------------------------------------------------------- #
#  Runtime state
# --------------------------------------------------------------------------- #

MODEL_CACHE: dict[str, tuple] = {}   # model_name -> (model, device, compute_type)
SESSION_HISTORY: list[dict] = []     # last few {label, transcript}
TRANSCRIBE_LOCK = threading.Lock()   # serializes jobs -> protects VRAM (6 GB)


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #

def _load_model(model_name: str):
    """Load (and cache) a faster-whisper model with automatic fallback.

    Returns:
        (model, device, compute_type)
    """
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    last_error = None
    for device, compute_type in DEVICE_PRECISION_PREFERENCE:
        try:
            print(f"[info] Loading '{model_name}' on {device} ({compute_type}) ...")
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=8 if device == "cpu" else 0,
                num_workers=1,
            )
            MODEL_CACHE[model_name] = (model, device, compute_type)
            return MODEL_CACHE[model_name]
        except Exception as err:  # noqa: BLE001 - we try every option before failing
            last_error = err
            print(f"[warn] {device}/{compute_type} failed: {err}")

    raise RuntimeError(
        "Could not load the model. Every device/precision attempt failed.\n"
        f"Last error: {last_error}\n\n"
        "Possible causes:\n"
        "  * No internet on the very first run (the model must be downloaded once,\n"
        "    afterwards it is cached locally and works fully offline).\n"
        "  * No NVIDIA GPU or CUDA runtime detected.\n"
        "  * Missing cuBLAS / cuDNN libraries (see README -> Troubleshooting)."
    )


# --------------------------------------------------------------------------- #
#  Core transcription
# --------------------------------------------------------------------------- #

def transcribe_audio(audio_path, mode, use_word_timestamps, progress=gr.Progress()):
    """Transcribe an uploaded audio file and return all UI outputs.

    Returns:
        (transcript, language_label, status, txt_path, srt_path, history_dd)
    """
    # ---- Input validation -------------------------------------------------- #
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")

    ext = Path(str(audio_path)).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise gr.Error(
            f"Unsupported file type '{ext or '(none)'}'.\n"
            f"Supported formats: {supported}"
        )

    if mode not in MODELS:
        raise gr.Error(f"Unknown mode '{mode}'.")

    model_name = MODELS[mode]

    # Serialize transcription jobs so two jobs can never OOM the 6 GB GPU.
    with TRANSCRIBE_LOCK:
        # ---- Model loading -------------------------------------------------- #
        progress(0.0, desc=f"Loading model '{model_name}' ...")
        try:
            model, device, compute_type = _load_model(model_name)
        except RuntimeError as err:
            raise gr.Error(str(err))
        except Exception as err:  # noqa: BLE001
            raise gr.Error(f"Unexpected error while loading the model:\n{err}")

        # Audio duration is needed for progress tracking and the speed stat.
        # In faster-whisper >= 1.2, decode_audio returns a float32 numpy array
        # resampled to 16 kHz, so duration is len(samples) / sampling_rate.
        duration = None
        try:
            samples = decode_audio(str(audio_path))
            duration = len(samples) / 16_000
        except Exception:  # noqa: BLE001 - duration is only a nicety
            duration = None

        # ---- Transcription -------------------------------------------------- #
        started = time.perf_counter()
        transcript_lines = []
        srt_entries = []
        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,                     # skip silence -> faster
                language=None,                       # auto-detect
                word_timestamps=use_word_timestamps,
            )

            for segment in segments:
                text = format_segment(segment, use_word_timestamps)
                if not text:
                    continue

                transcript_lines.append(f"{format_timestamp(segment.start)} {text}")
                srt_entries.append((segment.start, segment.end, text))

                # Live progress + ETA based on the fraction of audio done.
                if duration:
                    fraction = min(1.0, segment.end / max(duration, 1e-9))
                    elapsed_so_far = time.perf_counter() - started
                    eta = elapsed_so_far / max(fraction, 1e-9) * (1 - fraction)
                    progress(
                        fraction,
                        desc=(
                            f"Transcribing {format_timestamp(segment.end)} / "
                            f"{format_timestamp(duration)} · ETA ~{format_duration(eta)}"
                        ),
                    )
        except gr.Error:
            raise
        except Exception as err:  # noqa: BLE001
            raise gr.Error(
                "Transcription failed.\n\n"
                f"Error: {err}\n\n"
                "If this looks like an out-of-memory error, use Fast mode or "
                "see the README troubleshooting section."
            )
        elapsed = time.perf_counter() - started

    # ---- Assemble results --------------------------------------------------- #
    transcript = "\n".join(transcript_lines)
    if not transcript:
        transcript = "No speech detected in the audio file."

    language = getattr(info, "language", "unknown")
    language_prob = getattr(info, "language_probability", None)
    language_label = (
        f"{language} (probability: {language_prob:.1%})"
        if language_prob is not None
        else str(language)
    )

    engine = "GPU (CUDA)" if device == "cuda" else "CPU"
    status_lines = [
        f"Model:        {model_name}",
        f"Engine:       {engine} · {compute_type}",
        f"Language:     {language_label}",
        f"Audio length: {format_timestamp(duration) if duration else 'unknown'}",
        f"Processed in: {format_duration(elapsed)}",
        f"Speed:        {format_speed_ratio(duration, elapsed)}",
    ]
    status = "\n".join(status_lines)

    # ---- Write download files ------------------------------------------------ #
    base_name = (Path(str(audio_path)).stem or "transcript")[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = OUTPUT_DIR / f"{base_name}_{stamp}.txt"
    srt_path = OUTPUT_DIR / f"{base_name}_{stamp}.srt"
    txt_path.write_text(transcript, encoding="utf-8")
    srt_path.write_text(build_srt(srt_entries), encoding="utf-8")

    # ---- Session history ------------------------------------------------------ #
    SESSION_HISTORY.append({
        "label": f"{stamp} · {mode} · {language}",
        "transcript": transcript,
    })
    SESSION_HISTORY[:] = SESSION_HISTORY[-HISTORY_LIMIT:]
    history_choices = [entry["label"] for entry in SESSION_HISTORY]

    return (
        transcript,
        language_label,
        status,
        str(txt_path),
        str(srt_path),
        gr.Dropdown(choices=history_choices, value=history_choices[-1]),
    )


def load_history(label):
    """Restore a transcript from session history into the main text box."""
    if not label:
        return ""
    for entry in reversed(SESSION_HISTORY):
        if entry["label"] == label:
            return entry["transcript"]
    return ""


def clear_all():
    """Reset the UI back to a blank state (history is kept)."""
    history_choices = [entry["label"] for entry in SESSION_HISTORY]
    return (
        None,   # audio
        "",     # transcript
        "",     # language
        "",     # status
        None,   # .txt download
        None,   # .srt download
        gr.Dropdown(choices=history_choices),
    )


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #

THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(
    title="FoziScribe — Offline AI Transcription",
    theme=THEME,
    css="footer {display: none !important;}",
) as demo:
    gr.Markdown(
        "# FoziScribe\n"
        "### Fully offline AI audio transcription powered by faster-whisper — "
        "runs 100% on your laptop (no cloud, no API keys)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio file (drag & drop or click)",
                type="filepath",
                sources=["upload"],
            )
            mode_radio = gr.Radio(
                choices=[
                    ("Fast — 'small' model (fastest)", "fast"),
                    ("Accurate — 'large-v3-turbo' (best quality)", "accurate"),
                ],
                value="accurate",
                label="Mode",
                info="Accurate uses large-v3-turbo (~2.5 GB VRAM @ fp16) — fits your RTX 4050.",
            )
            word_ts_check = gr.Checkbox(
                label="Word-level timestamps",
                value=False,
                info="Add a timestamp after each individual word.",
            )
            transcribe_btn = gr.Button("Transcribe", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")

            status_box = gr.Textbox(
                label="Status",
                lines=7,
                interactive=False,
            )

            with gr.Row():
                txt_btn = gr.DownloadButton(label="Download .txt", variant="secondary")
                srt_btn = gr.DownloadButton(label="Download .srt", variant="secondary")

            history_dd = gr.Dropdown(
                label=f"Session history (last {HISTORY_LIMIT} transcripts)",
                choices=[],
                interactive=True,
            )
            history_btn = gr.Button("Load selected transcript")

        with gr.Column(scale=1):
            language_box = gr.Textbox(label="Detected language", interactive=False)
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=22,
                interactive=False,
            )

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, mode_radio, word_ts_check],
        outputs=[
            transcript_box, language_box, status_box,
            txt_btn, srt_btn, history_dd,
        ],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            audio_input, transcript_box, language_box, status_box,
            txt_btn, srt_btn, history_dd,
        ],
    )

    history_btn.click(
        fn=load_history,
        inputs=[history_dd],
        outputs=[transcript_box],
    )


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Starting FoziScribe ...")
    print(f"Transcriptions will be saved to: {OUTPUT_DIR}")
    demo.queue()
    demo.launch(
        inbrowser=True,          # auto-open the browser
        server_name="127.0.0.1",
        server_port=7860,
    )
