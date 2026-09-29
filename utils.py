"""Helper utilities for formatting transcription output.

Pure functions that turn faster-whisper segments into the FoziScribe-style
text and SRT formats used by the Gradio UI.
"""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    """Format seconds as a FoziScribe-style segment timestamp.

    Examples:
        format_timestamp(0)      -> "[0:00]"
        format_timestamp(2.4)    -> "[0:02]"
        format_timestamp(4.9)    -> "[0:04]"
        format_timestamp(3661)   -> "[1:01:01]"
    """
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes}:{secs:02d}]"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp, e.g. "00:00:02,400"."""
    ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_segment(segment, word_timestamps: bool = False) -> str:
    """Return the plain text of a faster-whisper segment.

    If ``word_timestamps`` is True and the segment carries per-word timing,
    each word is annotated with its own timestamp, e.g.
    "Hello [0:00] world [0:01]."
    """
    text = getattr(segment, "text", "") or ""
    text = text.strip()
    if not text:
        return ""

    if word_timestamps and getattr(segment, "words", None):
        words = []
        for word in segment.words:
            word_text = (getattr(word, "word", "") or "").strip()
            if not word_text:
                continue
            words.append(f"{word_text} {format_timestamp(word.start)}")
        if words:
            return " ".join(words)
    return text


def build_srt(entries) -> str:
    """Build a complete .srt document from a list of (start, end, text) tuples."""
    blocks = []
    for index, (start, end, text) in enumerate(entries, start=1):
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + "\n"


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration, e.g. "2m 05s"."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_speed_ratio(audio_seconds: float, elapsed: float) -> str:
    """Return how much faster than realtime transcription ran, e.g. "12.3x"."""
    if audio_seconds and elapsed > 0:
        return f"{audio_seconds / elapsed:.1f}x realtime"
    return "n/a"
