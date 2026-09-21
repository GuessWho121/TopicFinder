"""Whisper transcription and conversion to the project's JSON contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot create a transcript."""


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run Whisper and return its native result dictionary.

    Importing Whisper lazily lets the package's file/JSON utilities work even
    when ML dependencies are not installed yet.
    """
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Normalized audio file does not exist: {source}")
    try:
        from .audio_processor import ensure_ffmpeg_on_path
        ensure_ffmpeg_on_path()
    except Exception:
        pass

    try:
        import whisper
    except ImportError as error:
        raise TranscriptionError("Whisper is not installed. Run: pip install -r requirements.txt") from error
    try:
        model = whisper.load_model(model_name, device=device)
        options: dict[str, Any] = {"fp16": device == "cuda"}
        if language:
            options["language"] = language
        return model.transcribe(str(source), **options)
    except Exception as error:  # Whisper raises several backend-specific errors.
        raise TranscriptionError(f"Whisper transcription failed: {error}") from error


def format_transcript(
    result: dict[str, Any], *, audio_file: str, duration: float
) -> dict[str, Any]:
    """Map a native Whisper response to the stable downstream transcript schema."""
    segments = []
    for index, segment in enumerate(result.get("segments", [])):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if start < 0 or end < start:
            raise TranscriptionError(f"Whisper returned invalid timestamps in segment {index}.")
        segments.append({"id": len(segments), "start": start, "end": end, "text": text})
    return {
        "audio_file": audio_file,
        "duration": round(float(duration), 3),
        "language": result.get("language") or "unknown",
        "segments": segments,
    }
