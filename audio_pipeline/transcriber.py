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

    # Priority 1: faster-whisper (CTranslate2 INT8 backend with VAD filtering - 4x-8x faster on CPU)
    try:
        from faster_whisper import WhisperModel

        target_device = device or ("cuda" if _is_cuda_available() else "cpu")
        compute_type = "float16" if target_device == "cuda" else "int8"
        model = WhisperModel(model_name, device=target_device, compute_type=compute_type)

        kwargs: dict[str, Any] = {
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=400),
        }
        if language:
            kwargs["language"] = language

        segments_iter, info = model.transcribe(str(source), **kwargs)
        segments_list = []
        for s in segments_iter:
            txt = s.text.strip()
            if txt:
                segments_list.append({"start": float(s.start), "end": float(s.end), "text": txt})

        return {
            "language": getattr(info, "language", language or "en"),
            "segments": segments_list,
        }
    except ImportError:
        pass
    except Exception as fw_err:
        import logging
        logging.getLogger(__name__).warning("faster-whisper failed (%s); falling back to standard whisper", fw_err)

    # Priority 2: standard openai-whisper fallback
    try:
        import whisper
    except ImportError as error:
        raise TranscriptionError("Neither faster-whisper nor openai-whisper is installed.") from error
    try:
        model = whisper.load_model(model_name, device=device)
        options: dict[str, Any] = {"fp16": device == "cuda"}
        if language:
            options["language"] = language
        return model.transcribe(str(source), **options)
    except Exception as error:
        raise TranscriptionError(f"Whisper transcription failed: {error}") from error


def _is_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
