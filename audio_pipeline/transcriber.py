"""Whisper transcription and conversion to the project's JSON contract."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when Whisper cannot create a transcript."""


def _get_gpu_info() -> tuple[bool, str]:
    """Check if PyTorch has CUDA available and return (is_available, device_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
    except Exception:
        pass
    return False, "None"


def _setup_cuda_dlls() -> None:
    """Ensure torch lib is registered in Windows DLL search path so cublas DLLs are found."""
    import os
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
            current_path = os.environ.get("PATH", "")
            if torch_lib.lower() not in current_path.lower():
                os.environ["PATH"] = torch_lib + os.pathsep + current_path
    except Exception:
        pass


def _is_cublas12_loadable() -> bool:
    """Check if cublas64_12.dll is actually available for ctranslate2 (faster-whisper)."""
    _setup_cuda_dlls()
    try:
        import ctypes
        ctypes.CDLL("cublas64_12.dll")
        return True
    except Exception:
        return False


def _run_openai_whisper(
    source_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str,
) -> dict[str, Any]:
    """Run transcription using openai-whisper on PyTorch (CUDA GPU accelerated)."""
    import whisper

    loaded_model = None
    # Try requested model (e.g. 'base'), fallback to 'tiny' if requested is not yet downloaded
    for candidate in [model_name, "tiny"]:
        try:
            LOGGER.info("Loading OpenAI Whisper '%s' model on %s...", candidate, device)
            loaded_model = whisper.load_model(candidate, device=device)
            break
        except Exception as load_err:
            LOGGER.warning("Could not load whisper model '%s': %s", candidate, load_err)

    if loaded_model is None:
        raise TranscriptionError(f"Failed to load Whisper model '{model_name}'.")

    kwargs: dict[str, Any] = {
        "fp16": (device == "cuda"),
        "verbose": False,
    }
    if language:
        kwargs["language"] = language

    LOGGER.info("Transcribing on %s with OpenAI Whisper...", device)
    res = loaded_model.transcribe(str(source_path), **kwargs)

    segments = []
    for s in res.get("segments", []):
        txt = str(s.get("text", "")).strip()
        if txt:
            segments.append({
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": txt,
            })

    return {
        "language": res.get("language") or language or "en",
        "segments": segments,
    }


def _run_faster_whisper(
    source_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> dict[str, Any]:
    """Run transcription using faster-whisper (ctranslate2)."""
    from faster_whisper import WhisperModel

    kwargs: dict[str, Any] = {
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=400),
    }
    if language:
        kwargs["language"] = language

    mdl = WhisperModel(model_name, device=device, compute_type=compute_type)
    segs_iter, info = mdl.transcribe(str(source_path), **kwargs)
    segments = []
    for s in segs_iter:
        txt = s.text.strip()
        if txt:
            segments.append({"start": float(s.start), "end": float(s.end), "text": txt})
    return {
        "language": getattr(info, "language", language or "en"),
        "segments": segments,
    }


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run Whisper and return its native result dictionary.

    Device selection:
      - NVIDIA GPU detected -> cuda (PyTorch / OpenAI Whisper or faster-whisper)
      - No GPU / fallback   -> cpu / int8
    """
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Normalized audio file does not exist: {source}")

    try:
        from .audio_processor import ensure_ffmpeg_on_path
        ensure_ffmpeg_on_path()
    except Exception as e:
        LOGGER.warning("FFmpeg path setup warning: %s", e)

    has_cuda, gpu_name = _get_gpu_info()
    use_cuda = (device == "cuda") or (device is None and has_cuda)

    # 1. GPU Acceleration Path
    if use_cuda:
        LOGGER.info("NVIDIA GPU detected: %s. Enabling GPU acceleration!", gpu_name)

        # Check if faster-whisper has CUDA 12 runtime
        if _is_cublas12_loadable():
            try:
                LOGGER.info("Trying faster-whisper on CUDA...")
                return _run_faster_whisper(
                    source,
                    model_name=model_name,
                    language=language,
                    device="cuda",
                    compute_type="float16",
                )
            except Exception as fw_err:
                LOGGER.warning("faster-whisper CUDA failed (%s). Falling back to PyTorch Whisper on GPU.", fw_err)

        # Run native PyTorch Whisper on CUDA (utilizes RTX 4060 GPU directly)
        try:
            LOGGER.info("Running OpenAI Whisper on NVIDIA GPU (%s)...", gpu_name)
            return _run_openai_whisper(
                source,
                model_name=model_name,
                language=language,
                device="cuda",
            )
        except Exception as whisper_gpu_err:
            LOGGER.warning("OpenAI Whisper GPU execution failed: %s. Falling back to CPU.", whisper_gpu_err)

    # 2. CPU Fallback Path
    LOGGER.info("Running transcription on CPU...")
    try:
        return _run_faster_whisper(
            source,
            model_name=model_name,
            language=language,
            device="cpu",
            compute_type="int8",
        )
    except Exception as fw_cpu_err:
        LOGGER.warning("faster-whisper CPU failed (%s). Trying OpenAI Whisper on CPU...", fw_cpu_err)

    try:
        return _run_openai_whisper(
            source,
            model_name=model_name,
            language=language,
            device="cpu",
        )
    except Exception as whisper_cpu_err:
        raise TranscriptionError(f"All transcription backends failed. Last error: {whisper_cpu_err}") from whisper_cpu_err


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
            raise TranscriptionError(
                f"Whisper returned invalid timestamps in segment {index}."
            )
        segments.append({"id": len(segments), "start": start, "end": end, "text": text})
    return {
        "audio_file": audio_file,
        "duration": round(float(duration), 3),
        "language": result.get("language") or "unknown",
        "segments": segments,
    }
