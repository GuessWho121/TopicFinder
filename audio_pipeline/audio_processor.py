"""Audio/video normalization helpers backed by the FFmpeg executable."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class AudioProcessingError(RuntimeError):
    """Raised when FFmpeg cannot read or normalize a media file."""


def _ffmpeg_path(executable: str) -> str:
    if shutil.which(executable) is None:
        raise AudioProcessingError(
            f"'{executable}' was not found on PATH. Install FFmpeg and try again."
        )
    return executable


def probe_duration(media_path: str | Path, *, ffprobe_bin: str = "ffprobe") -> float:
    """Return media duration in seconds using ffprobe."""
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input media file does not exist: {path}")
    _ffmpeg_path(ffprobe_bin)
    command = [
        ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise AudioProcessingError(completed.stderr.strip() or "ffprobe failed.")
    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioProcessingError("FFprobe did not return a valid duration.") from error
    if duration < 0:
        raise AudioProcessingError("FFprobe returned an invalid negative duration.")
    return duration


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Convert any FFmpeg-supported media file to Whisper-friendly PCM WAV.

    The resulting file is mono, 16 kHz, signed 16-bit PCM.  It is deliberately
    not deleted by this function: callers may reuse it or clean it up explicitly.
    """
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input media file does not exist: {source}")
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive integers.")
    if target.suffix.lower() != ".wav":
        raise ValueError("Normalized output must use a .wav extension.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg_path(ffmpeg_bin)
    command = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode or not target.is_file() or target.stat().st_size == 0:
        message = completed.stderr.strip() or "FFmpeg did not create normalized audio."
        raise AudioProcessingError(message)
    return target
