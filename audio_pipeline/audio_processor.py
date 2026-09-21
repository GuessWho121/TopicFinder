"""Audio/video normalization helpers backed by the FFmpeg executable.

FFmpeg resolution order:
  1. System PATH  (ffmpeg / ffprobe)
  2. imageio-ffmpeg bundled binary (pip install imageio-ffmpeg)
  3. Known local install paths (VSCode extensions, common Windows locations)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class AudioProcessingError(RuntimeError):
    """Raised when FFmpeg cannot read or normalize a media file."""


# ---------------------------------------------------------------------------
# FFmpeg / FFprobe resolution
# ---------------------------------------------------------------------------

def ensure_ffmpeg_on_path() -> str:
    """Ensure a valid ffmpeg binary is in os.environ['PATH'] and return its path.

    This is critical on Windows because Whisper directly invokes `ffmpeg` via subprocess
    without a full path.
    """
    import os
    exe = _find_ffmpeg()
    exe_path = Path(exe).resolve()
    bin_dir = exe_path.parent

    # If the binary has a non-standard name (e.g. from imageio-ffmpeg), make sure
    # an ffmpeg.exe alias exists in the same folder
    if os.name == "nt":
        target = bin_dir / "ffmpeg.exe"
        if not target.is_file() and exe_path.is_file() and exe_path.name.lower() != "ffmpeg.exe":
            try:
                shutil.copyfile(str(exe_path), str(target))
            except Exception:
                pass

    dir_str = str(bin_dir)
    current_path = os.environ.get("PATH", "")
    if dir_str.lower() not in current_path.lower():
        os.environ["PATH"] = dir_str + os.pathsep + current_path

    return exe


def _find_ffmpeg() -> str:
    """Return a usable ffmpeg executable path, or raise AudioProcessingError."""
    # 1. System PATH
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass

    # 3. Known local paths (VSCode extensions, common installs)
    _candidates = [
        Path.home() / ".vscode" / "extensions",
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
        Path("C:/Program Files (x86)/ffmpeg/bin"),
    ]
    for base in _candidates:
        if not base.exists():
            continue
        if base.name == "extensions":
            # Search one level deep for any kilo/cursor/etc extension bin
            for p in sorted(base.glob("*/bin/ffmpeg.exe"), reverse=True):
                if p.is_file():
                    return str(p)
        else:
            candidate = base / "ffmpeg.exe"
            if candidate.is_file():
                return str(candidate)

    raise AudioProcessingError(
        "FFmpeg was not found. Install it and add it to PATH, "
        "or run: pip install imageio-ffmpeg"
    )


# Automatically configure PATH on module import
try:
    ensure_ffmpeg_on_path()
except Exception:
    pass


def _find_ffprobe() -> str | None:
    """Return a usable ffprobe executable path, or None if only ffmpeg is available."""
    if shutil.which("ffprobe"):
        return "ffprobe"

    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        for name in ("ffprobe.exe", "ffprobe"):
            probe = Path(exe).parent / name
            if probe.is_file():
                return str(probe)
    except Exception:
        pass

    try:
        ffmpeg_exe = _find_ffmpeg()
        for name in ("ffprobe.exe", "ffprobe"):
            probe = Path(ffmpeg_exe).parent / name
            if probe.is_file():
                return str(probe)
    except AudioProcessingError:
        pass

    return None  # ffprobe not available; probe_duration will use ffmpeg -i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_duration(media_path: str | Path, *, ffprobe_bin: str | None = None) -> float:
    """Return media duration in seconds.

    Uses ffprobe if available; falls back to parsing ``ffmpeg -i`` stderr output
    when only the imageio-ffmpeg binary (which ships without ffprobe) is present.
    """
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input media file does not exist: {path}")

    ffprobe = ffprobe_bin or _find_ffprobe()

    if ffprobe:
        # Happy path: proper ffprobe binary
        command = [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise AudioProcessingError(completed.stderr.strip() or "ffprobe failed.")
        try:
            duration = float(json.loads(completed.stdout)["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AudioProcessingError("FFprobe did not return a valid duration.") from error
    else:
        # Fallback: parse duration from ffmpeg -i stderr
        import re
        ffmpeg = _find_ffmpeg()
        completed = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True, text=True, check=False,
        )
        # ffmpeg -i always exits non-zero when no output is given; use stderr
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", completed.stderr)
        if not match:
            raise AudioProcessingError("Could not parse duration from ffmpeg output.")
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        duration = h * 3600 + m * 60 + s

    if duration < 0:
        raise AudioProcessingError("FFmpeg returned an invalid negative duration.")
    return duration


def normalize_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    ffmpeg_bin: str | None = None,
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

    ffmpeg = ffmpeg_bin or _find_ffmpeg()
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode or not target.is_file() or target.stat().st_size == 0:
        message = completed.stderr.strip() or "FFmpeg did not create normalized audio."
        raise AudioProcessingError(message)
    return target
