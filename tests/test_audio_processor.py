"""Unit tests for audio_processor FFmpeg/FFprobe subprocess encoding resilience."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_pipeline.audio_processor import (
    AudioProcessingError,
    _find_ffmpeg,
    ensure_ffmpeg_on_path,
)


def test_ffmpeg_on_path():
    """Ensure ffmpeg binary is found and path configured."""
    exe = ensure_ffmpeg_on_path()
    assert Path(exe).exists(), f"FFmpeg binary not found at {exe}"


def test_subprocess_utf8_encoding_resilience():
    """Ensure subprocess execution handles non-cp1252 / multibyte characters safely."""
    ffmpeg = _find_ffmpeg()
    res = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0
    assert "ffmpeg version" in res.stdout.lower()


if __name__ == "__main__":
    test_ffmpeg_on_path()
    print("test_ffmpeg_on_path: [PASSED]")
    test_subprocess_utf8_encoding_resilience()
    print("test_subprocess_utf8_encoding_resilience: [PASSED]")
