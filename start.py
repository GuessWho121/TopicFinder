"""
start.py - Unified Launcher for TopicFinder.

Preflight checks:
1. Verifies required dependencies and installs missing packages if requested.
2. Ensures project data directories exist (data/input, data/output, data/processed).
3. Verifies FFmpeg availability.
4. Launches the full-stack Streamlit web app in the default browser.

Usage:
    python start.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()

REQUIRED_PACKAGES = [
    ("streamlit", "streamlit"),
    ("plotly", "plotly"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("sentence_transformers", "sentence-transformers"),
    ("whisper", "openai-whisper"),
]


def ensure_directories():
    """Ensure required data directories exist."""
    for folder in ["input", "output", "processed"]:
        ( _ROOT / "data" / folder ).mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible in PATH."""
    return shutil.which("ffmpeg") is not None


def check_and_install_dependencies():
    """Check required packages and prompt/install if missing."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[TopicFinder] Missing dependencies detected: {', '.join(missing)}")
        print(f"[TopicFinder] Installing missing packages...")

        # 1. Try uv if available
        if shutil.which("uv"):
            try:
                subprocess.run(["uv", "pip", "install"] + missing, check=True)
                print("[TopicFinder] Dependencies installed successfully via uv.")
                return
            except Exception:
                pass

        # 2. Try standard python -m pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            print("[TopicFinder] Dependencies installed successfully via pip.")
            return
        except subprocess.CalledProcessError:
            # 3. Try bootstrapping pip via ensurepip
            try:
                subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
                print("[TopicFinder] Dependencies installed successfully via bootstrapped pip.")
                return
            except Exception as e:
                print(f"[TopicFinder] Warning: Auto-install failed ({e}).")
                print("Please run manually: uv pip install -r requirements.txt")


def launch_app():
    """Launch Streamlit dashboard."""
    ensure_directories()
    check_and_install_dependencies()

    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("[TopicFinder] Notice: FFmpeg is not found in PATH.")
        print("               Pre-transcribed demo files will work instantly.")
        print("               To transcribe new raw audio/video files, install FFmpeg.")

    port = 8501
    app_path = _ROOT / "app.py"

    print("\n" + "=" * 60)
    print(" 🎙️  TopicFinder: Podcast & Lecture Key Moment Extractor")
    print(f" 🚀  Launching Web Dashboard at: http://localhost:{port}")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        "4096",
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        subprocess.run(cmd, cwd=str(_ROOT))
    except KeyboardInterrupt:
        print("\n[TopicFinder] Dashboard stopped.")


if __name__ == "__main__":
    launch_app()
