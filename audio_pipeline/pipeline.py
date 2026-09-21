"""Command-line entry point for the audio-to-timestamped-transcript pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .audio_processor import normalize_audio, probe_duration
from .transcriber import format_transcript, transcribe_audio

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    device: str | None = None,
    keep_normalized_audio: bool = True,
    force: bool = False,
) -> Path:
    """Create ``transcript.json`` from an audio or video recording."""
    source = Path(input_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    processed_dir = destination.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_audio = processed_dir / f"{source.stem}.normalized.wav"
    transcript_path = destination / "transcript.json"

    if transcript_path.exists() and not force:
        LOGGER.info("Transcript already exists at %s. Skipping transcription.", transcript_path)
        return transcript_path

    LOGGER.info("Normalizing %s", source.name)
    normalize_audio(source, normalized_audio)
    duration = probe_duration(normalized_audio)
    LOGGER.info("Transcribing %.2f seconds with Whisper '%s'", duration, model_name)
    whisper_result = transcribe_audio(
        normalized_audio, model_name=model_name, language=language, device=device
    )
    transcript = format_transcript(
        whisper_result, audio_file=str(source), duration=duration
    )
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not keep_normalized_audio:
        normalized_audio.unlink(missing_ok=True)
    LOGGER.info("Wrote %s (%d segments)", transcript_path, len(transcript["segments"]))
    return transcript_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a timestamped Whisper transcript from media.")
    parser.add_argument("input_file", help="Path to an audio or video file supported by FFmpeg.")
    parser.add_argument("--output-dir", default="data/output", help="Directory for transcript.json.")
    parser.add_argument("--model", default="base", help="Whisper model name (default: base).")
    parser.add_argument("--language", help="Optional ISO language code, e.g. en.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Whisper device (default: automatic).")
    parser.add_argument("--discard-wav", action="store_true", help="Delete normalized WAV after transcription.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        output = run_pipeline(
            args.input_file, args.output_dir, model_name=args.model, language=args.language,
            device=args.device, keep_normalized_audio=not args.discard_wav,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
