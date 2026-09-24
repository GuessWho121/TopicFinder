"""
run.py - Unified end-to-end command-line runner for TopicFinder.

Executes the entire 3-stage pipeline in a single command:
  1. audio_pipeline: Media extraction & Whisper transcription -> data/output/transcript.json
  2. topic_pipeline: Semantic chunking & topic detection       -> data/output/topic_segments.json
  3. llm_pipeline:   LLM chapter synthesis & format exports     -> data/output/chapters.json

Usage:
  python run.py data/input/lecture.mp4
  python run.py data/input/podcast.mp3 --format youtube
  python run.py data/input/lecture.wav --format markdown --backend gemini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add repo root to path
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audio_pipeline.pipeline import run_pipeline as run_audio_stage
from topic_pipeline.detector import TopicDetector
from llm_pipeline.analyzer import ChapterAnalyzer
from llm_pipeline import formatters

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("TopicFinder")


def run_full_pipeline(
    input_media: str | Path,
    output_dir: str | Path = "data/output",
    *,
    whisper_model: str = "base",
    sensitivity: str = "medium",
    backend: str = "auto",
    gemini_api_key: str | None = None,
    export_format: str = "all",
) -> dict[str, Any]:
    source = Path(input_media).expanduser().resolve()
    dest = Path(output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    LOGGER.info("==================================================")
    LOGGER.info("  TopicFinder: End-to-End Execution")
    LOGGER.info("  Input: %s", source.name)
    LOGGER.info("==================================================")

    # Stage 1: Audio & Transcription
    LOGGER.info("[Stage 1/3] Running Whisper Transcription (%s)...", whisper_model)
    transcript_path = run_audio_stage(source, dest, model_name=whisper_model, force=True)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    LOGGER.info("✓ Stage 1 complete: %d segments transcribed.", len(transcript_data.get("segments", [])))

    # Stage 2: Semantic Chunking & Topic Detection
    LOGGER.info("[Stage 2/3] Running Topic Boundary Detection (sensitivity='%s')...", sensitivity)
    detector = TopicDetector(sensitivity=sensitivity)
    topic_results = detector.process(transcript_data)
    topic_path = dest / "topic_segments.json"
    topic_path.write_text(json.dumps(topic_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("✓ Stage 2 complete: %d topic sections identified.", topic_results["total_sections"])

    # Stage 3: LLM Chapter Analysis
    LOGGER.info("[Stage 3/3] Running LLM Chapter Synthesis (backend='%s')...", backend)
    analyzer = ChapterAnalyzer(backend=backend, gemini_api_key=gemini_api_key)
    chapters_data = analyzer.analyze_all(topic_results)
    chapters_path = dest / "chapters.json"
    chapters_path.write_text(json.dumps(chapters_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("✓ Stage 3 complete: %d chapters generated.", len(chapters_data.get("chapters", [])))

    # Export Formats
    yt_text = formatters.to_youtube_chapters(chapters_data)
    (dest / "youtube_chapters.txt").write_text(yt_text + "\n", encoding="utf-8")

    md_text = formatters.to_markdown(chapters_data)
    (dest / "summary_notes.md").write_text(md_text + "\n", encoding="utf-8")

    vtt_text = formatters.to_webvtt(chapters_data)
    (dest / "chapters.vtt").write_text(vtt_text + "\n", encoding="utf-8")

    p20_text = formatters.to_podcasting20_json(chapters_data)
    (dest / "podcasting20_chapters.json").write_text(p20_text + "\n", encoding="utf-8")

    LOGGER.info("==================================================")
    LOGGER.info("✓ All outputs saved to: %s", dest)
    LOGGER.info("==================================================")

    if export_format == "youtube":
        print("\n--- YouTube Description Chapters ---\n" + yt_text)
    elif export_format == "markdown":
        print("\n--- Markdown Summary ---\n" + md_text)
    elif export_format == "vtt":
        print("\n--- WebVTT Cue Track ---\n" + vtt_text)
    elif export_format == "all":
        print("\n--- Generated Chapters Preview ---\n" + yt_text)

    return chapters_data


def main():
    parser = argparse.ArgumentParser(description="TopicFinder: End-to-end Audio/Video to Chapter Pipeline")
    parser.add_argument("input_media", help="Path to audio or video file (.mp3, .wav, .mp4, etc.)")
    parser.add_argument("--output-dir", "-o", default="data/output", help="Directory to save generated outputs")
    parser.add_argument("--model", "-m", default="base", choices=["tiny", "base", "small"], help="Whisper model size")
    parser.add_argument("--sensitivity", "-s", default="medium", choices=["coarse", "medium", "fine"], help="Topic sensitivity")
    parser.add_argument("--backend", "-b", default="auto", choices=["auto", "gemini", "ollama", "heuristic"], help="LLM backend")
    parser.add_argument("--api-key", help="Gemini API Key (optional)")
    parser.add_argument("--format", "-f", default="all", choices=["youtube", "markdown", "vtt", "all"], help="Console preview format")
    args = parser.parse_args()

    try:
        run_full_pipeline(
            args.input_media,
            output_dir=args.output_dir,
            whisper_model=args.model,
            sensitivity=args.sensitivity,
            backend=args.backend,
            gemini_api_key=args.api_key,
            export_format=args.format,
        )
    except Exception as e:
        LOGGER.error("Execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
