"""Tests for the topic segmentation pipeline."""

import sys
import json
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_pipeline.detector import TopicDetector


def test_topic_detection_on_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    assert fixture_path.is_file(), f"Fixture not found at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    detector = TopicDetector(window_sec=75.0, step_sec=25.0)
    result = detector.process(transcript_data)

    assert result["total_sections"] >= 2, f"Expected >= 2 sections, got {result['total_sections']}"
    assert "similarity_curve" in result
    assert len(result["similarity_curve"]["timestamps"]) == len(result["similarity_curve"]["similarities"])
    assert len(result["sections"]) == result["total_sections"]

    for sec in result["sections"]:
        assert sec["start_time"] >= 0.0
        assert sec["end_time"] > sec["start_time"]
        assert len(sec["text"]) > 0
        assert "timestamp_str" in sec


if __name__ == "__main__":
    test_topic_detection_on_fixture()
    print("[OK] test_topic_detection_on_fixture passed.")
