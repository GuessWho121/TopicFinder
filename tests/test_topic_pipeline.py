"""Tests for the 100% topic segmentation and key moment extraction pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic_pipeline.detector import TopicDetector
from topic_pipeline.pipeline import run_pipeline


def test_topic_detection_100_percent():
    fixture_path = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    assert fixture_path.is_file(), f"Fixture not found at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    # 1. Test Standard Sensitivity
    detector = TopicDetector(sensitivity="medium")
    result = detector.process(transcript_data)

    assert result["total_sections"] >= 2, f"Expected >= 2 sections, got {result['total_sections']}"
    assert "similarity_curve" in result
    assert len(result["similarity_curve"]["timestamps"]) == len(result["similarity_curve"]["similarities"])
    assert len(result["sections"]) == result["total_sections"]

    # 2. Verify Key Moment Ranking & Badges
    ranks = [sec["key_moment_rank"] for sec in result["sections"]]
    assert sorted(ranks) == list(range(1, len(result["sections"]) + 1)), "Ranks must be a unique permutation 1..N"

    for sec in result["sections"]:
        assert 0.0 <= sec["key_moment_score"] <= 1.0
        assert sec["key_moment_badge"] in ["🔥 Major Shift", "⭐ Key Concept", "📌 Core Topic"]
        assert "metrics" in sec
        assert sec["start_time"] >= 0.0
        assert sec["end_time"] > sec["start_time"]
        assert len(sec["text"]) > 0
        assert "timestamp_str" in sec
        badge_clean = sec['key_moment_badge'].encode('ascii', 'ignore').decode('ascii').strip()
        print(f"  [{sec['timestamp_str']}] (Rank #{sec['key_moment_rank']} - [{badge_clean}] - Score: {sec['key_moment_score']}) {sec['text'][:55]}...")

    # 3. Test Boundary Snapping
    segment_timestamps = set()
    for s in transcript_data["segments"]:
        segment_timestamps.add(round(s["start"], 2))
        segment_timestamps.add(round(s["end"], 2))

    for sec in result["sections"][1:]:
        # Snapped boundary must match a segment start/end
        assert round(sec["start_time"], 2) in segment_timestamps, f"Boundary {sec['start_time']} not snapped to transcript segments"

    # 4. Test Sensitivity Scaling (Coarse vs Fine)
    coarse_res = TopicDetector(sensitivity="coarse").process(transcript_data)
    fine_res = TopicDetector(sensitivity="fine").process(transcript_data)
    assert fine_res["total_sections"] >= coarse_res["total_sections"], "Fine sensitivity must produce >= coarse sections"

    print("\n[OK] All 100% topic pipeline unit & integration tests passed successfully.")


if __name__ == "__main__":
    test_topic_detection_100_percent()
