"""Unit tests for the LLM pipeline (Person 3).

All tests use the heuristic backend so they run offline with zero API keys.
Run with:
    python tests/test_llm_pipeline.py
or:
    pytest tests/test_llm_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on path and stdout/stderr handle UTF-8 on Windows
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llm_pipeline.analyzer import (
    ChapterAnalyzer,
    _heuristic_analyze,
    _parse_llm_json_list,
)
from llm_pipeline.formatters import (
    to_markdown,
    to_podcasting2_json,
    to_raw_json,
    to_webvtt,
    to_youtube_chapters,
    to_zip_bundle,
)

_DEMO_PATH = Path(__file__).parent.parent / "demo_data" / "sample_topic_segments.json"


def _load_demo() -> dict:
    assert _DEMO_PATH.is_file(), f"Demo fixture not found: {_DEMO_PATH}"
    with open(_DEMO_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Analyzer tests ─────────────────────────────────────────────────────────

def test_heuristic_analyze_single_section():
    """Heuristic backend returns all required keys for one section."""
    result = _heuristic_analyze("Self-attention computes pairwise dot products. This allows parallel training. The transformer architecture scales efficiently.")
    assert "title" in result, "Missing 'title'"
    assert "summary" in result, "Missing 'summary'"
    assert isinstance(result["takeaways"], list), "'takeaways' must be a list"
    assert isinstance(result["key_quotes"], list), "'key_quotes' must be a list"
    assert len(result["title"]) > 0, "Title should not be empty"
    print(f"  title: {result['title']}")
    print(f"  summary: {result['summary'][:60]}...")


def test_heuristic_analyze_empty_text():
    """Heuristic handles empty text gracefully."""
    result = _heuristic_analyze("")
    assert "title" in result


def test_parse_llm_json_list_formats():
    """_parse_llm_json_list correctly extracts lists from raw, fenced, and nested formats."""
    # 1. Plain array
    raw1 = '[{"section_id": 1, "title": "Introduction", "summary": "Start of talk."}]'
    res1 = _parse_llm_json_list(raw1)
    assert len(res1) == 1 and res1[0]["title"] == "Introduction"

    # 2. Markdown fenced json
    raw2 = '```json\n[{"section_id": 2, "title": "Key Concept", "summary": "Core topic."}]\n```'
    res2 = _parse_llm_json_list(raw2)
    assert len(res2) == 1 and res2[0]["title"] == "Key Concept"

    # 3. Wrapped in dict under 'chapters'
    raw3 = '{"chapters": [{"section_id": 3, "title": "Wrap Up", "summary": "Concluding thoughts."}]}'
    res3 = _parse_llm_json_list(raw3)
    assert len(res3) == 1 and res3[0]["title"] == "Wrap Up"


def test_chapter_analyzer_heuristic_all_sections():
    """ChapterAnalyzer(heuristic) enriches all sections from demo fixture."""
    topic_segments = _load_demo()
    analyzer = ChapterAnalyzer(backend="heuristic")
    assert analyzer.active_backend == "heuristic"

    chapters_data = analyzer.analyze_all(topic_segments)

    n_sections = len(topic_segments.get("sections", []))
    chapters = chapters_data.get("chapters", [])
    assert len(chapters) == n_sections, f"Expected {n_sections} chapters, got {len(chapters)}"

    for ch in chapters:
        assert ch.get("title"), f"Chapter {ch.get('section_id')} missing title"
        assert ch.get("summary") is not None, f"Chapter {ch.get('section_id')} missing summary"
        assert isinstance(ch.get("takeaways"), list)
        assert isinstance(ch.get("key_quotes"), list)
        assert "key_moment_rank" in ch
        assert "start_time" in ch
        assert "end_time" in ch
        print(f"  [{ch['timestamp_str']}] {ch['title']}")


def test_chapter_analyzer_rank_assignment():
    """Ranks are assigned correctly (highest score → rank 1)."""
    topic_segments = _load_demo()
    analyzer = ChapterAnalyzer(backend="heuristic")
    chapters_data = analyzer.analyze_all(topic_segments)
    chapters = chapters_data.get("chapters", [])

    ranks = [ch["key_moment_rank"] for ch in chapters]
    assert 1 in ranks, "Rank 1 should be present"
    assert len(set(ranks)) == len(ranks), "All ranks should be unique"

    # The chapter with highest key_moment_score should have rank 1
    best = max(chapters, key=lambda c: c.get("key_moment_score", 0))
    assert best["key_moment_rank"] == 1, f"Best score chapter should be rank 1, got {best['key_moment_rank']}"


# ── Formatter tests ────────────────────────────────────────────────────────

def _get_chapters_data() -> dict:
    topic_segments = _load_demo()
    analyzer = ChapterAnalyzer(backend="heuristic")
    return analyzer.analyze_all(topic_segments)


def test_youtube_chapters_format():
    """YouTube chapters start at 00:00 and have one line per chapter."""
    chap_data = _get_chapters_data()
    result = to_youtube_chapters(chap_data)
    lines = [l for l in result.strip().splitlines() if l.strip()]
    assert lines[0].startswith("00:00"), f"First line must start with 00:00, got: {lines[0]}"
    assert len(lines) == len(chap_data["chapters"]), "Should have one line per chapter"
    print(f"  YouTube chapters:\n{result}")


def test_webvtt_format():
    """WebVTT output starts with WEBVTT header and contains --> cues."""
    chap_data = _get_chapters_data()
    result = to_webvtt(chap_data)
    assert result.startswith("WEBVTT"), "VTT must start with WEBVTT"
    assert " --> " in result, "VTT must contain --> cue markers"
    print(f"  WebVTT (first 200 chars): {result[:200]}")


def test_podcasting2_json_format():
    """Podcasting 2.0 JSON has required version and chapters keys."""
    chap_data = _get_chapters_data()
    result = to_podcasting2_json(chap_data)
    assert "version" in result, "Podcasting 2.0 JSON must have 'version'"
    assert "chapters" in result, "Podcasting 2.0 JSON must have 'chapters'"
    assert isinstance(result["chapters"], list)
    assert len(result["chapters"]) == len(chap_data["chapters"])
    for ch in result["chapters"]:
        assert "startTime" in ch
        assert "title" in ch


def test_markdown_format():
    """Markdown output contains headings and timestamps."""
    chap_data = _get_chapters_data()
    result = to_markdown(chap_data)
    assert "##" in result, "Markdown should contain ## headings"
    assert "---" in result, "Markdown should contain horizontal rules"
    assert "Summary:" in result, "Markdown should contain Summary labels"
    print(f"  Markdown (first 300 chars):\n{result[:300]}")


def test_raw_json_format():
    """Raw JSON is valid JSON with chapters key."""
    chap_data = _get_chapters_data()
    result = to_raw_json(chap_data)
    parsed = json.loads(result)
    assert "chapters" in parsed
    assert "total_chapters" in parsed




def test_zip_bundle_format():
    """ZIP bundle contains all 5 output assets and is non-empty."""
    import io
    import zipfile
    chap_data = _get_chapters_data()
    bundle_bytes = to_zip_bundle(chap_data)
    assert len(bundle_bytes) > 0, "ZIP bundle must not be empty"
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "youtube_chapters.txt" in names
        assert "chapters.json" in names
        assert "chapters.vtt" in names
        assert "study_notes.md" in names
        assert "topicfinder_output.json" in names


def test_gemini_analyze_batch_mocked():
    """Verify _gemini_analyze_batch processes all sections in a single call."""
    from unittest.mock import MagicMock
    from llm_pipeline.analyzer import _gemini_analyze_batch
    import google.genai

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "chapters": [
            {
                "section_id": 1,
                "title": "Starting New Habits",
                "summary": "Trying something new for 30 days is memorable.",
                "takeaways": ["Take daily action", "Memories stick"],
                "key_quotes": ["30 days is just about right"]
            },
            {
                "section_id": 2,
                "title": "Growing Self Confidence",
                "summary": "Challenges boost adventurousness.",
                "takeaways": ["Bike to work", "Hike Kilimanjaro"],
                "key_quotes": ["You can do anything for 30 days"]
            }
        ]
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    original_client = google.genai.Client

    try:
        google.genai.Client = MagicMock(return_value=mock_client)
        sections = [
            {"section_id": 1, "start_time": 0.0, "end_time": 75.0, "text": "Section 1 text"},
            {"section_id": 2, "start_time": 75.0, "end_time": 125.0, "text": "Section 2 text"},
        ]
        res = _gemini_analyze_batch(sections, api_key="test-key")
        assert len(res) == 2
        assert res[0]["title"] == "Starting New Habits"
        assert res[1]["title"] == "Growing Self Confidence"
        assert len(res[0]["takeaways"]) == 2
        # Verify exactly ONE API call was made for both sections
        assert mock_client.models.generate_content.call_count == 1
    finally:
        google.genai.Client = original_client


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_heuristic_analyze_single_section,
        test_heuristic_analyze_empty_text,
        test_chapter_analyzer_heuristic_all_sections,
        test_chapter_analyzer_rank_assignment,
        test_youtube_chapters_format,
        test_webvtt_format,
        test_podcasting2_json_format,
        test_markdown_format,
        test_raw_json_format,
        test_zip_bundle_format,
    ]
    failed = 0
    for test in tests:
        try:
            print(f"\n>> {test.__name__}")
            test()
            print(f"  [PASSED]")
        except Exception as exc:
            print(f"  [FAILED]: {exc}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)
