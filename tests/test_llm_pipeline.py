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

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_pipeline.analyzer import ChapterAnalyzer, _heuristic_analyze
from llm_pipeline.formatters import (
    to_markdown,
    to_podcasting2_json,
    to_raw_json,
    to_webvtt,
    to_youtube_chapters,
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
