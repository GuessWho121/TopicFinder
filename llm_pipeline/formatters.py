"""Multi-format exporters for TopicFinder chapter data.

All functions are pure and stateless — they take the ``chapters_data`` dict
produced by ``ChapterAnalyzer.analyze_all()`` and return formatted strings
or dicts ready for download or display.
"""

from __future__ import annotations

import json
from typing import Any


def _fmt_timestamp(seconds: float) -> str:
    """Convert a float seconds value to ``HH:MM:SS`` or ``MM:SS`` string."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT ``HH:MM:SS.mmm`` format."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def to_youtube_chapters(chapters_data: dict[str, Any]) -> str:
    """Return a YouTube chapter description string.

    Format::

        00:00 Chapter Title
        01:30 Another Chapter
        ...

    The first chapter is always pinned to ``00:00`` per YouTube requirements.
    """
    chapters = chapters_data.get("chapters", [])
    lines = []
    for i, ch in enumerate(chapters):
        ts = _fmt_timestamp(ch.get("start_time", 0.0))
        # YouTube requires first chapter at 00:00
        if i == 0:
            ts = "00:00"
        title = ch.get("title", f"Chapter {ch.get('section_id', i + 1)}")
        lines.append(f"{ts} {title}")
    return "\n".join(lines)


def to_podcasting2_json(chapters_data: dict[str, Any]) -> dict[str, Any]:
    """Return a Podcasting 2.0 chapters.json compatible dict.

    Spec: https://github.com/Podcastindex-org/podcast-namespace/blob/main/chapters/jsonChapters.md
    """
    chapters = chapters_data.get("chapters", [])
    pc2_chapters = []
    for ch in chapters:
        entry: dict[str, Any] = {
            "startTime": ch.get("start_time", 0.0),
            "title": ch.get("title", ""),
        }
        summary = ch.get("summary", "")
        if summary:
            entry["img"] = None  # placeholder; callers may add chapter artwork URLs
            entry["url"] = None
        pc2_chapters.append(entry)

    return {
        "version": "1.2.0",
        "chapters": pc2_chapters,
    }


def to_webvtt(chapters_data: dict[str, Any]) -> str:
    """Return a WebVTT chapter track string.

    The resulting ``.vtt`` file can be used as a ``<track kind="chapters">``
    element in an HTML5 video/audio player.
    """
    chapters = chapters_data.get("chapters", [])
    lines = ["WEBVTT", ""]
    for i, ch in enumerate(chapters, start=1):
        start = _fmt_timestamp_vtt(ch.get("start_time", 0.0))
        end = _fmt_timestamp_vtt(ch.get("end_time", ch.get("start_time", 0.0) + 1))
        title = ch.get("title", f"Chapter {i}")
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(title)
        lines.append("")
    return "\n".join(lines)


def to_markdown(chapters_data: dict[str, Any]) -> str:
    """Return a Markdown study guide string.

    Includes timestamp, title, summary, key takeaways, and key quotes for
    each chapter. Suitable for lecture notes or podcast summaries.
    """
    audio_file = chapters_data.get("audio_file", "Unknown source")
    duration = chapters_data.get("duration", 0.0)
    chapters = chapters_data.get("chapters", [])

    lines = [
        f"# 📚 Study Notes — {audio_file}",
        "",
        f"> **Duration:** {_fmt_timestamp(duration)}  "
        f"**Chapters:** {len(chapters)}",
        "",
        "---",
        "",
    ]

    for ch in chapters:
        ts = _fmt_timestamp(ch.get("start_time", 0.0))
        title = ch.get("title", "Untitled")
        summary = ch.get("summary", "")
        takeaways = ch.get("takeaways", [])
        key_quotes = ch.get("key_quotes", [])
        score = ch.get("key_moment_score", 0.0)
        rank = ch.get("key_moment_rank", 0)

        lines.append(f"## [{ts}] {title}")
        lines.append("")
        if summary:
            lines.append(f"**Summary:** {summary}")
            lines.append("")
        if takeaways:
            lines.append("**Key Takeaways:**")
            for t in takeaways:
                lines.append(f"- {t}")
            lines.append("")
        if key_quotes:
            lines.append("**Key Quotes:**")
            for q in key_quotes:
                lines.append(f"> _{q}_")
            lines.append("")
        lines.append(f"_Key Moment Score: {score:.2f} | Rank: #{rank}_")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def to_raw_json(chapters_data: dict[str, Any], *, indent: int = 2) -> str:
    """Return a pretty-printed JSON string of the full chapters output."""
    return json.dumps(chapters_data, ensure_ascii=False, indent=indent)
