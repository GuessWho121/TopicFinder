"""LLM-backed chapter analyzer with cascading backend fallback.

Priority order:
  1. Gemini Flash (google-generativeai + GEMINI_API_KEY)
  2. Ollama local  (requests + running ollama server)
  3. Heuristic     (pure Python, zero deps)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

LOGGER = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a podcast and lecture chapter generator.

Given the transcript of a single topic section, output ONLY a valid JSON object with these exact keys:
- "title": a punchy 3-6 word chapter heading
- "summary": 1-2 sentences describing the main concept
- "takeaways": a list of 2-3 actionable or insightful bullet points
- "key_quotes": a list of 1-2 memorable verbatim phrases from the text

Respond with ONLY the JSON object, no markdown fences, no explanation.

Transcript section:
\"\"\"
{text}
\"\"\"
"""

_BATCH_PROMPT_TEMPLATE = """You are an expert podcast and lecture chapter editor.

Below is the full transcript of a recording, split into {total_sections} consecutive candidate topic sections with timestamps.
Having the entire recording context from beginning to end, analyze each section and generate high-quality, distinct, non-repetitive chapter titles, summaries, key takeaways, and memorable verbatim quotes.

Sections:
{sections_text}

Output ONLY a valid JSON object with a single "chapters" array matching this exact schema:
{{
  "chapters": [
    {{
      "section_id": <int matching the section id>,
      "title": "<punchy 3-6 word chapter heading>",
      "summary": "<1-2 sentences describing the main concept in this section>",
      "takeaways": [
        "<actionable or insightful bullet point 1>",
        "<actionable or insightful bullet point 2>"
      ],
      "key_quotes": [
        "<memorable verbatim phrase from this section's text>"
      ]
    }}
  ]
}}

Ensure every section_id from 1 to {total_sections} is included in the "chapters" list in chronological order.
Respond with ONLY the JSON object, no markdown fences, no explanation.
"""


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


# ---------------------------------------------------------------------------
# Backend: Heuristic (pure Python)
# ---------------------------------------------------------------------------

def _heuristic_analyze(text: str) -> dict[str, Any]:
    """Deterministic fallback: no LLM required."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # Title: first 6 words of first sentence, title-cased
    raw_words = sentences[0].split() if sentences else ["Untitled", "Section"]
    title_words = raw_words[:6]
    title = " ".join(title_words).rstrip(".,!?")

    # Summary: first 2 sentences
    summary = " ".join(sentences[:2]) if len(sentences) >= 2 else sentences[0] if sentences else ""

    # Takeaways: sentences 3–5 (if available), else fallback messages
    takeaway_sents = sentences[2:5] if len(sentences) >= 3 else sentences
    takeaways = [s for s in takeaway_sents[:3]] or ["Review this section for key insights."]

    # Key quotes: longest sentence(s) (most informative)
    sorted_sents = sorted(sentences, key=len, reverse=True)
    key_quotes = [sorted_sents[0]] if sorted_sents else [text[:120]]

    return {
        "title": title,
        "summary": summary,
        "takeaways": takeaways,
        "key_quotes": key_quotes,
    }


# ---------------------------------------------------------------------------
# Backend: Gemini Flash
# ---------------------------------------------------------------------------

def _gemini_analyze(text: str, api_key: str, model: str = "models/gemini-3.5-flash-lite") -> dict[str, Any]:
    try:
        from google import genai
    except ImportError as err:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai") from err

    client = genai.Client(api_key=api_key)
    prompt = _PROMPT_TEMPLATE.format(text=text[:3000])  # Guard token limits
    
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    
    return _parse_llm_json(response.text)


def _gemini_analyze_batch(
    sections: list[dict[str, Any]],
    api_key: str,
    model: str = "models/gemini-3.5-flash-lite",
) -> list[dict[str, Any]]:
    """Analyze all sections in a single unified Gemini API call with full recording context."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as err:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai") from err

    client = genai.Client(api_key=api_key)

    formatted_sections = []
    for idx, s in enumerate(sections):
        sid = s.get("section_id", idx + 1)
        ts_str = s.get("timestamp_str", f"{s.get('start_time', 0.0):.1f}s")
        text = s.get("text", "").strip()
        formatted_sections.append(
            f"--- Section {sid} (Timestamp: {ts_str}) ---\n{text}"
        )

    full_text = "\n\n".join(formatted_sections)
    prompt = _BATCH_PROMPT_TEMPLATE.format(
        total_sections=len(sections),
        sections_text=full_text,
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    parsed = _parse_llm_json(response.text)
    chapters_list = parsed.get("chapters", [])
    if not isinstance(chapters_list, list) or len(chapters_list) == 0:
        raise ValueError("Gemini batch response did not contain a valid 'chapters' list.")

    by_id = {ch.get("section_id"): ch for ch in chapters_list if isinstance(ch, dict)}
    enriched = []
    for idx, s in enumerate(sections):
        sid = s.get("section_id", idx + 1)
        data = by_id.get(sid) or (chapters_list[idx] if idx < len(chapters_list) else {})
        title = data.get("title") or f"Chapter {sid}"
        summary = data.get("summary") or ""
        takeaways = data.get("takeaways") if isinstance(data.get("takeaways"), list) else []
        key_quotes = data.get("key_quotes") if isinstance(data.get("key_quotes"), list) else []

        enriched.append({
            **s,
            "title": str(title).strip(),
            "summary": str(summary).strip(),
            "takeaways": takeaways,
            "key_quotes": key_quotes,
            "key_moment_rank": 0,
        })
    return enriched


# ---------------------------------------------------------------------------
# Backend: Ollama local
# ---------------------------------------------------------------------------

def _ollama_analyze(text: str, model: str = "gemma:2b", base_url: str = "http://localhost:11434") -> dict[str, Any]:
    try:
        import requests  # type: ignore
    except ImportError as err:
        raise RuntimeError("requests not installed. Run: pip install requests") from err

    prompt = _PROMPT_TEMPLATE.format(text=text[:3000])
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    return _parse_llm_json(raw)


def _check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests  # type: ignore
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main ChapterAnalyzer class
# ---------------------------------------------------------------------------

class ChapterAnalyzer:
    """Analyzes topic sections and generates chapter titles, summaries, and takeaways.

    Args:
        backend: One of ``"auto"``, ``"gemini"``, ``"ollama"``, ``"heuristic"``.
            ``"auto"`` tries Gemini → Ollama → heuristic in that order.
        gemini_api_key: Gemini API key. Falls back to ``GEMINI_API_KEY`` env var.
        gemini_model: Gemini model identifier (default: ``models/gemini-3.5-flash-lite``).
        ollama_model: Ollama model name (default: ``gemma:2b``).
        ollama_base_url: Ollama server base URL (default: ``http://localhost:11434``).
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        gemini_api_key: str | None = None,
        gemini_model: str = "models/gemini-3.5-flash-lite",
        ollama_model: str = "gemma:2b",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.backend = backend.lower()
        self._gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self._gemini_model = gemini_model
        self._ollama_model = ollama_model
        self._ollama_base_url = ollama_base_url
        self._active_backend: str = self._resolve_backend()
        self._backend_errors: list[str] = []

    def _resolve_backend(self) -> str:
        if self.backend == "heuristic":
            return "heuristic"
        if self.backend == "gemini":
            if not self._gemini_api_key:
                LOGGER.warning("Gemini backend requested but GEMINI_API_KEY not set. Falling back to heuristic.")
                return "heuristic"
            return "gemini"
        if self.backend == "ollama":
            if _check_ollama_available(self._ollama_base_url):
                return "ollama"
            LOGGER.warning("Ollama not reachable at %s. Falling back to heuristic.", self._ollama_base_url)
            return "heuristic"
        # backend == "auto"
        if self._gemini_api_key:
            LOGGER.info("Auto-detected Gemini backend (API key present).")
            return "gemini"
        if _check_ollama_available(self._ollama_base_url):
            LOGGER.info("Auto-detected Ollama backend at %s.", self._ollama_base_url)
            return "ollama"
        LOGGER.info("No LLM backend available. Using heuristic fallback.")
        return "heuristic"

    @property
    def active_backend(self) -> str:
        """Return the resolved backend name being used."""
        return self._active_backend

    def analyze_section(self, section: dict[str, Any]) -> dict[str, Any]:
        """Analyze a single topic section and return LLM-generated chapter metadata.

        Args:
            section: A section dict from ``topic_segments.json`` with keys
                ``section_id``, ``start_time``, ``end_time``, ``text``, etc.

        Returns:
            The original section dict merged with ``title``, ``summary``,
            ``takeaways``, and ``key_quotes`` fields.
        """
        text = section.get("text", "")
        if not text.strip():
            llm_data: dict[str, Any] = {
                "title": "Untitled Section",
                "summary": "No transcript text available for this section.",
                "takeaways": [],
                "key_quotes": [],
            }
        else:
            try:
                if self._active_backend == "gemini":
                    llm_data = _gemini_analyze(text, self._gemini_api_key, self._gemini_model)
                elif self._active_backend == "ollama":
                    llm_data = _ollama_analyze(text, self._ollama_model, self._ollama_base_url)
                else:
                    llm_data = _heuristic_analyze(text)
            except Exception as exc:
                err_msg = str(exc)
                LOGGER.warning("LLM backend '%s' failed: %s. Falling back to heuristic.", self._active_backend, err_msg)
                self._backend_errors.append(err_msg)
                llm_data = _heuristic_analyze(text)

        # Validate / sanitise output keys
        llm_data.setdefault("title", "Untitled Section")
        llm_data.setdefault("summary", "")
        if not isinstance(llm_data.get("takeaways"), list):
            llm_data["takeaways"] = []
        if not isinstance(llm_data.get("key_quotes"), list):
            llm_data["key_quotes"] = []

        return {**section, **llm_data}

    def analyze_all(self, topic_segments: dict[str, Any]) -> dict[str, Any]:
        """Process all sections in a ``topic_segments.json`` payload.

        Args:
            topic_segments: Full parsed JSON from Person 2's pipeline output.

        Returns:
            A ``chapters.json``-compatible dict with top-level metadata plus
            a ``chapters`` list of enriched section dicts.
        """
        sections = topic_segments.get("sections", [])
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chapters: list[dict[str, Any]] = []

        # 1. High-speed single-call full-context analysis for Gemini
        if self._active_backend == "gemini" and sections:
            LOGGER.info(
                "Analyzing all %d sections in a single unified Gemini API call with full context...",
                len(sections),
            )
            try:
                chapters = _gemini_analyze_batch(sections, self._gemini_api_key, self._gemini_model)
                LOGGER.info("✓ Gemini full-context batch analysis completed successfully in 1 call.")
            except Exception as exc:
                err_msg = str(exc)
                LOGGER.warning(
                    "Gemini unified batch analysis failed: %s. Falling back to section-by-section analysis.",
                    err_msg,
                )
                self._backend_errors.append(f"Gemini batch fallback: {err_msg}")
                chapters = []

        # 2. Section-by-section fallback (Ollama, heuristic, or Gemini batch failure)
        if not chapters and sections:
            max_workers = 1 if self._active_backend == "gemini" else min(8, len(sections) or 1)

            if max_workers > 1 and len(sections) > 1:
                LOGGER.info("Analyzing %d sections in parallel (workers=%d)...", len(sections), max_workers)
                chapters_by_idx: dict[int, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self.analyze_section, section): i
                        for i, section in enumerate(sections)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            chapter = future.result()
                        except Exception as exc:
                            LOGGER.warning("Section %d analysis failed: %s", idx, exc)
                            chapter = _heuristic_analyze(sections[idx].get("text", ""))
                        chapter["key_moment_rank"] = 0
                        chapters_by_idx[idx] = chapter
                chapters = [chapters_by_idx[i] for i in range(len(sections))]
            else:
                for i, section in enumerate(sections):
                    LOGGER.info("Analyzing section %d/%d ...", i + 1, len(sections))
                    chapter = self.analyze_section(section)
                    chapter["key_moment_rank"] = 0
                    chapters.append(chapter)
                    if self._active_backend == "gemini" and i < len(sections) - 1:
                        import time
                        time.sleep(2.0)

        # Rank by key_moment_score
        sorted_idx = sorted(range(len(chapters)), key=lambda i: chapters[i].get("key_moment_score", 0), reverse=True)
        for rank, idx in enumerate(sorted_idx, start=1):
            chapters[idx]["key_moment_rank"] = rank

        actual_backend = "heuristic" if self._backend_errors else self._active_backend
        return {
            "audio_file": topic_segments.get("audio_file", ""),
            "duration": topic_segments.get("duration", 0.0),
            "total_chapters": len(chapters),
            "backend_used": actual_backend,
            "backend_errors": self._backend_errors,
            "similarity_curve": topic_segments.get("similarity_curve", {}),
            "chapters": chapters,
        }
