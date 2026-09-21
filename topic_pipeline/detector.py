"""Semantic chunking, dense embeddings, and topic boundary detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class TopicDetectionError(RuntimeError):
    """Raised when semantic chunking or topic boundary detection fails."""


class TopicDetector:
    """Detects topic boundaries and key moments in timestamped transcripts."""

    def __init__(
        self,
        *,
        window_sec: float = 75.0,
        step_sec: float = 25.0,
        threshold_percentile: float = 60.0,
        min_section_sec: float = 25.0,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if window_sec <= 0 or step_sec <= 0:
            raise ValueError("window_sec and step_sec must be positive floats.")
        if min_section_sec <= 0:
            raise ValueError("min_section_sec must be a positive float.")
        if not (0.0 <= threshold_percentile <= 100.0):
            raise ValueError("threshold_percentile must be between 0.0 and 100.0.")

        self.window_sec = window_sec
        self.step_sec = step_sec
        self.threshold_percentile = threshold_percentile
        self.min_section_sec = min_section_sec
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            # ponytail: fallback to TF-IDF when sentence-transformers is missing; upgrade: pip install sentence-transformers
            self._model = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384))
        if self._model:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=384, stop_words="english")
        try:
            mat = vec.fit_transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        except Exception:
            return np.ones((len(texts), 384)) / np.sqrt(384)

    def create_windows(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        total_duration = segments[-1]["end"]
        windows = []
        cur = 0.0

        while cur < total_duration:
            w_end = cur + self.window_sec
            w_segs = [s for s in segments if s["end"] > cur and s["start"] < w_end]
            text = " ".join(s["text"] for s in w_segs).strip()
            if text:
                windows.append({
                    "start": cur,
                    "end": min(w_end, total_duration),
                    "text": text,
                    "segments": w_segs,
                })
            cur += self.step_sec
        return windows

    def process(self, transcript_data: dict[str, Any]) -> dict[str, Any]:
        """Process transcript JSON into structured topic sections and similarity curve."""
        segments = transcript_data.get("segments", [])
        if not segments:
            return {
                "audio_file": transcript_data.get("audio_file", ""),
                "duration": transcript_data.get("duration", 0.0),
                "total_sections": 0,
                "similarity_curve": {"timestamps": [], "similarities": [], "threshold": 0.0, "boundaries": []},
                "sections": [],
            }

        windows = self.create_windows(segments)
        if len(windows) <= 1:
            return {
                "audio_file": transcript_data.get("audio_file", ""),
                "duration": transcript_data.get("duration", segments[-1]["end"]),
                "total_sections": 1,
                "similarity_curve": {"timestamps": [0.0], "similarities": [1.0], "threshold": 0.5, "boundaries": []},
                "sections": [{
                    "section_id": 1,
                    "start_time": segments[0]["start"],
                    "end_time": segments[-1]["end"],
                    "timestamp_str": "00:00",
                    "text": " ".join(s["text"] for s in segments),
                    "key_moment_score": 1.0,
                    "segment_count": len(segments),
                }],
            }

        embeddings = self._embed([w["text"] for w in windows])

        # 1. Cosine similarity between consecutive windows
        raw_sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
        raw_sims = [max(0.0, min(1.0, s)) for s in raw_sims]
        timestamps = [windows[i + 1]["start"] for i in range(len(raw_sims))]

        # 2. Moving average smoothing (kernel = 3)
        smoothed = [
            float(np.mean(raw_sims[max(0, i - 1) : min(len(raw_sims), i + 2)]))
            for i in range(len(raw_sims))
        ]

        # 3. Valley depth scoring: depth[i] = (left_peak - sim[i]) + (right_peak - sim[i])
        depths = []
        for i, sim in enumerate(smoothed):
            left = max(smoothed[: i + 1]) if i > 0 else sim
            right = max(smoothed[i:]) if i < len(smoothed) - 1 else sim
            depths.append((left - sim) + (right - sim))

        # 4. Adaptive thresholding
        pos_depths = [d for d in depths if d > 0]
        threshold = float(np.percentile(pos_depths, self.threshold_percentile)) if pos_depths else 0.2

        # 5. Topic boundary detection
        boundaries = [0.0]
        last_t = 0.0
        for i, depth in enumerate(depths):
            t = timestamps[i]
            if depth >= threshold and (t - last_t) >= self.min_section_sec:
                boundaries.append(t)
                last_t = t

        # 6. Form topic sections & rank key moments
        sections = []
        for idx in range(len(boundaries)):
            s_start = boundaries[idx]
            s_end = boundaries[idx + 1] if idx + 1 < len(boundaries) else segments[-1]["end"]
            sec_segs = [s for s in segments if s["start"] >= s_start and s["start"] < s_end]
            if not sec_segs:
                continue

            sec_text = " ".join(s["text"] for s in sec_segs)
            words = sec_text.lower().split()
            unique_ratio = len(set(words)) / max(1, len(words))
            score = round(min(1.0, 0.6 + (0.4 * unique_ratio)), 2)

            mins, secs = int(s_start // 60), int(s_start % 60)
            sections.append({
                "section_id": idx + 1,
                "start_time": round(s_start, 2),
                "end_time": round(s_end, 2),
                "timestamp_str": f"{mins:02d}:{secs:02d}",
                "text": sec_text,
                "key_moment_score": score,
                "segment_count": len(sec_segs),
            })

        return {
            "audio_file": transcript_data.get("audio_file", ""),
            "duration": transcript_data.get("duration", segments[-1]["end"]),
            "total_sections": len(sections),
            "similarity_curve": {
                "timestamps": [round(t, 2) for t in timestamps],
                "similarities": [round(s, 3) for s in smoothed],
                "threshold": round(threshold, 3),
                "boundaries": [round(b, 2) for b in boundaries[1:]],
            },
            "sections": sections,
        }
