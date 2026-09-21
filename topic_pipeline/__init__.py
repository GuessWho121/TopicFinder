"""Topic segmentation package for detecting topic boundaries in speech transcripts."""

from __future__ import annotations

from .detector import TopicDetector, TopicDetectionError
from .pipeline import run_pipeline

__all__ = ["TopicDetector", "TopicDetectionError", "run_pipeline"]
