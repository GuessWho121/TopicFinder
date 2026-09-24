# 🎓 50% Milestone Showcase & Faculty Evaluation Guide

> **Branch:** `checkpoint-50-percent`  
> **Project Title:** TopicFinder — Podcast & Lecture "Key Moment" Extractor

---

## 📌 1. Overview of the 50% Milestone Prototype

For this milestone, the team has built and integrated a complete **end-to-end prototype**. 
This branch includes **both the working codebase and pre-generated visible output artifacts** from a real audio recording.

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│         STAGE 1           │      │         STAGE 2           │      │         STAGE 3           │
│   Audio & Transcription   │ ───► │ Semantic Topic Detection  │ ───► │  LLM Chapter Synthesis   │
│                           │      │                           │      │                           │
│ • FFmpeg Audio Normalizer │      │ • 60–90s Context Windowing│      │ • Chapter Title Generator │
│ • Whisper Speech-to-Text  │      │ • all-MiniLM-L6-v2 Vectors│      │ • 2-Sentence Summaries    │
│ • Millisecond Timestamps  │      │ • Cosine Similarity Curve │      │ • Actionable Takeaways    │
│ • Output: transcript.json │      │ • Otsu Boundary Snapping  │      │ • Multi-Format Exporters  │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 📂 2. Visible Showcase Outputs in this Branch

All pre-generated outputs from the sample lecture/podcast recording are tracked in the [`showcase_output/`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/) directory:

| Artifact File | Stage & Description | Key Contents |
| :--- | :--- | :--- |
| [`stage1_transcript.json`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/stage1_transcript.json) | **Stage 1 (Audio ➔ Transcript)** | 24 timestamped speech segments with precise `start`, `end`, and `text`. |
| [`stage2_topic_segments.json`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/stage2_topic_segments.json) | **Stage 2 (Topic Segmentation)** | Cosine similarity data points $(t_k, \text{sim}_k)$, Otsu threshold, detected boundary marks, and ranked key moments. |
| [`stage3_chapters.json`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/stage3_chapters.json) | **Stage 3 (Structured Chapters)** | Generated chapter titles, summaries, bullet takeaways, and quotes. |
| [`youtube_chapters.txt`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/youtube_chapters.txt) | **Export (YouTube Timestamps)** | Ready-to-use YouTube video description chapter marks (`00:00 Intro...`). |
| [`study_summary_notes.md`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/study_summary_notes.md) | **Export (Markdown Study Guide)** | Structured lecture revision notes with summaries and bullet points. |
| [`chapters.vtt`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/chapters.vtt) | **Export (WebVTT Cue Track)** | Standard WebVTT chapter track for video/audio players. |
| [`podcasting2_chapters.json`](file:///c:/Users/aksha/Documents/Projects/slp/TopicFinder/showcase_output/podcasting2_chapters.json) | **Export (Podcasting 2.0)** | Standard JSON chapter metadata for podcast players. |

---

## 🖥️ 3. How to Launch the Live Interactive Showcase

Run the unified dashboard:

```bash
python start.py
```
*(Or double-click `launch.bat` on Windows)*

### In the Web Dashboard:
1. Click **"Load demo data"** in the sidebar.
2. Observe the **interactive Plotly Cosine Similarity Curve** showing exactly where topic shifts occurred.
3. Browse the **Generated Chapters & Key Moments** with importance scores and rankings.
4. Test the **Searchable Transcript** and **1-Click YouTube Description Exporter**.

---

## ⚡ 4. How to Run Each Pipeline Stage in the Terminal

### Run Stage 2 (Semantic Chunking & Topic Detection):
```bash
python -m topic_pipeline.pipeline --input tests/fixtures/sample_transcript.json --output-dir data/output
```

### Run Stage 3 (LLM Chapter Synthesis):
```bash
python -m llm_pipeline.analyzer --input data/output/topic_segments.json --output-dir data/output
```

### Run Full End-to-End CLI:
```bash
python run.py tests/fixtures/sample_transcript.json --format youtube
```
