# 🎯 50% Implementation Milestone & Faculty Showcase Guide

> **Document Version:** 1.0  
> **Target Milestone:** Midterm / 50% Project Progress Evaluation  
> **Project Title:** TopicFinder — Podcast & Lecture "Key Moment" Extractor

---

## 📌 1. Milestone Objective

The objective of the 50% milestone is to demonstrate a **fully functioning vertical slice (end-to-end prototype)** of the pipeline. 

By this checkpoint, the team must showcase how raw audio is transformed through each person's module into timestamped, structured chapters with interactive visualization on a working dashboard.

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│     PERSON 1 CHECKPOINT   │      │     PERSON 2 CHECKPOINT   │      │     PERSON 3 CHECKPOINT   │
│                           │      │                           │      │                           │
│  • Audio Ingestion/FFmpeg │ ───► │  • 60-90s Chunking        │ ───► │  • LLM Summarization      │
│  • Whisper Transcription  │      │  • all-MiniLM Embeddings  │      │  • Streamlit Dashboard    │
│  • Timestamped Segments   │      │  • Cosine Similarity Plot │      │  • Similarity Graph (UI)  │
│  • Output: transcript.json│      │  • Output: boundaries.json│      │  • YouTube Chapters Export│
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 👥 2. Individual 50% Implementation Deliverables

---

### 👤 Person 1: Audio + Transcription Pipeline

#### 🎯 Goal:
Accept an audio/video file, extract clean 16kHz mono audio, and produce a time-aligned transcript in standard JSON format.

#### 📦 50% Code & Deliverables:
1. **`person1_audio/audio_utils.py`**:
   - Accepts `.mp3`, `.wav`, or `.mp4` video.
   - Extracts mono 16kHz WAV using `ffmpeg` / `pydub`.
   - Normalizes audio volume to prevent quiet segments from missing transcription.
2. **`person1_audio/transcription.py`**:
   - Runs OpenAI Whisper (`base` or `tiny` model) or `faster-whisper`.
   - Extracts segment-level start/end timestamps and clean text.
3. **Showcase Asset (`sample_data/transcript.json`)**:
   - Pre-computed transcript of a real 3–5 minute podcast or lecture clip for immediate zero-latency demonstration.

#### 📊 Tangible Output to Show Faculty:
A running terminal command producing formatted JSON with verified segment timestamps:
```json
{
  "source_file": "lecture_sample.mp3",
  "duration_seconds": 240.5,
  "segments": [
    { "start": 0.0, "end": 14.2, "text": "Welcome to CS229. Today we cover Attention Mechanisms." },
    { "start": 14.5, "end": 32.0, "text": "In traditional recurrent networks, sequential bottlenecks limited parallel training." },
    { "start": 32.5, "end": 58.0, "text": "Self-attention computes pairwise dot products across all sequence positions simultaneously." }
  ]
}
```

---

### 👤 Person 2: Chunking + Topic Boundary Detection

#### 🎯 Goal:
Take the transcript from Person 1, partition it into overlapping context windows, compute sentence embeddings, track semantic shifts using cosine distance, and identify chapter boundaries.

#### 📦 50% Code & Deliverables:
1. **`person2_chunking/embeddings.py`**:
   - Loads `sentence-transformers` with model `all-MiniLM-L6-v2`.
   - Vectorizes text chunks into 384-dimensional dense vectors.
2. **`person2_chunking/chunking.py`**:
   - Divides transcript into overlapping 60–90 second semantic chunks (e.g. 30s step size).
   - Computes cosine similarity between consecutive windows $W_i$ and $W_{i+1}$.
   - Applies a 3-point moving average filter to smooth conversational noise.
   - Computes valley depth scores to locate distinct topic boundaries.
3. **Showcase Asset (`sample_data/topic_segments.json`)**:
   - Pre-computed similarity points $(t_k, \text{sim}_k)$ and detected boundary timestamps.

#### 📊 Tangible Output to Show Faculty:
1. **Console Output / JSON** showing detected boundary timestamps:
   ```json
   {
     "similarity_curve": {
       "timestamps": [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0],
       "similarities": [0.89, 0.85, 0.34, 0.82, 0.79, 0.31, 0.88],
       "threshold": 0.50,
       "detected_boundaries": [90.0, 180.0]
     },
     "topic_sections": [
       { "id": 1, "start_time": 0.0, "end_time": 90.0, "text": "Introduction & RNN Bottlenecks..." },
       { "id": 2, "start_time": 90.0, "end_time": 180.0, "text": "Self-Attention Mathematics & Scaling..." },
       { "id": 3, "start_time": 180.0, "end_time": 240.0, "text": "FlashAttention & Hardware Optimization..." }
     ]
   }
   ```
2. **Similarity Curve Evidence:** A numerical or plotted demonstration showing that when the topic shifted from *RNN Bottlenecks* to *FlashAttention*, the cosine similarity plummeted from `0.85` down to `0.34`.

---

### 👤 Person 3: LLM Analysis & Streamlit Dashboard

#### 🎯 Goal:
Synthesize concise chapter titles and takeaways using an LLM, build the interactive Streamlit dashboard, render the similarity curve, and provide one-click YouTube chapter exports.

#### 📦 50% Code & Deliverables:
1. **`person3_dashboard/llm_analysis.py`**:
   - Structured prompt that takes each topic segment from Person 2 and outputs:
     - Short, punchy **Title** (3–6 words).
     - 1–2 sentence **Summary**.
     - 2 bullet point **Key Takeaways**.
   - Integrates with local Ollama (`gemma:2b` / `llama3`) or Gemini Flash API (with heuristic fallback).
2. **`person3_dashboard/formatters.py`**:
   - Formats chapters into YouTube timestamp description format (`00:00 Intro\n01:30 ...`).
3. **`app.py` (Streamlit Dashboard)**:
   - **Audio Player:** Standard `st.audio()` widget.
   - **Interactive Plotly Graph:** Line plot of cosine similarity over time with red vertical dashed lines marking topic shifts.
   - **Chapter Cards:** Displaying timestamp badge, title, summary, and takeaways.
   - **Copy-Paste Export Box:** Formatted YouTube chapters with instant copy.
   - **"Load Demo Sample" Button:** Instantly populates the entire dashboard for presentation reliability.

#### 📊 Tangible Output to Show Faculty:
A live, running Streamlit web application running at `http://localhost:8501`.

---

## 🎬 3. Faculty Demonstration Script (3–5 Minute Presentation Flow)

When presenting the 50% progress to faculty, follow this structured demo:

### ⏱️ Minute 1: Overview & Person 1 (Audio ➔ Transcript)
- **Speaker:** Person 1
- **Action:**
  1. Show a raw 4-minute audio clip (`sample_audio/lecture_clip.mp3`).
  2. Run `python person1_audio/transcription.py --input sample_audio/lecture_clip.mp3`.
  3. Show the resulting `transcript.json` with timestamped segments.
  4. **Key Point to Highlight:** *"We preserve millisecond-level start and end timestamps for every sentence, enabling precise audio navigation later in the pipeline."*

### ⏱️ Minute 2: Person 2 (Semantic Chunking & Topic Shifts)
- **Speaker:** Person 2
- **Action:**
  1. Run `python person2_chunking/chunking.py --input transcript.json`.
  2. Point out the similarity calculation between adjacent 60s windows using `all-MiniLM-L6-v2`.
  3. Show the similarity curve output: highlight the valley at `t = 90.0s` where similarity dropped below threshold `0.50`.
  4. **Key Point to Highlight:** *"Instead of arbitrary fixed-length splitting, we detect natural semantic shifts where the context changes dynamically."*

### ⏱️ Minute 3: Person 3 (LLM Synthesis & Live Streamlit Dashboard)
- **Speaker:** Person 3
- **Action:**
  1. Open the browser showing the running **Streamlit Dashboard** (`http://localhost:8501`).
  2. Show the **Plotly Similarity Curve** with red boundary markers matching Person 2's calculations.
  3. Show the **Generated Chapter Cards** with titles and summaries generated by the LLM.
  4. Play the audio in the web player and click on a chapter timestamp (e.g. `01:30`) to demonstrate audio alignment.
  5. Show the **YouTube Chapters export box** ready for copy-pasting.
  6. **Key Point to Highlight:** *"The creator gets an instant, ready-to-use YouTube description and interactive study notes with zero manual editing."*

---

## 📋 4. 50% Milestone Evaluation Checklist

Use this rubric checklist to ensure full readiness before your presentation:

| Item | Requirement | Status | Owner |
| :--- | :--- | :---: | :---: |
| **Audio Preprocessing** | Audio extracted to 16kHz mono WAV from audio/video input | 🔲 | Person 1 |
| **Whisper Transcription** | Generates valid JSON with segment-level `start`, `end`, and `text` | 🔲 | Person 1 |
| **Sample Demo Dataset** | 1 lecture clip & 1 podcast clip pre-transcribed in `sample_data/` | 🔲 | Person 1 |
| **MiniLM Embeddings** | Generates dense embeddings using `all-MiniLM-L6-v2` | 🔲 | Person 2 |
| **Cosine Similarity Curve**| Computes $(t, \text{sim})$ points across sliding 60–90s windows | 🔲 | Person 2 |
| **Topic Boundary Detection**| Detects valleys / topic shift points with adaptive thresholding | 🔲 | Person 2 |
| **LLM Chapter Synthesizer**| Generates 3–6 word titles, summaries, and key takeaways | 🔲 | Person 3 |
| **Similarity Curve Plot** | Plotly line chart embedded in Streamlit showing topic boundary lines | 🔲 | Person 3 |
| **Streamlit Web UI** | Functional UI with audio player, chapter list, and YouTube exporter | 🔲 | Person 3 |
| **End-to-End Demo Script** | 3-minute scripted team walkthrough prepared for faculty | 🔲 | All Team |

---

## 🚀 5. Roadmap from 50% to 100% (Final Submission)

What will be added after the 50% showcase:
1. **Timestamp Snapping:** Fine-grained alignment to sentence ends and natural audio pauses.
2. **Key Moment Scoring:** Mathematical ranking of chapters based on information density and keyword emphasis.
3. **Full Multi-Format Export:** Podcasting 2.0 `chapters.json`, WebVTT `.vtt` cue tracks, and downloadable Markdown study guides.
4. **Interactive Timeline Seeker:** Two-way synchronized playback where the transcript highlights the active sentence as the audio plays.
