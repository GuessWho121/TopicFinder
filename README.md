# 🎙️ TopicFinder: Podcast & Lecture "Key Moment" Extractor

> **An automated speech intelligence pipeline that transcribes audio/video content, detects major thematic shifts via sliding-window semantic chunking, and synthesizes timestamped chapters, key takeaways, and visual topic graphs.**

---

## 📌 1. Project Overview

Creating manual chapters, timestamps, and study notes for long-form audio (podcasts, university lectures, webinars, conference talks) is tedious and time-consuming. 

**TopicFinder** solves this problem with an end-to-end multi-stage pipeline:
1. **Audio & Transcription Engine:** Ingests podcast/lecture media files (`.mp3`, `.wav`, `.mp4`, `.mkv`), normalizes audio, and produces word- and segment-level timestamped transcripts using Whisper.
2. **Semantic Topic Shift Detection:** Partitions transcripts into overlapping context windows (60–90s), computes dense embeddings using `all-MiniLM-L6-v2`, tracks cosine similarity transitions, applies depth scoring and dynamic thresholding to pinpoint topic boundaries.
3. **LLM Synthesis & Interactive Dashboard:** Uses LLMs (Gemma / Llama / Gemini) to generate punchy chapter titles, summaries, and key takeaways, visualizes the similarity curve, and renders an interactive Streamlit player with multi-format exports (YouTube, Podcasting 2.0, WebVTT, Markdown).

---

## 👥 2. Team Architecture & Responsibilities

The project is structured into three distinct, decoupled modules connected by standardized data contracts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PERSON 1: Audio + Transcription                     │
│                                                                         │
│  • Audio/Video Ingestion (.mp3, .wav, .mp4, .mkv)                       │
│  • FFmpeg 16kHz mono conversion & loudness normalization                │
│  • Whisper speech-to-text with start/end segment timestamps             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  Contract 1: transcript.json
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 PERSON 2: Chunking + Topic Detection                    │
│                                                                         │
│  • 60–90s sliding window segmentation with 30s step overlap            │
│  • Sentence-Transformers (all-MiniLM-L6-v2) embedding generation        │
│  • Cosine similarity curve & moving average smoothing                   │
│  • Valley depth scoring & adaptive threshold boundary detection         │
│  • Key moment ranking based on semantic shift magnitude & density       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  Contract 2: topic_segments.json
                                     │              similarity_curve.json
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PERSON 3: LLM Analysis + Streamlit Dashboard               │
│                                                                         │
│  • LLM Prompting (Gemma / Llama / Gemini) for titles & summaries        │
│  • Interactive Streamlit Dashboard with synced media playback           │
│  • Interactive Cosine Similarity & Topic Boundary curve (Plotly)        │
│  • Click-to-seek chapter cards with key quotes & takeaways              │
│  • 1-Click Exporters: YouTube, Podcasting 2.0, WebVTT, Markdown         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 3. Pipeline Architecture & Methodology

### Stage 1: Audio Preprocessing & Transcription (Person 1)
- **Audio Extraction:** Uses `ffmpeg` to extract mono 16kHz PCM WAV audio from uploaded video or raw audio containers.
- **Normalization:** Peak/EBU R128 loudness normalization to ensure consistent amplitude across diverse recording setups.
- **Transcription:** Runs OpenAI Whisper (`base` / `small` or `faster-whisper`) to generate time-aligned segments:
  $$\mathcal{S} = \{(t_{\text{start}}^{(i)}, t_{\text{end}}^{(i)}, \text{text}^{(i)})\}_{i=1}^N$$

### Stage 2: Semantic Chunking & Boundary Detection (Person 2)
- **Sliding Context Window:** Groups consecutive segments into windows $W_k$ spanning $60 - 90$ seconds with an overlap step of $30$ seconds.
- **Embedding Generation:** Vectorizes each window into a 384-dimensional dense representation using `sentence-transformers/all-MiniLM-L6-v2`:
  $$\mathbf{e}_k = \text{Embed}(W_k) \in \mathbb{R}^{384}$$
- **Cosine Similarity Transition:** Computes the similarity between adjacent context windows:
  $$\text{Sim}(W_k, W_{k+1}) = \frac{\mathbf{e}_k \cdot \mathbf{e}_{k+1}}{\|\mathbf{e}_k\| \|\mathbf{e}_{k+1}\|}$$
- **Depth Scoring & Boundary Selection:** Calculates valley depth scores:
  $$\text{Depth}(k) = (\text{Peak}_{\text{left}} - \text{Sim}_k) + (\text{Peak}_{\text{right}} - \text{Sim}_k)$$
  A topic boundary is declared where $\text{Depth}(k) \ge \tau_{\text{adaptive}}$ and duration constraints are met.

### Stage 3: LLM Structuring & Web Dashboard (Person 3)
- **Prompt Engineering:** Passes each segmented topic block to an instruction-tuned LLM (Gemma 2B/9B, Llama 3, or Gemini Flash) using structured schema output:
  - `title`: 3–6 word punchy chapter heading.
  - `summary`: 1–2 sentence conceptual summary.
  - `takeaways`: 2–3 actionable bullet points.
  - `key_quotes`: Memorable quotes from the speaker.
- **Interactive UI:** A Streamlit dashboard integrating audio scrubbing, interactive Plotly similarity charts with topic boundary markers, synchronized transcript search, and multi-format exporters.

---

## 📋 4. Standardized Data Contracts

### 📄 Contract 1: `transcript.json` (Person 1 ➔ Person 2)
```json
{
  "source_file": "ai_lecture_clip.mp3",
  "duration": 312.4,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 8.5,
      "text": "Welcome to today's lecture on transformer architectures and self-attention."
    },
    {
      "id": 1,
      "start": 8.8,
      "end": 19.2,
      "text": "Before we dive into multi-head attention, let's review why RNNs struggle with long sequences."
    }
  ]
}
```

### 📄 Contract 2: `topic_segments.json` (Person 2 ➔ Person 3)
```json
{
  "total_sections": 4,
  "similarity_curve": {
    "timestamps": [30.0, 60.0, 90.0, 120.0, 150.0, 180.0],
    "similarities": [0.88, 0.84, 0.38, 0.79, 0.41, 0.86],
    "threshold": 0.52,
    "boundary_timestamps": [90.0, 150.0]
  },
  "sections": [
    {
      "section_id": 1,
      "start_time": 0.0,
      "end_time": 90.0,
      "text": "Welcome to today's lecture... RNN sequential bottlenecks...",
      "key_moment_score": 0.91
    }
  ]
}
```

### 📄 Contract 3: Final Chapter Output & Exports (Person 3)
```json
{
  "chapters": [
    {
      "start_time": 0.0,
      "end_time": 90.0,
      "timestamp_str": "00:00",
      "title": "Introduction & RNN Bottlenecks",
      "summary": "Reviews historical sequential constraints in RNNs and introduces the need for parallelizable attention.",
      "takeaways": [
        "RNNs process tokens sequentially, preventing effective GPU parallelization.",
        "Vanishing gradients degrade long-range context preservation."
      ],
      "key_quotes": [
        "In traditional recurrent networks, sequential dependencies bottleneck training."
      ],
      "key_moment_rank": 1
    }
  ]
}
```

---

## 🚀 5. Quick Start & Installation

### Prerequisites
- Python 3.10+
- FFmpeg installed and added to system PATH (for media extraction)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/TopicFinder.git
cd TopicFinder

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

#### 2. Run via CLI (Headless Mode)
```bash
# Full end-to-end execution
python cli.py --input sample_audio/lecture.mp3 --output-dir ./output --format youtube

# Run individual modules
python person1_audio/transcription.py --input sample_audio/lecture.mp3 --output transcript.json
python person2_chunking/chunking.py --input transcript.json --output topic_segments.json
python person3_dashboard/llm_analysis.py --input topic_segments.json --format markdown
```

---

## 📂 6. Repository File Structure

```
TopicFinder/
├── docs/
│   └── 50_percent_checkpoint.md   # Faculty showcase milestone & evaluation criteria
├── sample_data/                   # Curated demo audio clips and reference transcripts
│   ├── lecture_clip.mp3
│   └── podcast_clip.mp3
├── person1_audio/
│   ├── __init__.py
│   ├── audio_utils.py             # FFmpeg conversion, loudness normalization
│   └── transcription.py           # Whisper STT pipeline with timestamp preservation
├── person2_chunking/
│   ├── __init__.py
│   ├── embeddings.py              # all-MiniLM-L6-v2 embedding generator
│   └── chunking.py                # Sliding window, cosine similarity, valley detection
├── person3_dashboard/
│   ├── __init__.py
│   ├── llm_analysis.py            # Chapter title, summary & key takeaway synthesizer
│   ├── formatters.py              # Exporters (YouTube, Podcasting 2.0, WebVTT, Markdown)
│   └── plot_utils.py              # Interactive Plotly similarity curve visualizer
├── app.py                         # Streamlit Interactive Web Dashboard
├── cli.py                         # Command-line interface for automation
├── test_pipeline.py               # Unit & integration test suite
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 📦 7. Supported Export Formats

| Format | Extension / Target | Use Case |
| :--- | :--- | :--- |
| **YouTube Chapters** | Description text (`00:00 Intro`) | Direct copy-paste into YouTube video descriptions |
| **Podcasting 2.0** | `chapters.json` | Modern podcast players (Apple Podcasts, Pocket Casts) |
| **WebVTT** | `.vtt` | Interactive video player chapter tracks (`<track kind="chapters">`) |
| **Markdown Notes** | `.md` | Study guides, lecture revision notes, and bullet summaries |
| **Raw JSON** | `.json` | Programmatic API access and downstream indexing |

---

## 👥 Contributors & Roles

- **Person 1:** Audio Preprocessing, FFmpeg Pipeline & Whisper Timestamped Transcription.
- **Person 2:** Sliding-Window Chunking, `all-MiniLM-L6-v2` Embeddings & Similarity Boundary Detection.
- **Person 3:** LLM Topic/Summary Generation, Multi-Format Exporters & Streamlit Dashboard.