"""TopicFinder — end-to-end Streamlit dashboard.

User input: one audio or video file.
The app runs all three pipeline stages automatically:
  1. audio_pipeline  : FFmpeg normalise + Whisper transcribe  → transcript.json
  2. topic_pipeline  : MiniLM embeddings + valley detection   → topic_segments.json
  3. llm_pipeline    : LLM chapter synthesis                  → chapters_data (in memory)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force reload internal modules to pick up latest fixes in running Streamlit session
import importlib
for mod in list(sys.modules.keys()):
    if mod.startswith("audio_pipeline.") or mod.startswith("topic_pipeline.") or mod.startswith("llm_pipeline."):
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

# Ensure FFmpeg is available on PATH for Whisper
try:
    from audio_pipeline.audio_processor import ensure_ffmpeg_on_path
    ensure_ffmpeg_on_path()
except Exception as e:
    logging.warning("Could not ensure FFmpeg on PATH: %s", e)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TopicFinder",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — same editorial dark palette, amber accent
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; font-size: 14px; color: #ededed; }

.stApp { background: #0e0e0e; }

[data-testid="stSidebar"] {
    background: #0e0e0e;
    border-right: 1px solid #1f1f1f;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 28px; }

.block-container { padding-top: 24px !important; max-width: 1180px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #282828;
    gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #606060;
    font-size: 13px; font-weight: 500; letter-spacing: 0.02em;
    padding: 10px 20px !important;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #a0a0a0 !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: #ededed !important; border-bottom-color: #d4a843 !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

/* ── Sidebar elements ── */
.sidebar-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.10em;
    text-transform: uppercase; color: #606060; margin: 20px 0 8px 0;
}
.wordmark { font-size: 15px; font-weight: 700; letter-spacing: -0.01em; color: #ededed; margin-bottom: 2px; }
.wordmark-sub { font-size: 11px; color: #606060; letter-spacing: 0.01em; }

/* ── Pipeline stage tracker ── */
.stage-list { margin: 0; padding: 0; list-style: none; }
.stage-item {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #1a1a1a; font-size: 12px; color: #606060;
}
.stage-item:last-child { border-bottom: none; }
.stage-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
    background: #282828; border: 1px solid #383838;
}
.stage-dot.done { background: #4caf7d; border-color: #4caf7d; }
.stage-dot.running { background: #d4a843; border-color: #d4a843; animation: pulse 1.2s infinite; }
.stage-dot.error { background: #e05252; border-color: #e05252; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.stage-name { flex: 1; }
.stage-name.done { color: #a0a0a0; }
.stage-name.running { color: #d4a843; }
.stage-time { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #606060; }

/* ── Page title ── */
.page-title { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: #ededed; margin: 0 0 4px 0; }
.page-sub { font-size: 13px; color: #606060; margin: 0 0 28px 0; }

/* ── Stat grid ── */
.stat-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 1px; background: #1f1f1f;
    border: 1px solid #1f1f1f; border-radius: 6px; overflow: hidden; margin-bottom: 28px;
}
.stat-cell { background: #161616; padding: 16px 20px; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 500; color: #ededed; line-height: 1; margin-bottom: 5px; }
.stat-key { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #606060; }

/* ── Chapter card ── */
.ch-card { border: 1px solid #282828; border-radius: 6px; background: #161616; margin-bottom: 10px; overflow: hidden; transition: border-color 0.15s; }
.ch-card:hover { border-color: #383838; }
.ch-head { display: flex; align-items: baseline; gap: 14px; padding: 14px 18px 12px; border-bottom: 1px solid #1f1f1f; }
.ch-ts { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 500; color: #d4a843; flex-shrink: 0; }
.ch-title { font-size: 14px; font-weight: 600; color: #ededed; flex: 1; min-width: 0; }
.ch-rank { font-size: 11px; color: #606060; flex-shrink: 0; font-family: 'JetBrains Mono', monospace; }
.ch-body { padding: 12px 18px 14px; }
.ch-summary { font-size: 13px; color: #a0a0a0; line-height: 1.65; margin: 0 0 12px 0; }
.ch-section-label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #606060; margin: 0 0 5px 0; }
.ch-takeaways { margin: 0; padding: 0; list-style: none; }
.ch-takeaways li { font-size: 13px; color: #a0a0a0; padding: 3px 0 3px 14px; position: relative; line-height: 1.55; }
.ch-takeaways li::before { content: ''; position: absolute; left: 0; top: 11px; width: 5px; height: 1px; background: #d4a843; }
.ch-quote { font-size: 13px; font-style: italic; color: #787878; border-left: 2px solid #282828; padding: 6px 12px; margin: 10px 0 0 0; line-height: 1.6; }
.ch-score { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500; padding: 2px 7px; border-radius: 3px; margin-left: auto; }
.score-hi { background: #0f2218; color: #4caf7d; }
.score-md { background: #201a06; color: #d4a843; }
.score-lo { background: #200e0e; color: #e05252; }

/* ── Status pills ── */
.status-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.pill { font-size: 10px; font-weight: 600; letter-spacing: 0.04em; padding: 3px 8px; border-radius: 3px; font-family: 'JetBrains Mono', monospace; }
.pill-ok  { background: #0f2218; color: #4caf7d; }
.pill-off { background: #1a1a1a; color: #606060; border: 1px solid #282828; }
.pill-amber { background: rgba(212,168,67,0.10); color: #d4a843; border: 1px solid rgba(212,168,67,0.2); }
.pill-err { background: #200e0e; color: #e05252; }

/* ── Transcript ── */
.transcript-wrap {
    border: 1px solid #282828; border-radius: 6px; background: #111;
    padding: 20px 22px; max-height: 520px; overflow-y: auto;
    line-height: 1.8; color: #a0a0a0; font-size: 13.5px;
}
.transcript-wrap::-webkit-scrollbar { width: 4px; }
.transcript-wrap::-webkit-scrollbar-track { background: #0e0e0e; }
.transcript-wrap::-webkit-scrollbar-thumb { background: #282828; border-radius: 2px; }
.tr-head { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #d4a843; font-weight: 500; margin: 18px 0 6px 0; display: flex; gap: 12px; align-items: center; }
.tr-title { font-size: 11px; color: #606060; font-family: 'Inter', sans-serif; }
.tr-divider { border: none; border-top: 1px solid #1f1f1f; margin: 16px 0; }
.search-hit { background: rgba(212,168,67,0.18); color: #ededed; border-radius: 2px; padding: 0 2px; }

/* ── Empty / info state ── */
.empty-state { border: 1px dashed #282828; border-radius: 6px; padding: 40px 32px; text-align: center; color: #606060; font-size: 13px; line-height: 1.7; }
.empty-state strong { color: #a0a0a0; display: block; margin-bottom: 6px; font-size: 14px; }

/* ── Seek banner ── */
.seek-banner { background: #161616; border: 1px solid #282828; border-radius: 5px; padding: 9px 14px; font-size: 13px; color: #a0a0a0; margin-bottom: 14px; display: flex; align-items: center; gap: 10px; }
.seek-ts { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #d4a843; font-weight: 500; }

/* ── Export ── */
.export-head { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #606060; margin: 0 0 3px 0; }
.export-desc { font-size: 12px; color: #606060; margin: 0 0 8px 0; }

/* ── Buttons ── */
.stButton > button {
    background: #1c1c1c !important; color: #ededed !important;
    border: 1px solid #282828 !important; border-radius: 5px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 6px 14px !important; transition: background 0.15s, border-color 0.15s !important;
}
.stButton > button:hover { background: #242424 !important; border-color: #383838 !important; }

.stDownloadButton > button {
    background: transparent !important; color: #d4a843 !important;
    border: 1px solid rgba(212,168,67,0.35) !important; border-radius: 5px !important;
    font-size: 12px !important; font-weight: 500 !important; padding: 5px 12px !important;
}
.stDownloadButton > button:hover { background: rgba(212,168,67,0.08) !important; border-color: rgba(212,168,67,0.6) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111 !important; border: 1px solid #282828 !important;
    border-radius: 5px !important; color: #ededed !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12.5px !important;
}
.stSelectbox > div > div { background: #111 !important; border: 1px solid #282828 !important; border-radius: 5px !important; color: #ededed !important; }
[data-testid="stFileUploader"] { border: 1px dashed #282828 !important; border-radius: 5px !important; background: #111 !important; }
.stAlert { border-radius: 5px !important; border: 1px solid #282828 !important; background: #161616 !important; color: #a0a0a0 !important; }
hr { border: none; border-top: 1px solid #1f1f1f !important; margin: 16px 0 !important; }
.stExpander { border: 1px solid #1f1f1f !important; border-radius: 5px !important; background: #161616 !important; }
.stExpander summary { font-size: 13px !important; font-weight: 500 !important; color: #a0a0a0 !important; padding: 12px 16px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_SS_DEFAULTS: dict[str, object] = {
    # pipeline outputs
    "transcript_data": None,       # parsed transcript.json dict
    "topic_segments": None,        # parsed topic_segments.json dict
    "chapters_data": None,         # ChapterAnalyzer.analyze_all() output
    # source file (saved to disk path)
    "source_path": None,           # Path | None
    "source_name": "",
    # stage timing
    "stage_times": {},             # stage_name -> elapsed_seconds
    "stage_errors": {},            # stage_name -> error message
    # media player
    "media_bytes": None,
    "media_mime": "audio/mpeg",
    "seek_time": 0.0,
    # llm
    "backend_label": "—",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    t = int(round(seconds))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _score_cls(score: float) -> str:
    return "score-hi" if score >= 0.8 else ("score-md" if score >= 0.6 else "score-lo")


def _pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{text}</span>'


def _stage_html(name: str, state: str, elapsed: float | None = None) -> str:
    """state: 'idle' | 'running' | 'done' | 'error'"""
    dot_cls = {"idle": "", "running": " running", "done": " done", "error": " error"}[state]
    name_cls = {"idle": "", "running": " running", "done": " done", "error": ""}[state]
    time_str = f"{elapsed:.1f}s" if elapsed is not None else ""
    return (
        f'<li class="stage-item">'
        f'<span class="stage-dot{dot_cls}"></span>'
        f'<span class="stage-name{name_cls}">{name}</span>'
        f'<span class="stage-time">{time_str}</span>'
        f'</li>'
    )


def _load_demo() -> None:
    demo_path = _ROOT / "demo_data" / "sample_topic_segments.json"
    if not demo_path.is_file():
        st.error("Demo fixture not found.")
        return
    with open(demo_path, encoding="utf-8") as f:
        data = json.load(f)
    st.session_state["topic_segments"] = data
    st.session_state["transcript_data"] = None
    st.session_state["chapters_data"] = None
    st.session_state["source_name"] = "demo_data/30_day_challenges.mp3"
    st.session_state["stage_times"] = {"transcription": None, "topic_detection": 0.0}
    st.session_state["stage_errors"] = {}
    st.session_state["backend_label"] = "—"
    st.toast("Demo loaded. Click 'Analyze chapters' to continue.")


def _save_uploaded_file(uploaded_file) -> Path:
    """Persist the Streamlit UploadedFile to data/input/ and return its path."""
    dest_dir = _ROOT / "data" / "input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_stage_1(source_path: Path, whisper_model: str, status_box) -> bool:
    """Run audio_pipeline: normalize + transcribe → transcript.json in data/output/"""
    from audio_pipeline.pipeline import run_pipeline as audio_run
    t0 = time.time()
    try:
        status_box.markdown(
            '<ul class="stage-list">'
            + _stage_html("Transcription (Whisper)", "running")
            + _stage_html("Topic detection (MiniLM)", "idle")
            + _stage_html("Chapter synthesis (LLM)", "idle")
            + '</ul>', unsafe_allow_html=True,
        )
        transcript_path = audio_run(
            source_path,
            _ROOT / "data" / "output",
            model_name=whisper_model,
            force=True,
        )
        with open(transcript_path, encoding="utf-8") as f:
            st.session_state["transcript_data"] = json.load(f)
        elapsed = time.time() - t0
        st.session_state["stage_times"]["transcription"] = elapsed
        st.session_state["stage_errors"].pop("transcription", None)
        return True
    except Exception as exc:
        import traceback
        with open(_ROOT / "data" / "debug_error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        st.session_state["stage_errors"]["transcription"] = str(exc)
        return False


def _run_stage_2(status_box, sensitivity: str = "medium") -> bool:
    """Run topic_pipeline: embeddings + valley detection → topic_segments.json"""
    from topic_pipeline.detector import TopicDetector
    t0 = time.time()
    try:
        elapsed_1 = st.session_state["stage_times"].get("transcription")
        status_box.markdown(
            '<ul class="stage-list">'
            + _stage_html("Transcription (Whisper)", "done", elapsed_1)
            + _stage_html("Topic detection (MiniLM)", "running")
            + _stage_html("Chapter synthesis (LLM)", "idle")
            + '</ul>', unsafe_allow_html=True,
        )
        transcript_data = st.session_state["transcript_data"]
        if transcript_data is None:
            # fall back to file on disk
            tp = _ROOT / "data" / "output" / "transcript.json"
            with open(tp, encoding="utf-8") as f:
                transcript_data = json.load(f)
            st.session_state["transcript_data"] = transcript_data

        detector = TopicDetector(sensitivity=sensitivity)
        result = detector.process(transcript_data)

        out_path = _ROOT / "data" / "output" / "topic_segments.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        st.session_state["topic_segments"] = result
        elapsed = time.time() - t0
        st.session_state["stage_times"]["topic_detection"] = elapsed
        st.session_state["stage_errors"].pop("topic_detection", None)
        return True
    except Exception as exc:
        st.session_state["stage_errors"]["topic_detection"] = str(exc)
        return False


def _run_stage_3(backend: str, api_key: str, status_box) -> bool:
    """Run llm_pipeline: chapter synthesis"""
    from llm_pipeline.analyzer import ChapterAnalyzer
    t0 = time.time()
    try:
        e1 = st.session_state["stage_times"].get("transcription")
        e2 = st.session_state["stage_times"].get("topic_detection")
        status_box.markdown(
            '<ul class="stage-list">'
            + _stage_html("Transcription (Whisper)", "done", e1)
            + _stage_html("Topic detection (MiniLM)", "done", e2)
            + _stage_html("Chapter synthesis (LLM)", "running")
            + '</ul>', unsafe_allow_html=True,
        )
        analyzer = ChapterAnalyzer(backend=backend, gemini_api_key=api_key or None)
        result = analyzer.analyze_all(st.session_state["topic_segments"])
        st.session_state["chapters_data"] = result
        st.session_state["backend_label"] = result.get("backend_used", backend)
        backend_errors = result.get("backend_errors", [])
        if backend_errors:
            st.session_state["stage_errors"]["llm_fallback"] = backend_errors[0]
        elapsed = time.time() - t0
        st.session_state["stage_times"]["llm"] = elapsed
        st.session_state["stage_errors"].pop("llm", None)
        return True
    except Exception as exc:
        st.session_state["stage_errors"]["llm"] = str(exc)
        return False


def _run_full_pipeline(
    source_path: Path,
    whisper_model: str,
    sensitivity: str,
    backend: str,
    api_key: str,
    status_box,
) -> None:
    """Run all three stages sequentially, updating status_box as we go."""
    ok = _run_stage_1(source_path, whisper_model, status_box)
    if not ok:
        err = st.session_state["stage_errors"].get("transcription", "Unknown error")
        st.error(f"Transcription failed: {err}")
        return
    ok = _run_stage_2(status_box, sensitivity=sensitivity)
    if not ok:
        err = st.session_state["stage_errors"].get("topic_detection", "Unknown error")
        st.error(f"Topic detection failed: {err}")
        return
    ok = _run_stage_3(backend, api_key, status_box)
    if not ok:
        err = st.session_state["stage_errors"].get("llm", "Unknown error")
        st.error(f"Chapter synthesis failed: {err}")
        return

    e1 = st.session_state["stage_times"].get("transcription")
    e2 = st.session_state["stage_times"].get("topic_detection")
    e3 = st.session_state["stage_times"].get("llm")
    status_box.markdown(
        '<ul class="stage-list">'
        + _stage_html("Transcription (Whisper)", "done", e1)
        + _stage_html("Topic detection (MiniLM)", "done", e2)
        + _stage_html("Chapter synthesis (LLM)", "done", e3)
        + '</ul>', unsafe_allow_html=True,
    )
    st.toast("Pipeline complete.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="wordmark">TopicFinder</div>'
        '<div class="wordmark-sub">Lecture &amp; Podcast Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── File upload ──
    st.markdown('<div class="sidebar-label">Input file</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Audio or video",
        type=["mp3", "wav", "mp4", "mkv", "m4a", "ogg", "webm"],
        label_visibility="collapsed",
        help="Upload any audio or video file. The pipeline handles everything from here.",
    )
    st.caption("MP3, WAV, M4A, OGG · MP4, MKV, WebM")

    if uploaded is not None:
        saved = _save_uploaded_file(uploaded)
        # Only reset pipeline results when a genuinely new file is selected.
        # file_id changes each time the user picks a different file; it stays
        # the same across Streamlit reruns (e.g. after st.rerun()), so results
        # produced by a completed pipeline run are not wiped on rerun.
        if uploaded.file_id != st.session_state.get("_upload_file_id"):
            st.session_state["_upload_file_id"] = uploaded.file_id
            st.session_state["source_path"] = saved
            st.session_state["source_name"] = uploaded.name
            mime_map = {
                ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
                ".m4a": "audio/mp4", ".mp4": "video/mp4", ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }
            ext = Path(uploaded.name).suffix.lower()
            st.session_state["media_mime"] = mime_map.get(ext, "audio/mpeg")
            st.session_state["media_bytes"] = saved.read_bytes()
            st.session_state["transcript_data"] = None
            st.session_state["topic_segments"] = None
            st.session_state["chapters_data"] = None
            st.session_state["stage_times"] = {}
            st.session_state["stage_errors"] = {}

    # ── Settings ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)

    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=0,
        label_visibility="collapsed",
    )
    _model_hint = {"tiny": "Fastest (~3–6 min / 30 min audio)", "base": "Balanced (~6–10 min)", "small": "Best quality (~12–18 min)"}
    st.caption(_model_hint[whisper_model])

    sens_choice = st.selectbox(
        "Topic sensitivity",
        ["Standard (Medium)", "Broad Overview (Coarse)", "Granular Highlights (Fine)"],
        index=0,
        label_visibility="collapsed",
    )
    sens_map = {
        "Standard (Medium)": "medium",
        "Broad Overview (Coarse)": "coarse",
        "Granular Highlights (Fine)": "fine",
    }
    sens_key = sens_map[sens_choice]

    backend_choice = st.selectbox(
        "LLM backend",
        ["Auto", "Gemini Flash", "Ollama", "Heuristic"],
        index=0,
        label_visibility="collapsed",
    )
    backend_map = {"Auto": "auto", "Gemini Flash": "gemini", "Ollama": "ollama", "Heuristic": "heuristic"}
    backend_key = backend_map[backend_choice]

    gemini_key = ""
    if backend_key in ("auto", "gemini"):
        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="AIza...  (or GEMINI_API_KEY env var)",
            value=os.getenv("GEMINI_API_KEY", ""),
            label_visibility="collapsed",
        )
        st.caption("Optional — falls back to heuristic if absent")

    # ── Process button ──
    st.markdown("<hr>", unsafe_allow_html=True)

    source_ready = st.session_state.get("source_path") is not None
    topics_ready = st.session_state.get("topic_segments") is not None
    analyzed = st.session_state.get("chapters_data") is not None

    if not source_ready:
        st.markdown(
            '<div style="font-size:12px;color:#606060;text-align:center;padding:8px 0;">Upload a file above to begin.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Stage tracker placeholder (updated during pipeline run)
        stage_placeholder = st.empty()

        # Show current stage state when idle
        e1 = st.session_state["stage_times"].get("transcription")
        e2 = st.session_state["stage_times"].get("topic_detection")
        e3 = st.session_state["stage_times"].get("llm")
        err = st.session_state.get("stage_errors", {})

        s1 = "done" if e1 is not None else ("error" if "transcription" in err else "idle")
        s2 = "done" if e2 is not None else ("error" if "topic_detection" in err else "idle")
        s3 = "done" if e3 is not None else ("error" if "llm" in err else "idle")

        stage_placeholder.markdown(
            '<ul class="stage-list">'
            + _stage_html("Transcription (Whisper)", s1, e1)
            + _stage_html("Topic detection (MiniLM)", s2, e2)
            + _stage_html("Chapter synthesis (LLM)", s3, e3)
            + '</ul>', unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col_run, col_chap = st.columns(2)
        with col_run:
            run_all = st.button("Process file", type="primary",
                                help="Runs transcription + topic detection + chapter analysis")
        with col_chap:
            run_chap = st.button("Re-analyze", 
                                 disabled=not topics_ready,
                                 help="Re-run only the LLM chapter synthesis (skips transcription)")

        if run_all:
            sp = st.session_state["source_path"]
            with st.spinner("Running pipeline..."):
                _run_full_pipeline(sp, whisper_model, sens_key, backend_key, gemini_key, stage_placeholder)
            st.rerun()

        if run_chap and topics_ready:
            status_ph2 = st.empty()
            with st.spinner("Re-analyzing chapters..."):
                _run_stage_3(backend_key, gemini_key, status_ph2)
            st.rerun()

    # ── Demo shortcut ──
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Load demo data"):
        _load_demo()
        st.rerun()

    # ── Status pills ──
    td = st.session_state.get("transcript_data")
    ts = st.session_state.get("topic_segments")
    cd = st.session_state.get("chapters_data")
    bl = st.session_state.get("backend_label", "—")
    src_name = st.session_state.get("source_name", "")

    if src_name:
        st.markdown(
            f'<div style="font-size:11px;color:#606060;margin-top:12px;word-break:break-all;">{src_name}</div>',
            unsafe_allow_html=True,
        )

    pills = []
    if ts: pills.append(_pill("topics ready", "pill-ok"))
    if cd: pills.append(_pill("chapters ready", "pill-ok"))
    if bl != "—": pills.append(_pill(bl, "pill-amber"))
    if pills:
        st.markdown(f'<div class="status-row">{"".join(pills)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
src_name = st.session_state.get("source_name", "")
title_suffix = f" — {src_name}" if src_name else ""
st.markdown(
    f'<h1 class="page-title">Analysis{title_suffix}</h1>'
    '<p class="page-sub">Upload an audio or video file. The pipeline transcribes, detects topic shifts, and generates structured chapters automatically.</p>',
    unsafe_allow_html=True,
)

tab_sim, tab_ch, tab_tr, tab_pl, tab_ex = st.tabs([
    "Similarity", "Chapters", "Transcript", "Player", "Export",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Similarity graph
# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    segs_data = st.session_state.get("topic_segments")
    chap_data = st.session_state.get("chapters_data")

    if segs_data is None:
        st.markdown(
            '<div class="empty-state"><strong>No data yet</strong>'
            'Upload a file and click "Process file" in the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            import plotly.graph_objects as go
        except ImportError:
            st.error("Install plotly: pip install plotly")
            st.stop()

        curve = segs_data.get("similarity_curve", {})
        timestamps = curve.get("timestamps", [])
        similarities = curve.get("similarities", [])
        threshold = curve.get("threshold", 0.5)
        boundaries = curve.get("boundaries", [])
        sections = segs_data.get("sections", [])
        duration = segs_data.get("duration", 0.0)

        n_ch = len(sections)
        avg_len = duration / max(1, n_ch)
        top_score = max((s.get("key_moment_score", 0) for s in sections), default=0.0)
        total_words = sum(len(s.get("text", "").split()) for s in sections)

        st.markdown(
            f"""
            <div class="stat-grid">
              <div class="stat-cell">
                <div class="stat-val">{_fmt_ts(duration)}</div>
                <div class="stat-key">Duration</div>
              </div>
              <div class="stat-cell">
                <div class="stat-val">{n_ch}</div>
                <div class="stat-key">Sections</div>
              </div>
              <div class="stat-cell">
                <div class="stat-val">{_fmt_ts(avg_len)}</div>
                <div class="stat-key">Avg length</div>
              </div>
              <div class="stat-cell">
                <div class="stat-val">{top_score:.2f}</div>
                <div class="stat-key">Top score</div>
              </div>
              <div class="stat-cell">
                <div class="stat-val">{total_words:,}</div>
                <div class="stat-key">Words</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        # Section bands
        all_starts = [0.0] + boundaries
        all_ends = boundaries + [duration]
        band_colors = ["rgba(212,168,67,0.05)", "rgba(255,255,255,0.02)"]
        for i, (bs, be) in enumerate(zip(all_starts, all_ends)):
            label = (
                chap_data["chapters"][i]["title"]
                if chap_data and i < len(chap_data.get("chapters", []))
                else f"Section {i + 1}"
            )
            fig.add_vrect(
                x0=bs, x1=be,
                fillcolor=band_colors[i % 2],
                layer="below", line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="#606060"),
            )

        # Similarity curve
        if timestamps and similarities:
            fig.add_trace(go.Scatter(
                x=timestamps, y=similarities,
                mode="lines",
                name="Similarity",
                line=dict(color="#d4a843", width=1.8, shape="spline", smoothing=0.6),
                fill="tozeroy", fillcolor="rgba(212,168,67,0.06)",
                hovertemplate="<b>%{x:.1f}s</b>  sim: %{y:.3f}<extra></extra>",
            ))

        # Threshold
        if timestamps:
            fig.add_trace(go.Scatter(
                x=[timestamps[0], timestamps[-1]],
                y=[threshold, threshold],
                mode="lines", name=f"Threshold {threshold:.2f}",
                line=dict(color="#606060", width=1, dash="dot"),
                hoverinfo="skip",
            ))

        # Boundary lines
        for bnd in boundaries:
            fig.add_vline(
                x=bnd, line=dict(color="#e05252", width=1.2, dash="dash"),
                annotation_text=_fmt_ts(bnd),
                annotation_position="top right",
                annotation=dict(font_size=10, font_color="#e05252"),
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12, color="#606060"),
            xaxis=dict(
                title=dict(text="Time (s)", font=dict(size=11, color="#606060")),
                gridcolor="#1a1a1a", zeroline=False, tickformat=".0f",
                tickfont=dict(size=11), linecolor="#282828",
            ),
            yaxis=dict(
                title=dict(text="Cosine similarity", font=dict(size=11, color="#606060")),
                range=[0, 1.08], gridcolor="#1a1a1a", zeroline=False,
                tickfont=dict(size=11), linecolor="#282828",
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11),
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            ),
            height=400, margin=dict(l=0, r=0, t=36, b=0), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not chap_data:
            st.markdown(
                '<div style="font-size:12px;color:#606060;margin-top:4px;">'
                'Run analysis in the sidebar to overlay chapter titles.</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Chapters
# ═══════════════════════════════════════════════════════════════════════════
with tab_ch:
    chap_data = st.session_state.get("chapters_data")
    topic_data = st.session_state.get("topic_segments")

    if chap_data is None:
        msg = (
            'Upload a file and click "Process file" to generate chapters.'
            if topic_data is None
            else 'Topic segments ready. Click "Process file" or "Re-analyze" to generate chapter titles.'
        )
        st.markdown(f'<div class="empty-state"><strong>No chapters yet</strong>{msg}</div>', unsafe_allow_html=True)
    else:
        chapters = chap_data.get("chapters", [])
        backend_errors = chap_data.get("backend_errors", [])
        if backend_errors:
            st.warning(f"LLM backend failed — fell back to heuristic. First error: {backend_errors[0]}")
        if not chapters:
            st.warning("No chapters found.")
        else:
            st.markdown(
                f'<div style="font-size:12px;color:#606060;margin-bottom:18px;">'
                f'{len(chapters)} chapters &nbsp;&middot;&nbsp; backend: {chap_data.get("backend_used", "—")}'
                f'</div>', unsafe_allow_html=True,
            )
            for ch in chapters:
                ts        = _fmt_ts(ch.get("start_time", 0.0))
                title     = ch.get("title", "Untitled")
                summary   = ch.get("summary", "")
                takeaways = ch.get("takeaways", [])
                key_quotes= ch.get("key_quotes", [])
                score     = ch.get("key_moment_score", 0.0)
                rank      = ch.get("key_moment_rank", 0)
                sec_id    = ch.get("section_id", 0)

                score_html = f'<span class="ch-score {_score_cls(score)}">{score:.2f}</span>'
                ta_html = ""
                if takeaways:
                    items = "".join(f"<li>{t}</li>" for t in takeaways)
                    ta_html = f'<div class="ch-section-label">Takeaways</div><ul class="ch-takeaways">{items}</ul>'
                kq_html = ""
                if key_quotes:
                    kq_html = '<div class="ch-section-label" style="margin-top:10px;">Key quote</div>'
                    kq_html += f'<div class="ch-quote">{key_quotes[0]}</div>'

                st.markdown(
                    f"""<div class="ch-card" id="ch-{sec_id}">
                      <div class="ch-head">
                        <span class="ch-ts">{ts}</span>
                        <span class="ch-title">{title}</span>
                        {score_html}
                        <span class="ch-rank">#{rank}</span>
                      </div>
                      <div class="ch-body">
                        {f'<p class="ch-summary">{summary}</p>' if summary else ''}
                        {ta_html}{kq_html}
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                col_btn, _ = st.columns([1, 7])
                with col_btn:
                    if st.button(f"Seek {ts}", key=f"seek_{sec_id}"):
                        st.session_state["seek_time"] = ch.get("start_time", 0.0)
                        st.toast(f"Seek set to {ts}. Switch to the Player tab.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Transcript
# ═══════════════════════════════════════════════════════════════════════════
with tab_tr:
    segs_data = st.session_state.get("topic_segments")
    chap_data = st.session_state.get("chapters_data")

    if segs_data is None:
        st.markdown(
            '<div class="empty-state"><strong>No transcript yet</strong>'
            'Run the pipeline to see the full transcript.</div>',
            unsafe_allow_html=True,
        )
    else:
        sections = segs_data.get("sections", [])
        full_text = " ".join(s.get("text", "") for s in sections)

        col_l, col_r = st.columns([3, 1])
        with col_l:
            search_q = st.text_input("Search", placeholder="Search transcript...", label_visibility="collapsed")
        with col_r:
            st.markdown(
                f'<div style="font-size:11px;color:#606060;padding-top:10px;text-align:right;">'
                f'{len(full_text.split()):,} words</div>', unsafe_allow_html=True,
            )

        html = ""
        for s in sections:
            sec_id = s.get("section_id", "")
            ts_str = s.get("timestamp_str", "")
            text   = s.get("text", "")

            ch_title = ""
            if chap_data:
                match = [c for c in chap_data.get("chapters", []) if c.get("section_id") == sec_id]
                if match:
                    ch_title = match[0].get("title", "")

            if search_q and search_q.lower() in text.lower():
                text = re.sub(
                    re.escape(search_q),
                    f'<span class="search-hit">{search_q}</span>',
                    text, flags=re.IGNORECASE,
                )

            title_span = f'<span class="tr-title">{ch_title}</span>' if ch_title else ""
            html += f'<div class="tr-head">{ts_str} {title_span}</div><div>{text}</div><hr class="tr-divider">'

        st.markdown(f'<div class="transcript-wrap">{html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Player
# ═══════════════════════════════════════════════════════════════════════════
with tab_pl:
    media_bytes = st.session_state.get("media_bytes")
    media_mime  = st.session_state.get("media_mime", "audio/mpeg")
    seek_time   = st.session_state.get("seek_time", 0.0)
    src_name    = st.session_state.get("source_name", "")

    if media_bytes is None:
        st.markdown(
            '<div class="empty-state"><strong>No media loaded</strong>'
            'Upload an audio or video file in the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        is_video = media_mime.startswith("video/")
        seek_ts = _fmt_ts(seek_time)

        st.markdown(
            f'<div style="font-size:12px;color:#606060;margin-bottom:14px;">{src_name}</div>',
            unsafe_allow_html=True,
        )
        if seek_time > 0:
            st.markdown(
                f'<div class="seek-banner">Seek target &nbsp;<span class="seek-ts">{seek_ts}</span>'
                f'&nbsp;&mdash;&nbsp;drag the slider to this position</div>',
                unsafe_allow_html=True,
            )

        if is_video:
            st.video(media_bytes, start_time=int(seek_time))
        else:
            st.audio(media_bytes, format=media_mime, start_time=int(seek_time))

        if seek_time > 0:
            tag = "video" if is_video else "audio"
            st.components.v1.html(
                f"""<script>(function(){{
                  var t={seek_time};
                  function s(){{var e=window.parent.document.querySelector('{tag}');
                  if(e){{e.currentTime=t;e.play();}}else{{setTimeout(s,300);}}}}s();
                }})();</script>""", height=0,
            )

        chap_data = st.session_state.get("chapters_data")
        if chap_data:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:10px;font-weight:600;letter-spacing:0.08em;'
                'text-transform:uppercase;color:#606060;margin-bottom:10px;">Chapter timeline</div>',
                unsafe_allow_html=True,
            )
            for ch in chap_data.get("chapters", []):
                ts = _fmt_ts(ch.get("start_time", 0.0))
                c1, c2 = st.columns([1, 7])
                with c1:
                    if st.button(ts, key=f"pl_{ch.get('section_id',0)}"):
                        st.session_state["seek_time"] = ch.get("start_time", 0.0)
                        st.rerun()
                with c2:
                    st.markdown(
                        f'<div style="font-size:13px;color:#a0a0a0;padding-top:6px;">{ch.get("title","")}</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB — Export
# ═══════════════════════════════════════════════════════════════════════════
with tab_ex:
    from llm_pipeline.formatters import (
        to_markdown, to_podcasting2_json, to_raw_json, to_webvtt, to_youtube_chapters,
    )

    chap_data = st.session_state.get("chapters_data")

    if chap_data is None:
        st.markdown(
            '<div class="empty-state"><strong>No analysis yet</strong>'
            'Run the pipeline to unlock export formats.</div>',
            unsafe_allow_html=True,
        )
    else:
        def _export_block(label, desc, content, fname, mime, key):
            st.markdown(
                f'<div class="export-head">{label}</div>'
                f'<div class="export-desc">{desc}</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([5, 1])
            with c1:
                st.text_area(label, content, height=130, key=f"ta_{key}", label_visibility="collapsed")
            with c2:
                st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                st.download_button("Download", data=content.encode(), file_name=fname,
                                   mime=mime, key=f"dl_{key}", use_container_width=True)
            st.markdown("<hr>", unsafe_allow_html=True)

        _export_block("YouTube chapters",
                      "Paste into your YouTube video description. First timestamp must be 00:00.",
                      to_youtube_chapters(chap_data), "youtube_chapters.txt", "text/plain", "yt")
        _export_block("Podcasting 2.0",
                      "chapters.json — compatible with Apple Podcasts, Pocket Casts, Podcasting 2.0.",
                      json.dumps(to_podcasting2_json(chap_data), indent=2),
                      "chapters.json", "application/json", "pc2")
        _export_block("WebVTT",
                      'Chapter track for HTML5 players — use as <track kind="chapters">.',
                      to_webvtt(chap_data), "chapters.vtt", "text/vtt", "vtt")
        _export_block("Markdown notes",
                      "Study guide with timestamps, summaries, takeaways, and key quotes.",
                      to_markdown(chap_data), "study_notes.md", "text/markdown", "md")
        _export_block("Raw JSON",
                      "Full machine-readable output for downstream processing.",
                      to_raw_json(chap_data), "topicfinder_output.json", "application/json", "raw")
