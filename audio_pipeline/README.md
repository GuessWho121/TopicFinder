# Audio + Transcription Foundation

This module converts an FFmpeg-supported audio or video recording into the
timestamped `transcript.json` consumed by downstream topic-analysis modules.
It intentionally performs no topic detection, summarization, or key-moment work.

## Setup

Install [FFmpeg](https://ffmpeg.org/download.html) and ensure both `ffmpeg` and
`ffprobe` are on your `PATH`. Then install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

From the project root:

```powershell
python -m audio_pipeline.pipeline data/input/lecture.mp4 --output-dir data/output --model base --language en
```

The first run downloads the selected Whisper model. The output directory contains
`transcript.json` and, unless `--discard-wav` is supplied, a normalized mono,
16 kHz PCM WAV file.

## Output contract

```json
{
  "audio_file": "D:\\TopicFinder\\data\\input\\lecture.mp4",
  "duration": 1800.12,
  "language": "en",
  "segments": [
    { "id": 0, "start": 0.0, "end": 5.4, "text": "Welcome to the lecture." }
  ]
}
```

Segment timestamps are measured in seconds from the start of the recording.
