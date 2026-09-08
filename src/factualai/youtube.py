"""YouTube video detection (Section 5.4 of the paper).

Downloads the highest-quality available audio track with yt-dlp, converts it
to MP3 via FFmpeg (through pydub), then passes it through the same
Faster-Whisper transcription pipeline used for uploaded audio files. Optional
extra: `pip install factualai[youtube]` (also requires the `ffmpeg` system
binary).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .asr import transcribe_audio


def download_audio(youtube_url: str, output_dir: str | None = None) -> str:
    """Download and convert a YouTube video's audio track to MP3.
    Returns the path to the resulting MP3 file."""
    import yt_dlp

    output_dir = output_dir or tempfile.mkdtemp()
    output_template = str(Path(output_dir) / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]
    return str(Path(output_dir) / f"{video_id}.mp3")


def transcribe_youtube_video(youtube_url: str) -> str:
    """End-to-end: download audio, convert to MP3, transcribe. Typical
    end-to-end latency is 15-30 seconds depending on video length and
    network conditions (Section 5.4)."""
    mp3_path = download_audio(youtube_url)
    return transcribe_audio(mp3_path)
