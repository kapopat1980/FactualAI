"""Audio input via Faster-Whisper ASR (Section 5.3 of the paper).

Transcribes MP3/WAV/etc. using Faster-Whisper (base model), an optimized
implementation of OpenAI's Whisper. Optional extra: `pip install
factualai[asr]`.
"""
from __future__ import annotations

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel("base")
    return _model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file and return the full transcription text."""
    model = _get_model()
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments)
