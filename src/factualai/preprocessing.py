"""Text preprocessing and multilingual normalization (Section 4.2 of the paper).

Incoming text is stripped of punctuation/special characters, lowercased, and
trimmed. If the detected source language is not English, the text is
translated to English via the Deep Translator library (Google Translate
backend) prior to embedding/classification. The original source language is
preserved so callers can surface a "Translated from <language>" indicator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover - exercised only if dependency missing
    detect = None
    LangDetectException = Exception

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreprocessResult:
    """Result of running the multilingual preprocessing pipeline."""

    clean_text: str
    original_language: str
    translated: bool


def clean_text(text: str) -> str:
    """Strip punctuation/special characters, lowercase, and collapse whitespace."""
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def detect_language(text: str) -> str:
    """Best-effort language detection. Returns 'en' if detection is unavailable
    or fails (e.g. on very short strings), matching the paper's fail-open design
    so the pipeline never blocks on a language-ID error."""
    if detect is None or not text.strip():
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_to_english(text: str, source_language: str) -> str:
    """Translate text to English using Deep Translator (Google Translate backend).

    Falls back to returning the original text unchanged if the translator
    dependency/service is unavailable, so a network hiccup degrades gracefully
    rather than raising.
    """
    if source_language == "en" or GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=source_language, target="en").translate(text)
    except Exception:
        return text


def preprocess(raw_text: str) -> PreprocessResult:
    """Run the full Section 4.2 pipeline: clean -> detect language -> translate.

    Language detection and translation happen on the *raw* text (translators
    need real sentence structure); cleaning is applied last, to the
    (possibly translated) English text, before it is passed to embedding.
    """
    language = detect_language(raw_text)
    translated_text = translate_to_english(raw_text, language)
    return PreprocessResult(
        clean_text=clean_text(translated_text),
        original_language=language,
        translated=language != "en",
    )
