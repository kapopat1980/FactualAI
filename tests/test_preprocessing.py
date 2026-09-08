import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factualai.preprocessing import clean_text, preprocess


def test_clean_text_strips_punctuation_and_lowercases():
    assert clean_text("Hello, World!!  Extra   spaces.") == "hello world extra spaces"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_preprocess_english_text_is_not_translated():
    result = preprocess("This is a plain English sentence.")
    assert result.original_language == "en"
    assert result.translated is False
    assert result.clean_text == "this is a plain english sentence"


def test_preprocess_short_string_fails_open_to_english(monkeypatch):
    # langdetect raises on very short/ambiguous strings; the pipeline should
    # fail open to English rather than raising, per the module's design.
    import factualai.preprocessing as preprocessing_module

    def raise_error(text):
        raise preprocessing_module.LangDetectException("err", "no features")

    monkeypatch.setattr(preprocessing_module, "detect", raise_error)
    result = preprocess("ok")
    assert result.original_language == "en"
