"""Tests for the weighted ensemble decision logic (Section 4.6).

These tests specifically exercise the two subtle behaviors that were
identified as inconsistent in earlier manuscript drafts and corrected before
submission:
  1. The LLM's (label, confidence) output must be converted into a
     directional LLM_fake_score before aggregation.
  2. On LLM timeout/unavailability, the ensemble must fall back to the RF
     score alone (never raise, never block).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factualai.ensemble import (
    RF_WEIGHT,
    LLM_WEIGHT,
    classify_with_ensemble,
    compute_ensemble,
    llm_verdict_to_fake_score,
)
from factualai.llm_reasoner import LLMReasoner, LLMUnavailableError, LLMVerdict


def test_weights_sum_to_one():
    assert RF_WEIGHT + LLM_WEIGHT == pytest.approx(1.0)


@pytest.mark.parametrize(
    "label,confidence,expected",
    [
        ("Fake", 0.9, 0.9),
        ("Real", 0.9, 0.1),
        ("Real", 0.6, 0.4),
        ("fake", 0.7, 0.7),
    ],
)
def test_llm_verdict_to_fake_score(label, confidence, expected):
    verdict = LLMVerdict(label=label, confidence=confidence, justification="")
    assert llm_verdict_to_fake_score(verdict) == pytest.approx(expected)


def test_compute_ensemble_matches_worked_example_from_paper():
    # Figure 3 of the paper: RF says "Real" at 64% confidence
    # (RF_fake_score = 0.36), LLM says "Real" at 90% confidence
    # (LLM_fake_score = 0.10). Ensemble should land on "Real" ~74%.
    rf_fake_score = 1 - 0.64
    llm_fake_score = 1 - 0.90
    label, confidence, ensemble_fake = compute_ensemble(rf_fake_score, llm_fake_score)
    assert label == "Real"
    assert confidence == pytest.approx(0.74, abs=0.01)


def test_compute_ensemble_boundary_favors_fake_at_exactly_half():
    label, confidence, ensemble_fake = compute_ensemble(0.5, 0.5)
    assert ensemble_fake == pytest.approx(0.5)
    assert label == "Fake"  # >= 0.5 threshold, per Section 4.6
    assert confidence == pytest.approx(0.5)


def test_classify_with_ensemble_uses_llm_when_available():
    class StubReasoner:
        def classify(self, text):
            return LLMVerdict(label="Fake", confidence=0.8, justification="stub reason")

    result = classify_with_ensemble("some article text", rf_fake_score=0.7, llm_reasoner=StubReasoner())

    assert result.used_llm is True
    assert result.llm_fake_score == pytest.approx(0.8)
    expected_ensemble = (0.7 * RF_WEIGHT) + (0.8 * LLM_WEIGHT)
    assert result.ensemble_fake_score == pytest.approx(expected_ensemble)
    assert result.llm_justification == "stub reason"


def test_classify_with_ensemble_falls_back_to_rf_on_llm_timeout():
    """This is the behavior described in Section 4.5 ('falls back to the
    Random Forest prediction alone') and reconciled with Section 4.6's
    aggregation formula: on timeout, LLM_fake_score is set equal to
    RF_fake_score, so ensemble_fake_score == rf_fake_score exactly."""

    class TimingOutReasoner:
        def classify(self, text):
            raise LLMUnavailableError("Ollama request timed out")

    result = classify_with_ensemble("some article text", rf_fake_score=0.83, llm_reasoner=TimingOutReasoner())

    assert result.used_llm is False
    assert result.llm_fake_score is None
    assert result.ensemble_fake_score == pytest.approx(0.83)
    assert result.label == "Fake"


def test_classify_with_ensemble_no_reasoner_configured():
    result = classify_with_ensemble("some article text", rf_fake_score=0.2, llm_reasoner=None)
    assert result.used_llm is False
    assert result.ensemble_fake_score == pytest.approx(0.2)
    assert result.label == "Real"


def test_llm_reasoner_raises_unavailable_on_bad_json(monkeypatch):
    import requests

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "not valid json"}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    reasoner = LLMReasoner()
    with pytest.raises(LLMUnavailableError):
        reasoner.classify("some article text")


def test_llm_reasoner_raises_unavailable_on_connection_error(monkeypatch):
    import requests

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)
    reasoner = LLMReasoner()
    with pytest.raises(LLMUnavailableError):
        reasoner.classify("some article text")


def test_unknown_label_is_an_abstention():
    verdict = LLMVerdict(label="Unknown", confidence=0.5, justification="cannot judge")
    assert llm_verdict_to_fake_score(verdict) is None


def test_classify_with_ensemble_falls_back_to_rf_when_llm_abstains():
    """An "Unknown" answer must not pull the score towards 0.5: the ensemble
    reduces to the Random Forest and the abstention is flagged (Section 4.4)."""

    class AbstainingReasoner:
        def classify(self, text):
            return LLMVerdict(label="Unknown", confidence=0.5, justification="transcript unclear")

    result = classify_with_ensemble("garbled text", rf_fake_score=0.25, llm_reasoner=AbstainingReasoner())
    assert result.used_llm is False
    assert result.llm_abstained is True
    assert result.ensemble_fake_score == pytest.approx(0.25)
    assert result.label == "Real"
    assert result.llm_justification == "transcript unclear"
