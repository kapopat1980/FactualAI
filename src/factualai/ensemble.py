"""Weighted ensemble decision (Section 4.6 of the paper).

    ensemble_fake = (RF_fake_score * 0.6) + (LLM_fake_score * 0.4)

The Random Forest receives the larger weight because it was found to be the
stronger standalone model on the held-out test set; the LLM contributes
contextual/world knowledge the RF cannot see.

LLM_fake_score conversion: the LLM (see `llm_reasoner.py`) returns a
(label, confidence) pair where confidence in [0.5, 1.0] is the confidence in
*whichever* label it chose -- not directly a P(fake) score. This module
converts that into a directional fake-score before aggregating:

    LLM_fake_score = confidence        if label == "Fake"
    LLM_fake_score = 1 - confidence    if label == "Real"

Fallback: if the LLM call times out or fails (`LLMUnavailableError`), or the
LLM abstains (label other than "Fake"/"Real", e.g. "Unknown"), the ensemble
substitutes the RF's own score for LLM_fake_score, so a request is never
blocked on the LLM and the ensemble reduces exactly to the Random Forest
prediction -- matching Section 4.4 of the revised paper. Abstentions are
flagged in the result (`llm_abstained`) so the interface can show them.

Weight: the deployed default is w = 0.6 for the Random Forest. The revised
paper (Section 7.5, Table 7) shows that with the 1B LLM the best validated
weight is w = 1.0 (LLM excluded from the decision), whereas with LLaMA 3.1 8B
a weight of about 0.6-0.8 improves accuracy. Set FACTUALAI_RF_WEIGHT to choose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, LLMUnavailableError, LLMVerdict

RF_WEIGHT = float(os.environ.get("FACTUALAI_RF_WEIGHT", "0.6"))
if not 0.0 <= RF_WEIGHT <= 1.0:
    raise ValueError("FACTUALAI_RF_WEIGHT must be between 0 and 1")
LLM_WEIGHT = 1.0 - RF_WEIGHT


@dataclass
class EnsembleResult:
    label: str  # "Fake" or "Real"
    confidence: float  # confidence in `label`, in [0, 1]
    ensemble_fake_score: float
    rf_fake_score: float
    llm_fake_score: float | None  # None if the LLM fallback path was used
    llm_justification: str | None
    used_llm: bool
    llm_abstained: bool = False  # LLM answered but declined to judge ("Unknown")


def llm_verdict_to_fake_score(verdict: LLMVerdict) -> float | None:
    """Convert an LLM (label, confidence) verdict into a directional
    P(fake)-style score compatible with RF_fake_score.

    Returns None when the LLM abstains ("Unknown" or any unrecognized label);
    the caller then falls back to the Random Forest score (Section 4.4)."""
    label = str(verdict.label).strip().capitalize()
    if label == "Fake":
        return verdict.confidence
    if label == "Real":
        return 1.0 - verdict.confidence
    return None


def compute_ensemble(rf_fake_score: float, llm_fake_score: float) -> tuple[str, float, float]:
    """Combine RF and LLM fake-scores into a final (label, confidence,
    ensemble_fake_score) triple."""
    ensemble_fake = (rf_fake_score * RF_WEIGHT) + (llm_fake_score * LLM_WEIGHT)
    if ensemble_fake >= 0.5:
        return "Fake", ensemble_fake, ensemble_fake
    return "Real", 1.0 - ensemble_fake, ensemble_fake


def classify_with_ensemble(
    text: str,
    rf_fake_score: float,
    llm_reasoner: LLMReasoner | None,
) -> EnsembleResult:
    """Run the full ensemble decision described in Section 4.6, including the
    30-second-timeout fallback to the Random Forest prediction alone."""
    if llm_reasoner is None:
        # No LLM configured at all: RF-only mode.
        label, confidence, ensemble_fake = compute_ensemble(rf_fake_score, rf_fake_score)
        return EnsembleResult(
            label=label,
            confidence=confidence,
            ensemble_fake_score=ensemble_fake,
            rf_fake_score=rf_fake_score,
            llm_fake_score=None,
            llm_justification=None,
            used_llm=False,
        )

    abstained = False
    try:
        verdict = llm_reasoner.classify(text)
        llm_fake_score = llm_verdict_to_fake_score(verdict)
        justification = verdict.justification
        used_llm = True
        if llm_fake_score is None:
            # LLM abstained: keep its explanation, but do not let it vote.
            llm_fake_score = rf_fake_score
            used_llm = False
            abstained = True
    except LLMUnavailableError:
        # Section 4.5 fallback: substitute the RF's own score for
        # LLM_fake_score so the request is never blocked on the LLM.
        llm_fake_score = rf_fake_score
        justification = None
        used_llm = False

    label, confidence, ensemble_fake = compute_ensemble(rf_fake_score, llm_fake_score)
    return EnsembleResult(
        label=label,
        confidence=confidence,
        ensemble_fake_score=ensemble_fake,
        rf_fake_score=rf_fake_score,
        llm_fake_score=llm_fake_score if used_llm else None,
        llm_justification=justification,
        used_llm=used_llm,
        llm_abstained=abstained,
    )
