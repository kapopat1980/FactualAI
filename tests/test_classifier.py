import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factualai.classifier import RandomForestNewsClassifier, FAKE_LABEL, REAL_LABEL
from factualai.embeddings import TfidfEmbedder

FAKE_SAMPLES = [
    "Aliens secretly control the government and hide it from citizens",
    "Miracle cure discovered that doctors do not want you to know about",
    "Celebrity fakes own death to escape taxes, insiders claim",
    "Scientists confirm the earth is actually flat after secret study",
    "Vaccine contains microchips to track the population, leaked memo shows",
] * 10

REAL_SAMPLES = [
    "The central bank raised interest rates by a quarter point on Wednesday",
    "Local council approves new budget for road maintenance this year",
    "Researchers publish peer-reviewed study on renewable energy efficiency",
    "The city council held a public hearing on the new transit plan",
    "Quarterly earnings report shows steady growth for the manufacturing sector",
] * 10


@pytest.fixture
def trained_classifier():
    texts = FAKE_SAMPLES + REAL_SAMPLES
    labels = [FAKE_LABEL] * len(FAKE_SAMPLES) + [REAL_LABEL] * len(REAL_SAMPLES)
    clf = RandomForestNewsClassifier(TfidfEmbedder(), n_estimators=50)
    clf.fit(texts, labels)
    return clf


def test_rf_fake_score_in_valid_range(trained_classifier):
    score = trained_classifier.rf_fake_score("Aliens control the government")
    assert 0.0 <= score <= 1.0


def test_rf_fake_score_raises_before_fit():
    clf = RandomForestNewsClassifier(TfidfEmbedder())
    with pytest.raises(RuntimeError):
        clf.rf_fake_score("some text")


def test_evaluate_returns_expected_keys(trained_classifier):
    metrics = trained_classifier.evaluate(FAKE_SAMPLES[:5] + REAL_SAMPLES[:5], [FAKE_LABEL] * 5 + [REAL_LABEL] * 5)
    assert set(metrics.keys()) == {"accuracy", "precision_fake", "recall_fake", "f1_fake"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_save_and_load_roundtrip(trained_classifier, tmp_path):
    save_path = tmp_path / "model.pkl"
    trained_classifier.save(save_path)

    reloaded = RandomForestNewsClassifier(trained_classifier.embedder)
    reloaded.load(save_path)
    score_original = trained_classifier.rf_fake_score("Aliens control the government")
    score_reloaded = reloaded.rf_fake_score("Aliens control the government")
    assert score_original == pytest.approx(score_reloaded)
