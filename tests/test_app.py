import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factualai.app import create_app


@pytest.fixture
def client():
    mock_classifier = MagicMock()
    mock_classifier.rf_fake_score.return_value = 0.83

    app = create_app(classifier=mock_classifier, llm_reasoner=None)
    app.testing = True
    return app.test_client()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_text_endpoint(client):
    response = client.post("/predict/text", json={"text": "Some news article text."})
    assert response.status_code == 200
    body = response.get_json()
    assert body["label"] in {"Fake", "Real"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["llm_confidence"] is None  # no LLM configured in this fixture


def test_predict_text_endpoint_rejects_empty_text(client):
    response = client.post("/predict/text", json={"text": "   "})
    assert response.status_code == 400


def test_stats_endpoint_counts_requests(client):
    client.post("/predict/text", json={"text": "First article."})
    client.post("/predict/text", json={"text": "Second article."})
    response = client.get("/stats")
    assert response.get_json()["total_requests"] == 2
