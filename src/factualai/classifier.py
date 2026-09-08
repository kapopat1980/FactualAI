"""Random Forest classifier over text embeddings (Section 4.4 of the paper).

An ensemble of decision trees, each trained on a bootstrap sample, classifying
the embeddings produced by `embeddings.py`. Class probabilities come directly
from the proportion of trees voting for each label.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from .embeddings import TextEmbedder

FAKE_LABEL = 1
REAL_LABEL = 0


class RandomForestNewsClassifier:
    """Thin wrapper around scikit-learn's RandomForestClassifier that operates
    on embeddings rather than raw text, and reports the RF_fake_score used by
    the weighted ensemble in `ensemble.py`."""

    def __init__(self, embedder: TextEmbedder, n_estimators: int = 200, random_state: int = 42):
        self.embedder = embedder
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, texts: list[str], labels: list[int]) -> "RandomForestNewsClassifier":
        embeddings = self.embedder.encode(texts)
        self.model.fit(embeddings, labels)
        self._is_fitted = True
        return self

    def rf_fake_score(self, text: str) -> float:
        """Return RF_fake_score = P(label == FAKE) for a single piece of text,
        as consumed by the ensemble formula in Section 4.6."""
        if not self._is_fitted:
            raise RuntimeError("Classifier has not been fitted yet.")
        embedding = self.embedder.encode([text])
        proba = self.model.predict_proba(embedding)[0]
        class_index = list(self.model.classes_).index(FAKE_LABEL)
        return float(proba[class_index])

    def evaluate(self, texts: list[str], labels: list[int]) -> dict:
        """Compute accuracy/precision/recall/F1 for the FAKE class, matching
        Table 2 of the paper."""
        embeddings = self.embedder.encode(texts)
        predictions = self.model.predict(embeddings)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=[FAKE_LABEL], average="binary", pos_label=FAKE_LABEL
        )
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_fake": precision,
            "recall_fake": recall,
            "f1_fake": f1,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str | Path) -> "RandomForestNewsClassifier":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._is_fitted = True
        return self


def train_test_split_dataset(texts: list[str], labels: list[int], test_size: float = 0.2, random_state: int = 42):
    """80/20 split matching Section 8.1 ('a held-out test set... 20% of the dataset')."""
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
