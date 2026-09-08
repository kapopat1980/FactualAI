"""Text vectorization (Section 4.3 of the paper).

The paper's production system encodes text as 768-dimensional embeddings
using the `all-mpnet-base-v2` SentenceTransformers model. That dependency
pulls in PyTorch and a ~420MB model download, which is unnecessary for
running the test suite or a lightweight CI job, so this module exposes a
common `TextEmbedder` interface with two backends:

  - `SentenceTransformerEmbedder` (default in production; requires the
    `sentence-transformers` extra: `pip install factualai[embeddings]`)
  - `TfidfEmbedder` (lightweight fallback used automatically if
    sentence-transformers is not installed, and used in the test suite)

Both expose the same `.encode(texts) -> np.ndarray` interface, so
`classifier.py` and `train.py` work unmodified regardless of backend.
"""
from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np


class TextEmbedder(Protocol):
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Wraps `all-mpnet-base-v2`, matching the paper's production pipeline."""

    MODEL_NAME = "all-mpnet-base-v2"
    DIM = 768

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # deferred import

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts)))


class TfidfEmbedder:
    """Lightweight stand-in embedder (no heavy ML runtime dependency).

    NOT used in the paper's reported results, which use
    SentenceTransformerEmbedder. Provided so the repository is runnable and
    testable in constrained environments (e.g. CI) without a multi-hundred-MB
    model download.
    """

    def __init__(self, max_features: int = 768):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if not self._fitted:
            # Fit-on-first-use so the interface matches SentenceTransformerEmbedder,
            # which needs no corpus-level fitting.
            return self.fit(texts).encode(texts)
        return np.asarray(self._vectorizer.transform(list(texts)).todense())


def get_default_embedder() -> TextEmbedder:
    """Return SentenceTransformerEmbedder if available, else TfidfEmbedder."""
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return TfidfEmbedder()
