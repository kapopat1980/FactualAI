#!/usr/bin/env python3
"""Train and evaluate the Random Forest classifier on the ISOT Fake News
Dataset, reproducing the Section 8.1 / Table 2 evaluation methodology
(80/20 stratified split, accuracy/precision/recall/F1 on the FAKE class).

Usage:
    python scripts/train.py --data-dir data/raw --output models/rf_classifier.pkl

Expects `Fake.csv` and `True.csv` in --data-dir (see data/README.md for
where to obtain the dataset -- it is not redistributed in this repository).
By default this script uses the lightweight TfidfEmbedder so it runs without
a PyTorch/sentence-transformers install; pass --embedder sentence-transformer
to reproduce the exact embedding backend used for the paper's reported
95% accuracy figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factualai.classifier import RandomForestNewsClassifier, train_test_split_dataset
from factualai.embeddings import SentenceTransformerEmbedder, TfidfEmbedder


def load_dataset(data_dir: Path) -> tuple[list[str], list[int]]:
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    texts = list(fake_df["title"].astype(str) + ". " + fake_df["text"].astype(str)) + list(
        true_df["title"].astype(str) + ". " + true_df["text"].astype(str)
    )
    labels = [1] * len(fake_df) + [0] * len(true_df)  # 1 = Fake, 0 = Real
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("models/rf_classifier.pkl"))
    parser.add_argument(
        "--embedder", choices=["tfidf", "sentence-transformer"], default="tfidf"
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir} ...")
    texts, labels = load_dataset(args.data_dir)
    print(f"Loaded {len(texts)} articles ({sum(labels)} fake, {len(labels) - sum(labels)} real).")

    train_texts, test_texts, train_labels, test_labels = train_test_split_dataset(texts, labels)
    print(f"Train: {len(train_texts)}  Test: {len(test_texts)} (20% held out, matching Section 8.1)")

    embedder = SentenceTransformerEmbedder() if args.embedder == "sentence-transformer" else TfidfEmbedder()
    classifier = RandomForestNewsClassifier(embedder, n_estimators=args.n_estimators)

    print("Training Random Forest classifier ...")
    classifier.fit(train_texts, train_labels)

    print("Evaluating on held-out test set ...")
    metrics = classifier.evaluate(test_texts, test_labels)
    print("\n=== Results (compare to Table 2) ===")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision (Fake): {metrics['precision_fake']:.4f}")
    print(f"Recall (Fake):    {metrics['recall_fake']:.4f}")
    print(f"F1 (Fake):        {metrics['f1_fake']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.output)
    print(f"\nSaved trained model to {args.output}")


if __name__ == "__main__":
    main()
