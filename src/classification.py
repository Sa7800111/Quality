from __future__ import annotations
from pathlib import Path
import json
import argparse
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import prepare

def train_classifier(data_dir: Path, models_dir: Path,
                     test_size=0.2, random_state=42, max_features=20000):
    models_dir.mkdir(parents=True, exist_ok=True)
    _, complaints, *_ = prepare(data_dir)
    if "complaint_text" not in complaints.columns or "severe" not in complaints.columns:
        raise ValueError("Expected 'complaint_text' and 'severe' after preprocessing.")
    X_text = complaints["complaint_text"].fillna("").astype(str)
    y = complaints["severe"].astype(int)

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english",
                                 ngram_range=(1,2), lowercase=True, strip_accents="unicode")
    X = vectorizer.fit_transform(X_text)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    yhat = clf.predict(Xte)
    report = classification_report(yte, yhat, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yhat).tolist()

    print("=== Classification Report ===")
    print(classification_report(yte, yhat, zero_division=0))
    print("Confusion Matrix:", cm)

    dump(vectorizer, models_dir / "vectorizer_tfidf.joblib")
    dump(clf,        models_dir / "classifier_logreg.joblib")
    with open(models_dir / "classifier_metrics.json","w",encoding="utf-8") as f:
        json.dump({"classes": sorted(list(np.unique(y))),
                   "report": report, "confusion_matrix": cm}, f, indent=2)
    print(f"Saved to {models_dir}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data-dir",   type=str, default=str(Path(r"E:\CustomerQuality\Data")))
    a.add_argument("--models-dir", type=str, default=str(Path(r"E:\CustomerQuality\models")))
    a.add_argument("--test-size", type=float, default=0.2)
    a.add_argument("--random-state", type=int, default=42)
    a.add_argument("--max-features", type=int, default=20000)
    args = a.parse_args()
    train_classifier(Path(args.data_dir), Path(args.models_dir),
                     test_size=args.test_size, random_state=args.random_state,
                     max_features=args.max_features)
