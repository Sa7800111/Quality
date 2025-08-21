from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# local
from preprocessing import prepare

def _top_terms_per_cluster(kmeans: KMeans, feature_names: np.ndarray, top_k: int = 15) -> list[list[str]]:
    centers = kmeans.cluster_centers_
    tops = []
    for i in range(centers.shape[0]):
        idx = np.argsort(centers[i])[::-1][:top_k]
        tops.append([feature_names[j] for j in idx])
    return tops

def cluster_tfidf(complaints: pd.DataFrame, models_dir: Path, n_clusters=8, max_features=20000):
    text = complaints["complaint_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(text)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    models_dir.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, models_dir / "kmeans_vectorizer_tfidf.joblib")
    dump(kmeans,    models_dir / "kmeans_tfidf.joblib")

    feature_names = np.array(vectorizer.get_feature_names_out())
    tops = _top_terms_per_cluster(kmeans, feature_names, top_k=15)
    with open(models_dir / "cluster_top_terms.txt", "w", encoding="utf-8") as f:
        for i, terms in enumerate(tops):
            f.write(f"Cluster {i}: {', '.join(terms)}\n")

    out = complaints.copy()
    out["cluster_tfidf"] = labels
    out.head(1000).to_csv(models_dir / "complaints_with_clusters_sample.csv", index=False)

    print(f"Saved TF-IDF KMeans + artifacts to {models_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Cluster complaint text with TF-IDF + KMeans")
    p.add_argument("--data-dir",   type=str, default=str(Path(r"E:\CustomerQuality\Data")))
    p.add_argument("--models-dir", type=str, default=str(Path(r"E:\CustomerQuality\models")))
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--max-features", type=int, default=20000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    _, complaints, *_ = prepare(Path(args.data_dir))
    cluster_tfidf(
        complaints=complaints,
        models_dir=Path(args.models_dir),
        n_clusters=args.n_clusters,
        max_features=args.max_features,
    )
