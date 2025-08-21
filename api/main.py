from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

MODELS_DIR = Path(r"E:\CustomerQuality\models")
app = FastAPI(title="Toyota Quality API", version="0.1.0")

_vectorizer = _classifier = _kvec = _kmeans = None

def get_classifier():
    global _vectorizer, _classifier
    if _vectorizer is None or _classifier is None:
        _vectorizer = load(MODELS_DIR / "vectorizer_tfidf.joblib")
        _classifier = load(MODELS_DIR / "classifier_logreg.joblib")
    return _vectorizer, _classifier

def get_kmeans():
    global _kvec, _kmeans
    if _kvec is None or _kmeans is None:
        _kvec = load(MODELS_DIR / "kmeans_vectorizer_tfidf.joblib")
        _kmeans = load(MODELS_DIR / "kmeans_tfidf.joblib")
    return _kvec, _kmeans

class PredictIn(BaseModel):
    complaint_text: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(inp: PredictIn):
    vec, clf = get_classifier()
    X = vec.transform([inp.complaint_text or ""])
    y = int(clf.predict(X)[0])
    proba = getattr(clf,"predict_proba",lambda z: None)(X)
    p1 = float(proba[0][1]) if proba is not None else None
    return {"severe": y, "prob_severe": p1}

@app.post("/cluster")
def cluster(inp: PredictIn):
    v, km = get_kmeans()
    X = v.transform([inp.complaint_text or ""])
    label = int(km.predict(X)[0])
    return {"cluster": label}
