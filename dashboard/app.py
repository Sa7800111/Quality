import streamlit as st
from pathlib import Path
from joblib import load

MODELS_DIR = Path(r"E:\CustomerQuality\models")
st.set_page_config(page_title="Toyota Quality Demo", layout="wide")

@st.cache_resource
def load_models():
    vec = load(MODELS_DIR / "vectorizer_tfidf.joblib")
    clf = load(MODELS_DIR / "classifier_logreg.joblib")
    kvec = load(MODELS_DIR / "kmeans_vectorizer_tfidf.joblib")
    km = load(MODELS_DIR / "kmeans_tfidf.joblib")
    return vec, clf, kvec, km

st.title("Customer Complaint Analyzer")
vec, clf, kvec, km = load_models()

txt = st.text_area("Paste complaint text:", height=180, placeholder="Engine stalled at highway speed…")
if st.button("Analyze"):
    X = vec.transform([txt or ""])
    y = int(clf.predict(X)[0])
    proba = getattr(clf,"predict_proba",lambda z: None)(X)
    p1 = float(proba[0][1]) if proba is not None else None
    Xk = kvec.transform([txt or ""])
    cl = int(km.predict(Xk)[0])
    st.subheader("Result")
    st.json({"severe": y, "prob_severe": p1, "cluster": cl})
