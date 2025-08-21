QualitySignal — NLP & Analytics for Customer Complaints
Turn raw complaint text into actionable signals for quality teams.
Severity prediction — estimate whether a complaint is likely severe
Topic clustering — group similar complaints to reveal themes
Serving — FastAPI endpoints (/predict, /cluster)
Dashboard — Streamlit app for non-technical users
EDA — quick data profiling & plots
Reusable preprocessing — prepare(DATA_DIR) returns a normalized DataFrame:

standardized make, model, modelyear
parsed dates
unified complaint_text
weak label severe := (crash ∨ fire ∨ injuries>0 ∨ deaths>0)

ML baselines
Severity: TF-IDF (1–2 grams) → Logistic Regression (class_weight="balanced")
Clustering: TF-IDF → KMeans (k configurable) with top terms per cluster
Artifacts to disk in models/ so API/UI load instantly 
Clean layout and scripts for reproducible training
