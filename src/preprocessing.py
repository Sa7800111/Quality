# src/preprocessing.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _to_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df

def _std_vehicle_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("make", "model"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    if "modelyear" in df.columns:
        df["modelyear"] = df["modelyear"].astype(str).str.extract(r"(\d{4})", expand=False)
    return df

def _resolve_base_dir(data_dir: str | Path) -> Path:
    """Use Data\\raw if it exists, otherwise Data."""
    p = Path(data_dir)
    return p / "raw" if (p / "raw").exists() else p

def _read_csv(base: Path, name: str) -> pd.DataFrame:
    fp = base / name
    if not fp.exists():
        raise FileNotFoundError(f"Expected file not found: {fp}")
    return pd.read_csv(fp)

def load_all(data_dir: str | Path):
    base = _resolve_base_dir(data_dir)
    car_models     = _read_csv(base, "car_models.csv")
    complaints     = _read_csv(base, "complaints.csv")
    ratings        = _read_csv(base, "ratings.csv")
    recalls        = _read_csv(base, "recalls.csv")
    investigations = _read_csv(base, "investigations.csv")
    return car_models, complaints, ratings, recalls, investigations

def clean_complaints(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_lower(df)
    for c in ("dateofincident", "datecomplaintfiled"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    crash = df.get("crash"); fire = df.get("fire")
    inj   = df.get("numberofinjuries"); death = df.get("numberofdeaths")

    severe = pd.Series(False, index=df.index)
    if crash is not None:
        severe |= crash.astype(str).str.lower().isin(["y","yes","true","1"])
    if fire is not None:
        severe |= fire.astype(str).str.lower().isin(["y","yes","true","1"])
    if inj is not None:
        severe |= pd.to_numeric(inj, errors="coerce").fillna(0) > 0
    if death is not None:
        severe |= pd.to_numeric(death, errors="coerce").fillna(0) > 0
    df["severe"] = severe.astype(int)

    df = _std_vehicle_cols(df)

    if "summary" in df.columns:
        df["complaint_text"] = df["summary"].astype(str)
    elif "complaint" in df.columns:
        df["complaint_text"] = df["complaint"].astype(str)
    else:
        df["complaint_text"] = ""

    return df

def clean_recalls(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_lower(df)
    if "reportreceiveddate" in df.columns:
        df["reportreceiveddate"] = pd.to_datetime(df["reportreceiveddate"], errors="coerce")
    df = _std_vehicle_cols(df)
    for c in ("summary", "consequence", "remedy"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def clean_investigations(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_lower(df)
    for c in ("odate", "cdate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = _std_vehicle_cols(df)
    if "compname" in df.columns and "component" not in df.columns:
        df["component"] = df["compname"]
    return df

def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_lower(df)
    df = _std_vehicle_cols(df)
    if "overallrating" in df.columns:
        df["overallrating"] = df["overallrating"].astype(str)
    return df

def prepare(data_dir: str | Path):
    car_models, complaints, ratings, recalls, investigations = load_all(data_dir)
    complaints     = clean_complaints(complaints)
    ratings        = clean_ratings(ratings)
    recalls        = clean_recalls(recalls)
    investigations = clean_investigations(investigations)
    return car_models, complaints, ratings, recalls, investigations
