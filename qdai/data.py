"""Data loading, preprocessing and summary statistics.

load order, the 3-row placeholder, preprocess_data() and the 80/20 split are
PRESERVED from the original main.py. The only changes are:
  * no Streamlit calls inside the ML functions (the UI reports status instead)
  * column names are stripped of stray whitespace (the public UCI export has a
    trailing tab on 'Daytime/evening attendance')
  * semicolon-delimited exports are detected automatically
  * pandas 3 string dtypes are treated like object columns
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATA_PATHS, RANDOM_STATE, TARGET_MAPPING, TEST_SIZE

_TEXT_DTYPES = ["object", "string"]


# ------------------------------------------------------------------ loading
@dataclass
class Loaded:
    df: pd.DataFrame
    source: str          # file name or path
    kind: str            # 'upload' | 'disk' | 'placeholder'
    notice: str = ""     # e.g. why an uploaded file was rejected


def _read_csv(src) -> pd.DataFrame:
    """Read a CSV from a path or a bytes buffer, tolerating ';' delimiters."""
    def attempt(**kw):
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, **kw)

    try:
        df = attempt(encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = attempt(encoding="latin-1")
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = attempt(sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _placeholder() -> pd.DataFrame:
    """PRESERVED: the minimal sample dataset from the original load_data()."""
    return pd.DataFrame({
        'id': [0, 1, 2],
        'Marital status': [1, 1, 1],
        'Application mode': [1, 17, 17],
        'Application order': [1, 1, 2],
        'Course': [9238, 9238, 9254],
        'Daytime/evening attendance': [1, 1, 1],
        'Previous qualification': [1, 1, 1],
        'Previous qualification (grade)': [126.0, 125.0, 137.0],
        'Nacionality': [1, 1, 1],
        "Mother's qualification": [1, 19, 3],
        "Father's qualification": [19, 19, 19],
        "Mother's occupation": [5, 9, 2],
        "Father's occupation": [5, 9, 3],
        'Admission grade': [122.6, 119.8, 144.7],
        'Displaced': [0, 1, 0],
        'Educational special needs': [0, 0, 0],
        'Debtor': [0, 0, 0],
        'Tuition fees up to date': [1, 1, 1],
        'Gender': [0, 0, 1],
        'Scholarship holder': [1, 0, 0],
        'Age at enrollment': [18, 18, 18],
        'International': [0, 0, 0],
        'Curricular units 1st sem (credited)': [0, 0, 0],
        'Curricular units 1st sem (enrolled)': [6, 6, 6],
        'Curricular units 1st sem (evaluations)': [6, 8, 0],
        'Curricular units 1st sem (approved)': [6, 4, 0],
        'Curricular units 1st sem (grade)': [14.5, 11.6, 0.0],
        'Curricular units 1st sem (without evaluations)': [0, 0, 0],
        'Curricular units 2nd sem (credited)': [0, 0, 0],
        'Curricular units 2nd sem (enrolled)': [6, 6, 6],
        'Curricular units 2nd sem (evaluations)': [7, 9, 0],
        'Curricular units 2nd sem (approved)': [6, 0, 0],
        'Curricular units 2nd sem (grade)': [12.428571428571429, 0.0, 0.0],
        'Curricular units 2nd sem (without evaluations)': [0, 0, 0],
        'Unemployment rate': [11.1, 11.1, 16.2],
        'Inflation rate': [0.6, 0.6, 0.3],
        'GDP': [2.02, 2.02, -0.92],
        'Target': ['Graduate', 'Dropout', 'Dropout'],
    })


@st.cache_data(show_spinner=False)
def load_dataset(upload_name: str | None = None, upload_bytes: bytes | None = None) -> Loaded:
    """Uploaded file first, then the known disk paths, then the placeholder."""
    notice = ""
    if upload_bytes is not None:
        try:
            df = _read_csv(io.BytesIO(upload_bytes))
            return Loaded(df, upload_name or "uploaded.csv", "upload")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            notice = f"Could not read {upload_name or 'the uploaded file'}: {exc}"

    for path in DATA_PATHS:
        try:
            return Loaded(_read_csv(path), path, "disk", notice)
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            notice = f"Could not read {path}: {exc}"

    return Loaded(_placeholder(), "built-in 3-row placeholder", "placeholder", notice)


def validate(df: pd.DataFrame) -> list[str]:
    """Return blocking problems with a dataset (empty list == usable)."""
    problems = []
    if "Target" not in df.columns:
        problems.append(
            "The dataset has no 'Target' column. Add a column named Target whose values are "
            "Dropout, Enrolled or Graduate."
        )
    else:
        unknown = sorted(set(df["Target"].dropna().astype(str)) - set(TARGET_MAPPING))
        if unknown:
            problems.append(
                f"Target contains values the model does not know: {', '.join(unknown[:5])}. "
                "Expected Dropout, Enrolled or Graduate."
            )
    if len(df) < 10:
        problems.append("The dataset has fewer than 10 rows, which is too small for a reliable model.")
    return problems


# ----------------------------------------------------------- preprocessing
def preprocess_data(df: pd.DataFrame):
    """PRESERVED from the original main.py (pandas-3 safe)."""
    processed_df = df.copy()
    if 'id' in processed_df.columns:
        processed_df = processed_df.drop('id', axis=1)
    processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=_TEXT_DTYPES).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Target']
    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])
    if 'Target' in processed_df.columns:
        processed_df['Target'] = processed_df['Target'].map(TARGET_MAPPING)
        processed_df['Target'] = processed_df['Target'].fillna(0)
    for col in processed_df.columns:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    processed_df.dropna(inplace=True)
    X = processed_df.drop('Target', axis=1)
    y = processed_df['Target']
    return X, y, processed_df


@dataclass
class Prepared:
    X: pd.DataFrame
    y: pd.Series
    processed: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dropped_rows: int = 0


@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> Prepared:
    X, y, processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return Prepared(X, y, processed, X_train, X_test, y_train, y_test,
                    dropped_rows=len(df) - len(processed))


# ------------------------------------------------------------- statistics
def summarize(df: pd.DataFrame, n_features: int | None = None) -> dict:
    """Headline numbers, computed exactly as the original dashboard did."""
    n = len(df)
    out = {
        "students": n,
        "columns": df.shape[1],
        "features": n_features if n_features is not None else max(df.shape[1] - 1, 0),
        "missing_pct": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if n else 0.0,
        "dropout_rate": None, "graduate_rate": None, "enrolled_rate": None,
        "counts": {},
    }
    if "Target" in df.columns and n:
        counts = df["Target"].astype(str).value_counts()
        out["counts"] = {k: int(v) for k, v in counts.items()}
        out["dropout_rate"] = float((df["Target"] == "Dropout").sum() / n * 100)
        out["graduate_rate"] = float((df["Target"] == "Graduate").sum() / n * 100)
        out["enrolled_rate"] = float((df["Target"] == "Enrolled").sum() / n * 100)
    return out


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in ("Target", "id")]
