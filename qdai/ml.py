"""Model training and evaluation. No Streamlit imports: pure ML.

PRESERVED: train_model() (RandomForest, identical hyper-parameters).
ADDED (real, computed from the hold-out split, nothing hard-coded):
  * evaluate(): accuracy, weighted precision / recall / F1, macro OvR ROC-AUC,
    per-class report and confusion matrix for any fitted model
  * train_baselines(): Logistic Regression and Gradient Boosting trained on the
    SAME split so the Models page can compare engines honestly
"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, REVERSE_TARGET, RF_PARAMS

PRIMARY_MODEL = "Random Forest"


def train_model(X_train, y_train):
    """PRESERVED: the Random Forest the whole application predicts with."""
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_baselines(X_train, y_train) -> tuple[dict, dict]:
    """Fit comparison models. Returns (models, errors) - a failure never blocks the primary model."""
    models, errors = {}, {}
    builders = {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
    }
    for name, build in builders.items():
        try:
            m = build()
            m.fit(X_train, y_train)
            models[name] = m
        except Exception as exc:  # noqa: BLE001
            errors[name] = str(exc)
    return models, errors


def class_names(model) -> list[str]:
    return [REVERSE_TARGET.get(int(c), f"Class {c}") for c in model.classes_]


def evaluate(model, X_test, y_test) -> dict:
    """Every number here is computed from the model's predictions on the hold-out set."""
    y_pred = model.predict(X_test)
    labels = list(model.classes_)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    roc = None
    try:
        proba = model.predict_proba(X_test)
        roc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=labels))
    except Exception:  # noqa: BLE001 - e.g. a class missing from the test split
        roc = None
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": roc,
        "report": report,
        "confusion": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "label_names": [REVERSE_TARGET.get(int(c), f"Class {c}") for c in labels],
        "n_test": int(len(y_test)),
    }


def fit_all(X_train, y_train, X_test, y_test, on_step=lambda label, state: None) -> dict:
    """Train + evaluate everything, reporting each REAL step through on_step()."""
    t0 = time.perf_counter()
    result = {"models": {}, "evals": {}, "errors": {}, "timings": {}}

    on_step("Random Forest", "running")
    t = time.perf_counter()
    rf = train_model(X_train, y_train)
    result["models"][PRIMARY_MODEL] = rf
    result["timings"][PRIMARY_MODEL] = time.perf_counter() - t
    on_step("Random Forest", "done")

    on_step("Comparison models", "running")
    base, errs = train_baselines(X_train, y_train)
    result["models"].update(base)
    result["errors"].update(errs)
    on_step("Comparison models", "done")

    on_step("Evaluation", "running")
    for name, m in result["models"].items():
        try:
            result["evals"][name] = evaluate(m, X_test, y_test)
        except Exception as exc:  # noqa: BLE001
            result["errors"][name] = str(exc)
    on_step("Evaluation", "done")

    result["seconds"] = time.perf_counter() - t0
    return result


def predict_row(model, columns: list[str], values: dict, fallback: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Build a one-row frame in training-column order and return class probabilities by name."""
    row = {c: values.get(c, fallback[c]) for c in columns}
    frame = pd.DataFrame([row], columns=columns).astype(float)
    proba = model.predict_proba(frame)[0]
    return frame, {n: float(p) for n, p in zip(class_names(model), proba)}
