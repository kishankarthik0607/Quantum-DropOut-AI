"""Explainability helpers (SHAP / LIME / permutation / partial dependence).

Behaviour PRESERVED from the original: SHAP TreeExplainer on the Random Forest,
global SHAP = mean |SHAP| averaged over classes on a 100-row sample
(random_state=42), permutation importance with n_repeats=5, LIME on the
training matrix.

Fixed while porting:
  * the original indexed SHAP output with the class LABEL and then described the
    result as "dropout risk" even for non-dropout predictions. Here every
    explanation names the class it explains and the class index is looked up
    from model.classes_.
  * LIME silently explained class index 1 (Graduate) because of its default
    `labels=(1,)`; here the explained class is always explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REVERSE_TARGET, TARGET_MAPPING


def class_index(model, label_name: str) -> int:
    """Position of an outcome (e.g. 'Dropout') inside model.classes_."""
    return list(model.classes_).index(TARGET_MAPPING[label_name])


def make_explainer(model):
    import shap
    return shap.TreeExplainer(model)


def _split_classes(sv, ci: int) -> np.ndarray:
    """Return an (n_samples, n_features) array for class index ci, for any SHAP output format."""
    if isinstance(sv, list):
        return np.asarray(sv[ci])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, ci]
    return sv


def _base_value(explainer, ci: int) -> float:
    ev = np.asarray(explainer.expected_value)
    return float(ev[ci]) if ev.ndim else float(ev)


def shap_local(explainer, model, row: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, float]:
    """SHAP contributions of one student towards ONE outcome.

    Values are in probability units (Random Forest + TreeExplainer), so
    base + sum(values) == the model's predicted probability for that outcome.
    """
    ci = class_index(model, outcome)
    vals = _split_classes(explainer.shap_values(row), ci)[0]
    frame = pd.DataFrame({
        "feature": list(row.columns),
        "value": row.iloc[0].to_numpy(dtype=float),
        "shap": np.asarray(vals, dtype=float).ravel(),
    })
    frame["abs"] = frame["shap"].abs()
    return frame.sort_values("abs", ascending=False).reset_index(drop=True), _base_value(explainer, ci)


def shap_global(explainer, model, X_train: pd.DataFrame, outcome: str | None = None,
                sample: int = 100) -> pd.DataFrame:
    """Mean |SHAP| per feature. outcome=None averages over all classes (original behaviour)."""
    n = min(sample, len(X_train))
    X_sample = X_train.sample(n=n, random_state=42)
    sv = explainer.shap_values(X_sample)
    if outcome is not None:
        per = np.abs(_split_classes(sv, class_index(model, outcome))).mean(axis=0)
    else:
        n_classes = len(model.classes_)
        per = np.mean([np.abs(_split_classes(sv, i)).mean(axis=0) for i in range(n_classes)], axis=0)
    out = pd.DataFrame({"feature": list(X_train.columns), "importance": np.asarray(per).ravel()})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def builtin_importance(model, feature_names) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importances_"):
        return None
    out = pd.DataFrame({"feature": list(feature_names), "importance": model.feature_importances_})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation(model, X_test, y_test, feature_names) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    out = pd.DataFrame({"feature": list(feature_names),
                        "importance": r.importances_mean, "std": r.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def lime_local(model, X_train: pd.DataFrame, row: pd.DataFrame, outcome: str,
               num_features: int = 15) -> list[tuple[str, float]]:
    from lime.lime_tabular import LimeTabularExplainer
    names = [REVERSE_TARGET.get(int(c), str(c)) for c in model.classes_]
    ci = class_index(model, outcome)
    explainer = LimeTabularExplainer(
        X_train.values, feature_names=list(X_train.columns), class_names=names,
        mode="classification", random_state=42)
    exp = explainer.explain_instance(
        row.iloc[0].to_numpy(dtype=float), model.predict_proba,
        num_features=num_features, labels=(ci,))
    return exp.as_list(label=ci)


def partial_dependence(model, X_ref: pd.DataFrame, feature: str, n_grid: int = 40,
                       n_students: int = 100) -> pd.DataFrame:
    """Average predicted probability per outcome as one feature sweeps its observed range.

    Uses up to n_students real students from the hold-out set as the background,
    so the curve reflects the population rather than one arbitrary record.
    """
    lo, hi = float(X_ref[feature].min()), float(X_ref[feature].max())
    grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    base = X_ref.sample(n=min(n_students, len(X_ref)), random_state=42).copy()
    names = [REVERSE_TARGET.get(int(c), str(c)) for c in model.classes_]
    rows = []
    for g in grid:
        tmp = base.copy()
        tmp[feature] = g
        rows.append(model.predict_proba(tmp).mean(axis=0))
    out = pd.DataFrame(rows, columns=names)
    out.insert(0, feature, grid)
    return out
