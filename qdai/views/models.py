"""Models: training console, real evaluation metrics, and explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import explain, ml
from ..config import REVERSE_TARGET, TEST_SIZE, RANDOM_STATE
from ..ui import charts as C
from ..ui.components import esc, html, insight, note, page_head, section, show, stat_strip, stretch
from ._common import get_explainer, primary_model


# ------------------------------------------------------------------ training
def _ledger(P, fit, error) -> None:
    def item(k, v, state=None):
        dot = f'<span class="qd-dot {state}"></span>' if state else ""
        return f'<div class="qd-ledger-i"><div class="qd-ledger-k">{esc(k)}</div><div class="qd-ledger-v">{dot}{esc(v)}</div></div>'
    if fit:
        t_state, t_txt = "done", "Complete"
        e_ok = bool(fit["res"]["evals"])
        e_state, e_txt = ("done", "Complete") if e_ok else ("failed", "Failed")
        secs = f'{fit["res"]["seconds"]:.1f} s'
    elif error:
        t_state, t_txt, e_state, e_txt, secs = "failed", "Failed", None, "Not run", "-"
    else:
        t_state, t_txt, e_state, e_txt, secs = None, "Not started", None, "Not started", "-"
    html('<div class="qd-ledger">' + item("Dataset", f"{len(P.X):,} records") + item("Features", f"{P.X.shape[1]}")
         + item("Hold-out split", f"{len(P.X_train):,} train · {len(P.X_test):,} test")
         + item("Training", t_txt, t_state) + item("Evaluation", e_txt, e_state) + item("Training time", secs) + "</div>")


def _train(P) -> None:
    with st.status("Training models", expanded=True) as status:
        def on_step(label, s):
            st.write(f"{label}: {'in progress' if s == 'running' else 'complete'}")
        try:
            res = ml.fit_all(P.X_train, P.y_train, P.X_test, P.y_test, on_step)
        except Exception as exc:  # noqa: BLE001
            st.session_state.train_error = str(exc)
            status.update(label="Training failed", state="error")
            return
    st.session_state.train_error = None
    st.session_state.explain_cache = {}
    st.session_state.pop("explainer", None)
    st.session_state.pred = None
    st.session_state.fit = {"res": res, "n_train": len(P.X_train), "n_test": len(P.X_test),
                            "feature_names": P.X.columns.tolist()}
    status.update(label=f"Training complete in {res['seconds']:.1f} s", state="complete", expanded=False)


def _rows(fit) -> None:
    res = fit["res"]
    order = [ml.PRIMARY_MODEL] + [m for m in res["evals"] if m != ml.PRIMARY_MODEL]
    out = []
    for name in order:
        ev = res["evals"].get(name)
        if not ev:
            continue
        prim = name == ml.PRIMARY_MODEL
        cells = ""
        for lab, key, pct in (("Accuracy", "accuracy", True), ("Precision", "precision", True), ("Recall", "recall", True),
                              ("F1", "f1", True), ("ROC-AUC", "roc_auc", False)):
            v = ev[key]
            txt = "n/a" if v is None else (f"{v * 100:.1f}%" if pct else f"{v:.3f}")
            w = 0 if v is None else v * 100
            cells += (f'<div><div class="qd-metric-val">{txt}</div><div class="qd-metric-lab">{lab}</div>'
                      f'<div class="qd-metric-bar"><i style="width:{w:.1f}%"></i></div></div>')
        role = "Prediction engine · used for every prediction and explanation" if prim else "Comparison model · not used for predictions"
        out.append(f'<div class="qd-model{" primary" if prim else ""}"><div class="qd-model-head">'
                   f'<div class="qd-model-name">{esc(name)}</div>'
                   f'<div class="qd-model-role{" primary" if prim else ""}">{role}</div></div>{cells}</div>')
    html("".join(out))
    n = fit["n_test"]
    st.caption(f"Measured on the {n:,} students held out from training ({int(TEST_SIZE * 100)}% split, random_state {RANDOM_STATE}). "
               "Precision, recall and F1 are weighted averages over the three outcomes; ROC-AUC is macro one-vs-rest.")
    for name, err in res["errors"].items():
        note(f"<b>{esc(name)} could not be evaluated.</b> {esc(err)}")


# ---------------------------------------------------------------- evaluation
def _evaluation(fit) -> None:
    res, model = fit["res"], primary_model(fit)
    ev = res["evals"][ml.PRIMARY_MODEL]
    section("Confusion matrix", "Where the prediction engine is right and where it is not.", first=True)
    mode = st.radio("Show", ["Counts", "Row percentages"], horizontal=True, key="cm_mode")
    show(C.confusion(ev["confusion"], ev["label_names"], mode != "Counts"))
    names, cm = ev["label_names"], ev["confusion"]
    if "Dropout" in names:
        i = names.index("Dropout")
        tot = int(cm[i].sum())
        if tot:
            insight(f"Of {tot:,} held-out students who actually dropped out, the model identified {int(cm[i, i]):,} "
                    f"({cm[i, i] / tot * 100:.1f}%). Of everyone it labelled Dropout, {cm[:, i][i] / max(cm[:, i].sum(), 1) * 100:.1f}% really did.")

    section("Per-outcome report", "Precision, recall and F1 for each outcome.")
    rep = ev["report"]
    rows = []
    for lab, nm in zip(ev["labels"], names):
        r = rep.get(str(lab)) or rep.get(str(float(lab))) or {}
        rows.append({"Outcome": nm, "Precision": r.get("precision"), "Recall": r.get("recall"),
                     "F1": r.get("f1-score"), "Students": int(r.get("support", 0))})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, **stretch())

    section("Feature importance", "Built-in importance of the Random Forest, top 15 features.")
    imp = explain.builtin_importance(model, fit["feature_names"])
    if imp is not None:
        show(C.hbar(imp.head(15), "importance", "feature", C.VIOLET, xtitle="Importance"))
        top = imp.head(3)["feature"].tolist()
        insight(f"The three most important features are {top[0]}, {top[1]} and {top[2]}.")


# ------------------------------------------------------------- explainability
def _global(fit, P) -> None:
    model = primary_model(fit)
    section("Global importance", "Which features matter most across all students.", first=True)
    method = st.radio("Method", ["Built-in importance", "SHAP", "Permutation"], horizontal=True, key="gx_method")
    scope = None
    if method == "SHAP":
        scope = st.radio("Scope", ["All outcomes", "Dropout only"], horizontal=True, key="gx_scope")
    key = (method, scope)
    cache = st.session_state.explain_cache
    if key not in cache:
        with st.spinner(f"Computing {method.lower()} importance..."):
            try:
                if method == "Built-in importance":
                    cache[key] = explain.builtin_importance(model, fit["feature_names"])
                elif method == "SHAP":
                    cache[key] = explain.shap_global(get_explainer(fit), model, P.X_train,
                                                     "Dropout" if scope == "Dropout only" else None)
                else:
                    cache[key] = explain.permutation(model, P.X_test, P.y_test, fit["feature_names"])
            except Exception as exc:  # noqa: BLE001
                note(f"<b>Could not compute {esc(method)}.</b> {esc(str(exc))} Try another method.", "bad")
                return
    df = cache[key]
    if df is None:
        note("This model does not expose built-in importance. Use SHAP or Permutation.")
        return
    top = df.head(15)
    show(C.hbar(top, "importance", "feature", C.VIOLET, xtitle={"Built-in importance": "Importance",
                "SHAP": "Mean |SHAP| (probability)", "Permutation": "Drop in accuracy when shuffled"}[method],
                err="std" if "std" in df.columns else None))
    lead = ", ".join(top["feature"].head(5))
    what = {"Built-in importance": "How much each feature reduces impurity across the forest's trees.",
            "SHAP": "Average size of each feature's push on the predicted probability, on a 100-student sample.",
            "Permutation": "How much accuracy falls on the test set when a feature's values are shuffled."}[method]
    insight(f"Top five: {lead}. {what}")


def _local(fit, P) -> None:
    model = primary_model(fit)
    names = ml.class_names(model)
    section("Explain a student", "Why the model reached its result for one held-out student.", first=True)
    idx = st.selectbox("Student from the test set", range(len(P.X_test)), format_func=lambda i: f"Student {i + 1}", key="lx_idx")
    row = P.X_test.iloc[[idx]]
    proba = model.predict_proba(row)[0]
    pred = names[int(np.argmax(proba))]
    actual = REVERSE_TARGET.get(int(P.y_test.iloc[idx]), "Unknown")
    stat_strip([(pred, "Predicted outcome"), (f"{proba.max() * 100:.1f}%", "Confidence"), (actual, "Recorded outcome")])
    outcome = st.radio("Explain the probability of", names, index=names.index(pred), horizontal=True, key=f"lx_out_{idx}")
    method = st.radio("Method", ["SHAP", "LIME"], horizontal=True, key="lx_method")
    try:
        if method == "SHAP":
            contrib, base = explain.shap_local(get_explainer(fit), model, row, outcome)
            show(C.waterfall(contrib, base, outcome))
            insight(f"Starting from a base rate of {base * 100:.1f}%, the features add up to {(base + contrib['shap'].sum()) * 100:.1f}% "
                    f"for {outcome}. Bars are percentage points.")
            top = contrib.head(5)
            for r in top.itertuples():
                st.markdown(f"**{r.feature}** (value {r.value:g}) {'raises' if r.shap > 0 else 'lowers'} the {outcome.lower()} "
                            f"probability by {abs(r.shap) * 100:.1f} points.")
        else:
            with st.spinner("Generating LIME explanation..."):
                items = explain.lime_local(model, P.X_train, row, outcome)
            df = pd.DataFrame(items, columns=["rule", "weight"]).sort_values("weight", key=np.abs, ascending=False)
            show(C.hbar(df, "weight", "rule", signed=True, xtitle=f"Local weight towards {outcome}"))
            insight("LIME fits a simple local model around this student. Positive weights push towards the selected outcome, negative weights away from it.")
    except Exception as exc:  # noqa: BLE001
        note(f"<b>Could not generate the explanation.</b> {esc(str(exc))}", "bad")


def _impact(fit, P) -> None:
    model = primary_model(fit)
    section("Feature impact", "How the predicted probabilities respond as one feature changes.", first=True)
    feat = st.selectbox("Feature", fit["feature_names"], key="fi_feat")
    curves = explain.partial_dependence(model, P.X_test, feat)
    mean_x = float(P.X[feat].mean())
    show(C.pdp(curves, feat, mean_x))
    if "Dropout" in curves.columns and len(curves) > 1:
        d = curves["Dropout"] * 100
        lo_i, hi_i = int(d.idxmin()), int(d.idxmax())
        insight(f"Averaged over {min(100, len(P.X_test))} held-out students, predicted dropout probability ranges from "
                f"{d.min():.0f}% (at {curves[feat][lo_i]:.3g}) to {d.max():.0f}% (at {curves[feat][hi_i]:.3g}). "
                "This shows how the model responds, not a causal effect.")
    s = P.X[feat]
    stat_strip([(f"{s.min():.4g}", "Min"), (f"{s.max():.4g}", "Max"), (f"{s.mean():.3g}", "Mean"), (f"{s.std():.3g}", "Std")])


def render(ctx) -> None:
    page_head("Model intelligence", 'Model<br><span class="dim">intelligence.</span>',
              "Train the prediction engine, check how it performs on students it has not seen, and see how it reads student-risk signals.")
    if ctx["problems"]:
        for p in ctx["problems"]:
            note(esc(p), "bad")
        return
    P = ctx["prepared"]
    if ctx["loaded"].kind == "placeholder":
        note("<b>Placeholder data.</b> With 3 rows the metrics below would be meaningless. Load your CSV on the Overview page before training.", "bad")

    section("Model training", "Prepare the prediction engine.", first=True)
    if st.button("Retrain models" if st.session_state.fit else "Train models", key="train_btn", type="primary"):
        _train(P)
    _ledger(P, st.session_state.fit, st.session_state.get("train_error"))
    if st.session_state.get("train_error"):
        note(f"<b>Training failed.</b> {esc(st.session_state.train_error)}", "bad")

    fit = st.session_state.fit
    if not fit:
        return
    section("Model results", "Random Forest is the prediction engine. The comparison models are trained on the same split.")
    _rows(fit)

    view = st.segmented_control("View", ["Evaluation", "Explainability"], default="Evaluation", required=True,
                                key="models_view", label_visibility="collapsed")
    if view == "Evaluation":
        _evaluation(fit)
    else:
        sub = st.segmented_control("Analysis", ["Global importance", "Explain a student", "Feature impact"],
                                   default="Global importance", required=True, key="models_ex", label_visibility="collapsed")
        {"Global importance": _global, "Explain a student": _local, "Feature impact": _impact}[sub](fit, P)
