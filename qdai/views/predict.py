"""Predict: assess one student.

Inputs   : every column the model was trained on is exposed (grouped in four sections; unknown
           columns appear under "Additional signals"). Ranges/defaults come from the dataset.
Output   : the real multiclass prediction, P(Dropout) as the risk score, SHAP contributions
           towards Dropout, and rule-based interventions built from those contributions.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import streamlit as st

from .. import explain, insights as I, interventions, ml
from ..config import OUTCOME_COLORS, REVERSE_TARGET, RISK_HIGH_MIN, RISK_LOW_MAX
from ..features import SECTIONS, code_label, spec_for
from ..ui.components import esc, html, insight, note, page_head, risk_band, section, stretch
from ._common import get_explainer, primary_model, require_fit


# ------------------------------------------------------------------ form model
def _kind(spec, s: pd.Series) -> str:
    """Widget kind actually used for a column (registry kind, downgraded if the data disagrees)."""
    if spec.kind == "binary" and not set(s.dropna().unique()) <= {0, 1}:
        return "int"
    if spec.kind in ("int", "code") and not all(float(v).is_integer() for v in s.dropna().unique()):
        return "float"
    return spec.kind


def _bounds(s: pd.Series, kind: str):
    if kind == "float":
        lo, hi = math.floor(float(s.min()) * 10) / 10, math.ceil(float(s.max()) * 10) / 10
        return lo, (hi if hi > lo else lo + 0.1)
    lo, hi = int(s.min()), int(s.max())
    return lo, (hi if hi > lo else lo + 1)


def _default(s: pd.Series, kind: str):
    if kind in ("binary", "code"):
        return int(s.mode().iloc[0])
    lo, hi = _bounds(s, kind)
    med = float(s.median())
    return min(max(round(med, 1) if kind == "float" else int(round(med)), lo), hi)


def _widget(spec, s: pd.Series) -> None:
    k, kind = f"f_{spec.name}", _kind(spec, s)
    st.session_state.setdefault(k, _default(s, kind))
    if kind == "binary":
        labels = dict(spec.options or ((0, "No"), (1, "Yes")))
        st.radio(spec.label, list(labels), format_func=lambda v: labels[v], horizontal=True, key=k, help=spec.help)
    elif kind == "code":
        opts = sorted(int(v) for v in s.dropna().unique())
        st.selectbox(spec.label, opts, format_func=lambda v, n=spec.name: code_label(n, v), key=k, help=spec.help)
    else:
        lo, hi = _bounds(s, kind)
        st.slider(spec.label, lo, hi, step=(1 if kind == "int" else (0.5 if hi - lo > 50 else 0.1)), key=k, help=spec.help)


def _fmt_value(name: str, s: pd.Series, v: float) -> str:
    spec = spec_for(name, s)
    kind = _kind(spec, s)
    if kind == "binary":
        return dict(spec.options or ((0, "No"), (1, "Yes"))).get(int(round(v)), f"{v:g}")
    if kind == "code":
        return code_label(name, v)
    return f"{v:.1f}" if kind == "float" else f"{int(round(v))}"


# ------------------------------------------------------------------- callbacks
def _load_sample(kind: str) -> None:
    P = st.session_state.get("_P")
    if P is None:
        return
    y = P.y_test
    if kind == "dropout":
        pool = y[y == 0].index
    elif kind == "graduate":
        pool = y[y == 1].index
    else:
        pool = y.index
    if len(pool) == 0:
        return
    idx = random.choice(list(pool))
    row = P.X_test.loc[idx]
    vals = {}
    for col in P.X.columns:
        spec = spec_for(col, P.X[col])
        kd = _kind(spec, P.X[col])
        v = float(row[col])
        lo, hi = _bounds(P.X[col], kd) if kd not in ("binary", "code") else (v, v)
        st.session_state[f"f_{col}"] = int(round(v)) if kd in ("binary", "code", "int") else min(max(v, lo), hi)
        vals[col] = float(st.session_state[f"f_{col}"])
    st.session_state.sample = {"values": vals, "outcome": REVERSE_TARGET.get(int(y.loc[idx]), "Unknown")}
    st.session_state.pred = None


def _reset_form() -> None:
    for k in [k for k in st.session_state if str(k).startswith("f_")]:
        del st.session_state[k]
    st.session_state.sample = None
    st.session_state.pred = None


# --------------------------------------------------------------------- analysis
def _analyze(fit, P, df, values: dict) -> None:
    model = primary_model(fit)
    cols = list(fit["feature_names"])
    frame, probs = ml.predict_row(model, cols, values, P.X.median())
    pred = {"values": values, "probs": probs, "contrib": None, "base": None, "plan": None, "error": None,
            "recorded": None}
    s = st.session_state.sample
    if s and all(abs(values[c] - s["values"].get(c, np.nan)) < 1e-6 for c in values):
        pred["recorded"] = s["outcome"]
    if "Dropout" in probs:
        try:
            contrib, base = explain.shap_local(get_explainer(fit), model, frame, "Dropout")
            pred["contrib"], pred["base"] = contrib, base
            pred["plan"] = interventions.build_plan(contrib[["feature", "value", "shap"]], values,
                                                    probs["Dropout"], I.graduate_median(df))
        except Exception as exc:  # noqa: BLE001
            pred["error"] = str(exc)
    st.session_state.pred = pred


# ---------------------------------------------------------------------- results
def _meter(p: float) -> str:
    lo, hi = RISK_LOW_MAX, RISK_HIGH_MIN
    widths = (lo * 100, (hi - lo) * 100, 100 - hi * 100)
    colors = ("#35D0C4", "#F0A24E", "#F0605D")
    active = 0 if p < lo else (1 if p < hi else 2)
    track = "".join(f'<span class="{"on" if i == active else ""}" style="flex:{w:.1f} 0 0;background:{c}"></span>'
                    for i, (w, c) in enumerate(zip(widths, colors)))
    scale = "".join(f'<span style="flex:{w:.1f} 0 0">{t}</span>' for w, t in zip(
        widths, (f"Low · under {lo * 100:.0f}%", f"Medium · {lo * 100:.0f}–{hi * 100:.0f}%", f"High · {hi * 100:.0f}%+")))
    return (f'<div class="qd-meter"><div class="qd-meter-mark" style="left:{p * 100:.1f}%"><b>{p * 100:.0f}%</b></div>'
            f'<div class="qd-meter-track">{track}</div><div class="qd-meter-scale">{scale}</div></div>')


def _result(pred, P) -> None:
    probs = pred["probs"]
    top = max(probs, key=probs.get)
    oc = OUTCOME_COLORS.get(top, "#F2F4F8")
    risk = ""
    if "Dropout" in probs:
        pd_ = probs["Dropout"]
        band, color = risk_band(pd_)
        risk = (f'<div class="qd-risk"><div><div class="qd-risk-num" style="color:{color}">{pd_ * 100:.0f}%</div>'
                f'<div class="qd-risk-k">Dropout risk score<br><span class="qd-small">Predicted probability of Dropout</span></div></div>'
                f'{_meter(pd_)}</div>')
        badge = f'<span class="qd-badge" style="color:{color}">{band}</span>'
    else:
        badge = ""
    html(f'<div class="qd-result"><div class="qd-outcome-k">Predicted outcome</div>'
         f'<div class="qd-outcome-row"><div class="qd-outcome" style="color:{oc}">{esc(top)}</div>{badge}</div>'
         f'<div class="qd-muted">The most likely of the three outcomes, at {probs[top] * 100:.1f}% probability.</div>{risk}</div>')
    st.caption(f"Risk bands are fixed presentation thresholds (under {RISK_LOW_MAX * 100:.0f}%, "
               f"{RISK_LOW_MAX * 100:.0f}–{RISK_HIGH_MIN * 100:.0f}%, {RISK_HIGH_MIN * 100:.0f}% and above), not calibrated cut-offs.")
    if pred["recorded"]:
        agree = pred["recorded"] == top
        note(f"<b>Real record from the test set.</b> This student's recorded outcome is <b>{esc(pred['recorded'])}</b>; "
             f"the model {'agrees' if agree else 'predicted differently'}.", "good" if agree else "")

    section("Probability by outcome", "The model's full output for this student.")
    rows = "".join(
        f'<div class="qd-prob"><span>{k}</span><div class="qd-prob-track"><i style="width:{probs[k] * 100:.1f}%;'
        f'background:{OUTCOME_COLORS[k]}"></i></div><span class="qd-prob-v">{probs[k] * 100:.1f}%</span></div>'
        for k in ("Dropout", "Enrolled", "Graduate") if k in probs)
    html(rows)


def _why(pred, P, fit) -> None:
    section("Why this result", "Each feature's push on the probability of Dropout, from the model's SHAP explanation.")
    c = pred["contrib"]
    if c is None:
        note(f"<b>The per-student explanation is unavailable.</b> {esc(pred['error'] or '')} "
             "Global feature importance is shown instead.", "")
        imp = explain.builtin_importance(primary_model(fit), fit["feature_names"]).head(6)
        html("".join(f'<div class="qd-prob"><span style="grid-column:1/3">{esc(r.feature)}</span>'
                     f'<span class="qd-prob-v">{r.importance:.3f}</span></div>' for r in imp.itertuples()))
        return
    top = c.head(8)
    scale = max(float(top["abs"].max()), 1e-9)
    rows = ""
    for i, r in enumerate(top.itertuples(), 1):
        w = abs(r.shap) / scale * 50
        left, color = (50, "#F0605D") if r.shap > 0 else (50 - w, "#35D0C4")
        rows += (f'<div class="qd-why"><span class="qd-why-n">{i}</span>'
                 f'<div class="qd-why-f">{esc(r.feature)}<small>{esc(_fmt_value(r.feature, P.X[r.feature], r.value))}</small></div>'
                 f'<div class="qd-why-track"><i style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></i></div>'
                 f'<div class="qd-why-v">{r.shap * 100:+.1f} pp<small>{"raises risk" if r.shap > 0 else "lowers risk"}</small></div></div>')
    html('<div class="qd-axis"><span></span><span></span><div><span>Lowers risk</span><span>Raises risk</span></div><span></span></div>' + rows)
    total = pred["base"] + float(c["shap"].sum())
    insight(f"The model starts from a base rate of {pred['base'] * 100:.1f}% Dropout. All {len(c)} features together move that to "
            f"{total * 100:.1f}%. Values are percentage points (pp); the eight largest are shown.")


def _next_steps(pred, P) -> None:
    plan = pred["plan"]
    section("What should happen next?", "Recommended institutional interventions, derived from the signals above.")
    if plan is None:
        note("Interventions need the per-student explanation, which is unavailable for this prediction.")
        return
    st.markdown(plan.headline)
    items = ""
    for it in plan.items:
        ev = "".join(f"<li>{esc(e)}</li>" for e in it.evidence)
        dr = "".join(f'<div class="qd-int-d"><span>{esc(f)}</span><span>{pp:+.1f} pp</span></div>' for f, pp in it.drivers)
        items += (f'<div class="qd-int"><div class="qd-int-n">{it.priority:02d}</div>'
                  f'<div><h3>{esc(it.title)}</h3><p>{esc(it.action)}</p><ul>{ev}</ul></div>'
                  f'<div><div class="qd-int-k">Model signals</div>{dr or "<span class=qd-small>Rule-based flag</span>"}</div></div>')
    html(items)
    if plan.context_note:
        note(esc(plan.context_note))
    if plan.protective:
        st.markdown("**Lowering the estimate for this student:** " + ", ".join(
            f"{f} = {_fmt_value(f, P.X[f], v)} ({pp:+.1f} pp)" for f, pp, v in plan.protective))
    st.caption("These are rule-based suggestions built from the model's signals and the values entered. "
               "They support an advisor's judgement and are not clinical or psychological advice.")


# ------------------------------------------------------------------------- page
def render(ctx) -> None:
    page_head("Predict", 'Assess<br><span class="dim">student risk.</span>',
              "Enter a student's academic and contextual information to estimate dropout risk, see why, and get suggested next steps.")
    if ctx["problems"]:
        for p in ctx["problems"]:
            note(esc(p), "bad")
        return
    fit = require_fit("Prediction")
    if fit is None:
        return
    P, df = ctx["prepared"], ctx["loaded"].df

    a, b, c, _ = st.columns([1, 1, 1, 1.4])
    a.button("Load a real student", key="s_rand", on_click=_load_sample, args=("random",), **stretch())
    b.button("Example: dropout", key="s_drop", on_click=_load_sample, args=("dropout",), **stretch())
    c.button("Example: graduate", key="s_grad", on_click=_load_sample, args=("graduate",), **stretch())
    st.caption("These fill the form with a real record from the held-out test set, so you can compare the model with what actually happened. Values stay editable.")

    known = {s.name for s in ctx_specs()}
    with st.form("predict_form", border=False):
        for n, (key, title, sub) in enumerate(SECTIONS, 1):
            cols_in = [c for c in P.X.columns if spec_for(c, P.X[c]).section == key]
            if not cols_in:
                continue
            html(f'<div class="qd-fsec"><div class="qd-fsec-n">{n:02d}</div><div class="qd-fsec-t">{esc(title)}</div>'
                 f'<div class="qd-fsec-s">{esc(sub)}</div></div>')
            grid = st.columns(3, gap="large")
            for i, col in enumerate(cols_in):
                with grid[i % 3]:
                    _widget(spec_for(col, P.X[col]), P.X[col])
        extra = [c for c in P.X.columns if c not in known]
        if extra:
            with st.expander(f"Additional signals ({len(extra)})"):
                grid = st.columns(3, gap="large")
                for i, col in enumerate(extra):
                    with grid[i % 3]:
                        _widget(spec_for(col, P.X[col]), P.X[col])
        st.write("")
        go = st.form_submit_button("Analyze student", type="primary")
    st.button("Reset form", key="reset_form", on_click=_reset_form)

    if go:
        with st.spinner("Analyzing..."):
            _analyze(fit, P, df, {c: float(st.session_state[f"f_{c}"]) for c in P.X.columns})
    pred = st.session_state.pred
    if pred:
        _result(pred, P)
        _why(pred, P, fit)
        _next_steps(pred, P)


def ctx_specs():
    from ..features import SPECS
    return SPECS
