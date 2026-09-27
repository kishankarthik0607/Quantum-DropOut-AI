"""Insights: what the data (and, once trained, the model) says. Everything is computed."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import explain, insights as I, ml, state
from ..config import RISK_HIGH_MIN, RISK_LOW_MAX
from ..ui import charts as C
from ..ui.components import esc, figure_title, html, insight, note, page_head, section, show
from .explore import _cols2, _lab, _rate_chart
from ._common import primary_model


def _drivers(df, fit) -> None:
    section("Dropout drivers", "Which measured features move most with dropout.", first=True)
    a, b = _cols2()
    with a:
        figure_title("Association with dropout", "Correlation with a 0/1 dropout indicator. Code-like columns are excluded.")
        corr = I.dropout_correlates(df).head(12)
        if len(corr):
            show(C.hbar(corr, "corr", "feature", signed=True, xtitle="Correlation", fmt="+.2f"))
            top = corr.iloc[0]
            insight(f"{top['feature']} has the strongest association ({top['corr']:+.2f}). "
                    "A negative value means higher values go with less dropout. Correlation is not causation.")
    with b:
        figure_title("What the trained model relies on", "Built-in Random Forest importance.")
        if fit:
            imp = explain.builtin_importance(primary_model(fit), fit["feature_names"]).head(12)
            show(C.hbar(imp, "importance", "feature", C.VIOLET, xtitle="Importance"))
            insight(f"Top three: {', '.join(imp['feature'].head(3))}.")
        else:
            note("Train the models on the Models page to see which features the prediction engine relies on.")
            st.button("Go to Models", key="ins_go_models", on_click=state.goto, args=("Models",))


def _risk_distribution(P, fit) -> None:
    section("Risk distribution", "How the trained model spreads held-out students across risk levels.")
    if not fit:
        note("Available after the models are trained. It uses the students held out from training, so it is not inflated by memorisation.")
        return
    model = primary_model(fit)
    names = ml.class_names(model)
    if "Dropout" not in names:
        return
    p = model.predict_proba(P.X_test)[:, names.index("Dropout")]
    actual = (P.y_test.to_numpy() == 0)
    show(C.prob_hist(p, RISK_LOW_MAX, RISK_HIGH_MIN))
    bands = [("Low", p < RISK_LOW_MAX), ("Medium", (p >= RISK_LOW_MAX) & (p < RISK_HIGH_MIN)), ("High", p >= RISK_HIGH_MIN)]
    rows = []
    for name, m in bands:
        n = int(m.sum())
        rows.append({"Risk level": name, "Students": n, "Share of test set": f"{n / len(p) * 100:.1f}%",
                     "Actually dropped out": f"{actual[m].mean() * 100:.1f}%" if n else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    hi = bands[2][1]
    if hi.sum() >= I.MIN_GROUP_N:
        insight(f"{int(hi.sum())} of {len(p)} held-out students ({hi.mean() * 100:.0f}%) fall in the High band; "
                f"{actual[hi].mean() * 100:.0f}% of them did drop out, against {actual.mean() * 100:.0f}% of all held-out students.")


def render(ctx) -> None:
    page_head("Insights", 'What the data<br><span class="dim">is telling us.</span>',
              "Patterns measured from the loaded dataset, plus how the trained model distributes risk. Associations, not causes.")
    if ctx["problems"]:
        for p in ctx["problems"]:
            note(esc(p), "bad")
        return
    df, P, fit = ctx["loaded"].df, ctx["prepared"], st.session_state.fit
    if ctx["loaded"].kind == "placeholder":
        note("<b>Placeholder data.</b> Findings need enough students per group, so most are withheld with 3 rows. Load your CSV on the Overview page.")

    found = I.headline_findings(df)
    section("Key findings", "Each comparison needs at least 30 students in both groups.", first=True)
    if found:
        html("".join(f'<div class="qd-find"><div class="qd-find-fig">{esc(f.figure)}</div>'
                     f'<div><h3>{esc(f.headline)}</h3><p>{esc(f.detail)}</p></div></div>' for f in found))
    else:
        note("No finding can be stated yet: the needed columns are missing or the groups are too small.")

    _drivers(df, fit)

    section("Academic trends", "Units passed and dropout.")
    a, b = _cols2()
    for ax, col, t in ((a, "Curricular units 1st sem (approved)", "First semester"),
                       (b, "Curricular units 2nd sem (approved)", "Second semester")):
        bands = I.unit_bands(df, col)
        if bands is not None:
            with ax:
                _rate_chart(df, bands, f"{t}: units approved", "Dropout rate by number of units approved.", "Units approved")

    section("Financial indicators", "Fees, debt and scholarships.")
    cols = st.columns(3, gap="large")
    for ax, col in zip(cols, ("Tuition fees up to date", "Debtor", "Scholarship holder")):
        if col in df.columns:
            with ax:
                _rate_chart(df, _lab(df, col), col, "Dropout rate.", "", height=270)

    section("Enrollment patterns", "Age, schedule and application order.")
    cols = st.columns(3, gap="large")
    ab = I.age_bands(df)
    for ax, (ser, title, x) in zip(cols, (
            (ab, "Age band", "Age"),
            (_lab(df, "Daytime/evening attendance") if "Daytime/evening attendance" in df else None, "Schedule", "Schedule"),
            (df["Application order"].astype(int) if "Application order" in df else None, "Application order", "Order"))):
        if ser is not None:
            with ax:
                _rate_chart(df, ser, title, "Dropout rate.", x, height=270)

    _risk_distribution(P, fit)
