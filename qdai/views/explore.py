"""Explore: the original Data Overview + EDA, reorganised as a data story.

Every chart is drawn from the loaded dataframe and every 'insight' sentence is
computed (see qdai/insights.py) - the generic hard-coded sentences of the
original app were replaced by measured ones.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import data, insights as I
from ..features import code_label
from ..ui import charts as C
from ..ui.components import esc, figure_title, insight, note, page_head, section, show, stat_strip, stretch

CHAPTERS = ["Outcomes", "Student profile", "Academic performance", "Enrollment patterns",
            "Financial indicators", "Relationships", "Data quality", "Feature lab"]

# PRESERVED from the original EDA: readable labels for 0/1 columns.
LABELS = {
    "Gender": {0: "Female", 1: "Male"},
    "Scholarship holder": {0: "No scholarship", 1: "Has scholarship"},
    "Displaced": {0: "Not displaced", 1: "Displaced"},
    "International": {0: "Domestic", 1: "International"},
    "Debtor": {0: "No debt", 1: "Has debt"},
    "Tuition fees up to date": {0: "Fees not up to date", 1: "Fees up to date"},
    "Educational special needs": {0: "No special needs", 1: "Has special needs"},
    "Daytime/evening attendance": {0: "Evening", 1: "Daytime"},
}


def _lab(df: pd.DataFrame, col: str) -> pd.Series:
    m = LABELS.get(col)
    return df[col].map(m).fillna(df[col]) if m else df[col]


def _base(df: pd.DataFrame) -> float:
    return float(I.is_dropout(df).mean() * 100)


def _rate_chart(df, series, title, sub, xtitle="", horizontal=False, height=300):
    t = I.rate_by(df, series)
    figure_title(title, sub)
    show(C.rate_bar(t, _base(df), xtitle, height, horizontal))
    insight(I.extremes_sentence(t, title))
    if not t["reliable"].all():
        st.caption(f"Grey bars have fewer than {I.MIN_GROUP_N} students and are excluded from the comparison sentence.")


def _cols2():
    return st.columns(2, gap="large")


# ------------------------------------------------------------------ chapters
def outcomes(df):
    section("Outcome distribution", "How the students in this dataset ended up.", first=True)
    s = data.summarize(df)
    show(C.outcome_bar(s["counts"]))
    parts = ", ".join(f"{v:,} {k.lower()}" for k, v in sorted(s["counts"].items(), key=lambda kv: -kv[1]))
    insight(f"{s['students']:,} students: {parts}. The observed dropout rate is {s['dropout_rate']:.1f}%.")

    section("Outcomes by factor", "Share of each outcome inside every group.")
    opts = [c for c in LABELS if c in df.columns]
    pick = st.multiselect("Factors to compare", opts, default=opts[:2], key="ex_factors")
    cols = _cols2()
    for i, col in enumerate(pick):
        with cols[i % 2]:
            figure_title(f"Outcome by {col.lower()}")
            ser = _lab(df, col)
            show(C.crosstab_pct(df, ser, col))
            insight(I.extremes_sentence(I.rate_by(df, ser), col))


def profile(df):
    section("Student profile", "Age, gender and background.", first=True)
    a, b = _cols2()
    if I.has(df, "Age at enrollment"):
        with a:
            figure_title("Age at enrollment", "Dotted line marks the mean age.")
            show(C.hist_single(df["Age at enrollment"], "Age (years)", kde=False))
            insight(f"Average age at enrollment is {df['Age at enrollment'].mean():.1f} years; "
                    f"the middle student enrolled at {df['Age at enrollment'].median():.0f}.")
        with b:
            _rate_chart(df, I.age_bands(df), "Dropout rate by age band", "Share of each band that dropped out.", "Age band")
    cols = _cols2()
    i = 0
    for col in ("Gender", "International", "Displaced", "Educational special needs"):
        if col in df.columns:
            with cols[i % 2]:
                _rate_chart(df, _lab(df, col), f"Dropout rate by {col.lower()}", "", col, height=260)
            i += 1


def academic(df):
    section("Academic performance", "Grades and units passed, by outcome.", first=True)
    parts = []
    tgt = df["Target"].astype(str)
    g, d = df[tgt == "Graduate"], df[tgt == "Dropout"]
    for col, label in (("Admission grade", "Admission grade, graduates"),
                       ("Curricular units 1st sem (grade)", "1st-semester grade, graduates")):
        if I.has(df, col) and len(g) and len(d):
            parts.append((f"{g[col].mean():.1f}", f"{label} (dropouts: {d[col].mean():.1f})"))
    if I.has(df, "Curricular units 1st sem (grade)"):
        hp = df[df["Curricular units 1st sem (grade)"] > 15]
        if len(hp) >= I.MIN_GROUP_N:
            parts.append((f"{(hp['Target'].astype(str) == 'Graduate').mean() * 100:.0f}%",
                          f"Graduated, of {len(hp):,} students with a 1st-semester grade above 15"))
    if parts:
        stat_strip(parts)
    cols = _cols2()
    i = 0
    for col, title in (("Admission grade", "Admission grade"),
                       ("Curricular units 1st sem (grade)", "First-semester grade"),
                       ("Curricular units 2nd sem (grade)", "Second-semester grade")):
        if I.has(df, col):
            with cols[i % 2]:
                figure_title(f"{title} by outcome", "Dotted lines mark each outcome's mean.")
                show(C.hist_by_outcome(df, col, 30))
                insight(I.mean_gap_sentence(df, col, title))
            i += 1

    section("Academic metric explorer", "Distribution of any grade or approved-units column, by outcome.")
    numeric = set(df.select_dtypes("number").columns)
    options = [c for c in df.columns if ("grade" in c.lower() or "approved" in c.lower()) and c in numeric]
    default = [c for c in ("Curricular units 1st sem (grade)", "Curricular units 1st sem (approved)") if c in options]
    pick = st.multiselect("Metrics", options, default=default, key="ex_metrics")
    cols = _cols2()
    for i, col in enumerate(pick):
        with cols[i % 2]:
            figure_title(col)
            show(C.box_by_outcome(df, col))
            insight(I.mean_gap_sentence(df, col, col))


def enrollment(df):
    section("Enrollment patterns", "How and when students entered.", first=True)
    a, b = _cols2()
    if I.has(df, "Daytime/evening attendance"):
        with a:
            _rate_chart(df, _lab(df, "Daytime/evening attendance"), "Dropout rate by schedule", "", "Attendance schedule")
    if I.has(df, "Application order"):
        with b:
            _rate_chart(df, df["Application order"].astype(int), "Dropout rate by application order",
                        "0 means the programme was the first choice.", "Application order")
    for col, title, sub in (("Course", "Dropout rate by course", "The twelve largest courses."),
                            ("Application mode", "Dropout rate by application mode",
                             "Ten most common modes; codes follow the dataset's own scheme.")):
        if I.has(df, col):
            figure_title(title, sub)
            s = df[col].map(lambda v, c=col: code_label(c, v))
            t = I.rate_by(df, s).sort_values("n", ascending=False).head(12 if col == "Course" else 10)
            t = t.sort_values("rate", ascending=False)
            show(C.rate_bar(t, _base(df), "", 420, horizontal=True))
            insight(I.extremes_sentence(t, col))


def financial(df):
    section("Financial indicators", "Fees, debt and scholarships.", first=True)
    cols = st.columns(3, gap="large")
    for ax, col in zip(cols, ("Tuition fees up to date", "Debtor", "Scholarship holder")):
        if col in df.columns:
            with ax:
                _rate_chart(df, _lab(df, col), col, "Dropout rate.", "", height=280)
    flags = I.risk_flags(df)
    if len(flags):
        section("Risk flags", "Simple, transparent flags and the dropout rate inside each group.")
        t = pd.DataFrame({"group": flags["flag"], "n": flags["students"],
                          "dropouts": (flags["students"] * flags["dropout_rate"] / 100).round(),
                          "rate": flags["dropout_rate"], "reliable": flags["students"] >= I.MIN_GROUP_N}
                         ).sort_values("rate", ascending=False)
        show(C.rate_bar(t, _base(df), "", 340, horizontal=True))
        top = t.iloc[0]
        insight(f"{top['group']}: {int(top['n']):,} students, of whom {top['rate']:.0f}% dropped out, "
                f"against {_base(df):.0f}% overall.")


def relationships(df):
    num = data.numeric_feature_cols(df)
    section("Distributions", "Histogram and density for up to four numeric columns.", first=True)
    default = [c for c in ("Age at enrollment", "Admission grade", "Curricular units 1st sem (grade)") if c in num][:3]
    pick = st.multiselect("Columns", num, default=default, max_selections=4, key="ex_dist")
    cols = _cols2()
    for i, col in enumerate(pick):
        with cols[i % 2]:
            figure_title(col)
            show(C.hist_single(df[col], col))
            insight(f"Mean {df[col].mean():.2f}, median {df[col].median():.2f}, standard deviation {df[col].std():.2f}.")
    section("Correlation", "How selected columns move together.")
    default = [c for c in ("Age at enrollment", "Admission grade", "Curricular units 1st sem (grade)",
                           "Curricular units 2nd sem (grade)") if c in num]
    pick = st.multiselect("Columns to correlate", num, default=default or num[:4], key="ex_corr")
    if len(pick) > 1:
        corr = df[pick].corr()
        a, b = st.columns([2, 1], gap="large")
        with a:
            show(C.corr_heatmap(corr))
        with b:
            pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                     for i in range(len(corr)) for j in range(i + 1, len(corr)) if abs(corr.iloc[i, j]) > 0.5]
            pairs.sort(key=lambda x: -abs(x[2]))
            figure_title("Strong pairs", "Absolute correlation above 0.5.")
            if pairs:
                for f1, f2, v in pairs[:6]:
                    st.markdown(f"**{f1}** and **{f2}**: {'positive' if v > 0 else 'negative'} ({v:.2f})")
            else:
                st.caption("No pair in this selection has an absolute correlation above 0.5.")
    else:
        st.caption("Select at least two columns.")


def quality(df):
    section("Data quality", "Completeness, types and structure.", first=True)
    s = data.summarize(df)
    stat_strip([(f"{s['students']:,}", "Rows"), (f"{s['columns']}", "Columns"),
                (f"{df.memory_usage().sum() / 1024 ** 2:.2f} MB", "Memory"),
                ("0%" if s["missing_pct"] == 0 else f"{s['missing_pct']:.2f}%", "Missing data")])
    a, b = _cols2()
    with a:
        figure_title("Missing values by column")
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False).rename_axis("feature").reset_index(name="missing")
        if len(miss):
            show(C.hbar(miss, "missing", "feature", C.CORAL, fmt=",.0f"))
            insight("Missing numeric values are filled with the column median before training (unchanged from the original pipeline).")
        else:
            note("<b>No missing values.</b> The dataset is complete, so no imputation is applied.", "good")
    with b:
        figure_title("Column types")
        dt = df.dtypes.astype(str).value_counts()
        show(C.simple_bar(dt.index, dt.values, 260, "Data type", "Columns"))
    demo = [c for c in ("Gender", "Age at enrollment", "Marital status", "Nacionality") if c in df.columns]
    acad = [c for c in df.columns if "grade" in c.lower() or "units" in c.lower()]
    eco = [c for c in ("Unemployment rate", "Inflation rate", "GDP") if c in df.columns]
    other = df.shape[1] - len(set(demo + acad + eco))
    insight(f"Feature families: {len(demo)} demographic, {len(acad)} academic, {len(eco)} economic, {other} other columns.")

    section("Columns", "Type, completeness and uniqueness of every column.")
    info = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Non-null": df.count().values,
                         "Null": df.isnull().sum().values, "Unique": df.nunique().values})
    st.dataframe(info, hide_index=True, **stretch())
    num = df.select_dtypes(include=["number"])
    if len(num.columns):
        with st.expander("Statistical summary of numeric columns"):
            st.dataframe(num.describe().T, **stretch())

    section("Sample records", "The first ten rows, outcome highlighted.")
    tint = {"Dropout": "background-color: rgba(240,96,93,.22)", "Graduate": "background-color: rgba(53,208,196,.20)",
            "Enrolled": "background-color: rgba(156,140,255,.22)"}
    sty = df.head(10).style
    fn = getattr(sty, "map", None) or sty.applymap
    st.dataframe(fn(lambda v: tint.get(v, ""), subset=["Target"]), **stretch())


def lab(df):
    section("Feature lab", "Pick any feature and see how dropout varies across its range.", first=True)
    feats = [c for c in df.columns if c not in ("Target", "id")]
    col = st.selectbox("Feature", feats, key="ex_lab")
    y = I.is_dropout(df).astype(float)
    if pd.api.types.is_numeric_dtype(df[col]):
        r = df[col].corr(y)
        stat_strip([(f"{r:+.2f}", "Correlation with dropout (0/1)"), (f"{df[col].mean():.2f}", "Mean"),
                    (f"{df[col].min():.4g} to {df[col].max():.4g}", "Range")])
        a, b = _cols2()
        with a:
            figure_title(f"{col} by outcome")
            show(C.hist_by_outcome(df, col, 30))
        with b:
            figure_title("Dropout rate across the range", "Five equal-width bins.")
            try:
                t = I.rate_by(df, I.numeric_bins(df[col]))
                show(C.rate_bar(t, _base(df), col, 320))
                insight(I.extremes_sentence(t, col))
            except ValueError as exc:
                st.caption(f"Cannot bin this feature: {exc}")
    else:
        t = I.rate_by(df, df[col])
        figure_title(f"Dropout rate by {col}")
        show(C.rate_bar(t, _base(df), col, 320))
        insight(I.extremes_sentence(t, col))


_MAP = {"Outcomes": outcomes, "Student profile": profile, "Academic performance": academic,
        "Enrollment patterns": enrollment, "Financial indicators": financial, "Relationships": relationships,
        "Data quality": quality, "Feature lab": lab}


def render(ctx) -> None:
    page_head("Explore", 'Explore<br><span class="dim">the signals.</span>',
              "Understand the academic, enrollment and financial patterns behind student outcomes, measured from the loaded dataset.")
    if ctx["problems"]:
        for p in ctx["problems"]:
            note(esc(p), "bad")
        return
    if ctx["loaded"].kind == "placeholder":
        note("<b>Placeholder data.</b> These charts describe the 3-row placeholder, not your students. Load your CSV on the Overview page.")
    ch = st.segmented_control("Chapter", CHAPTERS, default="Outcomes", required=True, key="explore_ch",
                              label_visibility="collapsed")
    _MAP[ch or "Outcomes"](ctx["loaded"].df)
