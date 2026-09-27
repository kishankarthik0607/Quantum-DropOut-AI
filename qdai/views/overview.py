"""Overview: hero, real dataset statistics, cohort field, data input, how it works, findings."""
from __future__ import annotations

import streamlit as st

from .. import data, insights, state
from ..config import OUTCOME_COLORS
from ..ui.components import esc, html, note, section, stat_strip, stretch, svg_img

_FIELD_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" fill="none">
<g stroke="rgba(255,255,255,.045)" stroke-width="1">%s</g>
<path d="M0 470 C 120 470, 160 300, 300 300 S 480 120, 600 110" stroke="#7C6CFF" stroke-opacity=".55" stroke-width="1.5"/>
<path d="M0 500 C 140 500, 190 340, 320 330 S 500 170, 600 150" stroke="#35D0C4" stroke-opacity=".32" stroke-width="1.5"/>
<path d="M0 430 C 110 430, 150 260, 290 250 S 470 80, 600 70" stroke="#A99CFF" stroke-opacity=".22" stroke-width="1.5"/>
<g fill="#A99CFF" fill-opacity=".7"><circle cx="300" cy="300" r="4"/><circle cx="480" cy="170" r="3"/><circle cx="160" cy="390" r="3"/></g>
</svg>""" % "".join(f'<path d="M{i * 60} 0V600M0 {i * 60}H600"/>' for i in range(11))
_FIELD = svg_img(_FIELD_SVG, cls="qd-hero-field")


def _cohort(counts: dict, total: int) -> None:
    """Dot field: the number of dots per outcome follows the real outcome shares."""
    cols, rows = 60, 8
    n = cols * rows
    order = [k for k in ("Graduate", "Enrolled", "Dropout") if k in counts]
    alloc = {k: int(counts[k] / total * n) for k in order}
    order_by_size = sorted(order, key=lambda k: counts[k], reverse=True)
    for i in range(n - sum(alloc.values())):
        alloc[order_by_size[i % len(order_by_size)]] += 1
    dots, idx = [], 0
    for k in order:
        for _ in range(alloc[k]):
            r, c = divmod(idx, cols)
            dots.append(f'<circle cx="{c * 14 + 7}" cy="{r * 14 + 7}" r="4.2" fill="{OUTCOME_COLORS[k]}"/>')
            idx += 1
    legend = "".join(f'<span><i style="background:{OUTCOME_COLORS[k]}"></i>{k} {counts[k] / total * 100:.1f}%</span>'
                     for k in order)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cols * 14} {rows * 14}">{"".join(dots)}</svg>')
    alt = "Cohort of students, one dot each, coloured by outcome: " + ", ".join(
        f"{k} {counts[k] / total * 100:.1f}%" for k in order)
    html(f'<div class="qd-cohort">{svg_img(svg, alt, style="width:100%;display:block")}'
         f'<div class="qd-legend">{legend}<span class="qd-small">Each dot represents about {total / n:.0f} students '
         f'({total:,} in total).</span></div></div>')


def _data_input(loaded, problems) -> None:
    section("Analyze your data", "Drop a CSV with a Target column (Dropout, Enrolled or Graduate). "
            "Without an upload the app looks for data/student_dropout_data.csv.")
    if loaded.notice:
        note(f"<b>Upload problem.</b> {esc(loaded.notice)} Check that the file is a CSV and try again.", "bad")
    up = st.file_uploader("Upload your Student Dropout CSV", type=["csv"],
                          key=f"uploader_{st.session_state.upload_nonce}")
    if up is not None:
        raw = up.getvalue()
        cur = st.session_state.upload
        if cur is None or (cur[0], len(cur[1])) != (up.name, len(raw)):
            st.session_state.upload = (up.name, raw)
            st.rerun()
    label = {"upload": "Uploaded file", "disk": "File from disk", "placeholder": "Placeholder"}[loaded.kind]
    n_feat = max(loaded.df.shape[1] - (1 if "Target" in loaded.df.columns else 0) - (1 if "id" in loaded.df.columns else 0), 0)
    html(f'<div class="qd-file"><div class="qd-file-k">{label} loaded</div>'
         f'<div class="qd-file-name">{esc(loaded.source)}</div>'
         f'<div class="qd-file-meta">{len(loaded.df):,} records · {n_feat} features</div></div>')
    if loaded.kind == "placeholder":
        note("<b>No dataset found.</b> The app is showing the built-in 3-row placeholder from the original project. "
             "Every number below describes those 3 rows, so upload your CSV or place it at data/student_dropout_data.csv.", "")
    if loaded.kind == "upload":
        st.button("Use the default dataset instead", key="revert_data", on_click=_revert)
    for p in problems:
        note(esc(p), "bad")


def _revert() -> None:
    st.session_state.upload = None
    st.session_state.upload_nonce += 1


def render(ctx) -> None:
    loaded, P, problems = ctx["loaded"], ctx["prepared"], ctx["problems"]
    df = loaded.df

    html("""<div class="qd-hero"><div>%s</div><div class="qd-hero-copy">
      <span class="qd-eyebrow qd-rise" style="--i:0">AI-powered student retention</span>
      <h1 class="qd-display qd-rise" style="--i:1">Understand<br>dropout risk<br><span class="dim">before it happens.</span></h1>
      <p class="qd-lede qd-rise" style="--i:2">Quantum_DropOut_AI analyzes academic, demographic and socioeconomic signals to identify
      students who may need timely support, then explains the result and suggests what to do next.</p></div></div>""" % _FIELD)
    with st.container(key="qd-hero-cta"):
        a, b, _ = st.columns([1, 1, 3])
        a.button("Run prediction", key="hero_run", type="primary", on_click=state.goto, args=("Predict",), **stretch())
        b.button("Explore insights", key="hero_explore", on_click=state.goto, args=("Insights",), **stretch())

    if not problems:
        s = data.summarize(df, P.X.shape[1])
        miss = "0%" if s["missing_pct"] == 0 else f'{s["missing_pct"]:.1f}%'
        stat_strip([(f'{s["students"]:,}', "Students analyzed"), (f'{s["dropout_rate"]:.1f}%', "Observed dropout rate"),
                    (f'{s["graduate_rate"]:.1f}%', "Graduate rate"), (miss, "Missing data")])
        if s["counts"]:
            _cohort(s["counts"], s["students"])

    _data_input(loaded, problems)

    section("How it works", "One path from raw records to a supported student.")
    html('<div class="qd-steps">'
         '<div class="qd-step"><div class="qd-step-n">01</div><h3>Load</h3><p>Bring in your student records. Completeness and structure are checked straight away.</p></div>'
         '<div class="qd-step"><div class="qd-step-n">02</div><h3>Explore</h3><p>See how outcomes differ across academic, enrollment and financial signals.</p></div>'
         '<div class="qd-step"><div class="qd-step-n">03</div><h3>Train</h3><p>Fit the prediction engine and read its accuracy on students it has not seen.</p></div>'
         '<div class="qd-step"><div class="qd-step-n">04</div><h3>Assess and act</h3><p>Score a student, see why, and get suggested institutional interventions.</p></div></div>')

    if not problems:
        found = insights.headline_findings(df)[:2]
        if found:
            section("What the data says", "Computed from the loaded dataset. Associations, not causes.")
            html("".join(f'<div class="qd-find"><div class="qd-find-fig">{esc(f.figure)}</div>'
                         f'<div><h3>{esc(f.headline)}</h3><p>{esc(f.detail)}</p></div></div>' for f in found))
            st.button("See all insights", key="ov_all_insights", on_click=state.goto, args=("Insights",))
