"""Plotly figures with one shared visual language. Each function draws exactly what it is given."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import OUTCOME_COLORS, OUTCOME_ORDER

INK, LINE, T2, T3 = "#07090F", "#1C2333", "#A3ACC0", "#7F89A0"
VIOLET, TEAL, CORAL, AMBER = "#7C6CFF", "#35D0C4", "#F0605D", "#F0A24E"
FONT = "Manrope, 'Manrope Local', system-ui, sans-serif"


def _base(fig: go.Figure, height: int = 340, legend: bool = True, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=8, t=8 if not legend else 34, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=T2, size=12),
        showlegend=legend, legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#111624", bordercolor="#2A3350", font=dict(family=FONT, color="#F2F4F8")),
        bargap=.28, **layout)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=T3), title_font=dict(color=T3, size=12))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=T3), title_font=dict(color=T3, size=12))
    return fig


def _oc(name: str) -> str:
    return OUTCOME_COLORS.get(name, VIOLET)


def outcome_bar(counts: dict) -> go.Figure:
    total = sum(counts.values()) or 1
    fig = go.Figure()
    for k in OUTCOME_ORDER:
        if k in counts:
            v = counts[k]
            fig.add_bar(y=[""], x=[v], orientation="h", name=k, marker_color=_oc(k),
                        text=[f"{k}  {v / total * 100:.1f}%"], textposition="inside", insidetextanchor="start",
                        textfont=dict(color="#07090F", size=13, family=FONT),
                        hovertemplate=f"{k}: %{{x:,}} students<extra></extra>")
    _base(fig, 96, legend=False, barmode="stack")
    fig.update_yaxes(visible=False); fig.update_xaxes(visible=False, range=[0, total])
    fig.update_layout(bargap=0, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def hist_by_outcome(df: pd.DataFrame, col: str, nbins: int = 30, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for k in OUTCOME_ORDER:
        s = df.loc[df["Target"].astype(str) == k, col].dropna()
        if len(s):
            fig.add_histogram(x=s, name=f"{k} (n={len(s):,})", nbinsx=nbins, marker_color=_oc(k), opacity=.62,
                              hovertemplate="%{x}: %{y} students<extra>" + k + "</extra>")
            fig.add_vline(x=float(s.mean()), line_dash="dot", line_color=_oc(k), line_width=1.5)
    _base(fig, height, barmode="overlay")
    fig.update_xaxes(title=col); fig.update_yaxes(title="Students")
    return fig


def hist_single(s: pd.Series, name: str, height: int = 300, kde: bool = True) -> go.Figure:
    s = s.dropna()
    fig = go.Figure(go.Histogram(x=s, nbinsx=30, marker_color=VIOLET, opacity=.55, name="Students",
                                 histnorm="probability density"))
    if kde and s.nunique() > 3:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(float(s.min()), float(s.max()), 200)
            fig.add_scatter(x=xs, y=gaussian_kde(s)(xs), mode="lines", line=dict(color=TEAL, width=2), name="Density")
        except Exception:  # noqa: BLE001
            pass
    fig.add_vline(x=float(s.mean()), line_dash="dot", line_color=T2, line_width=1.2)
    _base(fig, height, legend=False); fig.update_xaxes(title=name); fig.update_yaxes(title="Density")
    return fig


def rate_bar(table: pd.DataFrame, base_rate: float | None, xtitle: str = "", height: int = 300,
             horizontal: bool = False) -> go.Figure:
    t = table.copy()
    t["group"] = t["group"].astype(str)
    colors = [CORAL if r else "#3A4260" for r in t["reliable"]]
    text = [f"{r:.0f}%" for r in t["rate"]]
    custom = np.stack([t["n"], t["dropouts"]], axis=1)
    hover = "%{customdata[1]:.0f} of %{customdata[0]:,} students dropped out (%{y:.1f}%)<extra></extra>"
    if horizontal:
        fig = go.Figure(go.Bar(y=t["group"], x=t["rate"], orientation="h", marker_color=colors, text=text,
                               textposition="outside", customdata=custom,
                               hovertemplate=hover.replace("%{y", "%{x")))
        fig.update_yaxes(autorange="reversed", title=xtitle); fig.update_xaxes(title="Dropout rate (%)", range=[0, max(100, t["rate"].max() * 1.1)])
        if base_rate is not None:
            fig.add_vline(x=base_rate, line_dash="dot", line_color=T2, line_width=1)
    else:
        fig = go.Figure(go.Bar(x=t["group"], y=t["rate"], marker_color=colors, text=text, textposition="outside",
                               customdata=custom, hovertemplate=hover))
        fig.update_yaxes(title="Dropout rate (%)", range=[0, max(100, t["rate"].max() * 1.12)]); fig.update_xaxes(title=xtitle, type="category")
        if base_rate is not None:
            fig.add_hline(y=base_rate, line_dash="dot", line_color=T2, line_width=1,
                          annotation_text=f"All students {base_rate:.0f}%", annotation_font_color=T3, annotation_position="top right")
    return _base(fig, height, legend=False)


def crosstab_pct(df: pd.DataFrame, col: pd.Series, name: str, height: int = 320) -> go.Figure:
    ct = pd.crosstab(col, df["Target"].astype(str))
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for k in OUTCOME_ORDER:
        if k in pct.columns:
            fig.add_bar(x=pct.index.astype(str), y=pct[k], name=k, marker_color=_oc(k),
                        text=[f"{v:.0f}%" for v in pct[k]], textposition="inside", textfont=dict(color="#07090F"),
                        hovertemplate="%{x}: %{y:.1f}%<extra>" + k + "</extra>")
    _base(fig, height, barmode="stack"); fig.update_yaxes(title="Share of students (%)"); fig.update_xaxes(title=name, type="category")
    return fig


def box_by_outcome(df: pd.DataFrame, col: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for k in OUTCOME_ORDER:
        s = df.loc[df["Target"].astype(str) == k, col].dropna()
        if len(s):
            fig.add_box(y=s, name=k, marker_color=_oc(k), line_color=_oc(k), boxmean=True)
    _base(fig, height, legend=False); fig.update_yaxes(title=col)
    return fig


def corr_heatmap(corr: pd.DataFrame, height: int = 420) -> go.Figure:
    short = [c.replace("Curricular units ", "CU ") for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short, zmin=-1, zmax=1,
                               colorscale=[[0, CORAL], [.5, "#111624"], [1, TEAL]],
                               text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=11, color="#F2F4F8"),
                               hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=8, outlinewidth=0)))
    _base(fig, height, legend=False); fig.update_yaxes(autorange="reversed", showgrid=False); fig.update_xaxes(showgrid=False)
    return fig


def hbar(df: pd.DataFrame, x: str, y: str, color: str = VIOLET, height: int | None = None, xtitle: str = "",
         err: str | None = None, signed: bool = False, fmt: str = ".3f") -> go.Figure:
    d = df.iloc[::-1]
    colors = [(CORAL if v > 0 else TEAL) for v in d[x]] if signed else color
    fig = go.Figure(go.Bar(y=d[y], x=d[x], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=d[err], color=T3, thickness=1) if err else None,
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    _base(fig, height or max(240, 26 * len(d) + 40), legend=False)
    fig.update_xaxes(title=xtitle)
    return fig


def waterfall(contrib: pd.DataFrame, base: float, outcome: str, top: int = 10, height: int = 460) -> go.Figure:
    """SHAP waterfall in percentage points: base rate + contributions = predicted probability."""
    c = contrib.head(top)
    other = float(contrib["shap"].iloc[top:].sum())
    names = ["Base rate"] + [f.replace("Curricular units ", "CU ") for f in c["feature"]]
    vals = [base * 100] + list(c["shap"] * 100)
    measure = ["absolute"] + ["relative"] * len(c)
    if len(contrib) > top:
        names.append("All other features"); vals.append(other * 100); measure.append("relative")
    names.append(f"Predicted {outcome}"); vals.append(None); measure.append("total")
    up = CORAL if outcome == "Dropout" else VIOLET
    down = TEAL if outcome == "Dropout" else AMBER
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=vals, measure=measure, text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f}" if v is not None else "" for v, m in zip(vals, measure)],
        increasing=dict(marker=dict(color=up)), decreasing=dict(marker=dict(color=down)),
        totals=dict(marker=dict(color=VIOLET)), connector=dict(line=dict(color=LINE)),
        hovertemplate="%{y}: %{x:.1f} pp<extra></extra>"))
    _base(fig, height, legend=False); fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title=f"Probability of {outcome} (%)")
    return fig


def confusion(cm: np.ndarray, names: list[str], normalize: bool, height: int = 380) -> go.Figure:
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100 if normalize else cm
    text = [[f"{v:.0f}%" if normalize else f"{int(v)}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, text=text, texttemplate="%{text}", textfont=dict(size=15, color="#F2F4F8"),
                               colorscale=[[0, "#0B0E15"], [1, VIOLET]], showscale=False,
                               hovertemplate="Actual %{y} → predicted %{x}: %{text}<extra></extra>"))
    _base(fig, height, legend=False)
    fig.update_yaxes(autorange="reversed", title="Actual outcome", showgrid=False); fig.update_xaxes(title="Predicted outcome", showgrid=False, side="bottom")
    return fig


def pdp(curves: pd.DataFrame, feature: str, mean_x: float, height: int = 340) -> go.Figure:
    fig = go.Figure()
    for k in OUTCOME_ORDER:
        if k in curves.columns:
            fig.add_scatter(x=curves[feature], y=curves[k] * 100, mode="lines", name=k, line=dict(color=_oc(k), width=2.5))
    fig.add_vline(x=mean_x, line_dash="dot", line_color=T3, line_width=1, annotation_text="Mean", annotation_font_color=T3)
    _base(fig, height); fig.update_xaxes(title=feature); fig.update_yaxes(title="Average predicted probability (%)", range=[0, 100])
    return fig


def prob_hist(p: np.ndarray, low: float, high: float, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Histogram(x=p * 100, xbins=dict(start=0, end=100, size=5), marker_color=VIOLET, opacity=.8,
                                 hovertemplate="%{x}% dropout probability: %{y} students<extra></extra>"))
    for x, c in ((low * 100, AMBER), (high * 100, CORAL)):
        fig.add_vline(x=x, line_dash="dot", line_color=c, line_width=1.5)
    _base(fig, height, legend=False); fig.update_xaxes(title="Predicted dropout probability (%)", range=[0, 100]); fig.update_yaxes(title="Students")
    return fig


def simple_bar(x, y, height: int = 260, xtitle: str = "", ytitle: str = "", color: str = VIOLET, fmt: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=color, text=[f"{v:{fmt}}" for v in y], textposition="outside"))
    _base(fig, height, legend=False); fig.update_xaxes(title=xtitle, type="category"); fig.update_yaxes(title=ytitle)
    return fig
