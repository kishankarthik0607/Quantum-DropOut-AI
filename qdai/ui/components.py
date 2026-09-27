"""Small HTML building blocks that use the classes defined in styles/theme.css."""
from __future__ import annotations

from html import escape

import streamlit as st

from ..config import PAGES, RISK_HIGH_MIN, RISK_LOW_MAX

esc = escape


def html(s: str) -> None:
    body = " ".join(s.split())
    if body:                      # st.html rejects an empty body (e.g. a student with no interventions)
        st.html(body)


def svg_img(svg: str, alt: str = "", cls: str = "", style: str = "") -> str:
    """st.html strips inline <svg>, so SVGs are embedded as data-URI images (allowed by the sanitiser)."""
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    role = f'role="img" alt="{esc(alt)}"' if alt else 'alt="" aria-hidden="true"'
    return f'<img {role} class="{cls}" style="{style}" src="data:image/svg+xml;base64,{b64}">'


def brand(extra_class: str = "") -> str:
    return (f'<div class="qd-brand {extra_class}"><i class="qd-brand-mark"></i>'
            f'QUANTUM<span>_</span>DROPOUT<span>_</span>AI</div>')


def page_head(eyebrow: str, title: str, lede: str) -> None:
    """title may contain <span class="dim"> for the two-tone treatment."""
    html(f'<div class="qd-pagehead"><div><span class="qd-eyebrow">{esc(eyebrow)}</span>'
         f'<h1 class="qd-h1">{title}</h1></div><p class="qd-lede">{esc(lede)}</p></div>')


def section(title: str, note: str = "", first: bool = False) -> None:
    n = f'<div class="qd-small">{esc(note)}</div>' if note else ""
    html(f'<div class="qd-sec{" first" if first else ""}"><h2 class="qd-h2">{esc(title)}</h2>{n}</div>')


def figure_title(title: str, sub: str = "") -> None:
    s = f'<div class="qd-figure-sub">{esc(sub)}</div>' if sub else ""
    html(f'<div class="qd-figure-title">{esc(title)}</div>{s}')


def insight(text: str | None) -> None:
    if text:
        html(f'<div class="qd-insight">{esc(text)}</div>')


def note(text: str, kind: str = "") -> None:
    html(f'<div class="qd-note {kind}">{text}</div>')


def stat_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="qd-stat"><div class="qd-stat-num">{esc(v)}</div>'
                    f'<div class="qd-stat-label">{esc(k)}</div></div>' for v, k in items)
    html(f'<div class="qd-stats">{cells}</div>')


def show(fig, key: str | None = None) -> None:
    """Render a Plotly figure, across Streamlit versions."""
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def stretch() -> dict:
    """kwargs that make a widget fill its column in old and new Streamlit versions."""
    try:
        import inspect
        if "width" in inspect.signature(st.dataframe).parameters:
            return {"width": "stretch"}
    except Exception:  # noqa: BLE001
        pass
    return {"use_container_width": True}


def footer() -> None:
    html(f'<div class="qd-footer"><div>{brand()}'
         '<p>Predictive analytics for proactive student retention.</p></div>'
         '<div class="qd-footer-tags">Data-driven.<br>Explainable.<br>Intervention-focused.</div></div>')
    from .. import state
    with st.container(key="qd-footer-links"):
        cols = st.columns(len(PAGES))
        for c, p in zip(cols, PAGES):
            c.button(p, key=f"foot_{p}", type="tertiary", on_click=state.goto, args=(p,))


# -------------------------------------------------------------- risk helpers
def risk_band(p: float) -> tuple[str, str]:
    """(label, colour) for a dropout probability."""
    if p < RISK_LOW_MAX:
        return "Low risk", "#35D0C4"
    if p < RISK_HIGH_MIN:
        return "Medium risk", "#F0A24E"
    return "High risk", "#F0605D"
