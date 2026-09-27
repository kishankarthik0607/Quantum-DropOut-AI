"""Injects the design system (styles/theme.css) and page-change scroll reset."""
from __future__ import annotations

import streamlit as st

from ..config import ROOT

_CSS = ROOT / "styles" / "theme.css"


def inject() -> None:
    st.html(f"<style>{_CSS.read_text(encoding='utf-8')}</style>")


def scroll_top(token: int) -> None:
    """Scroll the Streamlit main pane to the top. `token` changes so it re-runs on each navigation."""
    js = ("<script>/*%d*/(function(){var d=window.parent.document;"
          "['section.main','[data-testid=stMain]','[data-testid=stAppViewContainer]'].forEach(function(s){"
          "var e=d.querySelector(s); if(e){e.scrollTo(0,0);}}); window.parent.scrollTo(0,0);})();</script>") % token
    try:
        st.html(js, unsafe_allow_javascript=True)
    except TypeError:
        import streamlit.components.v1 as components
        components.html(js, height=0)
