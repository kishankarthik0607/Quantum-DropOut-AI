"""Session state and navigation callbacks."""
from __future__ import annotations

import streamlit as st

from .config import PAGES

_DEFAULTS = dict(
    authed=False, user=None, page="Overview", upload=None, upload_nonce=0,
    fit=None, pred=None, sample=None, explain_cache=None, data_sig=None, last_page=None,
)


def init() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if st.session_state.explain_cache is None:
        st.session_state.explain_cache = {}
    st.session_state.setdefault("auth_mode", "Sign in")


def goto(page: str) -> None:
    """on_click / on_change callback: change the visible page."""
    if page in PAGES:
        st.session_state.page = page


def sync_nav() -> None:
    goto(st.session_state.get("nav_desktop", "Overview"))


def reset_model() -> None:
    st.session_state.fit = None
    st.session_state.pred = None
    st.session_state.sample = None
    st.session_state.explain_cache = {}
    st.session_state.pop("explainer", None)


def logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
