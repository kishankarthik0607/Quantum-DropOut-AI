"""Fixed top navigation. Styling lives in styles/theme.css (.st-key-qd-nav*)."""
from __future__ import annotations

import streamlit as st

from .. import state
from ..config import PAGES
from ..ui.components import brand, esc, html


def _initials(name: str) -> str:
    parts = [p for p in name.replace("@", " ").split() if p]
    return ("".join(p[0] for p in parts[:2]) or "U").upper()


def render() -> None:
    user = st.session_state.user or {"name": "User", "email": ""}
    st.session_state["nav_desktop"] = st.session_state.page   # page is the source of truth
    with st.container(key="qd-nav"):
        html(brand())
        st.radio("Navigate", PAGES, horizontal=True, key="nav_desktop",
                 label_visibility="collapsed", on_change=state.sync_nav)
        with st.container(key="qd-nav-actions"):
            st.button("Run prediction", key="qd-nav-run", type="primary", on_click=state.goto, args=("Predict",))
            with st.popover(_initials(user["name"]), key="qd-acct"):
                html(f'<div class="qd-acct-name">{esc(user["name"])}</div><div class="qd-acct-mail">{esc(user["email"])}</div>')
                st.button("Sign out", key="logout_desktop", on_click=state.logout, **_full())
            with st.popover("Menu", key="qd-menu"):
                for p in PAGES:
                    st.button(p, key=f"menu_{p}", type="tertiary", on_click=state.goto, args=(p,), **_full())
                st.button("Sign out", key="logout_mobile", on_click=state.logout, **_full())


def _full() -> dict:
    from ..ui.components import stretch
    return stretch()
