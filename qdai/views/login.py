"""Login / create-account entry point (styles: .qd-l-*, .qd-auth-*, .st-key-qd-login)."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from .. import auth, data
from ..ui.components import esc, html, stretch, svg_img


_MARK = svg_img('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/>'
                '<path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/></svg>', style="width:18px;height:18px")


def _left() -> None:
    loaded = data.load_dataset()
    if loaded.kind != "placeholder" and "Target" in loaded.df.columns:
        P = data.prepare(loaded.df)
        stats = [(f"{len(loaded.df):,}", "Students analyzed"), (f"{P.X.shape[1]}", "Risk signals tracked"),
                 (f"{loaded.df['Target'].nunique()}", "Outcome classes")]
    else:   # no dataset yet: neutral product facts, never invented numbers
        stats = [("AI-powered", "Student retention"), ("Data-driven", "Risk assessment"), ("Actionable", "Interventions")]
    cells = "".join(f'<div class="qd-l-stat"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for v, k in stats)
    html(f"""
      <div class="qd-l-brand qd-rise" style="--i:0">
        <div class="qd-l-mark">{_MARK}</div>
        <div class="qd-brand" style="font-size:1.15rem">QUANTUM<span>_</span>DROPOUT<span>_</span>AI</div>
      </div>
      <h1 class="qd-l-h1 qd-rise" style="--i:1">Predict Student<br><em>Dropout Risk</em><br>Before It Happens.</h1>
      <p class="qd-l-lede qd-rise" style="--i:2">Use machine learning to identify at-risk students early, understand the signals behind
      dropout risk, and support data-driven intervention.</p>
      <div class="qd-l-stats qd-rise" style="--i:3">{cells}</div>""")


def _signin() -> None:
    html('<h2 class="qd-auth-title">Welcome back.</h2>'
         '<p class="qd-auth-sub">Sign in to access the student retention intelligence dashboard.</p>')
    wait = auth.lockout_remaining(st.session_state)
    with st.form("signin_form", border=False):
        ident = st.text_input("Email or username", placeholder="Enter your email or username", key="si_ident")
        pw = st.text_input("Password", type="password", placeholder="Enter your password", key="si_pw")
        go = st.form_submit_button("Sign in", type="primary", **stretch())
    if go:
        if wait:
            st.session_state["_auth_err"] = f"Too many attempts. Try again in {wait} seconds."
        else:
            try:
                user = auth.authenticate(ident, pw)
                auth.record_success(st.session_state)
                st.session_state.update(authed=True, user=user, page="Overview")
                st.rerun()
            except auth.AuthError as exc:
                auth.record_failure(st.session_state)
                st.session_state["_auth_err"] = str(exc)


def _signup() -> None:
    html('<h2 class="qd-auth-title">Create your account.</h2>'
         '<p class="qd-auth-sub">Set up access to the student retention intelligence dashboard.</p>')
    with st.form("signup_form", border=False):
        name = st.text_input("Full name", placeholder="Enter your name", key="su_name")
        email = st.text_input("Email", placeholder="Enter your email", key="su_email")
        pw = st.text_input("Password", type="password", placeholder="Create a password", key="su_pw",
                           help="At least 8 characters, with a letter and a number.")
        pw2 = st.text_input("Confirm password", type="password", placeholder="Confirm your password", key="su_pw2")
        go = st.form_submit_button("Create account", type="primary", **stretch())
    if go:
        try:
            user = auth.register(name, email, pw, pw2)
            st.session_state.update(authed=True, user=user, page="Overview")
            st.rerun()
        except auth.AuthError as exc:
            st.session_state["_auth_err"] = str(exc)


def render() -> None:
    with st.container(key="qd-login"):
        left, right = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
        with left:
            _left()
        with right:
            with st.container(key="qd-auth-card"):
                mode = st.radio("Mode", ["Sign in", "Create account"], horizontal=True, key="auth_mode",
                                label_visibility="collapsed")
                err = st.session_state.pop("_auth_err", None)
                (_signin if mode == "Sign in" else _signup)()
                if err:
                    html(f'<div class="qd-note bad" role="alert">{esc(err)}</div>')
                html(f'<div class="qd-auth-foot">Quantum_DropOut_AI • Powered by Machine Learning • {datetime.now().year}'
                     '<br>Accounts are stored locally on this machine.</div>')
