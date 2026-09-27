"""Helpers shared by the Models and Predict views."""
from __future__ import annotations

import streamlit as st

from .. import explain, ml, state
from ..ui.components import note


def primary_model(fit):
    return fit["res"]["models"][ml.PRIMARY_MODEL]


def get_explainer(fit):
    """TreeExplainer for the primary model, built once per trained model."""
    if "explainer" not in st.session_state:
        st.session_state["explainer"] = explain.make_explainer(primary_model(fit))
    return st.session_state["explainer"]


def require_fit(what: str):
    """Return the trained-model bundle, or show a gate that sends the user to Models."""
    fit = st.session_state.fit
    if fit is None:
        note(f"<b>Train the models first.</b> {what} needs the trained Random Forest, and nothing has been trained in this session yet.")
        st.button("Go to Models", key=f"gate_{what[:6]}", type="primary", on_click=state.goto, args=("Models",))
    return fit
