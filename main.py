"""Quantum_DropOut_AI — entry point.  Run with:  streamlit run main.py

Flow: login -> authenticated shell (top navigation) -> Overview / Explore / Models / Predict / Insights.
The original single-file app is preserved untouched in legacy/main_original.py.
"""
import warnings

import streamlit as st

from qdai import state
from qdai.config import APP_NAME

st.set_page_config(page_title=APP_NAME, page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    from qdai.ui import theme
    state.init()
    theme.inject()
    if not st.session_state.authed:
        from qdai.views import login
        login.render()
        return

    from qdai import data
    from qdai.views import shell, overview, explore, models, predict, insights
    from qdai.ui.components import footer

    up = st.session_state.upload
    loaded = data.load_dataset(*(up if up else (None, None)))
    sig = (loaded.kind, loaded.source, len(loaded.df))
    if st.session_state.data_sig != sig:            # a different dataset invalidates trained models
        state.reset_model()
        st.session_state.data_sig = sig
    problems = data.validate(loaded.df)
    prepared = data.prepare(loaded.df) if not problems else None
    st.session_state["_P"] = prepared                # read by the "load a real student" callback

    shell.render()
    page = st.session_state.page
    if st.session_state.last_page != page:
        st.session_state.last_page = page
        st.session_state["_scroll"] = st.session_state.get("_scroll", 0) + 1
        theme.scroll_top(st.session_state["_scroll"])

    ctx = dict(loaded=loaded, prepared=prepared, problems=problems)
    {"Overview": overview, "Explore": explore, "Models": models,
     "Predict": predict, "Insights": insights}[page].render(ctx)
    footer()


main()
