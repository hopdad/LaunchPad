"""Settings page — departments, trailer capacity, fluff."""

import logging
import streamlit as st
from db_utils import db_connection, save_settings

logger = logging.getLogger(__name__)


def render():
    st.header("Settings")

    departments_raw = st.text_input(
        "Departments (comma-separated)",
        value=",".join(st.session_state["departments"]),
    )
    departments = [d.strip() for d in departments_raw.split(",") if d.strip()]
    if not departments:
        st.error("Enter at least one department.")
        st.stop()
    st.session_state["departments"] = departments

    trailer_capacity = st.number_input(
        "Trailer Capacity",
        value=st.session_state["trailer_capacity"],
        min_value=1,
    )
    st.session_state["trailer_capacity"] = trailer_capacity

    fluff = st.selectbox(
        "Fluff (extra cube buffer)",
        options=[50, 100, 150, 200, 250, 300],
        index=[50, 100, 150, 200, 250, 300].index(st.session_state["fluff"]),
    )
    st.session_state["fluff"] = fluff

    # Persist settings to DB
    try:
        with db_connection() as (conn, c):
            save_settings({
                "departments": st.session_state["departments"],
                "trailer_capacity": st.session_state["trailer_capacity"],
                "fluff": st.session_state["fluff"],
                "stores": st.session_state["stores"],
                "store_ready_times": st.session_state["store_ready_times"],
            }, conn, c)
    except Exception:
        logger.exception("Failed to persist settings")
        st.error("Could not save settings to database.")
