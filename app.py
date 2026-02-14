import logging
import streamlit as st

st.set_page_config(page_title="Peddle Sheet Generator", layout="wide")

import pandas as pd
import streamlit_authenticator as stauth
from db_utils import db_connection, load_settings, load_store_zones

# Pages (directory named "views" to avoid Streamlit auto-detection)
from views import (
    settings_page,
    configure_page,
    data_entry_page,
    summary_page,
    actuals_page,
    outputs_page,
    history_page,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Auth Setup ---
credentials = {
    "usernames": {
        "clerk1": {
            "name": "Clerk User",
            "password": "pass123",
            "role": "clerk"
        },
        "clerk2": {
            "name": "Clerk User 2",
            "password": "pass456",
            "role": "clerk"
        },
        "admin1": {
            "name": "Admin User",
            "password": "adminpass",
            "role": "admin"
        },
        "manager1": {
            "name": "Manager User",
            "password": "managerpass",
            "role": "admin"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="peddle_app",
    cookie_key="auth_key",
    cookie_expiry_days=30
)

try:
    authenticator.login(location='main')
except Exception:
    logger.exception("Login error")
    st.error("An error occurred during login.")

authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')
username = st.session_state.get('username')

if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()

st.success(f"Welcome, {name}! ({credentials['usernames'][username]['role'].capitalize()})")
user_role = credentials["usernames"][username]["role"]

st.title("Peddle Sheet Generator")
st.write("Streamlit-powered web app for daily peddle planning. Access via browser—no installs needed.")


# --- Session State Initialization ---
if "departments" not in st.session_state:
    try:
        with db_connection() as (conn, c):
            _saved = load_settings(c)
    except Exception:
        logger.exception("Failed to load settings from DB")
        _saved = {}
    st.session_state["departments"] = _saved.get("departments", ["882", "883", "MB"])
    st.session_state["stores"] = _saved.get("stores", [])
    st.session_state["store_ready_times"] = _saved.get("store_ready_times", {})
    st.session_state["trailer_capacity"] = _saved.get("trailer_capacity", 1600)
    st.session_state["fluff"] = _saved.get("fluff", 200)
    try:
        with db_connection() as (conn, c):
            st.session_state["store_zones"] = load_store_zones(c)
    except Exception:
        logger.exception("Failed to load store zones")
        st.session_state["store_zones"] = {}
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "runs_df" not in st.session_state:
    st.session_state["runs_df"] = pd.DataFrame()


# --- Sidebar Navigation (workflow order) ---
pages = ["Configure", "Data Entry", "Summary & Planning", "Outputs", "Actuals", "Settings"]
if user_role == "admin":
    pages.insert(5, "History")  # before Settings

with st.sidebar:
    st.image("logo.jpg", use_container_width=True)
    st.divider()
    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.divider()
    authenticator.logout(button_name="Logout", location="main")


# --- Page Router ---
_PAGE_MAP = {
    "Settings": settings_page.render,
    "Configure": configure_page.render,
    "Data Entry": data_entry_page.render,
    "Summary & Planning": summary_page.render,
    "Actuals": actuals_page.render,
    "Outputs": outputs_page.render,
    "History": history_page.render,
}

render_fn = _PAGE_MAP.get(selected_page)
if render_fn:
    render_fn()
