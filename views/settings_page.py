"""Settings page — departments, trailer capacity, fluff, store zones."""

import logging
import pandas as pd
import streamlit as st
from db_utils import db_connection, save_settings, save_store_zones, load_store_zones

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

    # --- Store Zone Management ---
    st.divider()
    st.header("Store Zones")
    st.caption(
        "Assign each store to a delivery zone. Overs from stores in the same zone "
        "will be grouped onto the same peddle run."
    )

    stores = st.session_state.get("stores", [])
    if not stores:
        st.info("Configure stores first (on the Configure page) before assigning zones.")
        return

    # Load existing zones
    try:
        with db_connection() as (conn, c):
            existing_zones = load_store_zones(c)
    except Exception:
        logger.exception("Failed to load store zones")
        existing_zones = {}

    # Build editable dataframe
    zone_data = []
    for s in stores:
        zone_data.append({"Store": s, "Zone": existing_zones.get(s, "")})

    zone_df = pd.DataFrame(zone_data)
    edited_zone_df = st.data_editor(
        zone_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Store": st.column_config.TextColumn("Store", disabled=True),
            "Zone": st.column_config.TextColumn("Zone", help="e.g. North, South, East, West"),
        },
        key="zone_editor",
    )

    if st.button("Save Zones", key="save_zones_btn"):
        zone_map = {}
        for _, row in edited_zone_df.iterrows():
            store = str(row["Store"]).strip()
            zone = str(row["Zone"]).strip()
            if store and zone:
                zone_map[store] = zone
        try:
            with db_connection() as (conn, c):
                save_store_zones(zone_map, conn, c)
            st.session_state["store_zones"] = zone_map
            st.success(f"Saved zones for {len(zone_map)} stores.")
        except Exception:
            logger.exception("Failed to save store zones")
            st.error("Could not save zones to database.")
