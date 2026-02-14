"""Settings page — departments, trailer capacity, fluff, store zones, store locations."""

import logging
import pandas as pd
import streamlit as st
from db_utils import (
    db_connection,
    save_settings,
    save_store_zones,
    load_store_zones,
    save_store_locations,
    load_store_locations,
)

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

    # --- DC Home Base Location ---
    st.divider()
    st.header("DC / Home Base Location")
    st.caption(
        "Enter the lat/lng of your distribution center. "
        "This is used to calculate distances for ordering stores within peddle runs."
    )

    saved_dc = st.session_state.get("dc_location") or (None, None)
    dc_col1, dc_col2 = st.columns(2)
    with dc_col1:
        dc_lat = st.number_input(
            "DC Latitude",
            value=saved_dc[0] if saved_dc[0] is not None else 0.0,
            format="%.6f",
            key="dc_lat_input",
        )
    with dc_col2:
        dc_lng = st.number_input(
            "DC Longitude",
            value=saved_dc[1] if saved_dc[1] is not None else 0.0,
            format="%.6f",
            key="dc_lng_input",
        )

    if st.button("Save DC Location", key="save_dc_btn"):
        st.session_state["dc_location"] = (dc_lat, dc_lng)
        try:
            with db_connection() as (conn, c):
                save_settings({"dc_location": [dc_lat, dc_lng]}, conn, c)
            st.success(f"DC location saved: ({dc_lat:.6f}, {dc_lng:.6f})")
        except Exception:
            logger.exception("Failed to save DC location")
            st.error("Could not save DC location.")

    # --- Store Locations ---
    st.divider()
    st.header("Store Locations")
    st.caption(
        "Enter lat/lng for each store. Stores in each peddle run will be sequenced "
        "by distance from the DC (furthest first by default)."
    )

    # Load existing locations
    try:
        with db_connection() as (conn, c):
            existing_locs = load_store_locations(c)
    except Exception:
        logger.exception("Failed to load store locations")
        existing_locs = {}

    loc_data = []
    for s in stores:
        loc = existing_locs.get(s, (0.0, 0.0))
        loc_data.append({"Store": s, "Latitude": loc[0], "Longitude": loc[1]})

    loc_df = pd.DataFrame(loc_data)
    edited_loc_df = st.data_editor(
        loc_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Store": st.column_config.TextColumn("Store", disabled=True),
            "Latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
            "Longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
        },
        key="location_editor",
    )

    if st.button("Save Locations", key="save_locations_btn"):
        loc_map = {}
        for _, row in edited_loc_df.iterrows():
            store = str(row["Store"]).strip()
            lat = float(row["Latitude"]) if pd.notna(row["Latitude"]) else 0.0
            lng = float(row["Longitude"]) if pd.notna(row["Longitude"]) else 0.0
            if store and (lat != 0.0 or lng != 0.0):
                loc_map[store] = (lat, lng)
        try:
            with db_connection() as (conn, c):
                save_store_locations(loc_map, conn, c)
            st.session_state["store_locations"] = loc_map
            st.success(f"Saved locations for {len(loc_map)} stores.")
        except Exception:
            logger.exception("Failed to save store locations")
            st.error("Could not save locations to database.")
