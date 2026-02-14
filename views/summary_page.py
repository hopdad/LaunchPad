"""Summary & Planning page."""

import logging
import pandas as pd
import streamlit as st
from datetime import datetime
from planning import auto_suggest_runs, extract_overs, auto_suggest_overs_runs
from summary import compute_summary
from db_utils import db_connection, load_store_zones, fetch_correction_factor

logger = logging.getLogger(__name__)


def render():
    df = st.session_state["df"]
    stores = st.session_state["stores"]
    trailer_capacity = st.session_state["trailer_capacity"]
    fluff = st.session_state["fluff"]

    if df.empty:
        st.info("No data yet. Go to Data Entry first.")
        return

    # Load correction factor and zones
    correction_factor = 1.0
    store_zones = {}
    try:
        with db_connection() as (conn, c):
            correction_factor = fetch_correction_factor(c)
            store_zones = load_store_zones(c)
    except Exception:
        logger.exception("Failed to load correction factor or zones")

    try:
        s = compute_summary(df, fluff, trailer_capacity)

        st.header("Summary")
        st.write(f"**Total Cube:** {s['total_cube']:.2f}")
        st.write(f"**Fluff:** {fluff}")
        st.write(f"**Total Cube (with fluff):** {s['total_cube_with_fluff']:.2f}")
        st.write(f"**Pallet Count:** {s['pallet_count']}")
        st.write(f"**Trailer Goal:** {s['trailer_goal']:.2f}")
        st.write(f"**Probable Trailers:** {s['probable_trailers']}")

        # Show correction factor when we have historical data
        if correction_factor != 1.0:
            adjusted_trailers = round(s['probable_trailers'] * correction_factor)
            st.write(f"**Correction Factor:** {correction_factor:.2f}x (from actuals)")
            st.write(f"**Adjusted Probable Trailers:** {adjusted_trailers}")

        st.header("Peddle Run Planning")
        auto_suggest = st.checkbox("Auto-suggest Peddle Runs (FFD bin-packing)")

        runs_df = pd.DataFrame()
        if auto_suggest:
            store_ready_times = st.session_state.get("store_ready_times") or {}
            runs = auto_suggest_runs(df, trailer_capacity, store_ready_times=store_ready_times)
            runs_df = pd.DataFrame(runs)
            runs_df["Run"] = runs_df.index + 1
            runs_df["Time"] = ""
            runs_df["Carrier"] = "Standard"
            runs_df["Fit Note"] = runs_df.apply(lambda r: "FITS" if r["Second Trailer"] == "No" else "", axis=1)
            st.write("Auto-Suggested Runs:")
            st.dataframe(runs_df)
        else:
            num_runs = st.number_input("Number of Peddle Runs", min_value=1, value=5)
            runs = []
            available_stores = stores.copy()
            assigned_stores = set()
            for i in range(num_runs):
                with st.expander(f"Run {i+1}"):
                    selected_stores = st.multiselect(f"Stores for Run {i+1}", options=available_stores)
                    carrier = st.selectbox(f"Carrier for Run {i+1}", ["Standard", "NTB"])
                    time_slot = st.time_input(f"Time for Run {i+1}", value=datetime.now().time())
                    run_cube = df[df["STORE"].isin(selected_stores)]["TOTAL"].sum()
                    second_trailer = "Yes" if run_cube > trailer_capacity else "No"
                    fit_note = st.text_input(f"Fit Note for Run {i+1}", value="FITS" if run_cube <= trailer_capacity else "")
                    runs.append({
                        "Run": i+1,
                        "Time": time_slot.strftime("%H:%M"),
                        "Stores": "/".join(selected_stores),
                        "Carrier": carrier,
                        "Total Cube": run_cube,
                        "Second Trailer": second_trailer,
                        "Fit Note": fit_note
                    })
                    assigned_stores.update(selected_stores)
                    available_stores = [s for s in available_stores if s not in selected_stores]

            # Show unassigned stores
            unassigned = [s for s in stores if s not in assigned_stores]
            if unassigned:
                st.warning(f"**Unassigned stores ({len(unassigned)}):** {', '.join(unassigned)}")

            runs_df = pd.DataFrame(runs)
            st.write("Manual Runs:")
            st.dataframe(runs_df)

        st.session_state["runs_df"] = runs_df

        # --- Overs-Based Peddle Runs ---
        st.divider()
        st.header("Overs Peddle Runs")

        overs_df = extract_overs(df, trailer_capacity)
        if overs_df.empty:
            st.info("No overs today — all stores fit within trailer capacity.")
        else:
            st.write(f"**{len(overs_df)} store(s) with overflow freight:**")
            st.dataframe(
                overs_df.rename(columns={"OVERS": "Overflow Cube"}),
                use_container_width=True,
                hide_index=True,
            )
            total_overs = overs_df["OVERS"].sum()
            st.write(f"**Total overflow cube:** {total_overs:.0f}")

            if not store_zones:
                st.warning(
                    "No store zones configured. Overs will be grouped into a single pool. "
                    "Assign zones in Settings to get route-optimized peddle runs."
                )

            overs_runs = auto_suggest_overs_runs(df, trailer_capacity, store_zones=store_zones)
            if overs_runs:
                overs_runs_df = pd.DataFrame(overs_runs)
                overs_runs_df.insert(0, "Run", range(1, len(overs_runs_df) + 1))
                st.write("**Suggested overs peddle runs:**")
                st.dataframe(overs_runs_df, use_container_width=True, hide_index=True)
                st.session_state["overs_runs_df"] = overs_runs_df
            else:
                st.session_state["overs_runs_df"] = pd.DataFrame()
    except Exception:
        logger.exception("Error in summary or planning")
        st.error("An error occurred computing the summary. Check your data.")
