"""History page (admin only) — historical data, trends, and accuracy metrics."""

import logging
import pandas as pd
import streamlit as st
from datetime import datetime
from db_utils import (
    db_connection,
    fetch_historical_data,
    fetch_actuals_for_date,
    fetch_per_store_history,
    fetch_correction_factor,
    fetch_per_store_correction,
)

logger = logging.getLogger(__name__)


def render():
    st.header("Historical Data")
    query_date = st.date_input("View Data For", datetime.today())

    try:
        with db_connection() as (conn, c):
            summaries, runs, totals = fetch_historical_data(query_date, c)
            actuals = fetch_actuals_for_date(query_date, c)
            store_history = fetch_per_store_history(c)

            # --- Store Summaries ---
            if not summaries.empty:
                st.subheader("Store Summaries")
                st.dataframe(summaries)
            else:
                st.info("No store data for this date.")

            # --- Peddle Runs ---
            if not runs.empty:
                st.subheader("Peddle Runs")
                st.dataframe(runs)

            # --- Projected vs Actual Comparison ---
            if not runs.empty and not actuals.empty:
                st.subheader("Projected vs Actual Trailers")
                merged = runs.merge(actuals, on="Run", how="left", suffixes=("", "_actual"))
                merged["Projected Trailers"] = merged["Second Trailer"].map(
                    lambda x: 2 if x == "Yes" else 1
                )
                merged["Actual Trailers"] = merged["Actual Trailers"].fillna(0).astype(int)
                merged["Variance"] = merged["Actual Trailers"] - merged["Projected Trailers"]

                display_cols = ["Run", "Stores", "Total Cube", "Projected Trailers", "Actual Trailers", "Variance"]
                st.dataframe(merged[[c for c in display_cols if c in merged.columns]])

                # Accuracy metrics
                total_projected = merged["Projected Trailers"].sum()
                total_actual = merged["Actual Trailers"].sum()
                if total_projected > 0:
                    accuracy = (1 - abs(total_actual - total_projected) / total_projected) * 100
                    accuracy = max(accuracy, 0)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Projected Trailers", int(total_projected))
                    col2.metric("Actual Trailers", int(total_actual))
                    col3.metric("Accuracy", f"{accuracy:.0f}%")
            elif not runs.empty:
                st.caption("No actuals recorded for this date yet.")

            # --- Cube Trends (all dates) ---
            if not totals.empty:
                st.subheader("Daily Cube Trends")
                totals["Date"] = pd.to_datetime(totals["Date"])
                st.line_chart(totals.set_index("Date"))

            # --- Per-Store Trends ---
            if not store_history.empty:
                st.subheader("Per-Store Cube Trends")
                store_history["Date"] = pd.to_datetime(store_history["Date"])
                pivot = store_history.pivot_table(
                    index="Date", columns="Store", values="Total", aggfunc="sum"
                ).fillna(0)
                if not pivot.empty:
                    st.line_chart(pivot)

            # --- Correction Factors ---
            st.divider()
            st.subheader("Cube Estimation Accuracy")
            correction = fetch_correction_factor(c)
            if correction != 1.0:
                delta_pct = (correction - 1.0) * 100
                direction = "over" if correction > 1.0 else "under"
                col1, col2 = st.columns(2)
                col1.metric(
                    "Overall Correction Factor",
                    f"{correction:.2f}x",
                    f"{delta_pct:+.0f}% (actuals run {direction})",
                )
                col2.caption(
                    "A factor > 1.0 means actuals use more trailers than projected. "
                    "This factor is automatically applied to future estimates."
                )

                # Per-store breakdown
                store_corrections = fetch_per_store_correction(c)
                if store_corrections:
                    st.write("**Per-Store Bias** (stores with 3+ data points):")
                    bias_df = pd.DataFrame([
                        {"Store": s, "Factor": f"{f:.2f}x",
                         "Bias": f"{(f-1)*100:+.0f}%"}
                        for s, f in sorted(store_corrections.items())
                    ])
                    st.dataframe(bias_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "Not enough historical data yet to compute a correction factor. "
                    "Enter actuals on the Actuals page after each shift to start building this."
                )

    except Exception:
        logger.exception("Error fetching historical data")
        st.error("Error loading historical data.")
