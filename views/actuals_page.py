"""Actuals page — enter what actually happened for the prior day."""

import logging
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from db_utils import db_connection, fetch_prior_peddles, save_actual_peddles

logger = logging.getLogger(__name__)


def render():
    st.header("Enter Actual Peddles (From Prior Day)")
    prior_date = (datetime.today() - timedelta(days=1)).date()
    st.write(f"Showing projections from {prior_date.strftime('%Y-%m-%d')}. Enter what actually happened.")

    try:
        with db_connection() as (conn, c):
            prior_peddles_df = fetch_prior_peddles(prior_date, c)

            if not prior_peddles_df.empty:
                st.dataframe(prior_peddles_df)

                actuals = []
                for _, row in prior_peddles_df.iterrows():
                    with st.expander(f"Run {row['Run']} ({row['Stores']}) - Projected: {row['Total Cube']} cube, Second Trailer: {row['Second Trailer']}"):
                        actual_trailers = st.number_input("Actual Trailers Used", min_value=1, value=1, key=f"actual_trailers_{row['Run']}")
                        actual_notes = st.text_input("Notes/Variances", key=f"actual_notes_{row['Run']}")
                        actuals.append({
                            "Run": row['Run'],
                            "Actual Trailers": actual_trailers,
                            "Actual Notes": actual_notes
                        })

                if st.button("Save Actuals"):
                    save_actual_peddles(pd.DataFrame(actuals), prior_date, conn, c)
                    st.success("Actuals saved! This data will help refine future estimates.")
            else:
                st.info("No prior day data found. Process today's sheet first.")
    except Exception:
        logger.exception("Error loading prior data")
        st.error("Error loading prior day data.")
