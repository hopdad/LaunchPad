"""Outputs page — PDF/Excel export and DB save."""

import logging
import streamlit as st
from datetime import datetime
from db_utils import db_connection, save_data_to_db, delete_draft
from exports import generate_pdf, generate_excel
from summary import compute_summary

logger = logging.getLogger(__name__)


def render():
    df = st.session_state["df"]
    departments = st.session_state["departments"]
    runs_df = st.session_state["runs_df"]
    fluff = st.session_state["fluff"]
    trailer_capacity = st.session_state["trailer_capacity"]

    if df.empty:
        st.info("No data yet. Go to Data Entry first.")
        return

    date = st.date_input("Sheet Date", st.session_state.get("_entry_date", datetime.today()))
    s = compute_summary(df, fluff, trailer_capacity)

    if st.button("Generate PDF Sheet"):
        try:
            pdf_output = generate_pdf(df, departments, runs_df, date, s["total_cube"], s["trailer_goal"], s["probable_trailers"])
            st.download_button("Download Peddle Sheet PDF", pdf_output, file_name=f"ped_sheet_{date.strftime('%Y%m%d')}.pdf", mime="application/pdf")
        except Exception:
            logger.exception("Error generating PDF")
            st.error("Error generating PDF.")

    try:
        excel_output = generate_excel(df)
        st.download_button("Download Data as Excel", excel_output, file_name="ped_data.xlsx")
    except Exception:
        logger.exception("Error generating Excel")
        st.error("Error generating Excel file.")

    if st.button("Save to DB"):
        try:
            username = st.session_state.get("username", "unknown")
            with db_connection() as (conn, c):
                save_data_to_db(df, departments, runs_df, date, conn, c)
                # Clean up the draft now that data is fully saved
                delete_draft(date.strftime("%Y-%m-%d"), username, conn, c)
                st.session_state["_last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Data saved to database!")
        except Exception:
            logger.exception("Error saving to DB")
            st.error("Error saving to database.")

    # Persistent "last saved" indicator
    if st.session_state.get("_last_saved"):
        st.caption(f"Last saved: {st.session_state['_last_saved']}")
