"""Data Entry page — manual entry, CSV upload, or OCR (with auto-save)."""

import logging
import pandas as pd
import streamlit as st
from db_utils import db_connection, save_draft, load_draft, delete_draft
from utils import parse_single_dept_ocr, run_ocr

logger = logging.getLogger(__name__)


def _auto_save(df, departments, entry_date):
    """Silently persist the current DataFrame as a draft."""
    username = st.session_state.get("username", "unknown")
    date_str = entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date)
    try:
        with db_connection() as (conn, c):
            save_draft(date_str, username, df, departments, conn, c)
    except Exception:
        logger.exception("Auto-save draft failed")


def render():
    departments = st.session_state["departments"]
    stores = st.session_state["stores"]
    trailer_capacity = st.session_state["trailer_capacity"]

    st.header("Cube Data Entry")

    # --- Clear / Undo buttons ---
    if not st.session_state["df"].empty:
        btn_cols = st.columns([1, 1, 4])
        with btn_cols[0]:
            if st.button("Clear Cube Data", type="secondary"):
                st.session_state["_confirm_clear"] = True
        with btn_cols[1]:
            if st.session_state.get("_df_backup") is not None:
                if st.button("Undo Clear"):
                    st.session_state["df"] = st.session_state.pop("_df_backup")
                    st.rerun()
        if st.session_state.get("_confirm_clear"):
            st.warning("Are you sure? This will erase all entered cube data.")
            yes_col, no_col, _ = st.columns([1, 1, 4])
            with yes_col:
                if st.button("Yes, clear it", type="primary"):
                    st.session_state["_df_backup"] = st.session_state["df"].copy()
                    st.session_state["df"] = pd.DataFrame()
                    st.session_state.pop("_confirm_clear", None)
                    st.rerun()
            with no_col:
                if st.button("Cancel"):
                    st.session_state.pop("_confirm_clear", None)
                    st.rerun()

    # --- Date association ---
    from datetime import datetime
    entry_date = st.date_input(
        "Data date",
        value=st.session_state.get("_entry_date", datetime.today()),
        key="_entry_date_input",
    )
    if st.session_state.get("_entry_date") != entry_date:
        st.session_state["_entry_date"] = entry_date
        st.session_state.pop("_draft_dismissed", None)

    # --- Check for saved draft ---
    username = st.session_state.get("username", "unknown")
    date_str = entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date)
    if st.session_state["df"].empty and not st.session_state.get("_draft_dismissed"):
        try:
            with db_connection() as (conn, c):
                draft_df, draft_depts, draft_time = load_draft(date_str, username, c)
            if draft_df is not None and not draft_df.empty:
                st.info(f"Draft found from {draft_time}. Resume where you left off?")
                resume_col, dismiss_col, _ = st.columns([1, 1, 4])
                with resume_col:
                    if st.button("Resume Draft", type="primary"):
                        st.session_state["df"] = draft_df
                        if draft_depts:
                            st.session_state["departments"] = draft_depts
                        st.rerun()
                with dismiss_col:
                    if st.button("Start Fresh"):
                        st.session_state["_draft_dismissed"] = True
                        st.rerun()
        except Exception:
            logger.exception("Error checking for draft")

    entry_method = st.selectbox("Entry Method", ("Upload Image/Screenshot/PDF (OCR)", "Manual Entry", "Upload CSV"))

    data = []

    if entry_method == "Upload CSV":
        csv_files = st.file_uploader("Upload one or more CSVs with cubes (columns: STORE, then each dept)", type=["csv"], accept_multiple_files=True)
        if csv_files:
            combined_csv_df = pd.DataFrame()
            for csv_file in csv_files:
                try:
                    csv_df = pd.read_csv(csv_file)
                    required_cols = ["STORE"] + departments
                    if not all(col in csv_df.columns for col in required_cols):
                        raise ValueError(f"Missing required columns: {', '.join(required_cols)}")
                    combined_csv_df = pd.concat([combined_csv_df, csv_df], ignore_index=True)
                except ValueError:
                    raise
                except Exception:
                    logger.exception("Error loading CSV %s", csv_file.name)
                    st.error(f"Error loading {csv_file.name}. Check the file format.")
                    continue

            if not combined_csv_df.empty:
                try:
                    combined_csv_df = combined_csv_df.groupby("STORE")[departments].sum().reset_index()
                    combined_csv_df["STORE"] = combined_csv_df["STORE"].astype(str)

                    extracted = {row["STORE"]: row for _, row in combined_csv_df.iterrows()}
                    for store in stores:
                        if store in extracted:
                            row = extracted[store]
                        else:
                            row = {"STORE": store}
                            for dept in departments:
                                row[dept] = 0.0
                            st.warning(f"Store {store} not in CSVs; defaulting to 0 cubes.")
                        data.append(row)
                    st.success("CSVs loaded and combined! Review and edit below.")
                except Exception:
                    logger.exception("Error combining CSVs")
                    st.error("Error combining CSV data. Check file contents.")

    elif entry_method == "Upload Image/Screenshot/PDF (OCR)":
        st.info(
            "Upload one image/screenshot/PDF per department. "
            "You can drag-and-drop files, or use your phone's camera to snap a photo."
        )
        ocr_engine = st.selectbox("OCR Engine", ("EasyOCR", "Tesseract"))

        dept_data = {}
        for dept in departments:
            st.subheader(f"Department: {dept}")
            upload_file = st.file_uploader(
                f"Upload image/screenshot/PDF for {dept}",
                type=["jpg", "png", "jpeg", "pdf"],
                key=f"ocr_upload_{dept}",
            )
            if upload_file:
                with st.spinner(f"Processing {dept} upload with {ocr_engine}..."):
                    try:
                        image_bytes = upload_file.read()
                        is_pdf = upload_file.type == "application/pdf"
                        all_results = run_ocr(image_bytes, ocr_engine, is_pdf=is_pdf)

                        store_cubes, skipped = parse_single_dept_ocr(all_results, stores)
                        dept_data[dept] = store_cubes

                        matched = len(store_cubes)
                        total = len(stores)
                        if matched:
                            st.success(f"Matched {matched}/{total} store(s) for {dept}.")
                        else:
                            st.warning(f"No stores matched for {dept}. Check the image quality.")
                        if skipped:
                            with st.expander(f"Unrecognized rows for {dept} ({len(skipped)})"):
                                for row_texts in skipped:
                                    st.text(", ".join(row_texts))
                    except Exception:
                        logger.exception("Error processing %s upload", dept)
                        st.error(f"Error processing {dept} upload. Check the file.")

        if dept_data:
            for store in stores:
                row = {"STORE": store}
                for dept in departments:
                    row[dept] = dept_data.get(dept, {}).get(store, 0.0)
                data.append(row)
            uploaded_depts = [d for d in departments if d in dept_data]
            st.success(f"Combined data from {len(uploaded_depts)} department upload(s). Review below.")

    else:
        for store in stores:
            with st.expander(f"Store {store}"):
                row = {"STORE": store}
                for dept in departments:
                    row[dept] = st.number_input(f"{dept} Cube for {store}", min_value=0, value=0, step=50, key=f"manual_{store}_{dept}")
                data.append(row)

    if data:
        try:
            df = pd.DataFrame(data)
            df["TOTAL"] = df[departments].sum(axis=1)
            df["DIFF"] = df["TOTAL"] - trailer_capacity
            st.session_state["df"] = df
        except Exception:
            logger.exception("Error processing entered data")
            st.error("Error processing data.")

    # Show editable preview with column type constraints
    if not st.session_state["df"].empty:
        st.write("Data Preview (Edit if needed):")
        col_config = {"STORE": st.column_config.TextColumn("STORE", disabled=True)}
        for dept in departments:
            col_config[dept] = st.column_config.NumberColumn(dept, min_value=0.0, format="%.1f")
        col_config["TOTAL"] = st.column_config.NumberColumn("TOTAL", disabled=True, format="%.1f")
        col_config["DIFF"] = st.column_config.NumberColumn("DIFF", disabled=True, format="%.1f")

        edited_df = st.data_editor(
            st.session_state["df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config=col_config,
        )
        # Recompute TOTAL and DIFF from edited department values
        for dept in departments:
            if dept in edited_df.columns:
                edited_df[dept] = pd.to_numeric(edited_df[dept], errors="coerce").fillna(0.0)
        if all(d in edited_df.columns for d in departments):
            edited_df["TOTAL"] = edited_df[departments].sum(axis=1)
            edited_df["DIFF"] = edited_df["TOTAL"] - trailer_capacity
        st.session_state["df"] = edited_df

    # --- Auto-save draft whenever we have data ---
    if not st.session_state["df"].empty:
        _auto_save(st.session_state["df"], departments, entry_date)
        st.caption("Draft auto-saved.")
