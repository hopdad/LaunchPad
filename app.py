import streamlit as st

st.set_page_config(page_title="Peddle Sheet Generator", layout="wide")

import pandas as pd
from math import ceil
import io
from datetime import datetime, timedelta
from utils import preprocess_image, parse_ocr_results
from db_utils import db_connection, save_data_to_db, fetch_historical_data, fetch_prior_peddles, save_actual_peddles
from planning import auto_suggest_runs
from exports import generate_pdf, generate_excel
import streamlit_authenticator as stauth
from pdf2image import convert_from_bytes
from PIL import Image
import easyocr
import pytesseract
import re


def _is_store_number(val: str) -> bool:
    """Return True if *val* looks like a store number (purely numeric, 1-5 digits)."""
    return bool(re.fullmatch(r"\d{1,5}", val))


def _normalize_time(val) -> str:
    """Best-effort convert a cell value to an HH:MM string."""
    if pd.isna(val):
        return "05:00"
    # datetime.time or Timestamp objects
    if hasattr(val, "strftime"):
        return val.strftime("%H:%M")
    s = str(val).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return "05:00"


def _extract_stores_from_file(store_file) -> dict[str, str]:
    """Extract store numbers and ready times from an uploaded CSV/Excel file.

    Finds columns named "Store" (exact, case-insensitive) and columns whose
    name starts with "R-Time", "R- Time", or "Ready Time" (case-insensitive).
    Supports multiple side-by-side tables on the same sheet (each with its
    own Store / R-Time pair).

    Returns an ordered dict of {store_number: ready_time} (e.g. {"32": "17:30"}).
    Stores without a matched ready-time column default to "05:00".
    """
    if store_file.name.endswith(".csv"):
        frames = [pd.read_csv(store_file, header=None)]
    else:
        xls = pd.read_excel(store_file, sheet_name=None, header=None)
        frames = list(xls.values())

    result: dict[str, str] = {}

    for frame in frames:
        # --- Locate the header row (first row containing a "store" cell) ---
        header_idx = None
        for idx, row in frame.iterrows():
            if any(str(v).strip().lower() == "store" for v in row.values):
                header_idx = idx
                break
        if header_idx is None:
            continue

        headers = [str(v).strip() for v in frame.iloc[header_idx].values]

        # --- Identify Store columns and R-Time columns by position ---
        store_cols: list[int] = []
        rtime_cols: list[int] = []
        for i, h in enumerate(headers):
            if h.lower() == "store":
                store_cols.append(i)
            else:
                # Normalise away spaces/dashes to match variants like
                # "R-Time", "R- Time", "Ready Time", "ReadyTime", etc.
                norm = re.sub(r"[\s\-]", "", h.lower())
                if norm.startswith("rtime") or norm.startswith("readytime"):
                    rtime_cols.append(i)

        if not store_cols:
            continue

        # Pair each Store column with the nearest R-Time column to its right
        pairs: list[tuple[int, int | None]] = []
        for sc in store_cols:
            matched = None
            for rc in rtime_cols:
                if rc > sc:
                    matched = rc
                    break
            pairs.append((sc, matched))

        # --- Extract data rows below the header ---
        data_rows = frame.iloc[header_idx + 1:]
        for sc, rc in pairs:
            for _, row in data_rows.iterrows():
                raw = str(row.iloc[sc]).strip()
                # Normalise pandas float representation "100.0" -> "100"
                if re.fullmatch(r"\d+\.0", raw):
                    raw = raw.split(".")[0]
                if not _is_store_number(raw):
                    continue
                if raw in result:
                    continue  # first occurrence wins
                ready_time = _normalize_time(row.iloc[rc]) if rc is not None else "05:00"
                result[raw] = ready_time

    return result


# Auth Setup (hardcoded for now; use secrets.toml in prod)
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
            "role": "admin"  # Same as admin for History access
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="peddle_app",
    cookie_key="auth_key",
    cookie_expiry_days=30
)

# Login Form
try:
    authenticator.login(location='main')
except Exception as e:
    st.error(e)

authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')
username = st.session_state.get('username')

if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()

# Logged In Successfully
st.success(f"Welcome, {name}! ({credentials['usernames'][username]['role'].capitalize()})")

# Get User Role
user_role = credentials["usernames"][username]["role"]

# App Title
st.title("Peddle Sheet Generator")
st.write("Streamlit-powered web app for daily peddle planning. Access via browser—no installs needed.")

# Hardcoded cube per pallet
pallet_cube = 50

# Initialize session state defaults
if "departments" not in st.session_state:
    st.session_state["departments"] = ["882", "883", "MB"]
if "stores" not in st.session_state:
    st.session_state["stores"] = []
if "store_ready_times" not in st.session_state:
    st.session_state["store_ready_times"] = {}
if "trailer_capacity" not in st.session_state:
    st.session_state["trailer_capacity"] = 1600
if "fluff" not in st.session_state:
    st.session_state["fluff"] = 200
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
if "runs_df" not in st.session_state:
    st.session_state["runs_df"] = pd.DataFrame()

# Sidebar navigation
pages = ["Settings", "Configure", "Data Entry", "Summary & Planning", "Actuals", "Outputs"]
if user_role == "admin":
    pages.append("History")

with st.sidebar:
    st.image("logo.jpg", use_container_width=True)
    st.divider()
    selected_page = st.radio("Navigation", pages, label_visibility="collapsed")
    st.divider()
    authenticator.logout(button_name="Logout", location="main")

# --- Settings Page ---
if selected_page == "Settings":
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

# --- Configure Page ---
if selected_page == "Configure":
    st.header("Configure Stores")

    time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

    # --- Current lineup summary (always visible, read-only) ---
    if st.session_state["stores"]:
        st.subheader(f"Current Lineup ({len(st.session_state['stores'])} stores)")
        st.info(", ".join(st.session_state["stores"]))
    else:
        st.warning("No stores configured yet. Use the editor or import a file below.")

    # --- Edit lineup manually (only when toggled) ---
    edit_lineup = st.checkbox("Edit Store Lineup", key="edit_lineup_toggle")
    if edit_lineup:
        st.caption("Add, remove, or reorder stores and ready times.")
        if st.session_state["stores"]:
            ready_times = [
                st.session_state["store_ready_times"].get(s, "05:00")
                for s in st.session_state["stores"]
            ]
            stores_edit_df = pd.DataFrame({"Store": st.session_state["stores"], "Ready Time": ready_times})
        else:
            stores_edit_df = pd.DataFrame({"Store": pd.Series(dtype="str"), "Ready Time": pd.Series(dtype="str")})

        edited_stores_df = st.data_editor(
            stores_edit_df,
            num_rows="dynamic",
            use_container_width=True,
            column_order=["Store", "Ready Time"],
            column_config={
                "Ready Time": st.column_config.SelectboxColumn(
                    "Ready Time",
                    options=time_options,
                    default="05:00",
                    required=True,
                )
            },
            key="stores_data_editor",
        )
        edited_stores = [s.strip() for s in edited_stores_df["Store"].astype(str).tolist() if s.strip() and s.strip() != "nan"]
        ready_time_list = edited_stores_df["Ready Time"].astype(str).tolist()
        store_names = edited_stores_df["Store"].astype(str).tolist()
        st.session_state["store_ready_times"] = {
            s.strip(): t for s, t in zip(store_names, ready_time_list)
            if s.strip() and s.strip() != "nan"
        }
        st.session_state["stores"] = edited_stores

    # --- Import from file (with diff review) ---
    with st.expander("Import stores from file"):
        store_file = st.file_uploader("Upload stores (CSV/Excel)", type=["csv", "xlsx"], key="store_file_uploader")
        if store_file:
            imported = _extract_stores_from_file(store_file)
            if imported:
                current_set = set(st.session_state["stores"])
                incoming_set = set(imported.keys())

                adds = sorted(incoming_set - current_set)
                removes = sorted(current_set - incoming_set)
                keeps = sorted(current_set & incoming_set)

                if not adds and not removes:
                    st.success("Uploaded store list matches current lineup. No changes needed.")
                else:
                    st.write("**Review proposed changes:**")

                    # Track which file we last diffed so checkboxes reset on new upload
                    file_id = f"{store_file.name}_{store_file.size}"
                    if st.session_state.get("_import_file_id") != file_id:
                        st.session_state["_import_file_id"] = file_id
                        st.session_state["_pending_adds"] = {s: True for s in adds}
                        st.session_state["_pending_removes"] = {s: True for s in removes}
                        st.session_state["_imported_ready_times"] = dict(imported)

                    if adds:
                        st.markdown(f"**Stores to ADD ({len(adds)}):**")
                        for s in adds:
                            rt = imported.get(s, "05:00")
                            st.session_state["_pending_adds"][s] = st.checkbox(
                                f"Add store {s} (ready {rt})",
                                value=st.session_state.get("_pending_adds", {}).get(s, True),
                                key=f"import_add_{s}",
                            )

                    if removes:
                        st.markdown(f"**Stores to REMOVE ({len(removes)}):**")
                        for s in removes:
                            st.session_state["_pending_removes"][s] = st.checkbox(
                                f"Remove store {s}",
                                value=st.session_state.get("_pending_removes", {}).get(s, True),
                                key=f"import_remove_{s}",
                            )

                    if keeps:
                        st.caption(f"Unchanged stores ({len(keeps)}): {', '.join(keeps)}")

                    if st.button("Apply Changes", key="apply_store_import"):
                        imported_times = st.session_state.get("_imported_ready_times", {})
                        new_stores = list(st.session_state["stores"])
                        # Apply adds
                        for s, checked in st.session_state.get("_pending_adds", {}).items():
                            if checked and s not in new_stores:
                                new_stores.append(s)
                        # Apply removes
                        for s, checked in st.session_state.get("_pending_removes", {}).items():
                            if checked and s in new_stores:
                                new_stores.remove(s)
                        # Set ready times from the file for new stores; update kept stores too
                        for s in new_stores:
                            if s in imported_times:
                                st.session_state["store_ready_times"][s] = imported_times[s]
                            elif s not in st.session_state["store_ready_times"]:
                                st.session_state["store_ready_times"][s] = "05:00"
                        # Clean up ready times for removed stores
                        st.session_state["store_ready_times"] = {
                            s: t for s, t in st.session_state["store_ready_times"].items()
                            if s in new_stores
                        }
                        added = sum(1 for v in st.session_state.get("_pending_adds", {}).values() if v)
                        removed = sum(1 for v in st.session_state.get("_pending_removes", {}).values() if v)
                        st.session_state["stores"] = new_stores
                        # Clean up temp state
                        for k in ("_import_file_id", "_pending_adds", "_pending_removes", "_imported_ready_times"):
                            st.session_state.pop(k, None)
                        st.success(f"Applied: {added} added, {removed} removed. {len(new_stores)} stores total.")
                        st.rerun()
            else:
                st.warning("No store numbers found. Make sure the file has a column named **Store**.")

    if not st.session_state["stores"]:
        st.warning("Add at least one store to get started.")

# Read settings from session state for all other pages
departments = st.session_state["departments"]
stores = st.session_state["stores"]
trailer_capacity = st.session_state["trailer_capacity"]
fluff = st.session_state["fluff"]
df = st.session_state["df"]
runs_df = st.session_state["runs_df"]

# --- Data Entry Page ---
if selected_page == "Data Entry":
    st.header("Cube Data Entry")
    entry_method = st.selectbox("Entry Method", ("Manual Entry", "Upload Image/Screenshot/PDF (OCR)", "Upload CSV"))

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
                except Exception as e:
                    st.error(f"Error loading {csv_file.name}: {e}")
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
                except Exception as e:
                    st.error(f"Error combining CSVs: {e}")

    elif entry_method == "Upload Image/Screenshot/PDF (OCR)":
        upload_files = st.file_uploader("Upload one or more images/screenshots or PDFs", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)
        ocr_engine = st.selectbox("OCR Engine", ("EasyOCR", "Tesseract"))
        if upload_files:
            combined_data = []
            with st.spinner("Processing uploads with OCR..."):
                for upload_file in upload_files:
                    try:
                        image_bytes = upload_file.read()
                        if upload_file.type == "application/pdf":
                            pdf_pages = convert_from_bytes(image_bytes)
                            for i, page in enumerate(pdf_pages):
                                st.info(f"Processing page {i+1} of {upload_file.name}")
                                preprocessed_img = preprocess_image(page)
                                if ocr_engine == "EasyOCR":
                                    reader = easyocr.Reader(['en'])
                                    results = reader.readtext(preprocessed_img)
                                else:
                                    tess_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
                                    results = []
                                    for j in range(len(tess_data['text'])):
                                        if float(tess_data['conf'][j]) > 0:
                                            x, y, w, h = tess_data['left'][j], tess_data['top'][j], tess_data['width'][j], tess_data['height'][j]
                                            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                                            text = tess_data['text'][j]
                                            conf = float(tess_data['conf'][j]) / 100.0
                                            results.append((bbox, text, conf))

                                extracted_texts = [text for _, text, _ in results]
                                st.write(f"Extracted Texts Preview for {upload_file.name} page {i+1}:")
                                st.text(", ".join(extracted_texts[:100]))

                                page_data = parse_ocr_results(results, departments, stores)
                                combined_data.extend(page_data)
                        else:
                            page_img = Image.open(io.BytesIO(image_bytes))
                            preprocessed_img = preprocess_image(page_img)
                            if ocr_engine == "EasyOCR":
                                reader = easyocr.Reader(['en'])
                                results = reader.readtext(preprocessed_img)
                            else:
                                tess_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
                                results = []
                                for j in range(len(tess_data['text'])):
                                    if float(tess_data['conf'][j]) > 0:
                                        x, y, w, h = tess_data['left'][j], tess_data['top'][j], tess_data['width'][j], tess_data['height'][j]
                                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                                        text = tess_data['text'][j]
                                        conf = float(tess_data['conf'][j]) / 100.0
                                        results.append((bbox, text, conf))

                            extracted_texts = [text for _, text, _ in results]
                            st.write(f"Extracted Texts Preview for {upload_file.name}:")
                            st.text(", ".join(extracted_texts[:100]))

                            file_data = parse_ocr_results(results, departments, stores)
                            combined_data.extend(file_data)
                    except Exception as e:
                        st.error(f"Error processing {upload_file.name}: {e}")
                        continue

                if combined_data:
                    try:
                        combined_df = pd.DataFrame(combined_data)
                        if not combined_df.empty:
                            combined_df = combined_df.groupby("STORE")[departments].sum().reset_index()
                            combined_df["STORE"] = combined_df["STORE"].astype(str)
                            extracted = {row["STORE"]: row for _, row in combined_df.iterrows()}
                            data = []
                            for store in stores:
                                if store in extracted:
                                    row = extracted[store]
                                else:
                                    row = {"STORE": store}
                                    for dept in departments:
                                        row[dept] = 0.0
                                    st.warning(f"Store {store} not in uploads; defaulting to 0 cubes.")
                                data.append(row)
                        st.success(f"Auto-filled from {len(upload_files)} files using {ocr_engine}! Edit below if needed.")
                    except Exception as e:
                        st.error(f"Error combining OCR data: {e}")

    else:
        for store in stores:
            with st.expander(f"Store {store}"):
                row = {"STORE": store}
                for dept in departments:
                    row[dept] = st.number_input(f"{dept} Cube for {store}", min_value=0.0, value=0.0, key=f"manual_{store}_{dept}")
                data.append(row)

    if data:
        try:
            df = pd.DataFrame(data)
            df["TOTAL"] = df[departments].sum(axis=1)
            df["DIFF"] = df["TOTAL"] - trailer_capacity
            st.write("Data Preview (Edit if needed):")
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            df = edited_df
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Error processing data: {e}")

# --- Summary & Planning Page ---
if selected_page == "Summary & Planning":
    if not df.empty:
        try:
            total_cube = df["TOTAL"].sum()
            total_cube_with_fluff = total_cube + fluff
            pallet_count = ceil(total_cube / pallet_cube)
            trailer_goal = total_cube_with_fluff / trailer_capacity
            probable_trailers = ceil(trailer_goal * 1.1)

            st.header("Summary")
            st.write(f"**Total Cube:** {total_cube:.2f}")
            st.write(f"**Fluff:** {fluff}")
            st.write(f"**Total Cube (with fluff):** {total_cube_with_fluff:.2f}")
            st.write(f"**Pallet Count:** {pallet_count}")
            st.write(f"**Trailer Goal:** {trailer_goal:.2f}")
            st.write(f"**Probable Trailers:** {probable_trailers}")

            st.header("Peddle Run Planning")
            auto_suggest = st.checkbox("Auto-suggest Peddle Runs (greedy packing)")

            runs_df = pd.DataFrame()
            if auto_suggest:
                runs = auto_suggest_runs(df, trailer_capacity)
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
                        available_stores = [s for s in available_stores if s not in selected_stores]

                runs_df = pd.DataFrame(runs)
                st.write("Manual Runs:")
                st.dataframe(runs_df)

            st.session_state["runs_df"] = runs_df
        except Exception as e:
            st.error(f"Error in summary or planning: {e}")
    else:
        st.info("No data yet. Go to Data Entry first.")

# --- Actuals Page ---
if selected_page == "Actuals":
    st.header("Enter Actual Peddles (From Prior Day)")
    prior_date = (datetime.today() - timedelta(days=1)).date()
    st.write(f"Showing projections from {prior_date.strftime('%Y-%m-%d')}. Enter what actually happened.")

    try:
        with db_connection() as (conn, c):
            prior_peddles_df = fetch_prior_peddles(prior_date, c)

            if not prior_peddles_df.empty:
                st.dataframe(prior_peddles_df)

                # Form to input actuals
                actuals = []
                for _, row in prior_peddles_df.iterrows():
                    with st.expander(f"Run {row['Run']} ({row['Stores']}) - Projected: {row['Total Cube']} cube, Second Trailer: {row['Second Trailer']}"):
                        actual_trailers = st.number_input(f"Actual Trailers Used", min_value=1, value=1, key=f"actual_trailers_{row['Run']}")
                        actual_notes = st.text_input(f"Notes/Variances", key=f"actual_notes_{row['Run']}")
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
    except Exception as e:
        st.error(f"Error loading prior data: {e}")

# --- Outputs Page ---
if selected_page == "Outputs":
    if not df.empty:
        date = st.date_input("Sheet Date", datetime.today())
        total_cube = df["TOTAL"].sum()
        total_cube_with_fluff = total_cube + fluff
        pallet_count = ceil(total_cube / pallet_cube)
        trailer_goal = total_cube_with_fluff / trailer_capacity
        probable_trailers = ceil(trailer_goal * 1.1)
        if st.button("Generate PDF Sheet"):
            try:
                pdf_output = generate_pdf(df, departments, runs_df, date, total_cube, trailer_goal, probable_trailers)
                st.download_button("Download Peddle Sheet PDF", pdf_output, file_name=f"ped_sheet_{date.strftime('%Y%m%d')}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

        try:
            excel_output = generate_excel(df)
            st.download_button("Download Data as Excel", excel_output, file_name="ped_data.xlsx")
        except Exception as e:
            st.error(f"Error generating Excel: {e}")

        if st.button("Save to DB"):
            try:
                with db_connection() as (conn, c):
                    save_data_to_db(df, departments, runs_df, date, conn, c)
                    st.success("Data saved to database!")
            except Exception as e:
                st.error(f"Error saving to DB: {e}")
    else:
        st.info("No data yet. Go to Data Entry first.")

# --- History Page (admin only) ---
if selected_page == "History" and user_role == "admin":
    st.header("Historical Data")
    query_date = st.date_input("View Data For", datetime.today())
    try:
        with db_connection() as (conn, c):
            summaries, runs, totals = fetch_historical_data(query_date, c)

            if not summaries.empty:
                st.subheader("Store Summaries")
                st.dataframe(summaries)

            if not runs.empty:
                st.subheader("Peddle Runs")
                st.dataframe(runs)

            if not totals.empty:
                st.subheader("Cube Trends")
                st.line_chart(totals.set_index('Date'))
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
