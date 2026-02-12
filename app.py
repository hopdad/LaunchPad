import streamlit as st
import pandas as pd
from math import ceil
import io
from fpdf import FPDF  # For PDFs
from datetime import datetime
import easyocr  # For EasyOCR
import pytesseract  # For Tesseract OCR
from PIL import Image  # For handling images from PDF
import pdf2image  # For converting PDF to images
import numpy as np  # For coordinate calculations and images
import cv2  # For OpenCV preprocessing
import sqlite3  # For database storage

# Initialize SQLite DB for persistent data storage
conn = sqlite3.connect('peddle_data.db', check_same_thread=False)  # Allow multi-thread access if needed
c = conn.cursor()

# Create tables if not exists (normalized for dynamic departments)
c.execute('''CREATE TABLE IF NOT EXISTS store_summaries
             (date TEXT, store TEXT, total REAL, diff REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS dept_cubes
             (date TEXT, store TEXT, dept TEXT, cube REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS peddle_runs
             (date TEXT, run INTEGER, time TEXT, stores TEXT, carrier TEXT, total_cube REAL, second_trailer TEXT, fit_note TEXT)''')
conn.commit()

# Function to preprocess image with OpenCV (now with deskewing)
def preprocess_image(img):
    # Assume img is np.array or PIL.Image; convert to np if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Deskewing: Detect edges and find skew angle
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Median angle for skew
        skew_angle = np.median(angles)
        if abs(skew_angle) < 45:  # Assume text lines are near horizontal
            # Rotate to correct skew
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            deskewed = cv2.warpAffine(blurred, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed = blurred
    else:
        deskewed = blurred
    
    # Apply adaptive thresholding to binarize (enhance text)
    thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Optional: Dilate to connect text components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated  # Return preprocessed image as np.array

# Function to parse OCR results into structured data (advanced: reconstruct table-like structure)
def parse_ocr_results(results, departments, stores):
    # results: list of (bbox, text, conf) where bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or equivalent
    # Normalize to (bbox, text, conf)
    sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))  # Top-left y, then x
    
    # Group into "rows" based on y-overlap (threshold for vertical alignment)
    rows = []
    current_row = []
    prev_y = None
    y_threshold = 30  # Increased slightly for better grouping; adjust as needed
    for bbox, text, conf in sorted_results:
        y = bbox[0][1]  # Top-left y
        if prev_y is None or abs(y - prev_y) < y_threshold:
            current_row.append((bbox, text, conf))
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda item: item[0][0][0]))  # Sort row by x
            current_row = [(bbox, text, conf)]
        prev_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda item: item[0][0][0]))
    
    # Now, parse rows into store data
    data = []
    known_stores = set(stores)
    dept_map = {dept.lower(): dept for dept in departments}  # Case-insensitive
    current_store = None
    row_dict = {}
    
    for row in rows:
        texts = [item[1].strip() for item in row]  # Extract texts in x-order
        if not texts:
            continue
        
        # Detect store number (usually first in row, numeric 1-4 digits)
        potential_store = texts[0]
        if potential_store.isdigit() and 1 <= len(potential_store) <= 4 and potential_store in known_stores:
            if current_store:  # Save previous row if any
                data.append(row_dict)
            current_store = potential_store
            row_dict = {"STORE": current_store}
            # Next values are dept cubes
            dept_idx = 0
            for text in texts[1:]:
                if text.replace('.', '').isdigit() and dept_idx < len(departments):  # Cube value
                    row_dict[departments[dept_idx]] = float(text)
                    dept_idx += 1
                elif text.lower() in dept_map:  # Dept label, skip
                    continue
                # Ignore TOTAL/DIFF for now; calculate later
        elif current_store:  # Continuation or other
            # If no store, perhaps append to previous (multi-line rows), but assume single-line per store
            pass
    
    # Append last row
    if current_store:
        data.append(row_dict)
    
    # Handle mismatches: Map to configured stores, fill missing with 0
    extracted = {d["STORE"]: d for d in data if "STORE" in d}
    final_data = []
    for store in stores:
        if store in extracted:
            row = extracted[store]
        else:
            row = {"STORE": store}
            for dept in departments:
                row[dept] = 0.0
            st.warning(f"Store {store} not found in scan; defaulting to 0 cubes.")
        final_data.append(row)
    
    return final_data

# App title and intro
st.title("Peddle Sheet Generator")
st.write("Web-hosted tool to create daily peddle sheets. Configure stores/departments, enter cube data from printed sheets (manual or via scanned image upload with advanced OCR using EasyOCR or Tesseract, and OpenCV preprocessing including deskewing), calculate trailer projections, and generate loader sheets with optional auto-suggested peddle runs. Data can now be saved to a local SQLite DB for accumulation and future features.")

# Step 0: Configuration (make it dynamic)
st.header("Configuration")
departments = st.text_input("Departments (comma-separated, e.g., 882,883,MB)", value="882,883,MB").split(",")
departments = [d.strip() for d in departments]

store_input_method = st.radio("How to add stores?", ("Manual List", "Upload Store List (CSV/Excel)"))
if store_input_method == "Upload Store List (CSV/Excel)":
    store_file = st.file_uploader("Upload store numbers (one per row)", type=["csv", "xlsx"])
    if store_file:
        if store_file.name.endswith(".csv"):
            stores_df = pd.read_csv(store_file, header=None)
        else:
            stores_df = pd.read_excel(store_file, header=None)
        stores = stores_df.iloc[:, 0].astype(str).tolist()
        st.write(f"Loaded {len(stores)} stores: {', '.join(stores[:5])}...")
else:
    stores_text = st.text_area("Enter store numbers (one per line, in desired order)")
    stores = [s.strip() for s in stores_text.split("\n") if s.strip()]

trailer_capacity = st.number_input("Trailer Cube Capacity", value=1600)
pallet_cube = st.number_input("Cube per Pallet", value=50)

# Step 1: Data Entry
st.header("Cube Data Entry")
entry_method = st.radio("Entry Method", ("Manual Entry", "Upload Scanned Image"))

if entry_method == "Upload Scanned Image":
    image_file = st.file_uploader("Upload scanned image of cube sheet", type=["jpg", "png", "jpeg", "pdf"])
    ocr_engine = st.selectbox("OCR Engine", ("EasyOCR", "Tesseract"))
    if image_file:
        image_bytes = image_file.read()
        file_type = image_file.type
        
        # Handle PDF: Convert to images
        pages = pdf2image.convert_from_bytes(image_bytes)
        if len(pages) > 1:
            st.info(f"PDF has {len(pages)} pages; processing first page only. For multi-page, extend code as needed.")
        page_img = pages[0]  # Take first page
        
        # Preprocess
        preprocessed_img = preprocess_image(page_img)
        
        # OCR based on engine
        if ocr_engine == "EasyOCR":
            reader = easyocr.Reader(['en'])
            results = reader.readtext(preprocessed_img)
            # Results: [(bbox, text, prob), ...]
        else:  # Tesseract
            tess_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
            results = []
            for i in range(len(tess_data['text'])):
                if int(tess_data['conf'][i]) > 0:  # Filter low conf
                    x, y, w, h = tess_data['left'][i], tess_data['top'][i], tess_data['width'][i], tess_data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    text = tess_data['text'][i]
                    conf = float(tess_data['conf'][i]) / 100.0
                    results.append((bbox, text, conf))
        
        # Preview raw results
        extracted_texts = [text for _, text, _ in results]
        st.write("Extracted Texts Preview:")
        st.text(", ".join(extracted_texts))
        
        # Advanced parsing
        data = parse_ocr_results(results, departments, stores)
        
        if data:
            st.success(f"Auto-filled from scan using {ocr_engine} with OpenCV preprocessing! Review and edit below.")
else:
    # Manual entry (as before)
    data = []
    for store in stores:
        with st.expander(f"Store {store}"):
            row = {"STORE": store}
            for dept in departments:
                row[dept] = st.number_input(f"{dept} Cube for {store}", min_value=0.0, value=0.0)
            data.append(row)

df = pd.DataFrame(data)
if not df.empty:
    # Calculate totals per store
    df["TOTAL"] = df[departments].sum(axis=1)
    df["DIFF"] = df["TOTAL"] - trailer_capacity

    st.write("Data Preview (Edit if needed):")
    edited_df = st.data_editor(df, num_rows="dynamic")  # Allow editing
    df = edited_df  # Update with edits

    # Grand totals
    total_cube = df["TOTAL"].sum()
    pallet_count = ceil(total_cube / pallet_cube)
    trailer_goal = total_cube / trailer_capacity
    probable_trailers = ceil(trailer_goal * 1.1)  # Example adjustment; tweak as needed

    st.header("Summary")
    st.write(f"**Total Cube:** {total_cube:.2f}")
    st.write(f"**Pallet Count:** {pallet_count} (rounded up)")
    st.write(f"**Trailer Goal:** {trailer_goal:.2f}")
    st.write(f"**Probable Trailers:** {probable_trailers} (rounded up with buffer)")

# Step 2: Peddle Run Planning
runs_df = pd.DataFrame()  # Initialize
if not df.empty:
    st.header("Peddle Run Planning")
    auto_suggest = st.checkbox("Auto-suggest Peddle Runs (greedy packing)")
    
    if auto_suggest:
        # Simple bin-packing: Sort by total descending, pack into bins <= capacity
        sorted_df = df.sort_values("TOTAL", ascending=False).copy()
        runs = []
        current_run = []
        current_cube = 0
        for _, row in sorted_df.iterrows():
            if current_cube + row["TOTAL"] <= trailer_capacity:
                current_run.append(row["STORE"])
                current_cube += row["TOTAL"]
            else:
                if current_run:
                    runs.append({"Stores": "/".join(current_run), "Total Cube": current_cube, "Second Trailer": "No" if current_cube <= trailer_capacity else "Yes"})
                current_run = [row["STORE"]]
                current_cube = row["TOTAL"]
        if current_run:
            runs.append({"Stores": "/".join(current_run), "Total Cube": current_cube, "Second Trailer": "No" if current_cube <= trailer_capacity else "Yes"})
        
        runs_df = pd.DataFrame(runs)
        runs_df["Run"] = runs_df.index + 1
        runs_df["Time"] = ""  # Placeholder
        runs_df["Carrier"] = "Standard"  # Default
        runs_df["Fit Note"] = "FITS" if runs_df["Second Trailer"] == "No" else ""
        st.write("Auto-Suggested Runs:")
        st.dataframe(runs_df)
    else:
        # Manual grouping
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

# Step 3: Generate Outputs
if not df.empty:
    st.header("Generate Outputs")
    date = st.date_input("Sheet Date", value=datetime.today())
    if st.button("Generate PDF Sheet"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        
        # Header
        pdf.cell(200, 10, txt=f"Peddle Sheet - {date.strftime('%m/%d/%Y')}", ln=1, align="C")
        
        # Data Table
        pdf.cell(200, 10, txt="Store Data:", ln=1)
        for _, row in df.iterrows():
            pdf.cell(200, 10, txt=f"STORE {row['STORE']}: {', '.join([f'{dept}: {row[dept]}' for dept in departments])}, TOTAL: {row['TOTAL']}, DIFF: {row['DIFF']}", ln=1)
        
        # Summaries
        pdf.cell(200, 10, txt=f"Total Cube: {total_cube:.2f}, Trailer Goal: {trailer_goal:.2f}, Probable: {probable_trailers}", ln=1)
        
        # Peddle Runs
        pdf.cell(200, 10, txt="Peddle Runs:", ln=1)
        for _, run in runs_df.iterrows():
            pdf.cell(200, 10, txt=f"({run['Time']}) {run['Stores']}, Carrier: {run['Carrier']}, Cube: {run['Total Cube']}, Second: {run['Second Trailer']}, Note: {run['Fit Note']}", ln=1)
        
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button("Download Peddle Sheet PDF", pdf_output, file_name=f"ped_sheet_{date.strftime('%Y%m%d')}.pdf", mime="application/pdf")
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    st.download_button("Download Data as Excel", output, file_name="ped_data.xlsx")

    # Step 4: Save Data to DB (for accumulation)
    if st.button("Save Data to Database"):
        date_str = date.strftime('%Y-%m-%d')
        
        # Save store summaries and dept cubes
        for _, row in df.iterrows():
            c.execute("INSERT INTO store_summaries VALUES (?, ?, ?, ?)",
                      (date_str, row['STORE'], row['TOTAL'], row['DIFF']))
            for dept in departments:
                if dept in row:
                    c.execute("INSERT INTO dept_cubes VALUES (?, ?, ?, ?)",
                              (date_str, row['STORE'], dept, row[dept]))
        
        # Save peddle runs
        for _, run in runs_df.iterrows():
            c.execute("INSERT INTO peddle_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (date_str, run['Run'], run.get('Time', ''), run['Stores'], run['Carrier'], run['Total Cube'], run['Second Trailer'], run.get('Fit Note', '')))
        
        conn.commit()
        st.success("Data saved to database! You can now accumulate data for future features like historical analytics.")
