# app.py
import streamlit as st
import pandas as pd
from math import ceil
import io
from fpdf import FPDF
from datetime import datetime
from utils import preprocess_image, parse_ocr_results
from db_utils import get_db_connection, save_data_to_db, fetch_historical_data
import streamlit as st

password = st.text_input("Enter Password", type="password")
if password != "your_secret_password":  # Replace with your chosen password
    st.error("Incorrect password.")
    st.stop()


# App Title
st.title("Peddle Sheet Generator")
st.write("Streamlit-powered web app for daily peddle planning. Access via browser—no installs needed.")

# Sidebar Config
with st.sidebar:
    st.header("Configuration")
    departments_input = st.text_input("Departments (e.g., 882,883,MB)", value="882,883,MB")
    try:
        departments = [d.strip() for d in departments_input.split(",") if d.strip()]
        if not departments:
            raise ValueError("No departments provided.")
    except Exception as e:
        st.error(f"Error parsing departments: {e}")
        departments = []

    store_input_method = st.radio("Add Stores", ("Manual", "Upload List"))
    stores = []
    if store_input_method == "Upload List":
        store_file = st.file_uploader("Upload stores (CSV/Excel)", type=["csv", "xlsx"])
        if store_file:
            try:
                stores_df = pd.read_csv(store_file) if store_file.name.endswith(".csv") else pd.read_excel(store_file)
                stores = stores_df.iloc[:, 0].astype(str).tolist()
            except Exception as e:
                st.error(f"Error loading store list: {e}")
    else:
        stores_text = st.text_area("Stores (one per line)")
        stores = [s.strip() for s in stores_text.split("\n") if s.strip()]
    
    trailer_capacity = st.number_input("Trailer Capacity", value=1600)
    pallet_cube = st.number_input("Cube per Pallet", value=50)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Entry", "Summary & Planning", "Outputs", "History"])

with tab1:
    st.header("Cube Data Entry")
    entry_method = st.selectbox("Entry Method", ("Manual Entry", "Upload Image/Screenshot/PDF (OCR)", "Upload CSV"))
    
    data = []
    df = pd.DataFrame()
    
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
                    combined_csv_df = combined_csv_df.groupby("STORE", as_index=False).sum(numeric_only=True)
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
                            pages = pdf2image.convert_from_bytes(image_bytes)
                            for i, page in enumerate(pages):
                                st.info(f"Processing page {i+1} of {upload_file.name}")
                                preprocessed_img = preprocess_image(page)
                                if ocr_engine == "EasyOCR":
                                    reader = easyocr.Reader(['en'])
                                    results = reader.readtext(preprocessed_img)
                                else:
                                    tess_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
                                    results = []
                                    for j in range(len(tess_data['text'])):
                                        if int(tess_data['conf'][j]) > 0:
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
                                    if int(tess_data['conf'][j]) > 0:
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
                            combined_df = combined_df.groupby("STORE", as_index=False).sum(numeric_only=True)
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
                    row[dept] = st.number_input(f"{dept} Cube for {store}", min_value=0.0, value=0.0)
                data.append(row)
    
    if data:
        try:
            df = pd.DataFrame(data)
            df["TOTAL"] = df[departments].sum(axis=1)
            df["DIFF"] = df["TOTAL"] - trailer_capacity
            st.write("Data Preview (Edit if needed):")
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            df = edited_df
        except Exception as e:
            st.error(f"Error processing data: {e}")

with tab2:
    if not df.empty:
        try:
            total_cube = df["TOTAL"].sum()
            pallet_count = ceil(total_cube / pallet_cube)
            trailer_goal = total_cube / trailer_capacity
            probable_trailers = ceil(trailer_goal * 1.1)
            
            st.header("Summary")
            st.write(f"**Total Cube:** {total_cube:.2f}")
            st.write(f"**Pallet Count:** {pallet_count}")
            st.write(f"**Trailer Goal:** {trailer_goal:.2f}")
            st.write(f"**Probable Trailers:** {probable_trailers}")
            
            st.header("Peddle Run Planning")
            auto_suggest = st.checkbox("Auto-suggest Peddle Runs (greedy packing)")
            
            runs_df = pd.DataFrame()
            if auto_suggest:
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
        except Exception as e:
            st.error(f"Error in summary or planning: {e}")

with tab3:
    if not df.empty:
        date = st.date_input("Sheet Date", datetime.today())
        if st.button("Generate PDF Sheet"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=10)
                
                pdf.cell(200, 10, txt=f"Peddle Sheet - {date.strftime('%m/%d/%Y')}", ln=1, align="C")
                
                pdf.cell(200, 10, txt="Store Data:", ln=1)
                for _, row in df.iterrows():
                    pdf.cell(200, 10, txt=f"STORE {row['STORE']}: {', '.join([f'{dept}: {row[dept]}' for dept in departments])}, TOTAL: {row['TOTAL']}, DIFF: {row['DIFF']}", ln=1)
                
                pdf.cell(200, 10, txt=f"Total Cube: {total_cube:.2f}, Trailer Goal: {trailer_goal:.2f}, Probable: {probable_trailers}", ln=1)
                
                pdf.cell(200, 10, txt="Peddle Runs:", ln=1)
                for _, run in runs_df.iterrows():
                    pdf.cell(200, 10, txt=f"({run['Time']}) {run['Stores']}, Carrier: {run['Carrier']}, Cube: {run['Total Cube']}, Second: {run['Second Trailer']}, Note: {run['Fit Note']}", ln=1)
                
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                pdf_output.seek(0)
                st.download_button("Download Peddle Sheet PDF", pdf_output, file_name=f"ped_sheet_{date.strftime('%Y%m%d')}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
        
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            st.download_button("Download Data as Excel", output, file_name="ped_data.xlsx")
        except Exception as e:
            st.error(f"Error generating Excel: {e}")
        
        if st.button("Save to DB"):
            try:
                conn, c = get_db_connection()
                save_data_to_db(df, departments, runs_df, date, conn, c)
                st.success("Data saved to database!")
            except Exception as e:
                st.error(f"Error saving to DB: {e}")
            finally:
                conn.close()

with tab4:
    st.header("Historical Data")
    query_date = st.date_input("View Data For", datetime.today())
    try:
        conn, c = get_db_connection()
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
    finally:
        conn.close()
