"""PDF and Excel export helpers."""

import io
import pandas as pd


def generate_pdf(df, departments, runs_df, date, total_cube, trailer_goal, probable_trailers):
    """Build a one-page landscape PDF peddle sheet and return it as a BytesIO object."""
    from fpdf import FPDF

    pdf = FPDF(orientation="L", format="Letter")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    # --- Title ---
    pdf.set_font("Arial", "B", 14)
    pdf.cell(page_w, 7, txt=f"Peddle Sheet - {date.strftime('%m/%d/%Y')}", ln=1, align="C")
    pdf.ln(2)

    # --- Summary line ---
    pdf.set_font("Arial", "", 8)
    pdf.cell(page_w, 5,
             txt=f"Total Cube: {total_cube:.2f}   |   Trailer Goal: {trailer_goal:.2f}   |   Probable Trailers: {probable_trailers}",
             ln=1, align="C")
    pdf.ln(2)

    # --- Store Data Table ---
    pdf.set_font("Arial", "B", 8)
    pdf.cell(page_w, 5, txt="Store Data", ln=1)

    store_cols = ["STORE"] + departments + ["TOTAL", "DIFF"]
    num_store_cols = len(store_cols)
    col_w = page_w / num_store_cols

    # Header
    pdf.set_font("Arial", "B", 7)
    for col in store_cols:
        pdf.cell(col_w, 5, str(col), border=1, align="C")
    pdf.ln()

    # Rows
    pdf.set_font("Arial", "", 7)
    for _, row in df.iterrows():
        for col in store_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.1f}"
            pdf.cell(col_w, 4, str(val), border=1, align="C")
        pdf.ln()

    pdf.ln(3)

    # --- Peddle Runs Table ---
    if not runs_df.empty:
        pdf.set_font("Arial", "B", 8)
        pdf.cell(page_w, 5, txt="Peddle Runs", ln=1)

        run_cols = [c for c in ["Run", "Time", "Stores", "Carrier", "Total Cube", "Second Trailer", "Fit Note"] if c in runs_df.columns]
        # Give more width to Stores column
        base_w = page_w / (len(run_cols) + 2)  # +2 to give Stores extra share
        run_col_widths = []
        for c in run_cols:
            if c == "Stores":
                run_col_widths.append(base_w * 3)
            else:
                run_col_widths.append(base_w)

        # Header
        pdf.set_font("Arial", "B", 7)
        for c, w in zip(run_cols, run_col_widths):
            pdf.cell(w, 5, str(c), border=1, align="C")
        pdf.ln()

        # Rows
        pdf.set_font("Arial", "", 7)
        for _, row in runs_df.iterrows():
            for c, w in zip(run_cols, run_col_widths):
                val = row.get(c, "")
                if isinstance(val, float):
                    val = f"{val:.1f}"
                pdf.cell(w, 4, str(val), border=1, align="C")
            pdf.ln()

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


def generate_excel(df):
    """Export a DataFrame to an Excel BytesIO buffer, configured to print on one page."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Peddle Data")

        ws = writer.sheets["Peddle Data"]
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 1
        ws.page_setup.paperSize = ws.PAPERSIZE_LETTER

        # Auto-size columns
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = max_len + 2

    buf.seek(0)
    return buf
