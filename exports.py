"""PDF and Excel export helpers."""

import io
import pandas as pd


def generate_pdf(df, departments, runs_df, date, total_cube, trailer_goal, probable_trailers):
    """Build a PDF peddle sheet and return it as a BytesIO object.

    Requires the ``fpdf2`` package (``pip install fpdf2``).

    Parameters
    ----------
    df : pd.DataFrame
        Store-level data with STORE, departments, TOTAL, DIFF columns.
    departments : list[str]
        Department names.
    runs_df : pd.DataFrame
        Peddle runs with Time, Stores, Carrier, Total Cube, Second Trailer, Fit Note.
    date : datetime.date
        Sheet date.
    total_cube, trailer_goal, probable_trailers
        Summary metrics.

    Returns
    -------
    io.BytesIO
        PDF file bytes, seeked to 0.
    """
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    pdf.cell(200, 10, txt=f"Peddle Sheet - {date.strftime('%m/%d/%Y')}", ln=1, align="C")

    pdf.cell(200, 10, txt="Store Data:", ln=1)
    for _, row in df.iterrows():
        dept_vals = ", ".join(f"{dept}: {row[dept]}" for dept in departments)
        pdf.cell(200, 10, txt=f"STORE {row['STORE']}: {dept_vals}, TOTAL: {row['TOTAL']}, DIFF: {row['DIFF']}", ln=1)

    pdf.cell(200, 10, txt=f"Total Cube: {total_cube:.2f}, Trailer Goal: {trailer_goal:.2f}, Probable: {probable_trailers}", ln=1)

    pdf.cell(200, 10, txt="Peddle Runs:", ln=1)
    for _, run in runs_df.iterrows():
        pdf.cell(200, 10,
                 txt=f"({run['Time']}) {run['Stores']}, Carrier: {run['Carrier']}, "
                     f"Cube: {run['Total Cube']}, Second: {run['Second Trailer']}, "
                     f"Note: {run['Fit Note']}",
                 ln=1)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


def generate_excel(df):
    """Export a DataFrame to an Excel BytesIO buffer.

    Parameters
    ----------
    df : pd.DataFrame
        Data to export.

    Returns
    -------
    io.BytesIO
        Excel file bytes, seeked to 0.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf
