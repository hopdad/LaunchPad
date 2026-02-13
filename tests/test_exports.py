import sys
import types
import importlib
import pytest
import pandas as pd
from datetime import date

# fpdf2 requires cryptography which may not be available in all environments.
# Stub it out so the exports module can still be imported for Excel tests.
_HAS_FPDF = False
try:
    _spec = importlib.util.find_spec("fpdf")
    if _spec is not None:
        import fpdf  # noqa: F401
        _HAS_FPDF = True
except Exception:
    pass

if not _HAS_FPDF:
    _fpdf_mod = types.ModuleType("fpdf")
    _fpdf_mod.FPDF = type("FPDF", (), {})  # type: ignore[attr-defined]
    sys.modules["fpdf"] = _fpdf_mod

from exports import generate_pdf, generate_excel


@pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not available in this environment")
class TestGeneratePdf:
    def _sample_df(self):
        return pd.DataFrame([
            {"STORE": "100", "882": 500.0, "883": 300.0, "TOTAL": 800.0, "DIFF": -800.0},
        ])

    def _sample_runs(self):
        return pd.DataFrame([
            {"Run": 1, "Time": "06:00", "Stores": "100", "Carrier": "Standard",
             "Total Cube": 800.0, "Second Trailer": "No", "Fit Note": "FITS"},
        ])

    def test_returns_bytes_io(self):
        buf = generate_pdf(
            self._sample_df(), ["882", "883"], self._sample_runs(),
            date(2025, 6, 15), 800.0, 0.5, 1,
        )
        assert buf.read(5) == b"%PDF-"

    def test_contains_date(self):
        buf = generate_pdf(
            self._sample_df(), ["882", "883"], self._sample_runs(),
            date(2025, 6, 15), 800.0, 0.5, 1,
        )
        content = buf.read()
        assert b"06/15/2025" in content

    def test_contains_store_data(self):
        buf = generate_pdf(
            self._sample_df(), ["882", "883"], self._sample_runs(),
            date(2025, 6, 15), 800.0, 0.5, 1,
        )
        content = buf.read()
        assert b"STORE 100" in content

    def test_multiple_stores(self):
        df = pd.DataFrame([
            {"STORE": "100", "882": 500.0, "TOTAL": 500.0, "DIFF": -1100.0},
            {"STORE": "200", "882": 300.0, "TOTAL": 300.0, "DIFF": -1300.0},
        ])
        buf = generate_pdf(df, ["882"], self._sample_runs(), date(2025, 6, 15), 800.0, 0.5, 1)
        content = buf.read()
        assert b"STORE 100" in content
        assert b"STORE 200" in content


class TestGenerateExcel:
    def test_returns_valid_excel(self):
        df = pd.DataFrame([{"STORE": "100", "TOTAL": 500.0}])
        buf = generate_excel(df)
        result = pd.read_excel(buf)
        assert len(result) == 1
        assert str(result.iloc[0]["STORE"]) == "100"

    def test_preserves_columns(self):
        df = pd.DataFrame([{"STORE": "100", "882": 500.0, "883": 300.0, "TOTAL": 800.0}])
        buf = generate_excel(df)
        result = pd.read_excel(buf)
        assert list(result.columns) == ["STORE", "882", "883", "TOTAL"]

    def test_multiple_rows(self):
        df = pd.DataFrame([
            {"STORE": "100", "TOTAL": 500.0},
            {"STORE": "200", "TOTAL": 300.0},
        ])
        buf = generate_excel(df)
        result = pd.read_excel(buf)
        assert len(result) == 2
