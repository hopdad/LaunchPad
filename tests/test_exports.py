import sys
import types
import pytest
import pandas as pd

# fpdf2 requires cryptography which may not be available in all environments.
# Stub it out so the exports module can still be imported for Excel tests.
_HAS_FPDF = False
try:
    import importlib
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

from exports import generate_excel


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
