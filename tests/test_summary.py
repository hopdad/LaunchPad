import pytest
import pandas as pd
from summary import compute_summary, PALLET_CUBE
from math import ceil


class TestComputeSummary:
    def _make_df(self, totals):
        return pd.DataFrame([{"STORE": str(i), "TOTAL": t} for i, t in enumerate(totals)])

    def test_basic_calculation(self):
        df = self._make_df([1000, 600])
        s = compute_summary(df, fluff=200, trailer_capacity=1600)
        assert s["total_cube"] == 1600
        assert s["total_cube_with_fluff"] == 1800
        assert s["pallet_count"] == ceil(1600 / PALLET_CUBE)
        assert s["trailer_goal"] == pytest.approx(1800 / 1600)
        assert s["probable_trailers"] == ceil(1800 / 1600 * 1.1)

    def test_zero_cube(self):
        df = self._make_df([0, 0])
        s = compute_summary(df, fluff=200, trailer_capacity=1600)
        assert s["total_cube"] == 0
        assert s["pallet_count"] == 0

    def test_single_store(self):
        df = self._make_df([500])
        s = compute_summary(df, fluff=100, trailer_capacity=1600)
        assert s["total_cube"] == 500
        assert s["total_cube_with_fluff"] == 600
