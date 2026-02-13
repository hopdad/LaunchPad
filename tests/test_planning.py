import pytest
import pandas as pd

from planning import auto_suggest_runs


class TestAutoSuggestRuns:
    def _make_df(self, stores_and_totals):
        """Helper: list of (store, total) -> DataFrame with STORE and TOTAL."""
        return pd.DataFrame([{"STORE": s, "TOTAL": t} for s, t in stores_and_totals])

    def test_single_store_fits(self):
        df = self._make_df([("100", 500)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Stores"] == "100"
        assert runs[0]["Total Cube"] == 500
        assert runs[0]["Second Trailer"] == "No"

    def test_two_stores_fit_one_run(self):
        df = self._make_df([("100", 800), ("200", 700)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert "100" in runs[0]["Stores"]
        assert "200" in runs[0]["Stores"]

    def test_two_stores_need_two_runs(self):
        df = self._make_df([("100", 1000), ("200", 900)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["STORE", "TOTAL"])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert runs == []

    def test_greedy_sorts_descending(self):
        """Largest stores should be placed first to pack efficiently."""
        df = self._make_df([("A", 100), ("B", 1500), ("C", 200)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        # B (1500) goes first, then C (200) won't fit with B, so B alone
        # Actually: B=1500 first. C=200: 1500+200=1700 > 1600 -> new run.
        # Then A=100: 200+100=300 <= 1600 -> fits with C.
        assert len(runs) == 2
        assert runs[0]["Stores"] == "B"
        assert runs[0]["Total Cube"] == 1500
        assert "C" in runs[1]["Stores"]
        assert "A" in runs[1]["Stores"]

    def test_exact_capacity_fit(self):
        df = self._make_df([("100", 800), ("200", 800)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Total Cube"] == 1600
        assert runs[0]["Second Trailer"] == "No"

    def test_single_store_exceeds_capacity(self):
        """A store that alone exceeds capacity should be its own run."""
        df = self._make_df([("100", 2000)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Total Cube"] == 2000
        assert runs[0]["Second Trailer"] == "Yes"

    def test_many_small_stores_packed(self):
        stores = [(str(i), 100) for i in range(20)]
        df = self._make_df(stores)
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        total = sum(r["Total Cube"] for r in runs)
        assert total == 2000
        # Should need 2 runs: 16 stores in first (1600), 4 in second (400)
        assert len(runs) == 2

    def test_all_runs_have_required_keys(self):
        df = self._make_df([("100", 500), ("200", 600), ("300", 700)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        for run in runs:
            assert "Stores" in run
            assert "Total Cube" in run
            assert "Second Trailer" in run

    def test_zero_cube_stores(self):
        df = self._make_df([("100", 0), ("200", 0)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Total Cube"] == 0
