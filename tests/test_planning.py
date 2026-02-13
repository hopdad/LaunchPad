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

    def test_ffd_sorts_descending(self):
        """FFD: largest stores placed first. Smaller stores back-fill earlier bins."""
        df = self._make_df([("A", 100), ("B", 1500), ("C", 200)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        # B=1500 first run. C=200: 1500+200=1700>1600 -> new run.
        # A=100: 1500+100=1600<=1600 -> fits with B!
        assert len(runs) == 2
        assert "B" in runs[0]["Stores"]
        assert "A" in runs[0]["Stores"]
        assert runs[0]["Total Cube"] == 1600
        assert runs[1]["Stores"] == "C"
        assert runs[1]["Total Cube"] == 200

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

    # --- FFD-specific tests ---

    def test_ffd_backfills_earlier_bins(self):
        """FFD should place small stores into earlier runs when they fit."""
        # Run 1: 1000, Run 2: 900. Then 500 fits in run 1 (1000+500=1500).
        # Then 400 fits in run 2 (900+400=1300). Only 2 runs needed.
        df = self._make_df([("A", 1000), ("B", 900), ("C", 500), ("D", 400)])
        runs = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs) == 2
        total = sum(r["Total Cube"] for r in runs)
        assert total == 2800

    # --- Ready time grouping tests ---

    def test_ready_times_group_stores(self):
        """Stores with different ready times should not share runs."""
        df = self._make_df([("A", 100), ("B", 100)])
        ready_times = {"A": "05:00", "B": "08:00"}
        runs = auto_suggest_runs(df, trailer_capacity=1600, store_ready_times=ready_times)
        # Different ready times -> separate groups -> separate runs
        assert len(runs) == 2

    def test_same_ready_time_shares_run(self):
        """Stores with the same ready time can share a run."""
        df = self._make_df([("A", 100), ("B", 100)])
        ready_times = {"A": "05:00", "B": "05:00"}
        runs = auto_suggest_runs(df, trailer_capacity=1600, store_ready_times=ready_times)
        assert len(runs) == 1

    def test_ready_times_chronological_order(self):
        """Earlier ready times should produce earlier runs."""
        df = self._make_df([("A", 100), ("B", 100)])
        ready_times = {"A": "08:00", "B": "05:00"}
        runs = auto_suggest_runs(df, trailer_capacity=1600, store_ready_times=ready_times)
        assert len(runs) == 2
        # B (05:00) should be first
        assert runs[0]["Stores"] == "B"
        assert runs[1]["Stores"] == "A"

    def test_none_ready_times_same_as_omitted(self):
        """Passing None for ready times should behave like omitting them."""
        df = self._make_df([("A", 100), ("B", 100)])
        runs_none = auto_suggest_runs(df, trailer_capacity=1600, store_ready_times=None)
        runs_empty = auto_suggest_runs(df, trailer_capacity=1600, store_ready_times={})
        runs_default = auto_suggest_runs(df, trailer_capacity=1600)
        assert len(runs_none) == len(runs_empty) == len(runs_default) == 1
