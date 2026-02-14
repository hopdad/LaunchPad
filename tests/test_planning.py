import pytest
import pandas as pd

from planning import (
    auto_suggest_runs,
    extract_overs,
    auto_suggest_overs_runs,
    haversine,
    sequence_stores,
    sequence_runs,
)


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


# --- Overs extraction tests ---

class TestExtractOvers:
    def _make_df(self, stores_and_totals):
        return pd.DataFrame([{"STORE": s, "TOTAL": t} for s, t in stores_and_totals])

    def test_no_overs_when_all_fit(self):
        df = self._make_df([("100", 500), ("200", 800)])
        overs = extract_overs(df, trailer_capacity=1600)
        assert len(overs) == 0

    def test_overs_for_exceeding_store(self):
        df = self._make_df([("100", 2000), ("200", 800)])
        overs = extract_overs(df, trailer_capacity=1600)
        assert len(overs) == 1
        assert overs.iloc[0]["STORE"] == "100"
        assert overs.iloc[0]["OVERS"] == 400  # 2000 - 1600

    def test_multiple_stores_with_overs(self):
        df = self._make_df([("100", 2000), ("200", 1800), ("300", 500)])
        overs = extract_overs(df, trailer_capacity=1600)
        assert len(overs) == 2
        total_overs = overs["OVERS"].sum()
        assert total_overs == 600  # 400 + 200

    def test_exact_capacity_no_overs(self):
        df = self._make_df([("100", 1600)])
        overs = extract_overs(df, trailer_capacity=1600)
        assert len(overs) == 0

    def test_empty_df(self):
        df = pd.DataFrame(columns=["STORE", "TOTAL"])
        overs = extract_overs(df, trailer_capacity=1600)
        assert len(overs) == 0

    def test_overs_columns(self):
        df = self._make_df([("100", 2000)])
        overs = extract_overs(df, trailer_capacity=1600)
        assert list(overs.columns) == ["STORE", "OVERS"]


# --- Overs run generation tests ---

class TestAutoSuggestOversRuns:
    def _make_df(self, stores_and_totals):
        return pd.DataFrame([{"STORE": s, "TOTAL": t} for s, t in stores_and_totals])

    def test_no_runs_when_no_overs(self):
        df = self._make_df([("100", 500), ("200", 800)])
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600)
        assert runs == []

    def test_single_store_overs_one_run(self):
        df = self._make_df([("100", 2000)])
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Stores"] == "100"
        assert runs[0]["Total Cube"] == 400  # only the overflow
        assert runs[0]["Zone"] == "Unzoned"

    def test_overs_grouped_by_zone(self):
        df = self._make_df([("100", 2000), ("200", 1800), ("300", 1900)])
        zones = {"100": "North", "200": "North", "300": "South"}
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600, store_zones=zones)
        # North: 100 (400 overs) + 200 (200 overs) = 600 -> 1 run
        # South: 300 (300 overs) -> 1 run
        assert len(runs) == 2
        north_runs = [r for r in runs if r["Zone"] == "North"]
        south_runs = [r for r in runs if r["Zone"] == "South"]
        assert len(north_runs) == 1
        assert len(south_runs) == 1
        assert north_runs[0]["Total Cube"] == 600
        assert south_runs[0]["Total Cube"] == 300

    def test_unzoned_stores_grouped_together(self):
        df = self._make_df([("100", 2000), ("200", 1800)])
        zones = {"100": "North"}  # 200 has no zone
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600, store_zones=zones)
        assert len(runs) == 2
        zones_in_runs = {r["Zone"] for r in runs}
        assert "North" in zones_in_runs
        assert "Unzoned" in zones_in_runs

    def test_no_zones_all_in_one_pool(self):
        df = self._make_df([("100", 2000), ("200", 1800)])
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600, store_zones=None)
        # 400 + 200 = 600 overs, fits in one run
        assert len(runs) == 1
        assert runs[0]["Total Cube"] == 600
        assert runs[0]["Zone"] == "Unzoned"

    def test_overs_runs_have_required_keys(self):
        df = self._make_df([("100", 2000)])
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600)
        for run in runs:
            assert "Stores" in run
            assert "Total Cube" in run
            assert "Second Trailer" in run
            assert "Zone" in run

    def test_large_overs_needs_second_trailer(self):
        """If a single store's overs exceed trailer capacity, flag it."""
        df = self._make_df([("100", 4000)])  # 2400 overs
        runs = auto_suggest_overs_runs(df, trailer_capacity=1600)
        assert len(runs) == 1
        assert runs[0]["Total Cube"] == 2400
        assert runs[0]["Second Trailer"] == "Yes"


# --- Haversine distance tests ---

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_known_distance(self):
        # NYC to LA ~2451 miles (great-circle)
        d = haversine(40.7128, -74.0060, 33.9425, -118.4081)
        assert 2400 < d < 2500

    def test_short_distance(self):
        # ~1 mile apart (roughly 0.0145 degrees lat)
        d = haversine(40.0, -74.0, 40.0145, -74.0)
        assert 0.9 < d < 1.1

    def test_symmetrical(self):
        d1 = haversine(40.0, -74.0, 33.0, -118.0)
        d2 = haversine(33.0, -118.0, 40.0, -74.0)
        assert abs(d1 - d2) < 0.001


# --- Sequence stores tests ---

class TestSequenceStores:
    # DC at origin-ish location (0,0), stores at varying distances
    DC = (40.0, -74.0)
    LOCATIONS = {
        "100": (40.1, -74.0),   # ~7 miles north
        "200": (41.0, -74.0),   # ~69 miles north
        "300": (40.5, -74.0),   # ~35 miles north
    }

    def test_furthest_first(self):
        result = sequence_stores(
            ["100", "200", "300"], self.LOCATIONS, self.DC, order="furthest_first"
        )
        assert result == ["200", "300", "100"]

    def test_closest_first(self):
        result = sequence_stores(
            ["100", "200", "300"], self.LOCATIONS, self.DC, order="closest_first"
        )
        assert result == ["100", "300", "200"]

    def test_missing_coords_appended_at_end(self):
        result = sequence_stores(
            ["100", "999", "200"], self.LOCATIONS, self.DC, order="furthest_first"
        )
        assert result[-1] == "999"
        assert result[0] == "200"

    def test_no_locations_returns_unchanged(self):
        result = sequence_stores(["100", "200"], {}, self.DC)
        assert result == ["100", "200"]

    def test_no_dc_returns_unchanged(self):
        result = sequence_stores(["100", "200"], self.LOCATIONS, None)
        assert result == ["100", "200"]

    def test_single_store(self):
        result = sequence_stores(["100"], self.LOCATIONS, self.DC)
        assert result == ["100"]

    def test_empty_list(self):
        result = sequence_stores([], self.LOCATIONS, self.DC)
        assert result == []


# --- Sequence runs tests ---

class TestSequenceRuns:
    DC = (40.0, -74.0)
    LOCATIONS = {
        "100": (40.1, -74.0),
        "200": (41.0, -74.0),
        "300": (40.5, -74.0),
    }

    def test_sequences_stores_in_run(self):
        runs = [{"Stores": "100/200/300", "Total Cube": 1000, "Second Trailer": "No"}]
        sequence_runs(runs, self.LOCATIONS, self.DC, order="furthest_first")
        assert runs[0]["Stores"] == "200/300/100"

    def test_multiple_runs_sequenced(self):
        runs = [
            {"Stores": "100/200", "Total Cube": 500, "Second Trailer": "No"},
            {"Stores": "300/100", "Total Cube": 600, "Second Trailer": "No"},
        ]
        sequence_runs(runs, self.LOCATIONS, self.DC, order="furthest_first")
        assert runs[0]["Stores"] == "200/100"
        # 300 is further than 100
        assert runs[1]["Stores"] == "300/100"

    def test_no_locations_unchanged(self):
        runs = [{"Stores": "100/200", "Total Cube": 500, "Second Trailer": "No"}]
        sequence_runs(runs, {}, self.DC)
        assert runs[0]["Stores"] == "100/200"

    def test_closest_first_order(self):
        runs = [{"Stores": "100/200/300", "Total Cube": 1000, "Second Trailer": "No"}]
        sequence_runs(runs, self.LOCATIONS, self.DC, order="closest_first")
        assert runs[0]["Stores"] == "100/300/200"
