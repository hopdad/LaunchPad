import pytest
import sqlite3
import pandas as pd
from datetime import datetime

from db_utils import (
    get_db_connection,
    db_connection,
    save_data_to_db,
    fetch_historical_data,
    fetch_prior_peddles,
    save_actual_peddles,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Provide a fresh in-memory-like SQLite DB for each test by patching the DB path."""
    db_path = str(tmp_path / "test_peddle.db")
    monkeypatch.setattr("db_utils.sqlite3.connect", _make_connect(db_path))
    conn, c = get_db_connection()
    yield conn, c
    conn.close()


def _make_connect(path):
    """Return a patched connect that always uses the given path."""
    original_connect = sqlite3.connect

    def patched_connect(db_name, **kwargs):
        return original_connect(path, **kwargs)

    return patched_connect


# --- get_db_connection tests ---

class TestGetDbConnection:
    def test_returns_connection_and_cursor(self, db):
        conn, c = db
        assert isinstance(conn, sqlite3.Connection)
        assert isinstance(c, sqlite3.Cursor)

    def test_creates_all_tables(self, db):
        conn, c = db
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        assert "store_summaries" in tables
        assert "dept_cubes" in tables
        assert "peddle_runs" in tables
        assert "actual_peddles" in tables

    def test_idempotent(self, db, tmp_path, monkeypatch):
        """Calling get_db_connection twice should not fail (CREATE IF NOT EXISTS)."""
        # Already called once in fixture; call again
        conn2, c2 = get_db_connection()
        c2.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c2.fetchall()}
        assert len(tables) >= 4
        conn2.close()


class TestDbConnectionContextManager:
    def test_closes_connection_on_exit(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_ctx.db")
        monkeypatch.setattr("db_utils.DB_PATH", db_path)
        with db_connection() as (conn, c):
            c.execute("SELECT 1")
        # After exiting the context, attempting a query should fail
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_closes_connection_on_exception(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_ctx_err.db")
        monkeypatch.setattr("db_utils.DB_PATH", db_path)
        with pytest.raises(ValueError):
            with db_connection() as (conn, c):
                raise ValueError("boom")
        with pytest.raises(Exception):
            conn.execute("SELECT 1")


# --- save_data_to_db tests ---

class TestSaveDataToDb:
    def _sample_df(self):
        return pd.DataFrame([
            {"STORE": "100", "882": 500.0, "883": 300.0, "MB": 200.0, "TOTAL": 1000.0, "DIFF": 600.0},
            {"STORE": "200", "882": 150.0, "883": 250.0, "MB": 350.0, "TOTAL": 750.0, "DIFF": 850.0},
        ])

    def _sample_runs(self):
        return pd.DataFrame([
            {"Run": 1, "Time": "06:00", "Stores": "100,200", "Carrier": "ABC",
             "Total Cube": 1750.0, "Second Trailer": "No", "Fit Note": "OK"},
        ])

    def test_saves_store_summaries(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        save_data_to_db(self._sample_df(), ["882", "883", "MB"], self._sample_runs(), date, conn, c)

        c.execute("SELECT * FROM store_summaries ORDER BY store")
        rows = c.fetchall()
        assert len(rows) == 2
        assert rows[0] == ("2025-06-15", "100", 1000.0, 600.0)
        assert rows[1] == ("2025-06-15", "200", 750.0, 850.0)

    def test_saves_dept_cubes(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        save_data_to_db(self._sample_df(), ["882", "883", "MB"], self._sample_runs(), date, conn, c)

        c.execute("SELECT * FROM dept_cubes WHERE store = '100' ORDER BY dept")
        rows = c.fetchall()
        assert len(rows) == 3
        cubes = {row[2]: row[3] for row in rows}
        assert cubes["882"] == 500.0
        assert cubes["883"] == 300.0
        assert cubes["MB"] == 200.0

    def test_saves_peddle_runs(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        save_data_to_db(self._sample_df(), ["882", "883", "MB"], self._sample_runs(), date, conn, c)

        c.execute("SELECT * FROM peddle_runs")
        rows = c.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "2025-06-15"
        assert rows[0][1] == 1
        assert rows[0][3] == "100,200"
        assert rows[0][5] == 1750.0

    def test_upsert_replaces_on_duplicate(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        save_data_to_db(self._sample_df(), ["882", "883", "MB"], self._sample_runs(), date, conn, c)

        # Save again with different totals
        df2 = pd.DataFrame([
            {"STORE": "100", "882": 999.0, "883": 0.0, "MB": 0.0, "TOTAL": 999.0, "DIFF": 601.0},
        ])
        save_data_to_db(df2, ["882", "883", "MB"], self._sample_runs(), date, conn, c)

        c.execute("SELECT total FROM store_summaries WHERE store = '100'")
        assert c.fetchone()[0] == 999.0


# --- fetch_historical_data tests ---

class TestFetchHistoricalData:
    def test_returns_three_dataframes(self, db):
        conn, c = db
        summaries, runs, totals = fetch_historical_data(datetime(2025, 6, 15), c)
        assert isinstance(summaries, pd.DataFrame)
        assert isinstance(runs, pd.DataFrame)
        assert isinstance(totals, pd.DataFrame)

    def test_empty_when_no_data(self, db):
        conn, c = db
        summaries, runs, totals = fetch_historical_data(datetime(2025, 6, 15), c)
        assert len(summaries) == 0
        assert len(runs) == 0
        assert len(totals) == 0

    def test_returns_saved_data(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        df = pd.DataFrame([
            {"STORE": "100", "882": 500.0, "TOTAL": 500.0, "DIFF": 1100.0},
        ])
        runs_df = pd.DataFrame([
            {"Run": 1, "Time": "06:00", "Stores": "100", "Carrier": "XYZ",
             "Total Cube": 500.0, "Second Trailer": "No", "Fit Note": ""},
        ])
        save_data_to_db(df, ["882"], runs_df, date, conn, c)

        summaries, runs, totals = fetch_historical_data(date, c)
        assert len(summaries) == 1
        assert summaries.iloc[0]["Store"] == "100"
        assert len(runs) == 1
        assert runs.iloc[0]["Carrier"] == "XYZ"

    def test_totals_aggregated_across_dates(self, db):
        conn, c = db
        for day, total in [(15, 500.0), (16, 800.0)]:
            date = datetime(2025, 6, day)
            df = pd.DataFrame([{"STORE": "100", "TOTAL": total, "DIFF": 0.0}])
            runs_df = pd.DataFrame([
                {"Run": 1, "Time": "", "Stores": "100", "Carrier": "",
                 "Total Cube": total, "Second Trailer": "No", "Fit Note": ""},
            ])
            save_data_to_db(df, [], runs_df, date, conn, c)

        _, _, totals = fetch_historical_data(datetime(2025, 6, 15), c)
        assert len(totals) == 2  # two dates


# --- fetch_prior_peddles tests ---

class TestFetchPriorPeddles:
    def test_returns_dataframe(self, db):
        conn, c = db
        result = fetch_prior_peddles(datetime(2025, 6, 14), c)
        assert isinstance(result, pd.DataFrame)

    def test_empty_when_no_data(self, db):
        conn, c = db
        result = fetch_prior_peddles(datetime(2025, 6, 14), c)
        assert len(result) == 0

    def test_returns_runs_for_date(self, db):
        conn, c = db
        date = datetime(2025, 6, 14)
        df = pd.DataFrame([{"STORE": "100", "TOTAL": 500.0, "DIFF": 0.0}])
        runs_df = pd.DataFrame([
            {"Run": 1, "Time": "07:00", "Stores": "100", "Carrier": "DEF",
             "Total Cube": 500.0, "Second Trailer": "No", "Fit Note": "tight"},
        ])
        save_data_to_db(df, [], runs_df, date, conn, c)

        result = fetch_prior_peddles(date, c)
        assert len(result) == 1
        assert result.iloc[0]["Carrier"] == "DEF"

    def test_does_not_return_other_dates(self, db):
        conn, c = db
        df = pd.DataFrame([{"STORE": "100", "TOTAL": 500.0, "DIFF": 0.0}])
        runs_df = pd.DataFrame([
            {"Run": 1, "Time": "", "Stores": "100", "Carrier": "",
             "Total Cube": 500.0, "Second Trailer": "No", "Fit Note": ""},
        ])
        save_data_to_db(df, [], runs_df, datetime(2025, 6, 14), conn, c)

        result = fetch_prior_peddles(datetime(2025, 6, 15), c)
        assert len(result) == 0


# --- save_actual_peddles tests ---

class TestSaveActualPeddles:
    def test_saves_actuals(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        actuals = pd.DataFrame([
            {"Run": 1, "Actual Trailers": 2, "Actual Notes": "Extra load"},
            {"Run": 2, "Actual Trailers": 1, "Actual Notes": "On target"},
        ])
        save_actual_peddles(actuals, date, conn, c)

        c.execute("SELECT * FROM actual_peddles ORDER BY run")
        rows = c.fetchall()
        assert len(rows) == 2
        assert rows[0] == ("2025-06-15", 1, 2, "Extra load")
        assert rows[1] == ("2025-06-15", 2, 1, "On target")

    def test_upsert_replaces_existing(self, db):
        conn, c = db
        date = datetime(2025, 6, 15)
        actuals1 = pd.DataFrame([{"Run": 1, "Actual Trailers": 2, "Actual Notes": "v1"}])
        save_actual_peddles(actuals1, date, conn, c)

        actuals2 = pd.DataFrame([{"Run": 1, "Actual Trailers": 3, "Actual Notes": "v2"}])
        save_actual_peddles(actuals2, date, conn, c)

        c.execute("SELECT actual_trailers, actual_notes FROM actual_peddles WHERE run = 1")
        row = c.fetchone()
        assert row == (3, "v2")

    def test_different_dates_coexist(self, db):
        conn, c = db
        for day in [14, 15]:
            actuals = pd.DataFrame([{"Run": 1, "Actual Trailers": day, "Actual Notes": f"day {day}"}])
            save_actual_peddles(actuals, datetime(2025, 6, day), conn, c)

        c.execute("SELECT * FROM actual_peddles")
        rows = c.fetchall()
        assert len(rows) == 2
