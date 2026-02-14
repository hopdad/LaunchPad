import io
import json
import logging
import pandas as pd
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = 'peddle_data.db'

_SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS store_summaries
       (date TEXT, store TEXT, total REAL, diff REAL,
        PRIMARY KEY (date, store))''',
    '''CREATE TABLE IF NOT EXISTS dept_cubes
       (date TEXT, store TEXT, dept TEXT, cube REAL,
        PRIMARY KEY (date, store, dept))''',
    '''CREATE TABLE IF NOT EXISTS peddle_runs
       (date TEXT, run INTEGER, time TEXT, stores TEXT,
        carrier TEXT, total_cube REAL, second_trailer TEXT, fit_note TEXT,
        PRIMARY KEY (date, run))''',
    '''CREATE TABLE IF NOT EXISTS actual_peddles
       (date TEXT, run INTEGER, actual_trailers INTEGER, actual_notes TEXT,
        PRIMARY KEY (date, run))''',
    '''CREATE TABLE IF NOT EXISTS app_settings
       (key TEXT PRIMARY KEY, value TEXT)''',
    '''CREATE TABLE IF NOT EXISTS store_zones
       (store TEXT PRIMARY KEY, zone TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS drafts
       (date TEXT, username TEXT, df_json TEXT NOT NULL,
        departments TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (date, username))''',
]


def get_db_connection():
    """Return (conn, cursor) with schema initialised.

    Kept for backward compatibility with existing callers.
    Prefer the ``db_connection`` context manager for new code.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    for ddl in _SCHEMA:
        c.execute(ddl)
    conn.commit()
    return conn, c


@contextmanager
def db_connection():
    """Context manager that yields (conn, cursor) and closes on exit."""
    conn, c = get_db_connection()
    try:
        yield conn, c
    finally:
        conn.close()


def save_data_to_db(df, departments, runs_df, date, conn, c):
    date_str = date.strftime('%Y-%m-%d')

    for _, row in df.iterrows():
        c.execute("INSERT OR REPLACE INTO store_summaries VALUES (?, ?, ?, ?)",
                  (date_str, row['STORE'], row['TOTAL'], row['DIFF']))
        for dept in departments:
            if dept in row:
                c.execute("INSERT OR REPLACE INTO dept_cubes VALUES (?, ?, ?, ?)",
                          (date_str, row['STORE'], dept, row[dept]))

    for _, run in runs_df.iterrows():
        c.execute("INSERT OR REPLACE INTO peddle_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (date_str, run['Run'], run.get('Time', ''), run['Stores'], run['Carrier'], run['Total Cube'], run['Second Trailer'], run.get('Fit Note', '')))

    conn.commit()


def fetch_historical_data(query_date, c):
    date_str = query_date.strftime('%Y-%m-%d')

    c.execute("SELECT * FROM store_summaries WHERE date = ?", (date_str,))
    summaries = pd.DataFrame(c.fetchall(), columns=['Date', 'Store', 'Total', 'Diff'])

    c.execute("SELECT * FROM peddle_runs WHERE date = ?", (date_str,))
    runs = pd.DataFrame(c.fetchall(), columns=['Date', 'Run', 'Time', 'Stores', 'Carrier', 'Total Cube', 'Second Trailer', 'Fit Note'])

    c.execute("SELECT date, SUM(total) FROM store_summaries GROUP BY date")
    totals = pd.DataFrame(c.fetchall(), columns=['Date', 'Total Cube'])

    return summaries, runs, totals


def fetch_prior_peddles(prior_date, c):
    date_str = prior_date.strftime('%Y-%m-%d')
    c.execute("SELECT * FROM peddle_runs WHERE date = ?", (date_str,))
    return pd.DataFrame(c.fetchall(), columns=['Date', 'Run', 'Time', 'Stores', 'Carrier', 'Total Cube', 'Second Trailer', 'Fit Note'])


def fetch_actuals_for_date(date, c):
    """Return actual peddle data for the given date as a DataFrame."""
    date_str = date.strftime('%Y-%m-%d')
    c.execute("SELECT * FROM actual_peddles WHERE date = ?", (date_str,))
    rows = c.fetchall()
    if not rows:
        return pd.DataFrame(columns=['Date', 'Run', 'Actual Trailers', 'Actual Notes'])
    return pd.DataFrame(rows, columns=['Date', 'Run', 'Actual Trailers', 'Actual Notes'])


def fetch_per_store_history(c):
    """Return per-store totals across all dates for trend analysis."""
    c.execute("SELECT date, store, total FROM store_summaries ORDER BY date, store")
    return pd.DataFrame(c.fetchall(), columns=['Date', 'Store', 'Total'])


def save_actual_peddles(actuals_df, date, conn, c):
    date_str = date.strftime('%Y-%m-%d')
    for _, row in actuals_df.iterrows():
        c.execute("INSERT OR REPLACE INTO actual_peddles VALUES (?, ?, ?, ?)",
                  (date_str, row['Run'], row['Actual Trailers'], row['Actual Notes']))
    conn.commit()


def save_settings(settings_dict, conn, c):
    """Persist a dict of app settings to the database.

    Each value is JSON-encoded so lists/dicts round-trip cleanly.
    """
    for key, value in settings_dict.items():
        c.execute("INSERT OR REPLACE INTO app_settings VALUES (?, ?)",
                  (key, json.dumps(value)))
    conn.commit()


def load_settings(c):
    """Return all saved settings as a plain dict (values JSON-decoded)."""
    c.execute("SELECT key, value FROM app_settings")
    return {row[0]: json.loads(row[1]) for row in c.fetchall()}


# --- Drafts (auto-save in-progress data entry) ---

def save_draft(date_str, username, df, departments, conn, c):
    """Auto-save in-progress cube data so it can be resumed on any device."""
    from datetime import datetime as _dt
    df_json = df.to_json(orient="records")
    depts_json = json.dumps(departments)
    now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT OR REPLACE INTO drafts VALUES (?, ?, ?, ?, ?)",
        (date_str, username, df_json, depts_json, now),
    )
    conn.commit()


def load_draft(date_str, username, c):
    """Load a saved draft for the given date and user.

    Returns (df, departments, updated_at) or (None, None, None) if no draft.
    """
    c.execute(
        "SELECT df_json, departments, updated_at FROM drafts WHERE date = ? AND username = ?",
        (date_str, username),
    )
    row = c.fetchone()
    if not row:
        return None, None, None
    df = pd.read_json(io.StringIO(row[0]), orient="records")
    if "STORE" in df.columns:
        df["STORE"] = df["STORE"].astype(str)
    departments = json.loads(row[1])
    return df, departments, row[2]


def delete_draft(date_str, username, conn, c):
    """Remove a draft after data has been fully saved."""
    c.execute("DELETE FROM drafts WHERE date = ? AND username = ?", (date_str, username))
    conn.commit()


# --- Store zones ---

def save_store_zones(zone_map, conn, c):
    """Persist a {store: zone} mapping, replacing all existing rows."""
    c.execute("DELETE FROM store_zones")
    for store, zone in zone_map.items():
        c.execute("INSERT INTO store_zones VALUES (?, ?)", (store, zone))
    conn.commit()


def load_store_zones(c):
    """Return {store: zone} dict."""
    c.execute("SELECT store, zone FROM store_zones")
    return {row[0]: row[1] for row in c.fetchall()}


# --- Correction factor ---

def fetch_correction_factor(c, lookback_days=30):
    """Compute a cube correction multiplier from recent projected vs actual data.

    Looks at the last *lookback_days* of data where both projections and actuals
    exist.  Returns a float multiplier (e.g. 1.15 means actuals use 15% more
    trailers than projected).  Returns 1.0 when there is insufficient data.
    """
    c.execute(
        """
        SELECT p.total_cube, p.second_trailer, a.actual_trailers
        FROM peddle_runs p
        JOIN actual_peddles a ON p.date = a.date AND p.run = a.run
        WHERE p.date >= date('now', ?)
        """,
        (f"-{lookback_days} days",),
    )
    rows = c.fetchall()
    if not rows:
        return 1.0

    total_projected_trailers = 0
    total_actual_trailers = 0
    for total_cube, second_trailer, actual_trailers in rows:
        projected = 2 if second_trailer == "Yes" else 1
        total_projected_trailers += projected
        total_actual_trailers += actual_trailers

    if total_projected_trailers == 0:
        return 1.0

    return total_actual_trailers / total_projected_trailers


def fetch_per_store_correction(c, lookback_days=30):
    """Compute per-store cube bias from historical data.

    Returns {store: multiplier} where multiplier > 1 means the store
    consistently uses more trailers than projected.
    Only includes stores with at least 3 data points.
    """
    c.execute(
        """
        SELECT p.stores, p.total_cube, p.second_trailer, a.actual_trailers
        FROM peddle_runs p
        JOIN actual_peddles a ON p.date = a.date AND p.run = a.run
        WHERE p.date >= date('now', ?)
        """,
        (f"-{lookback_days} days",),
    )
    rows = c.fetchall()
    if not rows:
        return {}

    # Accumulate per-store projected vs actual (split evenly across run stores)
    from collections import defaultdict
    store_proj = defaultdict(list)
    store_actual = defaultdict(list)
    for stores_str, total_cube, second_trailer, actual_trailers in rows:
        stores = [s.strip() for s in stores_str.replace(",", "/").split("/") if s.strip()]
        if not stores:
            continue
        projected = 2.0 if second_trailer == "Yes" else 1.0
        share_proj = projected / len(stores)
        share_actual = actual_trailers / len(stores)
        for s in stores:
            store_proj[s].append(share_proj)
            store_actual[s].append(share_actual)

    result = {}
    for store in store_proj:
        if len(store_proj[store]) < 3:
            continue
        sum_proj = sum(store_proj[store])
        sum_actual = sum(store_actual[store])
        if sum_proj > 0:
            result[store] = sum_actual / sum_proj

    return result
