import json
import pandas as pd
import sqlite3
from contextlib import contextmanager
from datetime import datetime

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
]


def get_db_connection():
    """Return (conn, cursor) with schema initialised.

    Kept for backward compatibility with existing callers.
    Prefer the ``db_connection`` context manager for new code.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        for ddl in _SCHEMA:
            c.execute(ddl)
        conn.commit()
        return conn, c
    except Exception as e:
        raise RuntimeError(f"Error connecting to DB: {e}")


@contextmanager
def db_connection():
    """Context manager that yields (conn, cursor) and closes on exit."""
    conn, c = get_db_connection()
    try:
        yield conn, c
    finally:
        conn.close()


def save_data_to_db(df, departments, runs_df, date, conn, c):
    try:
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
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Error saving to DB: {e}")


def fetch_historical_data(query_date, c):
    try:
        date_str = query_date.strftime('%Y-%m-%d')

        c.execute("SELECT * FROM store_summaries WHERE date = ?", (date_str,))
        summaries = pd.DataFrame(c.fetchall(), columns=['Date', 'Store', 'Total', 'Diff'])

        c.execute("SELECT * FROM peddle_runs WHERE date = ?", (date_str,))
        runs = pd.DataFrame(c.fetchall(), columns=['Date', 'Run', 'Time', 'Stores', 'Carrier', 'Total Cube', 'Second Trailer', 'Fit Note'])

        c.execute("SELECT date, SUM(total) FROM store_summaries GROUP BY date")
        totals = pd.DataFrame(c.fetchall(), columns=['Date', 'Total Cube'])

        return summaries, runs, totals
    except Exception as e:
        raise RuntimeError(f"Error fetching historical data: {e}")


def fetch_prior_peddles(prior_date, c):
    try:
        date_str = prior_date.strftime('%Y-%m-%d')
        c.execute("SELECT * FROM peddle_runs WHERE date = ?", (date_str,))
        return pd.DataFrame(c.fetchall(), columns=['Date', 'Run', 'Time', 'Stores', 'Carrier', 'Total Cube', 'Second Trailer', 'Fit Note'])
    except Exception as e:
        raise RuntimeError(f"Error fetching prior peddles: {e}")


def save_actual_peddles(actuals_df, date, conn, c):
    try:
        date_str = date.strftime('%Y-%m-%d')
        for _, row in actuals_df.iterrows():
            c.execute("INSERT OR REPLACE INTO actual_peddles VALUES (?, ?, ?, ?)",
                      (date_str, row['Run'], row['Actual Trailers'], row['Actual Notes']))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Error saving actual peddles: {e}")


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
