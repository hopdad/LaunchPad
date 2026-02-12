import pandas as pd
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('peddle_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS store_summaries (date TEXT, store TEXT, total REAL, diff REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS dept_cubes (date TEXT, store TEXT, dept TEXT, cube REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS peddle_runs (date TEXT, run INTEGER, time TEXT, stores TEXT, carrier TEXT, total_cube REAL, second_trailer TEXT, fit_note TEXT)''')
    conn.commit()
    return conn, c

def save_data_to_db(df, departments, runs_df, date, conn, c):
    date_str = date.strftime('%Y-%m-%d')
    
    for _, row in df.iterrows():
        c.execute("INSERT INTO store_summaries VALUES (?, ?, ?, ?)",
                  (date_str, row['STORE'], row['TOTAL'], row['DIFF']))
        for dept in departments:
            if dept in row:
                c.execute("INSERT INTO dept_cubes VALUES (?, ?, ?, ?)",
                          (date_str, row['STORE'], dept, row[dept]))
    
    for _, run in runs_df.iterrows():
        c.execute("INSERT INTO peddle_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
