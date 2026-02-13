"""Peddle run planning logic (bin-packing)."""

import pandas as pd


def auto_suggest_runs(df, trailer_capacity):
    """Greedy bin-packing: sort stores by total cube descending, pack into runs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``STORE`` and ``TOTAL``.
    trailer_capacity : float
        Maximum cube per trailer/run.

    Returns
    -------
    list[dict]
        Each dict has keys: Stores, Total Cube, Second Trailer.
    """
    sorted_df = df.sort_values("TOTAL", ascending=False).copy()
    runs = []
    current_run = []
    current_cube = 0

    for _, row in sorted_df.iterrows():
        if current_cube + row["TOTAL"] <= trailer_capacity:
            current_run.append(str(row["STORE"]))
            current_cube += row["TOTAL"]
        else:
            if current_run:
                runs.append({
                    "Stores": "/".join(current_run),
                    "Total Cube": current_cube,
                    "Second Trailer": "No" if current_cube <= trailer_capacity else "Yes",
                })
            current_run = [str(row["STORE"])]
            current_cube = row["TOTAL"]

    if current_run:
        runs.append({
            "Stores": "/".join(current_run),
            "Total Cube": current_cube,
            "Second Trailer": "No" if current_cube <= trailer_capacity else "Yes",
        })

    return runs
