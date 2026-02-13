"""Peddle run planning logic (bin-packing)."""

import pandas as pd


def auto_suggest_runs(df, trailer_capacity, store_ready_times=None):
    """First-Fit Decreasing bin-packing with optional ready-time grouping.

    When *store_ready_times* is provided, stores are first grouped by their
    ready-time window (each unique time forms a group).  Within each group
    the stores are sorted largest-first and packed using FFD.  Groups are
    processed in chronological order so early-ready stores ship first.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``STORE`` and ``TOTAL``.
    trailer_capacity : float
        Maximum cube per trailer/run.
    store_ready_times : dict | None
        Mapping of store number (str) to ready time (str, e.g. "05:00").
        If ``None`` or empty, all stores are treated as a single group.

    Returns
    -------
    list[dict]
        Each dict has keys: Stores, Total Cube, Second Trailer.
    """
    if df.empty:
        return []

    if not store_ready_times:
        return _ffd_pack(df, trailer_capacity)

    # Group stores by ready time, process chronologically
    df = df.copy()
    df["_ready"] = df["STORE"].astype(str).map(
        lambda s: store_ready_times.get(s, "05:00")
    )

    runs = []
    for _time, group_df in sorted(df.groupby("_ready"), key=lambda x: x[0]):
        runs.extend(_ffd_pack(group_df, trailer_capacity))

    return runs


def _ffd_pack(df, trailer_capacity):
    """First-Fit Decreasing: try each store in the first run it fits."""
    sorted_df = df.sort_values("TOTAL", ascending=False)
    runs = []  # list of {"stores": [str], "cube": float}

    for _, row in sorted_df.iterrows():
        store = str(row["STORE"])
        cube = row["TOTAL"]
        placed = False
        for run in runs:
            if run["cube"] + cube <= trailer_capacity:
                run["stores"].append(store)
                run["cube"] += cube
                placed = True
                break
        if not placed:
            runs.append({"stores": [store], "cube": cube})

    return [
        {
            "Stores": "/".join(r["stores"]),
            "Total Cube": r["cube"],
            "Second Trailer": "No" if r["cube"] <= trailer_capacity else "Yes",
        }
        for r in runs
    ]
