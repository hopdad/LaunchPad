"""Peddle run planning logic (bin-packing + overs routing)."""

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


# --- Overs-based peddle run planning ---


def extract_overs(df, trailer_capacity):
    """Identify stores with overflow and return a DataFrame of overs.

    Returns a DataFrame with columns: STORE, OVERS (the cube amount beyond
    what fits on the direct trailer).  Only stores with TOTAL > trailer_capacity
    are included.
    """
    if df.empty:
        return pd.DataFrame(columns=["STORE", "OVERS"])

    overs = df[df["TOTAL"] > trailer_capacity].copy()
    if overs.empty:
        return pd.DataFrame(columns=["STORE", "OVERS"])

    overs = overs[["STORE", "TOTAL"]].copy()
    overs["OVERS"] = overs["TOTAL"] - trailer_capacity
    return overs[["STORE", "OVERS"]].reset_index(drop=True)


def auto_suggest_overs_runs(df, trailer_capacity, store_zones=None):
    """Generate peddle runs from overflow freight, grouped by zone.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``STORE`` and ``TOTAL``.
    trailer_capacity : float
        Maximum cube per trailer/run.
    store_zones : dict | None
        Mapping of store number (str) to zone name (str).
        If ``None`` or empty, all overs go into a single pool.

    Returns
    -------
    list[dict]
        Each dict has keys: Stores, Total Cube, Second Trailer, Zone.
    """
    overs_df = extract_overs(df, trailer_capacity)
    if overs_df.empty:
        return []

    # Build a df suitable for FFD packing (STORE + TOTAL columns)
    pack_df = overs_df.rename(columns={"OVERS": "TOTAL"})

    if not store_zones:
        runs = _ffd_pack(pack_df, trailer_capacity)
        for r in runs:
            r["Zone"] = "Unzoned"
        return runs

    # Group by zone, pack each zone separately
    pack_df = pack_df.copy()
    pack_df["_zone"] = pack_df["STORE"].astype(str).map(
        lambda s: store_zones.get(s, "Unzoned")
    )

    all_runs = []
    for zone, group_df in sorted(pack_df.groupby("_zone"), key=lambda x: x[0]):
        zone_runs = _ffd_pack(group_df, trailer_capacity)
        for r in zone_runs:
            r["Zone"] = zone
        all_runs.extend(zone_runs)

    return all_runs
