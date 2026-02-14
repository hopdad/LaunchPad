"""Peddle run planning logic (bin-packing + overs routing + distance sequencing)."""

import math
import pandas as pd


# --- Haversine distance ---

def haversine(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in miles between two (lat, lng) points."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sequence_stores(stores, store_locations, dc_location, order="furthest_first"):
    """Sort a list of store numbers by distance from the DC.

    Parameters
    ----------
    stores : list[str]
        Store numbers to sequence.
    store_locations : dict
        {store: (lat, lng)} mapping.
    dc_location : tuple
        (lat, lng) of the distribution center / home base.
    order : str
        ``"furthest_first"`` (default) or ``"closest_first"``.

    Returns
    -------
    list[str]
        Store numbers sorted by distance.  Stores without coordinates
        are appended at the end in their original order.
    """
    if not dc_location or not store_locations:
        return stores

    dc_lat, dc_lng = dc_location
    with_dist = []
    without_coords = []

    for s in stores:
        loc = store_locations.get(s)
        if loc:
            d = haversine(dc_lat, dc_lng, loc[0], loc[1])
            with_dist.append((s, d))
        else:
            without_coords.append(s)

    descending = order == "furthest_first"
    with_dist.sort(key=lambda x: x[1], reverse=descending)

    return [s for s, _ in with_dist] + without_coords


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


def sequence_runs(runs, store_locations, dc_location, order="furthest_first"):
    """Apply distance sequencing to the Stores field of each run dict.

    Modifies runs in-place and returns them for convenience.
    If locations are missing, runs are returned unchanged.
    """
    if not store_locations or not dc_location:
        return runs
    for r in runs:
        stores = [s.strip() for s in r["Stores"].split("/") if s.strip()]
        ordered = sequence_stores(stores, store_locations, dc_location, order)
        r["Stores"] = "/".join(ordered)
    return runs


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
