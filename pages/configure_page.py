"""Configure page — manage store lineup and ready times."""

import logging
import re
import pandas as pd
import streamlit as st
from db_utils import db_connection, save_settings

logger = logging.getLogger(__name__)


def _is_store_number(val: str) -> bool:
    """Return True if *val* looks like a store number (purely numeric, 1-5 digits)."""
    return bool(re.fullmatch(r"\d{1,5}", val))


def _normalize_time(val) -> str:
    """Best-effort convert a cell value to an HH:MM string."""
    if pd.isna(val):
        return "05:00"
    if hasattr(val, "strftime"):
        return val.strftime("%H:%M")
    s = str(val).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return "05:00"


def _extract_stores_from_file(store_file) -> dict[str, str]:
    """Extract store numbers and ready times from an uploaded CSV/Excel file."""
    if store_file.name.endswith(".csv"):
        frames = [pd.read_csv(store_file, header=None)]
    else:
        xls = pd.read_excel(store_file, sheet_name=None, header=None)
        frames = list(xls.values())

    result: dict[str, str] = {}

    for frame in frames:
        header_idx = None
        for idx, row in frame.iterrows():
            if any(str(v).strip().lower() == "store" for v in row.values):
                header_idx = idx
                break
        if header_idx is None:
            continue

        headers = [str(v).strip() for v in frame.iloc[header_idx].values]

        store_cols: list[int] = []
        rtime_cols: list[int] = []
        for i, h in enumerate(headers):
            if h.lower() == "store":
                store_cols.append(i)
            else:
                norm = re.sub(r"[\s\-]", "", h.lower())
                if norm.startswith("rtime") or norm.startswith("readytime"):
                    rtime_cols.append(i)

        if not store_cols:
            continue

        pairs: list[tuple[int, int | None]] = []
        for sc in store_cols:
            matched = None
            for rc in rtime_cols:
                if rc > sc:
                    matched = rc
                    break
            pairs.append((sc, matched))

        data_rows = frame.iloc[header_idx + 1:]
        for sc, rc in pairs:
            for _, row in data_rows.iterrows():
                raw = str(row.iloc[sc]).strip()
                if re.fullmatch(r"\d+\.0", raw):
                    raw = raw.split(".")[0]
                if not _is_store_number(raw):
                    continue
                if raw in result:
                    continue
                ready_time = _normalize_time(row.iloc[rc]) if rc is not None else "05:00"
                result[raw] = ready_time

    return result


def render():
    st.header("Configure Stores")

    time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

    # --- Current lineup summary ---
    if st.session_state["stores"]:
        st.subheader(f"Current Lineup ({len(st.session_state['stores'])} stores)")
        st.info(", ".join(st.session_state["stores"]))
    else:
        st.warning("No stores configured yet. Use the editor or import a file below.")

    # --- Edit lineup manually ---
    edit_lineup = st.checkbox("Edit Store Lineup", key="edit_lineup_toggle")
    if edit_lineup:
        st.caption("Add, remove, or reorder stores and ready times.")
        if st.session_state["stores"]:
            ready_times = [
                st.session_state["store_ready_times"].get(s, "05:00")
                for s in st.session_state["stores"]
            ]
            stores_edit_df = pd.DataFrame({"Store": st.session_state["stores"], "Ready Time": ready_times})
        else:
            stores_edit_df = pd.DataFrame({"Store": pd.Series(dtype="str"), "Ready Time": pd.Series(dtype="str")})

        edited_stores_df = st.data_editor(
            stores_edit_df,
            num_rows="dynamic",
            use_container_width=True,
            column_order=["Store", "Ready Time"],
            column_config={
                "Ready Time": st.column_config.SelectboxColumn(
                    "Ready Time",
                    options=time_options,
                    default="05:00",
                    required=True,
                )
            },
            key="stores_data_editor",
        )
        edited_stores = [s.strip() for s in edited_stores_df["Store"].astype(str).tolist() if s.strip() and s.strip() != "nan"]
        ready_time_list = edited_stores_df["Ready Time"].astype(str).tolist()
        store_names = edited_stores_df["Store"].astype(str).tolist()
        st.session_state["store_ready_times"] = {
            s.strip(): t for s, t in zip(store_names, ready_time_list)
            if s.strip() and s.strip() != "nan"
        }
        st.session_state["stores"] = edited_stores
        try:
            with db_connection() as (conn, c):
                save_settings({
                    "stores": st.session_state["stores"],
                    "store_ready_times": st.session_state["store_ready_times"],
                }, conn, c)
        except Exception:
            logger.exception("Failed to persist store settings")

    # --- Import from file ---
    with st.expander("Import stores from file"):
        store_file = st.file_uploader("Upload stores (CSV/Excel)", type=["csv", "xlsx"], key="store_file_uploader")
        if store_file:
            imported = _extract_stores_from_file(store_file)
            if imported:
                current_set = set(st.session_state["stores"])
                incoming_set = set(imported.keys())

                adds = sorted(incoming_set - current_set)
                removes = sorted(current_set - incoming_set)
                keeps = sorted(current_set & incoming_set)

                if not adds and not removes:
                    st.success("Uploaded store list matches current lineup. No changes needed.")
                else:
                    st.write("**Review proposed changes:**")

                    file_id = f"{store_file.name}_{store_file.size}"
                    if st.session_state.get("_import_file_id") != file_id:
                        st.session_state["_import_file_id"] = file_id
                        st.session_state["_pending_adds"] = {s: True for s in adds}
                        st.session_state["_pending_removes"] = {s: True for s in removes}
                        st.session_state["_imported_ready_times"] = dict(imported)

                    if adds:
                        st.markdown(f"**Stores to ADD ({len(adds)}):**")
                        for s in adds:
                            rt = imported.get(s, "05:00")
                            st.session_state["_pending_adds"][s] = st.checkbox(
                                f"Add store {s} (ready {rt})",
                                value=st.session_state.get("_pending_adds", {}).get(s, True),
                                key=f"import_add_{s}",
                            )

                    if removes:
                        st.markdown(f"**Stores to REMOVE ({len(removes)}):**")
                        for s in removes:
                            st.session_state["_pending_removes"][s] = st.checkbox(
                                f"Remove store {s}",
                                value=st.session_state.get("_pending_removes", {}).get(s, True),
                                key=f"import_remove_{s}",
                            )

                    if keeps:
                        st.caption(f"Unchanged stores ({len(keeps)}): {', '.join(keeps)}")

                    if st.button("Apply Changes", key="apply_store_import"):
                        imported_times = st.session_state.get("_imported_ready_times", {})
                        new_stores = list(st.session_state["stores"])
                        for s, checked in st.session_state.get("_pending_adds", {}).items():
                            if checked and s not in new_stores:
                                new_stores.append(s)
                        for s, checked in st.session_state.get("_pending_removes", {}).items():
                            if checked and s in new_stores:
                                new_stores.remove(s)
                        for s in new_stores:
                            if s in imported_times:
                                st.session_state["store_ready_times"][s] = imported_times[s]
                            elif s not in st.session_state["store_ready_times"]:
                                st.session_state["store_ready_times"][s] = "05:00"
                        st.session_state["store_ready_times"] = {
                            s: t for s, t in st.session_state["store_ready_times"].items()
                            if s in new_stores
                        }
                        added = sum(1 for v in st.session_state.get("_pending_adds", {}).values() if v)
                        removed = sum(1 for v in st.session_state.get("_pending_removes", {}).values() if v)
                        st.session_state["stores"] = new_stores
                        try:
                            with db_connection() as (conn, c):
                                save_settings({
                                    "stores": st.session_state["stores"],
                                    "store_ready_times": st.session_state["store_ready_times"],
                                }, conn, c)
                        except Exception:
                            logger.exception("Failed to persist imported stores")
                        for k in ("_import_file_id", "_pending_adds", "_pending_removes", "_imported_ready_times"):
                            st.session_state.pop(k, None)
                        st.success(f"Applied: {added} added, {removed} removed. {len(new_stores)} stores total.")
                        st.rerun()
            else:
                st.warning("No store numbers found. Make sure the file has a column named **Store**.")

    if not st.session_state["stores"]:
        st.warning("Add at least one store to get started.")
