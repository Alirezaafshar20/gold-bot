"""
Sync pending entry zones to MT5 chart rectangles via MQL5/Files/smc_zones.csv.

Requires SMCZoneDrawer.ex5 attached to the symbol chart (see mql5/SMCZoneDrawer.mq5).
"""
from __future__ import annotations

import os
import time

ZONE_FILE = "smc_zones.csv"


def _files_dir():
    try:
        import MetaTrader5 as mt5
        info = mt5.terminal_info()
        if info and info.data_path:
            return os.path.join(info.data_path, "MQL5", "Files")
    except Exception:
        pass
    return None


def zone_file_name(tf_tag: str = "M15") -> str:
    """Per-TF file so two bots (M5+M15) don't overwrite each other's zones.
    M15 keeps the legacy name; attach a second EA on the M5 chart with
    input ZoneFile=smc_zones_M5.csv."""
    t = str(tf_tag).upper()
    return ZONE_FILE if t in ("", "M15") else f"smc_zones_{t}.csv"


def zones_csv_path(tf_tag: str = "M15"):
    d = _files_dir()
    return os.path.join(d, zone_file_name(tf_tag)) if d else None


def _to_epoch(ts) -> int:
    if ts is None:
        return int(time.time())
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp())
    import pandas as pd
    return int(pd.Timestamp(ts).timestamp())


def _write_text_atomic(path: str, text: str, retries: int = 8, delay: float = 0.05) -> bool:
    """Write via temp file + replace; retry when MT5 EA briefly locks the CSV."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    last_err = None
    for attempt in range(retries):
        try:
            with open(tmp, "w", encoding="ascii", newline="\n") as f:
                f.write(text)
            os.replace(tmp, path)
            return True
        except OSError as exc:
            last_err = exc
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt + 1 < retries:
                time.sleep(delay)
    if last_err is not None:
        raise last_err
    return False


def sync_zones(states, also_structural: bool = True, tf_tag: str = "M15"):
    """Write all pending zones to smc_zones.csv for the MQL5 drawer EA."""
    path = zones_csv_path(tf_tag)
    if not path:
        return False
    lines = ["kind,symbol,zone_id,direction,t_start,t_end,p_lo,p_hi,p_distal,tag"]
    now = int(time.time())
    for st in states:
        sym = st.sym
        for z in st.pending_zones:
            t_end = _to_epoch(z.get("expires")) if z.get("expires") else now + 3600
            t_start = now - 900
            tag = str(z.get("tag", "?")).replace(",", "_")
            lines.append(
                f"entry,{sym},{z['id']},{z['direction']},{t_start},{t_end},"
                f"{z['zone_lo']:.5f},{z['zone_hi']:.5f},{z['distal']:.5f},{tag}")
            if also_structural:
                p_lo = min(z["proximal"], z["distal"])
                p_hi = max(z["proximal"], z["distal"])
                lines.append(
                    f"struct,{sym},{z['id']},{z['direction']},{t_start},{t_end},"
                    f"{p_lo:.5f},{p_hi:.5f},{z['distal']:.5f},{tag}")
        # Zones that already produced a trade stay visible (kind=filled) so
        # every executed entry can be traced back to its rectangle.
        for z in getattr(st, "filled_zones", []) or []:
            t_fill = _to_epoch(z.get("filled_at"))
            t_start = t_fill - 3600
            t_end = t_fill + 1800
            tag = str(z.get("tag", "?")).replace(",", "_")
            lines.append(
                f"filled,{sym},{z['id']},{z['direction']},{t_start},{t_end},"
                f"{z['zone_lo']:.5f},{z['zone_hi']:.5f},{z['distal']:.5f},{tag}")
        # Invalidated zones stay too (kind=dead) so the chart matches the
        # CLI history even when a zone lived only a few minutes.
        for z in getattr(st, "dead_zones", []) or []:
            t_die = _to_epoch(z.get("died_at"))
            t_start = t_die - 3600
            t_end = t_die + 1800
            tag = str(z.get("tag", "?")).replace(",", "_")
            lines.append(
                f"dead,{sym},{z['id']},{z['direction']},{t_start},{t_end},"
                f"{z['zone_lo']:.5f},{z['zone_hi']:.5f},{z['distal']:.5f},{tag}")
    _write_text_atomic(path, "\n".join(lines) + "\n")
    return True


def clear_zones(tf_tag: str = "M15"):
    path = zones_csv_path(tf_tag)
    if not path:
        return False
    _write_text_atomic(
        path,
        "kind,symbol,zone_id,direction,t_start,t_end,p_lo,p_hi,p_distal,tag\n")
    return True
