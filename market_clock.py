"""One canonical clock for backtest, replay and live.

Two feeds arrive on different clocks:

  Dukascopy CSV      naive UTC
  MT5 copy_rates_*   the broker's SERVER wall clock (UTC+3 on WM Markets),
                     handed over as an epoch so it *looks* like UTC

Nothing used to reconcile them, which had two consequences. Hour-gated rules
(ICT silver bullet / judas / NY open) fired three hours away from the session
they name, and `idx.normalize()` bucketed a different 24h window in each engine
— "yesterday's high" matched on barely half of all bars.

Everything now indexes in true UTC. Session hours are resolved through a real
timezone so DST moves with the exchange, and the trading day rolls over at a
configured UTC hour rather than at UTC midnight.
"""
import time
from zoneinfo import ZoneInfo

import pandas as pd

# The broker offset is re-asked periodically rather than once per process.
# European brokers sit on UTC+3 through summer and UTC+2 after the October DST
# change; a bot that has been up for weeks would otherwise keep the stale
# summer offset and mislabel every bar by an hour.
_OFFSET_TTL_SEC = 6 * 3600

_offset_cache = {"h": None, "at": 0.0, "warned": False}


def _utc_now():
    """Naive UTC. Timestamp.utcnow() is deprecated and goes away in pandas 4."""
    return pd.Timestamp.now("UTC").tz_localize(None)


def _detect_offset(mt5=None):
    """Ask the terminal for the broker clock. None when it cannot be read."""
    try:
        if mt5 is None:
            import MetaTrader5 as mt5
        for sym in ("XAUUSD@", "XAUUSD", "EURUSD"):
            if not mt5.symbol_select(sym, True):
                continue
            tick = mt5.symbol_info_tick(sym)
            if tick is None or not tick.time:
                continue
            server = pd.Timestamp(tick.time, unit="s")
            delta = (server - _utc_now()).total_seconds() / 3600.0
            return round(delta * 4) / 4          # brokers use quarter hours
    except Exception:
        return None
    return None


def _cfg():
    import floating_config as FC
    return getattr(FC, "CLOCK", {})


def session_tz():
    return ZoneInfo(_cfg().get("session_tz", "America/New_York"))


def day_anchor_hour():
    return int(_cfg().get("day_anchor_hour", 21))


def server_utc_offset(mt5=None):
    """Hours the broker's bar labels run ahead of UTC.

    A configured value wins. Otherwise the terminal is asked, and re-asked
    every few hours so a DST change on the broker side is picked up.

    A failed read never becomes a cached 0.0. That was the dangerous case: one
    unreachable tick at startup used to pin the offset at zero for the whole
    session, leaving bars on broker time while every session filter and day
    boundary believed they were UTC — silently, because the exception was
    swallowed. A failure now falls back to the last known good value and says
    so out loud.
    """
    cfg = _cfg().get("server_utc_offset")
    if cfg is not None:
        return float(cfg)

    now = time.time()
    cached = _offset_cache["h"]
    if cached is not None and now - _offset_cache["at"] < _OFFSET_TTL_SEC:
        return cached

    off = _detect_offset(mt5)
    if off is None:
        if cached is not None:
            return cached
        if not _offset_cache["warned"]:
            _offset_cache["warned"] = True
            print("[CLOCK] could not read the broker clock — treating bars as "
                  "already UTC. Set floating_config.CLOCK['server_utc_offset'] "
                  "explicitly if this persists.")
        return 0.0

    if cached is None:
        print(f"[CLOCK] broker server clock is UTC{off:+.2f}h; bars normalised to UTC")
    elif off != cached:
        print(f"[CLOCK] broker UTC offset changed {cached:+.2f}h -> {off:+.2f}h "
              f"(DST?) — bars re-normalised")
    _offset_cache["h"] = off
    _offset_cache["at"] = now
    return off


def to_utc(index, offset_hours):
    """Shift broker-clock timestamps back onto UTC."""
    idx = pd.DatetimeIndex(index)
    if not offset_hours:
        return idx
    return idx - pd.Timedelta(hours=float(offset_hours))


def session_hours(index):
    """Hour of day on the session exchange clock (DST-aware)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(session_tz()).hour


def session_hour(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(session_tz()).hour


def day_key(index):
    """Trading-day bucket: rolls over at day_anchor_hour UTC, not UTC midnight."""
    idx = pd.DatetimeIndex(index)
    return (idx - pd.Timedelta(hours=day_anchor_hour())).normalize()


def week_key(index):
    idx = pd.DatetimeIndex(index)
    return (idx - pd.Timedelta(hours=day_anchor_hour())).to_period("W")
