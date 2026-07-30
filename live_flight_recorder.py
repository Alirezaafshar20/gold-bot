"""
Live Flight Recorder — append-only JSONL of every zone decision.

This is the ground-truth tape for exact_replay + parity scoring.
One line = one event. Paths are per-TF (M5/M15) via set_tf().

Events:
  session | bar | cand | armed | skip_register | limit_armed | limit_cancel
  fill | skip_fill | invalidate | expire | touch | open_seen | close

`bar` and `cand` are the decision journal: one `bar` per closed candle with a
fingerprint of the exact window the engine saw, and one `cand` per rejected
setup with the gate that killed it. Live and replay both emit them from the
shared register_zones, so parity_diff can tell a data mismatch (different
fingerprint) from a logic mismatch (same fingerprint, different verdict).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()
_TF = "M15"
_PATH_OVERRIDE: str | None = None
_ENABLED = True
_JOURNAL = True
# Set by live_portfolio to its injectable clock, so events logged without an
# explicit ts land on replay's simulated time instead of real wall time.
_CLOCK = None

DEFAULT_DIR = "reports"


def set_tf(tf: str) -> None:
    global _TF
    _TF = str(tf or "M15").upper()


def set_path(path: str | None) -> None:
    global _PATH_OVERRIDE
    _PATH_OVERRIDE = path


def set_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def set_journal(on: bool) -> None:
    """Toggle the per-bar / per-candidate decision journal."""
    global _JOURNAL
    _JOURNAL = bool(on)


def journal_on() -> bool:
    return bool(_ENABLED and _JOURNAL)


def set_clock(fn) -> None:
    """Zero-arg callable returning 'now' for events logged without a ts."""
    global _CLOCK
    _CLOCK = fn


def recorder_path(tf: str | None = None) -> str:
    if _PATH_OVERRIDE:
        return _PATH_OVERRIDE
    t = str(tf or _TF).upper()
    if t in ("", "M15"):
        return os.path.join(DEFAULT_DIR, "flight_recorder.jsonl")
    return os.path.join(DEFAULT_DIR, f"flight_recorder_{t}.jsonl")


def _iso(ts) -> str:
    if ts is None:
        if _CLOCK is not None:
            try:
                return _CLOCK().isoformat(timespec="seconds")
            except Exception:
                pass
        return dt.datetime.now().isoformat(timespec="seconds")
    if isinstance(ts, dt.datetime):
        return ts.isoformat(timespec="seconds")
    try:
        import pandas as pd
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


def _tick_snapshot(symbol: str | None = None) -> dict:
    try:
        import MetaTrader5 as mt5
        if symbol:
            t = mt5.symbol_info_tick(symbol)
            if t:
                return {"bid": float(t.bid), "ask": float(t.ask)}
    except Exception:
        pass
    return {}


def log_event(event: str, *, asset: str = "", tf: str | None = None,
              rule: str = "", side: str = "", reason: str = "",
              zone_id: str = "", ts=None, symbol: str | None = None,
              **fields: Any) -> None:
    """Append one JSONL event. Never raises into the trading loop."""
    if not _ENABLED:
        return
    try:
        tfs = str(tf or _TF).upper()
        row = {
            "ts": _iso(ts),
            "event": str(event),
            "asset": asset,
            "tf": tfs,
            "rule": rule,
            "side": side,
            "reason": reason,
            "zone_id": zone_id,
        }
        tick = _tick_snapshot(symbol)
        if tick:
            row["bid"] = tick["bid"]
            row["ask"] = tick["ask"]
        for k, v in fields.items():
            if v is None:
                continue
            if hasattr(v, "isoformat"):
                row[k] = _iso(v)
            elif isinstance(v, (int, float, str, bool)):
                row[k] = v
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    row[k] = str(v)
        path = recorder_path(tfs)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        try:
            print(f"[FLIGHT] log failed: {exc}")
        except Exception:
            pass


def frame_fingerprint(df, upto=None) -> dict:
    """Identify the exact bar window an engine decided on.

    Returned as n/first/last/sha so a diff can separate "both engines saw the
    same candles and disagreed" from "they were never looking at the same
    data". `upto` excludes the forming bar, whose values are live-only noise.
    """
    try:
        import numpy as np
        sub = df if upto is None else df.iloc[: int(upto) + 1]
        cols = [c for c in ("open", "high", "low", "close") if c in sub.columns]
        arr = np.ascontiguousarray(
            np.round(sub[cols].to_numpy(dtype="float64"), 5))
        return {
            "n": int(len(sub)),
            "first": _iso(sub.index[0]),
            "last": _iso(sub.index[-1]),
            "sha": hashlib.sha1(arr.tobytes()).hexdigest()[:12],
        }
    except Exception:
        return {}


def log_session(*, asset: str, tf: str, mode: str, **extra) -> None:
    """Startup marker: which config this tape was produced under."""
    log_event("session", asset=asset, tf=tf, reason=mode, **extra)


def log_bar(st, *, df, upto, ts, gate: str, htf: dict | None = None, **extra):
    """One line per closed candle: the window, the regime, the verdict."""
    if not journal_on():
        return
    fp = frame_fingerprint(df, upto)
    fields = {
        "gate": gate,
        "n_bars": fp.get("n"),
        "win_first": fp.get("first"),
        "win_last": fp.get("last"),
        "win_sha": fp.get("sha"),
    }
    for tf_name, hfp in (htf or {}).items():
        key = str(tf_name).lower()
        fields[f"{key}_n"] = hfp.get("n")
        fields[f"{key}_last"] = hfp.get("last")
        fields[f"{key}_sha"] = hfp.get("sha")
    fields.update(extra)
    log_event("bar", asset=st.key, tf=st.tf_tag, symbol=st.sym, ts=ts,
              reason=gate, **fields)


def log_cand(st, *, rule, side, reason, ts, **extra):
    """A setup the detectors produced but a gate rejected."""
    if not journal_on():
        return
    # No symbol= on purpose: this fires many times per bar and a tick snapshot
    # would mean an MT5 round trip per rejected candidate.
    log_event("cand", asset=st.key, tf=st.tf_tag,
              rule=rule, side=side, reason=reason, ts=ts, **extra)


def log_armed(st, z, closed_time, **extra):
    log_event(
        "armed",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=z.get("tag") or z.get("rule"), side=z.get("direction"),
        zone_id=z.get("id", ""), ts=closed_time,
        proximal=z.get("proximal"), distal=z.get("distal"),
        zone_lo=z.get("zone_lo"), zone_hi=z.get("zone_hi"),
        sl=z.get("sl"), tp=z.get("tp"), risk=z.get("risk"),
        signal_bar=z.get("signal_bar"), etol=z.get("etol"),
        **extra,
    )


def log_skip_register(st, *, rule, side, reason, closed_time, **extra):
    log_event(
        "skip_register",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=rule, side=side, reason=reason, ts=closed_time, **extra,
    )


def log_skip_fill(st, z, reason: str, **extra):
    log_event(
        "skip_fill",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=z.get("tag") or z.get("rule"), side=z.get("direction"),
        zone_id=z.get("id", ""), reason=reason,
        proximal=z.get("proximal"), sl=z.get("sl"), tp=z.get("tp"),
        **extra,
    )


def log_fill(st, z, *, entry: float, via: str, ticket=None, **extra):
    log_event(
        "fill",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=z.get("tag") or z.get("rule"), side=z.get("direction"),
        zone_id=z.get("id", ""), reason=via,
        entry=entry, sl=z.get("sl"), tp=z.get("tp"),
        proximal=z.get("proximal"), ticket=ticket, via=via,
        **extra,
    )


def log_invalidate(st, z, *, close: float, bar_time, **extra):
    log_event(
        "invalidate",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=z.get("tag") or z.get("rule"), side=z.get("direction"),
        zone_id=z.get("id", ""), ts=bar_time,
        close=close, distal=z.get("distal"), proximal=z.get("proximal"),
        **extra,
    )


def log_limit(st, z, event: str, **extra):
    # Callers may pass an explicit ticket (cancel pops it off the zone first).
    ticket = extra.pop("ticket", None) or z.get("limit_ticket")
    log_event(
        event,
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=z.get("tag") or z.get("rule"), side=z.get("direction"),
        zone_id=z.get("id", ""),
        proximal=z.get("proximal"), sl=z.get("sl"), tp=z.get("tp"),
        ticket=ticket, **extra,
    )


def log_close(st, *, rule, side, entry, exit_px, R, pnl, reason, ticket=None, **extra):
    log_event(
        "close",
        asset=st.key, tf=st.tf_tag, symbol=st.sym,
        rule=rule, side=side, reason=reason,
        entry=entry, exit=exit_px, R=R, pnl_usd=pnl, ticket=ticket,
        **extra,
    )


def read_events(path: str | None = None, tf: str | None = None,
                day: str | None = None) -> list[dict]:
    path = path or recorder_path(tf)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if day:
                ts = str(ev.get("ts", ""))
                if not ts.startswith(day):
                    continue
            out.append(ev)
    return out


def summarize(events: list[dict]) -> dict:
    from collections import Counter
    c = Counter(e.get("event") for e in events)
    return {
        "n": len(events),
        "by_event": dict(c),
        "fills": c.get("fill", 0),
        "armed": c.get("armed", 0),
        "skips": c.get("skip_register", 0) + c.get("skip_fill", 0),
    }
