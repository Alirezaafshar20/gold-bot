"""
Exact / path replay from Flight Recorder tape + M1 bars.

Modes:
  tape  — replay decisions exactly as logged (fills at journal prices)
  sim   — from `armed` events only, re-simulate market-in-band fills on M1
          then score vs tape fills (fill parity) and M1 exits (path parity)

Usage:
  # After live (or live_replay --record) has written reports/flight_recorder*.jsonl:
  python exact_replay.py --day 2026-07-20 --dual --mode sim
  python exact_replay.py --day 2026-07-20 --tf M5 --mode tape

Parity scores printed:
  decision — armed ideas covered (tf|rule|side multiset)
  fill     — timed fill match vs tape
  path     — exit reason match on M1 for matched fills
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import MetaTrader5 as mt5
import strategy as S
import symbol_profiles as P
import mt5_data as M
import live_flight_recorder as FR
from live_smc import TF_MIN, price_in_entry_zone


def _norm_rule(raw: str) -> str:
    s = str(raw or "").upper().replace("SMC-", "").split("@", 1)[0]
    s = s.replace("_", "-")
    if s in ("CHREV", "CH-REV-L"):
        return "CH-REV"
    if s in ("WYCKSOW",):
        return "WYCK-SOW"
    return s


def _norm_side(raw: str) -> str:
    s = str(raw or "").lower()
    if s in ("buy", "long", "0"):
        return "long"
    if s in ("sell", "short", "1"):
        return "short"
    return s


def _idea_key(e: dict) -> tuple:
    return (
        str(e.get("tf", "")).upper(),
        _norm_rule(e.get("rule", "")),
        _norm_side(e.get("side", "")),
    )


def load_tape(day: str, tfs: list[str], paths: list[str] | None = None) -> list[dict]:
    events = []
    if paths:
        for p in paths:
            events.extend(FR.read_events(p, day=day))
    else:
        for tf in tfs:
            events.extend(FR.read_events(tf=tf, day=day))
    events.sort(key=lambda e: str(e.get("ts", "")))
    return events


def _spread_half(spread: float) -> float:
    return max(float(spread), 0.0) * 0.5


def simulate_from_armed(armed: list[dict], m1: pd.DataFrame, spread: float,
                        max_hold_bars: int, tf_min: int) -> list[dict]:
    """Market-in-band fill on M1 after arm, then M1 exit path."""
    ctx = S.M1Ctx(m1, m1.index)  # exit helpers need ctx aligned; use m1 index
    # Rebuild a proper M1Ctx with dummy signal index = m1 index
    trades = []
    h = _spread_half(spread)
    for z in armed:
        t0 = pd.Timestamp(z["ts"])
        side = _norm_side(z.get("side"))
        zlo = float(z.get("zone_lo", z.get("proximal", 0)))
        zhi = float(z.get("zone_hi", z.get("proximal", 0)))
        sl = float(z.get("sl", 0))
        tp = z.get("tp")
        tp = float(tp) if tp not in (None, "") else None
        proximal = float(z.get("proximal", (zlo + zhi) / 2))
        # scan M1 after arm
        mask = m1.index > t0
        fill_ts = None
        entry = None
        for ts, row in m1.loc[mask].iterrows():
            lo, hi = float(row["low"]), float(row["high"])
            mid = float(row["close"])
            bid, ask = mid - h, mid + h
            tick_hit = price_in_entry_zone(side, zlo, zhi, bid, ask)
            wick_hit = lo <= zhi and hi >= zlo
            if not (tick_hit or wick_hit):
                # invalidate-ish: skip for now (tape has invalidate events)
                continue
            if side == "long":
                entry = ask if tick_hit else min(max(lo, zlo), zhi)
            else:
                entry = bid if tick_hit else min(max(hi, zlo), zhi)
            risk = (entry - sl) if side == "long" else (sl - entry)
            if risk <= 0:
                continue
            fill_ts = pd.Timestamp(ts)
            break
        if fill_ts is None or entry is None:
            continue
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        # exit on M1
        j0 = int(m1.index.searchsorted(fill_ts))
        jend = min(len(m1), j0 + max(1, max_hold_bars * max(tf_min, 1)))
        # use strategy exit helpers via arrays
        class _Ctx:
            pass
        c = _Ctx()
        c.n = len(m1)
        c.h = m1["high"].to_numpy(dtype=float)
        c.l = m1["low"].to_numpy(dtype=float)
        c.c = m1["close"].to_numpy(dtype=float)
        c.t = m1.index.to_numpy()
        if side == "long":
            R, jx, reason = S._exit_long_m1(
                c, j0, jend, entry, sl, risk, tp=tp,
                use_trail=False, breakeven_only=True, trig=1.0)
        else:
            R, jx, reason = S._exit_short_m1(
                c, j0, jend, entry, sl, risk, tp=tp,
                use_trail=False, breakeven_only=True, trig=1.0)
        trades.append({
            "tf": z.get("tf"),
            "rule": _norm_rule(z.get("rule")),
            "side": side,
            "entry_time": fill_ts,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "R": R,
            "exit": reason,
            "exit_time": pd.Timestamp(m1.index[min(jx, len(m1) - 1)]),
            "zone_id": z.get("zone_id", ""),
            "proximal": proximal,
            "source": "sim",
        })
    return trades


def trades_from_tape(events: list[dict]) -> list[dict]:
    """Build trade list from fill (+ optional close) events on the tape."""
    fills = [e for e in events if e.get("event") == "fill"]
    closes = [e for e in events if e.get("event") == "close"]
    used_close = set()
    out = []
    for f in fills:
        side = _norm_side(f.get("side"))
        rule = _norm_rule(f.get("rule"))
        tf = str(f.get("tf", "")).upper()
        et = pd.Timestamp(f.get("ts"))
        entry = float(f.get("entry", f.get("proximal", 0)))
        # match a close by rule/side shortly after
        reason = "open"
        R = None
        exit_t = None
        for i, c in enumerate(closes):
            if i in used_close:
                continue
            if _norm_rule(c.get("rule")) != rule or _norm_side(c.get("side")) != side:
                continue
            if str(c.get("tf", tf)).upper() != tf:
                continue
            ct = pd.Timestamp(c.get("ts"))
            if ct < et:
                continue
            reason = c.get("reason") or "unknown"
            R = c.get("R")
            exit_t = ct
            used_close.add(i)
            break
        out.append({
            "tf": tf, "rule": rule, "side": side,
            "entry_time": et, "entry": entry,
            "sl": f.get("sl"), "tp": f.get("tp"),
            "R": R, "exit": reason, "exit_time": exit_t,
            "zone_id": f.get("zone_id", ""),
            "proximal": f.get("proximal"),
            "source": "tape",
        })
    return out


def score_decision(armed_tape: list[dict], armed_or_sim: list[dict]) -> dict:
    cg = Counter(_idea_key(e) for e in armed_tape)
    cr = Counter(_idea_key(e) for e in armed_or_sim)
    keys = set(cg) | set(cr)
    overlap = sum(min(cg[k], cr[k]) for k in keys)
    n_gt = sum(cg.values()) or 1
    n_rp = sum(cr.values()) or 1
    recall = overlap / n_gt * 100
    prec = overlap / n_rp * 100
    f1 = (2 * recall * prec / (recall + prec)) if (recall + prec) else 0.0
    return {"overlap": overlap, "recall": recall, "precision": prec, "score": f1,
            "n_gt": int(sum(cg.values())), "n_rp": int(sum(cr.values()))}


def score_fill(tape_fills: list[dict], sim_fills: list[dict],
               time_tol_min: float = 30.0, entry_tol: float = 8.0) -> dict:
    matched = 0
    used = set()
    details = []
    for g in sorted(tape_fills, key=lambda x: x["entry_time"]):
        best = None
        best_sc = None
        for i, r in enumerate(sim_fills):
            if i in used:
                continue
            if r["tf"] != g["tf"] or r["side"] != g["side"] or r["rule"] != g["rule"]:
                continue
            dtm = abs((r["entry_time"] - g["entry_time"]).total_seconds()) / 60.0
            if dtm > time_tol_min:
                continue
            dpx = abs(float(r["entry"]) - float(g["entry"]))
            if dpx > entry_tol:
                continue
            sc = dtm / time_tol_min + dpx / max(entry_tol, 1e-9)
            if best_sc is None or sc < best_sc:
                best_sc, best = sc, i
        if best is None:
            details.append({"gt": g, "status": "miss"})
            continue
        used.add(best)
        matched += 1
        r = sim_fills[best]
        details.append({
            "gt": g, "sim": r, "status": "ok",
            "dt_min": abs((r["entry_time"] - g["entry_time"]).total_seconds()) / 60.0,
            "d_entry": float(r["entry"]) - float(g["entry"]),
        })
    n_gt = len(tape_fills)
    n_rp = len(sim_fills)
    recall = matched / n_gt * 100 if n_gt else 0.0
    prec = matched / n_rp * 100 if n_rp else 0.0
    f1 = (2 * recall * prec / (recall + prec)) if (recall + prec) else 0.0
    return {"matched": matched, "n_gt": n_gt, "n_rp": n_rp,
            "recall": recall, "precision": prec, "score": f1, "details": details}


def score_path(fill_details: list[dict]) -> dict:
    """Among timed fill matches that have exit on both sides, compare exit reason."""
    ok = 0
    n = 0
    for d in fill_details:
        if d.get("status") != "ok":
            continue
        g, r = d["gt"], d.get("sim")
        if not r:
            continue
        ge, re = str(g.get("exit") or ""), str(r.get("exit") or "")
        if not ge or ge in ("open", "") or not re:
            continue
        n += 1
        # normalize
        def norm(x):
            x = x.lower()
            if x in ("sl", "stop"):
                return "sl"
            if x in ("tp",):
                return "tp"
            if x in ("be", "breakeven"):
                return "be"
            if "time" in x or x == "eod":
                return "time"
            return x
        if norm(ge) == norm(re):
            ok += 1
    score = ok / n * 100 if n else 0.0
    return {"n": n, "ok": ok, "score": score}


def fetch_m1(asset: str, day: str, days: int = 3):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 failed: {mt5.last_error()}")
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    spread = P.live_spread(sym, prof, mt5)
    # enough history
    _, _, m1, _ = M.fetch_pair(sym, "M5", mt5=mt5, days=max(days, 2))
    # filter around day
    d0 = pd.Timestamp(day)
    d1 = d0 + pd.Timedelta(days=1)
    # keep warmup before day
    m1 = m1[(m1.index >= d0 - pd.Timedelta(days=1)) & (m1.index < d1 + pd.Timedelta(hours=6))]
    return sym, spread, m1


def main():
    ap = argparse.ArgumentParser(description="Exact/path replay from flight recorder + M1")
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tf", default="", help="M5 or M15 (default: both with --dual)")
    ap.add_argument("--dual", action="store_true", help="Load M5+M15 tapes")
    ap.add_argument("--mode", choices=["tape", "sim"], default="sim")
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--days", type=int, default=3, help="M1 history window")
    ap.add_argument("--time-tol", type=float, default=30.0)
    ap.add_argument("--entry-tol", type=float, default=8.0)
    ap.add_argument("--jsonl", action="append", default=None,
                    help="Explicit flight jsonl path (repeatable)")
    args = ap.parse_args()

    if args.dual or not args.tf:
        tfs = ["M5", "M15"]
    else:
        tfs = [args.tf.upper().replace("5M", "M5").replace("15M", "M15")]

    tape = load_tape(args.day, tfs, args.jsonl)
    if not tape:
        print(f"\n  No flight events for {args.day} in {tfs}.")
        print("  Run live_portfolio (or: python live_replay.py --days 2 --dual --record)")
        print(f"  Expected files: {[FR.recorder_path(t) for t in tfs]}")
        sys.exit(1)

    armed = [e for e in tape if e.get("event") == "armed"]
    tape_fills = trades_from_tape(tape)
    print(f"\n  EXACT REPLAY  |  day={args.day}  mode={args.mode}  tfs={','.join(tfs)}")
    print(f"  Tape events: {len(tape)}  armed={len(armed)}  fills={len(tape_fills)}")
    print(f"  Summary: {FR.summarize(tape)}")

    if args.mode == "tape":
        print("\n  --- TAPE FILLS (as logged) ---")
        for i, t in enumerate(tape_fills, 1):
            et = pd.Timestamp(t["entry_time"]).strftime("%H:%M")
            print(f"  {i:>3}  {t['tf']:<4} {t['rule']:<10} {t['side']:<5} "
                  f"{et}  @{t['entry']:.2f}  exit={t['exit']}")
        print("\n  Tape mode reproduces logged decisions; use --mode sim for M1 fill/path scores.\n")
        mt5.shutdown()
        return

    sym, spread, m1 = fetch_m1(args.asset, args.day, args.days)
    print(f"  M1 bars: {len(m1)}  symbol={sym}  spread={spread}")

    # Simulate fills from armed zones on M1
    sim_all = []
    for tf in tfs:
        armed_tf = [e for e in armed if str(e.get("tf", "")).upper() == tf]
        tf_min = TF_MIN.get(tf, 5)
        max_hold = 48  # bars of signal tf approx
        sim_all.extend(
            simulate_from_armed(armed_tf, m1, spread, max_hold, tf_min))

    dec = score_decision(armed, armed)  # tape vs itself = 100; also report sim idea cover
    dec_sim = score_decision(armed, [
        {"tf": t["tf"], "rule": t["rule"], "side": t["side"]} for t in sim_all
    ])
    fill = score_fill(tape_fills, sim_all, args.time_tol, args.entry_tol)
    # Enrich tape fills with exit from sim match for path score when tape has close
    path = score_path(fill["details"])

    print(f"\n  === PARITY SCORES {args.day} ===")
    print(f"  Decision (armed ideas → sim fills cover):  "
          f"recall={dec_sim['recall']:.1f}%  prec={dec_sim['precision']:.1f}%  "
          f"F1={dec_sim['score']:.1f}%")
    print(f"  Fill     (tape fill ↔ M1 sim fill):        "
          f"recall={fill['recall']:.1f}%  prec={fill['precision']:.1f}%  "
          f"F1={fill['score']:.1f}%  matched={fill['matched']}/{fill['n_gt']}")
    print(f"  Path     (exit reason on matched fills):   "
          f"{path['score']:.1f}%  ({path['ok']}/{path['n']})")
    # Trust: emphasize covering tape fills (recall) + path; F1 penalizes
    # extra sim fills from armed-but-unfilled zones (expected until invalidate
    # timing is fully shared).
    if path["n"] > 0:
        trust = (0.25 * dec_sim["score"] + 0.45 * fill["recall"]
                 + 0.15 * fill["score"] + 0.15 * path["score"])
        print(f"  TRUST SCORE (dec/fill-recall/fill-F1/path): {trust:.1f}%")
    else:
        trust = 0.4 * dec_sim["score"] + 0.6 * fill["recall"]
        print(f"  TRUST SCORE (dec/fill-recall; path n/a): {trust:.1f}%")
    gate = 85.0
    print(f"  Target ≥ {gate:.0f}% → {'PASS' if trust >= gate else 'BELOW TARGET'}")

    miss = [d for d in fill["details"] if d["status"] == "miss"]
    if miss:
        print(f"\n  --- FILL MISS ({len(miss)}) ---")
        for d in miss[:12]:
            g = d["gt"]
            et = pd.Timestamp(g["entry_time"]).strftime("%H:%M")
            print(f"  ✗ {g['tf']} {g['rule']} {g['side']} {et} @{g['entry']:.2f}")
    print()
    mt5.shutdown()
    if trust < gate:
        sys.exit(2)


if __name__ == "__main__":
    main()
