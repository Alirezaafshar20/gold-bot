"""Are the stops wide enough for the excursions the market actually makes?

regime_audit measures how far price travels for and against a direction after
the label is set. This tool measures where the book's stops sit on that same
scale, so the two can be put side by side.

The comparison matters because the two are set independently. Stop distance
comes out of zone geometry — the distal edge of whatever structure the rule
found, plus a buffer — and nothing in that calculation knows how far gold
normally retraces before continuing. If a cell's typical adverse excursion is
1.9 ATR and the rule habitually places its stop 1.2 ATR away, the trade is
resolved by noise before the edge the label identified has a chance to appear,
and no amount of rule selection fixes it.

The headline number is `stopped %`: the share of bars in that cell whose
adverse excursion alone would have taken out a stop at the book's typical
distance, before any favourable move. It is an upper bound rather than a
prediction, since a real entry is a limit fill at a better price than the close
this is measured from, but it is directly comparable between long and short —
which is what the question needs.

Usage: python stop_audit.py reports/rp_j_m15.jsonl [more.jsonl ...]
"""
import collections
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dukascopy_loader as DK
import floating_config as FC
import regime as R
from regime_audit import atr, build_map, forward_excursions, resample

HORIZON = 12       # hours; the span regime_audit found the label most additive


def fills(path):
    """Every fill, carrying its arm-time regime and, where it can be chained,
    the R it eventually closed at.

    Geometry lives on the fill event and the outcome lives on the close event,
    so the two have to be joined for "did stop width predict the result" to be
    answerable. They do not share a key: a position's close carries an 8000000xx
    ticket and an empty zone_id, while the zone that produced it is only named on
    the fill. The link is ordering — a fill is followed by the open_seen for the
    position it created — which is the same chain journal_attrib walks.
    """
    zone_regime, by_zone, tk2zone = {}, {}, {}
    pending = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        ev = e.get("event")
        if ev == "armed":
            zone_regime[e.get("zone_id")] = e.get("regime")
        elif ev == "fill":
            entry, sl, tp = e.get("entry"), e.get("sl"), e.get("tp")
            if entry is None or sl is None:
                continue
            zid = e.get("zone_id")
            by_zone[zid] = {
                "ts": pd.Timestamp(e["ts"]),
                "side": e.get("side"), "rule": e.get("rule"),
                "regime": zone_regime.get(zid) or "?",
                "entry": float(entry), "sl": float(sl),
                "tp": float(tp) if tp is not None else np.nan,
                "R": np.nan,
            }
            pending = zid
        elif ev == "open_seen":
            tk = e.get("ticket")
            if tk is not None and tk not in tk2zone and pending is not None:
                tk2zone[tk] = pending
            pending = None
        elif ev == "close":
            z = tk2zone.get(e.get("ticket")) or e.get("zone_id")
            if z in by_zone:
                by_zone[z]["R"] = float(e.get("R") or 0.0)
    return list(by_zone.values())


def width_vs_outcome(tr):
    """Split each side into stop-width terciles and score them.

    Stop distance is not a free parameter — it falls out of the zone the rule
    found — so it doubles as a description of the setup. If the terciles score
    differently, stop width is a usable filter on entries the rules already
    produce, which is cheaper than replacing a rule.
    """
    print()
    print("  -- does stop width predict the outcome? --")
    for side in ("long", "short"):
        s = [t for t in tr if t["side"] == side
             and np.isfinite(t.get("sl_atr", np.nan)) and np.isfinite(t["R"])]
        if len(s) < 15:
            continue
        w = np.array([t["sl_atr"] for t in s])
        q = np.quantile(w, [0.0, 1 / 3, 2 / 3])
        print(f"\n  {side}: {len(s)} trades, terciles split at "
              f"{q[1]:.2f} and {q[2]:.2f} ATR")
        print(f"  {'bucket':<16} {'n':>4} {'WR%':>7} {'PF':>7} {'netR':>8} "
              f"{'mean stop':>10}")
        print("  " + "-" * 56)
        for lo, hi, nm in ((q[0], q[1], "tight"), (q[1], q[2], "mid"),
                           (q[2], np.inf, "wide")):
            b = [t for t in s if lo <= t["sl_atr"] < hi] if np.isfinite(hi) \
                else [t for t in s if t["sl_atr"] >= lo]
            if not b:
                continue
            r = [t["R"] for t in b]
            gp = sum(x for x in r if x > 0)
            gl = -sum(x for x in r if x < 0)
            p = gp / gl if gl > 0 else float("inf")
            ps = "    inf" if p == float("inf") else f"{p:7.2f}"
            print(f"  {nm + ' stop':<16} {len(b):>4} "
                  f"{100 * sum(1 for x in r if x > 0) / len(b):>7.1f} {ps} "
                  f"{sum(r):>+8.2f} {np.mean([t['sl_atr'] for t in b]):>10.2f}")


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    tr = []
    for p in paths:
        tr += fills(p)
    if not tr:
        raise SystemExit("no fills found in those journals")

    lo = min(t["ts"] for t in tr)
    hi = max(t["ts"] for t in tr)
    print("=" * 92)
    print(f"  STOP DISTANCE vs MARKET EXCURSION — {len(tr)} fills   "
          f"{lo.date()} .. {hi.date()}")
    print("=" * 92)

    start = (lo - pd.Timedelta(days=120)).strftime("%Y-%m")
    print(f"  loading H1 from {start} for ATR and excursions ...")
    _, m1 = DK.load_pair("XAUUSD", start=start, end=None, tf="M15")
    h1 = resample(m1, "1h")
    h4 = resample(m1, "4h")
    high = h1["high"].to_numpy(float)
    low = h1["low"].to_numpy(float)
    close = h1["close"].to_numpy(float)
    a = atr(high, low, close)
    idx = h1.index
    fwd_hi, fwd_lo, _ = forward_excursions(high, low, close, HORIZON)
    rmap = build_map(h1, h4, FC.REGIME["regime_detector"])
    labels = np.asarray(rmap.labels[:len(close)], dtype=object)

    with np.errstate(invalid="ignore", divide="ignore"):
        mae_long = (close - fwd_lo) / a
        mae_short = (fwd_hi - close) / a

    for t in tr:
        j = int(idx.searchsorted(t["ts"], side="right")) - 1
        t["atr"] = float(a[j]) if 0 <= j < len(a) else np.nan
        if t["atr"] and np.isfinite(t["atr"]) and t["atr"] > 0:
            t["sl_atr"] = abs(t["entry"] - t["sl"]) / t["atr"]
            t["tp_atr"] = abs(t["tp"] - t["entry"]) / t["atr"] \
                if np.isfinite(t["tp"]) else np.nan
        else:
            t["sl_atr"] = t["tp_atr"] = np.nan

    print()
    print(f"  {'cell':<22} {'fills':>6} {'stop':>7} {'target':>8} {'RR':>6}   "
          f"{'mkt MAE':>8} {'stopped %':>10}")
    print("  " + "-" * 74)

    cells = collections.defaultdict(list)
    for t in tr:
        cells[(t["side"], t["regime"])].append(t)

    rows = []
    for (side, reg) in sorted(cells, key=lambda k: (k[0], k[1])):
        rows_c = [t for t in cells[(side, reg)] if np.isfinite(t.get("sl_atr", np.nan))]
        if not rows_c:
            continue
        sl_m = float(np.mean([t["sl_atr"] for t in rows_c]))
        tp_m = float(np.nanmean([t["tp_atr"] for t in rows_c]))
        mae = mae_long if side == "long" else mae_short
        m = np.isfinite(mae) & np.isfinite(a) & (a > 0)
        if reg != "?":
            m &= (labels == reg)
        pool = mae[m]
        mkt = float(np.mean(pool)) if len(pool) else np.nan
        stopped = 100.0 * float(np.mean(pool >= sl_m)) if len(pool) else np.nan
        rows.append((side, reg, len(rows_c), sl_m, tp_m, mkt, stopped))
        flag = "  <<< stop inside the noise" if stopped > 55 else ""
        print(f"  {side + ' in ' + reg:<22} {len(rows_c):>6} {sl_m:>7.2f} "
              f"{tp_m:>8.2f} {tp_m/sl_m if sl_m else 0:>6.2f}   "
              f"{mkt:>8.2f} {stopped:>9.1f}%{flag}")

    print()
    print("  -- pooled by side --")
    print(f"  {'side':<22} {'fills':>6} {'stop':>7} {'target':>8} {'RR':>6}   "
          f"{'mkt MAE':>8} {'stopped %':>10}")
    print("  " + "-" * 74)
    for side in ("long", "short"):
        rows_c = [t for t in tr
                  if t["side"] == side and np.isfinite(t.get("sl_atr", np.nan))]
        if not rows_c:
            continue
        sl_m = float(np.mean([t["sl_atr"] for t in rows_c]))
        tp_m = float(np.nanmean([t["tp_atr"] for t in rows_c]))
        mae = mae_long if side == "long" else mae_short
        pool = mae[np.isfinite(mae)]
        mkt = float(np.mean(pool))
        stopped = 100.0 * float(np.mean(pool >= sl_m))
        print(f"  {side:<22} {len(rows_c):>6} {sl_m:>7.2f} {tp_m:>8.2f} "
              f"{tp_m/sl_m if sl_m else 0:>6.2f}   {mkt:>8.2f} {stopped:>9.1f}%")

    width_vs_outcome(tr)

    print()
    print(f"  stop / target : mean distance from entry, in ATR(H1,14) at fill time")
    print(f"  mkt MAE       : mean adverse excursion over {HORIZON}h for that side")
    print(f"                  and label, measured on every bar of the archive")
    print(f"  stopped %     : share of those bars whose adverse excursion alone")
    print(f"                  reaches the book's typical stop for that cell")
    print()
    print("  A cell where the stop sits below the market's own MAE is resolved by")
    print("  noise. Comparing the long and short rows of that column says whether")
    print("  the two sides are being held to the same standard.")


if __name__ == "__main__":
    main()
