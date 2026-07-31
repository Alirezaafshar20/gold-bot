"""Catalogue every 60-day window in the archive by what the market actually did.

Two windows were used to reach the current conclusions: one where gold fell 8.8%
and one where it rose 28.6%. That is enough to separate a rule that loses to the
market from one that loses regardless, but it says nothing about a sideways
market, and a sideways market is where the book spends 60% of its bars.

This tool describes the whole archive so windows can be chosen on evidence rather
than convenience. For each candidate it reports the net price move, the realised
efficiency of that move, and how the regime detector labelled it — because "range"
has two meanings that matter separately here. A window can be flat end-to-end
while trending hard in both directions along the way (efficient, and the detector
will say TREND), or it can genuinely chop (inefficient, detector says RANGE). Only
the second kind tests what a range window is supposed to test.

Usage
  python window_census.py                  # every 60-day window, stepped monthly
  python window_census.py --days 60 --step 30
  python window_census.py --pick 6         # suggest a stratified set
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dukascopy_loader as DK
import floating_config as FC
from regime_audit import build_map, resample


def census(h1, h4, days, step):
    rmap = build_map(h1, h4, FC.REGIME["regime_detector"])
    labels = np.asarray(rmap.labels[:len(h1)], dtype=object)
    close = h1["close"].to_numpy(float)
    idx = pd.DatetimeIndex(h1.index)

    out = []
    end = idx[-1]
    cur = idx[0] + pd.Timedelta(days=days)
    while cur <= end:
        lo = cur - pd.Timedelta(days=days)
        m = (idx >= lo) & (idx <= cur)
        n = int(m.sum())
        if n < days * 8:          # a 60d window holds ~1000 tradable H1 bars
            cur += pd.Timedelta(days=step)
            continue
        c = close[m]
        lab = labels[m]
        net = c[-1] / c[0] - 1.0
        path = np.abs(np.diff(c)).sum()
        er = abs(c[-1] - c[0]) / path if path > 0 else 0.0
        cov = {k: 100.0 * float(np.mean(lab == k))
               for k in ("TREND_UP", "TREND_DOWN", "RANGE")}
        out.append({
            "end": cur, "start": lo, "bars": n, "net": net * 100, "er": er,
            "tu": cov["TREND_UP"], "td": cov["TREND_DOWN"], "rg": cov["RANGE"],
            "dd": 100.0 * float((np.maximum.accumulate(c) - c).max()
                                / np.maximum.accumulate(c).max()),
        })
        cur += pd.Timedelta(days=step)
    return out


def kind(w):
    """Label the window the way a person would describe it."""
    if w["net"] >= 8 and w["er"] >= 0.18:
        return "RISING"
    if w["net"] <= -6 and w["er"] >= 0.15:
        return "FALLING"
    if abs(w["net"]) <= 6 and w["er"] <= 0.12:
        return "CHOPPY"
    if w["rg"] >= 68:
        return "RANGEY"
    return "MIXED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--start", default="2022-04")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--pick", type=int, default=0)
    args = ap.parse_args()

    print("=" * 92)
    print(f"  WINDOW CENSUS — {args.asset}, {args.days}-day windows "
          f"stepped {args.step}d from {args.start}")
    print("=" * 92)
    _, m1 = DK.load_pair(args.asset, start=args.start, end=None, tf="M15")
    h1 = resample(m1, "1h")
    h4 = resample(m1, "4h")
    print(f"  H1 bars {len(h1)}   {h1.index[0].date()} .. {h1.index[-1].date()}")

    ws = census(h1, h4, args.days, args.step)
    print()
    print(f"  {'--end':<12} {'kind':<9} {'net %':>7} {'path ER':>8} {'maxDD %':>8}   "
          f"{'TU %':>6} {'TD %':>6} {'RG %':>6}")
    print("  " + "-" * 78)
    for w in ws:
        k = kind(w)
        mark = "  *" if k in ("CHOPPY", "RANGEY") else ""
        print(f"  {str(w['end'].date()):<12} {k:<9} {w['net']:>+7.1f} "
              f"{w['er']:>8.3f} {w['dd']:>8.1f}   "
              f"{w['tu']:>6.1f} {w['td']:>6.1f} {w['rg']:>6.1f}{mark}")

    from collections import Counter
    print()
    print("  window kinds: " + "  ".join(
        f"{k} {v}" for k, v in Counter(kind(w) for w in ws).most_common()))

    if args.pick:
        print()
        print("=" * 92)
        print(f"  SUGGESTED STRATIFIED SET ({args.pick} windows)")
        print("=" * 92)
        print("  Non-overlapping, spread across kinds, so no single market mood")
        print("  can dominate the verdict.")
        chosen, used = [], []
        order = ["FALLING", "CHOPPY", "RISING", "RANGEY", "MIXED"]
        pool = {k: sorted([w for w in ws if kind(w) == k],
                          key=lambda w: -abs(w["net"]) if k != "CHOPPY" else w["er"])
                for k in order}
        while len(chosen) < args.pick:
            progressed = False
            for k in order:
                if len(chosen) >= args.pick:
                    break
                for w in pool[k]:
                    if any(not (w["end"] <= s or w["start"] >= e)
                           for s, e in used):
                        continue
                    chosen.append((k, w))
                    used.append((w["start"], w["end"]))
                    pool[k].remove(w)
                    progressed = True
                    break
            if not progressed:
                break
        print()
        for k, w in sorted(chosen, key=lambda p: p[1]["end"]):
            print(f"  --end {w['end'].date()}   {k:<9} net {w['net']:>+6.1f}%  "
                  f"ER {w['er']:.3f}  RG {w['rg']:.0f}%")
        print()
        print("  Paste into a run loop as --end values; --days "
              f"{args.days} covers each window.")


if __name__ == "__main__":
    main()
