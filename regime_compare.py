"""
Compare regime detectors on the noise-vs-lag trade-off.

For each detector it reports, over the SAME H1 series:
  switches      : how many times the regime label changed  (lower = less noisy)
  avg run (bars): average bars the regime stays put         (higher = stickier)
  flips/1000bar : switches normalised                        (the noise metric)
  coverage %    : share of bars in UP / DOWN / RANGE

The plain single-threshold 'kaufman' whipsaws near its cut-off; 'ma' is sticky
but lags; 'hybrid' (ER + slope + hysteresis + confirmation) aims for the middle:
few switches WITHOUT the MA lag.

  python regime_compare.py                 # all assets it can find locally
  python regime_compare.py --asset XAUUSD
"""
import argparse
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import regime as RG
import calibrate_long as CL
from oos_analyze import _h1_from_m15_csv, _M15_FALLBACK


def get_h1(asset, start):
    try:
        _, _, htf = CL.load_dukascopy(asset, start)
        h1 = htf.get("H1")
        if h1 is not None and len(h1) and pd.Timestamp(h1.index[-1]) >= pd.Timestamp("2025-06-01"):
            return h1, "dukascopy"
    except Exception:
        pass
    for path in _M15_FALLBACK.get(asset, []):
        h1 = _h1_from_m15_csv(path)
        if h1 is not None and len(h1) > 100:
            return h1, f"fallback {path}"
    return None, None


def switches(labels):
    return int(sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]))


def summarize(name, labels, win):
    lab = labels[win:]
    n = len(lab)
    sw = switches(lab)
    avg_run = n / (sw + 1) if n else 0
    cov = Counter(lab)
    tot = sum(cov.values()) or 1
    up = 100 * cov.get("TREND_UP", 0) / tot
    dn = 100 * cov.get("TREND_DOWN", 0) / tot
    rg = 100 * cov.get("RANGE", 0) / tot
    print(f"  {name:<22}{sw:>9}{avg_run:>12.1f}{1000*sw/n:>14.1f}"
          f"    UP {up:>4.0f}%  DOWN {dn:>4.0f}%  RANGE {rg:>4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--from", dest="start", default=CL.DEFAULT_FROM)
    ap.add_argument("--win", type=int, default=40)
    args = ap.parse_args()

    assets = [args.asset.upper()] if args.asset \
        else ["XAUUSD", "US500", "BRENT", "BTCUSD"]

    for a in assets:
        h1, src = get_h1(a, args.start)
        if h1 is None:
            print(f"\n{a}: no local H1 data — skip")
            continue
        print(f"\n{'='*80}\n  {a}   H1 bars={len(h1):,}   {h1.index[0]:%Y-%m-%d}"
              f" -> {h1.index[-1]:%Y-%m-%d}   ({src})\n{'='*80}")
        print(f"  {'detector':<22}{'switches':>9}{'avg run':>12}"
              f"{'flips/1000':>14}")
        print("  " + "-" * 78)
        variants = [
            ("ma (laggy baseline)", dict(detector="ma", ema_period=20)),
            ("kaufman (single-thr)", dict(detector="kaufman", er_trend=0.35)),
            ("kaufman+confirm3", dict(detector="kaufman", er_trend=0.35, confirm=3)),
            ("slope", dict(detector="slope", slope_k=0.0006)),
            ("HYBRID (recommended)", dict(detector="hybrid", er_hi=0.40,
                                          er_lo=0.25, confirm=2)),
        ]
        for name, kw in variants:
            try:
                rm = RG.RegimeMap(h1, win=args.win, **kw)
                summarize(name, rm.labels, args.win)
            except Exception as e:
                print(f"  {name:<22} ERROR {e}")

    print("\n  READ: lower 'flips/1000' + higher 'avg run' = less noisy. The goal")
    print("  is HYBRID sitting between 'ma' (few flips but laggy) and 'kaufman'")
    print("  (responsive but noisy), keeping ER's low lag without the whipsaw.")


if __name__ == "__main__":
    main()
