"""Does the regime label predict anything? Measured over the full M1 archive.

Every gate in this project trusts the regime label: `regime_allows` refuses a
direction the label disagrees with, and the RANGE macro veto refuses the
counter-bias side on top of that. None of that has ever been measured directly
against price. It has only ever been measured indirectly, through the profit of
a book that also depends on five detectors, an entry model and an exit model —
so a bad label and a bad rule are indistinguishable in that number.

This tool measures the label alone, with no rules and no trades involved.

Method
------
For every H1 bar the detector labels, look at the next `horizon` hours and
measure how far price travelled in favour of the direction that label implies
versus how far it travelled against it:

    long  : MFE = max(high) - close   MAE = close - min(low)
    short : MFE = close - min(low)    MAE = max(high) - close

Both are divided by ATR at the labelling bar, so a $20 excursion in a calm
market and a $20 excursion in a violent one are not treated as equal, and
periods at $1,800 gold compare with periods at $4,000.

    edge ratio = mean(MFE) / mean(MAE)

Edge ratio is the standard sizing-free test of a directional bias: it asks
whether a trade taken in the labelled direction gets more room to profit than
room to lose, before any stop, target or entry rule is chosen. 1.00 means the
label told you nothing. It is deliberately not a win rate: a win rate depends
on where the stop sits, which is the rule's job, not the label's.

The number that matters is not the edge ratio itself but the gap between the
edge ratio on labelled bars and the ALL row, which is what an unfiltered
strategy would have got for free. A label that scores 1.15 while the market
scores 1.15 unfiltered is contributing nothing, however profitable it looks.

Usage
-----
  python regime_audit.py                       # full archive, current detector
  python regime_audit.py --detectors           # compare all detectors
  python regime_audit.py --by-year             # stability over time
  python regime_audit.py --start 2024-01 --end 2026-07
"""
import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dukascopy_loader as DK
import floating_config as FC
import regime as R

HORIZONS = (4, 8, 12, 24)   # hours; MAX_HOLD is 96 bars = 8h on M5, 24h on M15
WARMUP = 800                # H1 bars before the first label is trusted


def resample(m1, rule):
    """Same aggregation the calibration path uses, so labels transfer."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = m1.resample(rule, label="left", closed="left", origin="epoch").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def atr(high, low, close, period=14):
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()


def forward_excursions(high, low, close, horizon):
    """Max high and min low over the next `horizon` bars, per bar.

    Reversed rolling window: rolling(h).max() on a reversed series gives, at
    each position, the max over the following h bars of the original.
    """
    n = len(close)
    rev_hi = pd.Series(high[::-1]).rolling(horizon, min_periods=horizon).max()
    rev_lo = pd.Series(low[::-1]).rolling(horizon, min_periods=horizon).min()
    fwd_hi = rev_hi.to_numpy()[::-1]
    fwd_lo = rev_lo.to_numpy()[::-1]
    # shift by one so the labelling bar itself is excluded
    fwd_hi = np.concatenate([fwd_hi[1:], [np.nan]])
    fwd_lo = np.concatenate([fwd_lo[1:], [np.nan]])
    fwd_close = np.concatenate([close[horizon:], np.full(min(horizon, n), np.nan)])[:n]
    return fwd_hi, fwd_lo, fwd_close


def forward_er(close, horizon):
    """Kaufman ER looking forward: did the next `horizon` bars actually trend?"""
    s = pd.Series(close)
    net = (s.shift(-horizon) - s).abs()
    path = s.diff().abs().shift(-horizon).rolling(horizon).sum()
    return (net / path.replace(0.0, np.nan)).to_numpy()


def edge(mfe, mae):
    m = np.isfinite(mfe) & np.isfinite(mae)
    if m.sum() < 30:
        return float("nan"), float("nan"), float("nan"), 0
    a, b = mfe[m], mae[m]
    denom = b.mean()
    return (a.mean() / denom if denom > 0 else float("nan"),
            a.mean(), b.mean(), int(m.sum()))


def build_map(h1, h4, detector):
    """Regime map for a named detector, using the live config where it applies."""
    o = FC.REGIME
    if detector == "mtf":
        return R.MtfRegimeMap(
            h1, h4, h4_win=o["regime_h4_win"], h1_win=o["regime_win"],
            er_hi=o["regime_er_hi"], er_lo=o["regime_er_lo"],
            confirm=o["regime_confirm"], h4_lookback=o["regime_h4_lookback"],
            structure_break=o["regime_structure_break"],
            brk_win=o["regime_brk_win"], shock_win=o["regime_shock_win"],
            shock_baseline=o["regime_shock_baseline"])
    return R.RegimeMap(
        h1, detector=detector, win=o["regime_win"],
        er_trend=o["regime_er_trend"], slope_k=o["regime_slope_k"],
        er_hi=o["regime_er_hi"], er_lo=o["regime_er_lo"],
        confirm=o["regime_confirm"], hysteresis=(detector == "hybrid"))


def stability(labels):
    """Mean bars a label survives, and how many times it changes."""
    runs, cur, ln = [], labels[0], 1
    for x in labels[1:]:
        if x == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur, ln = x, 1
    runs.append((cur, ln))
    by = {}
    for lab, n in runs:
        by.setdefault(lab, []).append(n)
    return len(runs) - 1, {k: float(np.mean(v)) for k, v in by.items()}


def audit(h1, rmap, horizon, label_slice=None, tag=""):
    high = h1["high"].to_numpy(float)
    low = h1["low"].to_numpy(float)
    close = h1["close"].to_numpy(float)
    a = atr(high, low, close)
    fwd_hi, fwd_lo, _ = forward_excursions(high, low, close, horizon)
    fer = forward_er(close, horizon)

    labels = np.asarray(rmap.labels[:len(close)], dtype=object)
    with np.errstate(invalid="ignore", divide="ignore"):
        up_mfe = (fwd_hi - close) / a
        up_mae = (close - fwd_lo) / a
        dn_mfe = (close - fwd_lo) / a
        dn_mae = (fwd_hi - close) / a

    ok = np.zeros(len(close), dtype=bool)
    ok[WARMUP:] = True
    if label_slice is not None:
        sel = np.zeros(len(close), dtype=bool)
        sel[label_slice] = True
        ok &= sel
    ok &= np.isfinite(a) & (a > 0) & np.isfinite(fwd_hi) & np.isfinite(fwd_lo)

    rows = []
    for name, mask, mfe, mae in (
            ("ALL bars  long", ok, up_mfe, up_mae),
            ("ALL bars  short", ok, dn_mfe, dn_mae),
            ("TREND_UP  long", ok & (labels == "TREND_UP"), up_mfe, up_mae),
            ("TREND_UP  short", ok & (labels == "TREND_UP"), dn_mfe, dn_mae),
            ("TREND_DOWN short", ok & (labels == "TREND_DOWN"), dn_mfe, dn_mae),
            ("TREND_DOWN long", ok & (labels == "TREND_DOWN"), up_mfe, up_mae),
            ("RANGE     long", ok & (labels == "RANGE"), up_mfe, up_mae),
            ("RANGE     short", ok & (labels == "RANGE"), dn_mfe, dn_mae),
    ):
        e, m, v, n = edge(np.where(mask, mfe, np.nan), np.where(mask, mae, np.nan))
        fe = float(np.nanmean(np.where(mask, fer, np.nan))) if n else float("nan")
        rows.append((name, n, e, m, v, fe))
    return rows


def print_rows(rows, base_up, base_dn, title):
    print(f"\n  {title}")
    print(f"  {'label / side':<20} {'bars':>7} {'edge':>7} {'vs ALL':>8} "
          f"{'MFE':>7} {'MAE':>7} {'fwd ER':>8}")
    print("  " + "-" * 68)
    for name, n, e, m, v, fe in rows:
        base = base_up if name.endswith("long") else base_dn
        rel = e - base if np.isfinite(e) and np.isfinite(base) else float("nan")
        flag = ""
        if np.isfinite(rel):
            flag = "  <<<" if rel > 0.08 else ("  xxx" if rel < -0.08 else "")
        print(f"  {name:<20} {n:>7} {e:>7.3f} {rel:>+8.3f} "
              f"{m:>7.2f} {v:>7.2f} {fe:>8.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--start", default="2022-04")
    ap.add_argument("--end", default=None)
    ap.add_argument("--detectors", action="store_true",
                    help="compare every detector, not just the configured one")
    ap.add_argument("--by-year", action="store_true",
                    help="split the configured detector's result by year")
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()

    horizons = (args.horizon,) if args.horizon else HORIZONS

    print("=" * 78)
    print(f"  REGIME LABEL AUDIT — {args.asset}  H1  "
          f"{args.start} .. {args.end or 'latest'}")
    print("=" * 78)
    print("  Loading M1 archive and resampling to H1/H4 ...")
    _, m1 = DK.load_pair(args.asset, start=args.start, end=args.end, tf="M15")
    h1 = resample(m1, "1h")
    h4 = resample(m1, "4h")
    print(f"  H1 bars: {len(h1)}   H4 bars: {len(h4)}   "
          f"{h1.index[0].date()} .. {h1.index[-1].date()}")
    px0, px1 = float(h1['close'].iloc[0]), float(h1['close'].iloc[-1])
    print(f"  price {px0:.0f} -> {px1:.0f}  ({(px1/px0-1)*100:+.1f}% over the span)")

    dets = ("mtf", "kaufman", "hybrid", "donchian", "slope") if args.detectors \
        else (FC.REGIME["regime_detector"],)

    for det in dets:
        rmap = build_map(h1, h4, det)
        cov = rmap.coverage()
        flips, runlen = stability(np.asarray(rmap.labels[WARMUP:], dtype=object))
        print()
        print("=" * 78)
        print(f"  DETECTOR: {det}"
              + ("   (the one live uses)" if det == FC.REGIME["regime_detector"] else ""))
        print("=" * 78)
        print(f"  coverage   " + "   ".join(f"{k} {v:.1f}%" for k, v in cov.items()))
        print(f"  changes    {flips} label changes over {len(h1)-WARMUP} bars"
              f"   mean run: "
              + "  ".join(f"{k} {v:.0f}h" for k, v in sorted(runlen.items())))
        if hasattr(rmap, "macro_coverage"):
            mc = rmap.macro_coverage()
            print(f"  H4 macro   " + "   ".join(f"{k} {v:.1f}%" for k, v in mc.items()))

        for h in horizons:
            rows = audit(h1, rmap, h)
            base_up = rows[0][2]
            base_dn = rows[1][2]
            print_rows(rows, base_up, base_dn, f"horizon {h}h")

    if args.by_year:
        det = FC.REGIME["regime_detector"]
        rmap = build_map(h1, h4, det)
        years = pd.DatetimeIndex(h1.index).year
        h = 12
        print()
        print("=" * 78)
        print(f"  STABILITY OVER TIME — detector {det}, horizon {h}h")
        print("  Does the label keep working, or did it only work in some years?")
        print("=" * 78)
        print(f"  {'year':<6} {'bars':>6} {'TU long':>9} {'TD short':>9} "
              f"{'RG long':>9} {'RG short':>9} {'ALL long':>9} {'ALL short':>9}")
        print("  " + "-" * 74)
        for y in sorted(set(years)):
            idx = np.where(years == y)[0]
            if len(idx) < 500:
                continue
            rows = audit(h1, rmap, h, label_slice=idx)
            d = {r[0]: r[2] for r in rows}
            print(f"  {y:<6} {len(idx):>6} "
                  f"{d['TREND_UP  long']:>9.3f} {d['TREND_DOWN short']:>9.3f} "
                  f"{d['RANGE     long']:>9.3f} {d['RANGE     short']:>9.3f} "
                  f"{d['ALL bars  long']:>9.3f} {d['ALL bars  short']:>9.3f}")
        print()
        print("  A label with real predictive power beats its ALL column in most")
        print("  years. One that only beats it in one or two years was fitted to")
        print("  those years by the calibration, not discovered in the market.")

    print()
    print("  edge   = mean(MFE)/mean(MAE) in ATR units for that side")
    print("  vs ALL = edge minus the same side's unfiltered edge. This is the")
    print("           only column that matters: it is what the label ADDS.")
    print("  fwd ER = realised Kaufman efficiency over the horizon; high means")
    print("           the market really did travel in a straight line.")
    print("  <<< label helps that side    xxx label actively hurts that side")


if __name__ == "__main__":
    main()
