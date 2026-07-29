"""
Post-hoc analysis of an OOS run — answers two questions honestly, using ONLY
the already-saved trade ledger (no re-calibration, no download):

  1. FIXED-RISK view: strip the compounding snowball. Each trade risks a fixed
     % of a CONSTANT base, so the numbers reflect the real per-trade edge, not
     an inflated balance curve. (Exact, because pnl scales linearly with risk$:
     net_fixed = (pnl_usd / risk_usd_original) * fixed_risk_usd .)

  2. REGIME split: tag every test trade with the market regime at entry
     (Kaufman ER on H1, same as calibration) and show how much of the profit
     comes from TREND_UP vs TREND_DOWN vs RANGE — i.e. is the edge just riding
     the 2025 uptrend, or does it hold across regimes?

  python oos_analyze.py
  python oos_analyze.py --csv reports/oos_trades.csv --base 1000
"""
import argparse
import csv
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import portfolio_config as C
import calibrate_long as CL
import regime as RG

BASE = 1000.0


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["pnl"] = float(r["pnl_usd"])
        r["risk_usd"] = float(r["risk_usd"])
        r["R"] = float(r["R"])
        r["ts"] = pd.Timestamp(r["entry_time"])
        # net per unit-$ risked (already includes spread); compounding-independent
        r["netR"] = r["pnl"] / r["risk_usd"] if r["risk_usd"] else 0.0
    rows.sort(key=lambda r: r["ts"])
    return rows


def curve_stats(nets, base):
    """Equity stats for a stream of per-trade $ P&L on a FIXED base (no compound)."""
    bal = base
    peak = base
    dd = 0.0
    wins = gp = gl = 0
    gpp = gll = 0.0
    for x in nets:
        bal += x
        peak = max(peak, bal)
        dd = max(dd, (peak - bal) / peak if peak > 0 else 0)
        if x > 0:
            wins += 1
            gpp += x
        else:
            gll += -x
    n = len(nets)
    pf = gpp / gll if gll > 0 else (999.0 if gpp > 0 else 0.0)
    return {"final": bal, "ret": (bal / base - 1) * 100, "profit": bal - base,
            "n": n, "wr": 100.0 * wins / n if n else 0, "pf": pf,
            "max_dd": dd * 100}


def fixed_risk_nets(rows):
    """Rescale each trade's P&L to a CONSTANT risk = risk_pct * BASE (no snowball)."""
    out = []
    for r in rows:
        rp = C.RISK_MAP.get(r["asset"], 0.01)
        fixed_risk_usd = rp * BASE
        out.append(r["netR"] * fixed_risk_usd)
    return out


# Fallback broker M15 files (used only if full Dukascopy M1 is absent on this
# machine). Must cover the TEST window to be useful.
_M15_FALLBACK = {
    "XAUUSD": ["data/XAUUSD@_M15.csv", "data/XAUUSD_M15.csv"],
    "BTCUSD": ["data/BTCUSD@_M15.csv", "data/BTCUSD_M15.csv"],
    "US500":  ["data/US500@_M15.csv", "data/SP500m_M15.csv"],
    "BRENT":  ["data/BRENT@_M15.csv", "data/UKBRENT.Q26_M15.csv"],
}


def _h1_from_m15_csv(path):
    import os
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample("1h").agg(agg).dropna(subset=["open", "high", "low", "close"])


def build_regime_maps(assets, start):
    maps = {}
    reg_cfg = {"detector": "kaufman", "tf": "H1", "win": 40,
               "er_trend": 0.35, "slope_k": 0.0006}
    for a in assets:
        try:
            _, _, htf = CL.load_dukascopy(a, start)
            h1 = htf.get("H1")
            # only trust dukascopy here if it actually reaches the test period
            if h1 is not None and len(h1) and pd.Timestamp(h1.index[-1]) >= pd.Timestamp("2025-06-01"):
                maps[a] = CL.build_regime_map(htf, reg_cfg)
                print(f"  regime map ready: {a}  (dukascopy)", flush=True)
                continue
        except Exception:
            pass
        # fallback: build H1 regime from a broker M15 file present on disk
        made = False
        for path in _M15_FALLBACK.get(a, []):
            h1 = _h1_from_m15_csv(path)
            if h1 is not None and len(h1) > 100:
                maps[a] = RG.RegimeMap(h1, detector="kaufman", win=40,
                                       er_trend=0.35, slope_k=0.0006)
                print(f"  regime map ready: {a}  (fallback {path}, "
                      f"{h1.index[0]:%Y-%m-%d}->{h1.index[-1]:%Y-%m-%d})", flush=True)
                made = True
                break
        if not made:
            print(f"  regime map SKIP {a}: no full data on this machine", flush=True)
            maps[a] = None
    return maps


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/oos_trades.csv")
    ap.add_argument("--from", dest="start", default=CL.DEFAULT_FROM)
    ap.add_argument("--base", type=float, default=BASE)
    ap.add_argument("--last-days", dest="last_days", type=int, default=None,
                    help="only analyse trades in the last N days of the ledger "
                         "(e.g. 30 for the last month)")
    ap.add_argument("--since", default=None,
                    help="only analyse trades on/after this date (YYYY-MM-DD)")
    args = ap.parse_args()
    BASE = args.base

    rows = load_rows(args.csv)
    if not rows:
        print("No trades in CSV.")
        return
    last_ts = max(r["ts"] for r in rows)
    win_lo = None
    if args.since:
        win_lo = pd.Timestamp(args.since)
    if args.last_days:
        cand = last_ts - pd.Timedelta(days=args.last_days)
        win_lo = cand if win_lo is None else max(win_lo, cand)
    if win_lo is not None:
        rows = [r for r in rows if r["ts"] >= win_lo]
        if not rows:
            print(f"No trades on/after {win_lo:%Y-%m-%d}.")
            return

    assets = sorted({r["asset"] for r in rows})
    print("=" * 80)
    hdr = f"  OOS POST-HOC ANALYSIS   ({len(rows)} trades, base ${BASE:,.0f})"
    if win_lo is not None:
        hdr += f"\n  WINDOW: {win_lo:%Y-%m-%d} -> {last_ts:%Y-%m-%d}"
    print(hdr)
    print("=" * 80)

    # ---------------------------------------------------------------- 1. FIXED RISK
    print("\n" + "=" * 80)
    print("  1) FIXED-RISK VIEW  (no compounding — the honest edge)")
    print("=" * 80)
    nets = fixed_risk_nets(rows)
    st = curve_stats(nets, BASE)
    print(f"  Risk per trade: fixed % of ${BASE:,.0f} "
          f"({', '.join(f'{a} {C.RISK_MAP.get(a,0.01)*100:.1f}%' for a in assets)})")
    print(f"  Trades: {st['n']}  |  WR: {st['wr']:.1f}%  |  PF: {st['pf']:.2f}  "
          f"|  Max DD: {st['max_dd']:.1f}%")
    print(f"  ${BASE:,.0f}  ->  ${st['final']:,.0f}   "
          f"Profit ${st['profit']:+,.0f}  ({st['ret']:+.1f}%)")
    print(f"  (vs the compounded report which snowballed to +9489% — same trades)")

    print(f"\n  per asset (fixed risk):")
    print(f"  {'asset':<8}{'n':>6}{'WR%':>7}{'totR':>9}{'avgR':>8}{'net$':>12}{'ret%':>9}")
    print("  " + "-" * 60)
    by_a = defaultdict(list)
    for r, net in zip(rows, nets):
        by_a[r["asset"]].append((r, net))
    for a in assets:
        items = by_a[a]
        totR = sum(r["R"] for r, _ in items)
        net = sum(x for _, x in items)
        w = sum(1 for r, _ in items if r["pnl"] > 0)
        n = len(items)
        print(f"  {a:<8}{n:>6}{100*w/n:>7.1f}{totR:>+9.1f}{totR/n:>+8.3f}"
              f"{net:>+12.0f}{100*net/BASE:>+9.1f}")

    # ---------------------------------------------------------------- 2. REGIME
    print("\n" + "=" * 80)
    print("  2) REGIME SPLIT  (Kaufman ER on H1 — where does the edge come from?)")
    print("=" * 80)
    maps = build_regime_maps(assets, args.start)

    for r in rows:
        rm = maps.get(r["asset"])
        if rm is None:
            r["regime"] = "NO_COV"
            continue
        tmin = pd.Timestamp(rm.t[0])
        tmax = pd.Timestamp(rm.t[-1])
        if r["ts"] < tmin or r["ts"] > tmax:
            r["regime"] = "NO_COV"
        else:
            r["regime"] = rm.at(r["ts"])

    # overall regime breakdown, on FIXED-risk P&L
    print(f"\n  {'regime':<12}{'n':>7}{'share':>7}{'WR%':>7}{'totR':>9}"
          f"{'avgR':>8}{'net$(fix)':>12}")
    print("  " + "-" * 62)
    reg_tot = defaultdict(lambda: {"n": 0, "w": 0, "R": 0.0, "net": 0.0})
    for r, net in zip(rows, nets):
        g = reg_tot[r["regime"]]
        g["n"] += 1
        g["w"] += (r["pnl"] > 0)
        g["R"] += r["R"]
        g["net"] += net
    N = len(rows)
    for reg in list(RG.REGIMES) + ["NO_COV"]:
        g = reg_tot[reg]
        if not g["n"]:
            continue
        label = reg if reg != "NO_COV" else "NO_COV*"
        print(f"  {label:<12}{g['n']:>7}{100*g['n']/N:>6.0f}%{100*g['w']/g['n']:>7.1f}"
              f"{g['R']:>+9.1f}{g['R']/g['n']:>+8.3f}{g['net']:>+12.0f}")
    if reg_tot["NO_COV"]["n"]:
        print(f"  * NO_COV = trades whose asset has no full data on THIS machine "
              f"(run on the strong PC for BRENT/US500 & the tail).")

    # per asset x regime (net R — compounding independent)
    print(f"\n  per-asset avg R by regime (compounding-independent):")
    print(f"  {'asset':<8}{'UP n/avgR':>18}{'DOWN n/avgR':>18}{'RANGE n/avgR':>18}")
    print("  " + "-" * 62)
    ar = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for r in rows:
        c = ar[r["asset"]][r["regime"]]
        c[0] += 1
        c[1] += r["R"]
    for a in assets:
        cells = ""
        for reg in RG.REGIMES:
            n, R = ar[a][reg]
            cells += f"{n:>6}/{(R/n if n else 0):>+8.3f}   " if n else f"{'—':>13}   "
        print(f"  {a:<8}{cells}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 80)
    up = reg_tot["TREND_UP"]["net"]
    dn = reg_tot["TREND_DOWN"]["net"]
    rg = reg_tot["RANGE"]["net"]
    tot = up + dn + rg
    print("  READ (over trades with regime coverage only):")
    if tot != 0:
        print(f"    TREND_UP contributes {100*up/tot:+.0f}% of fixed-risk profit, "
              f"TREND_DOWN {100*dn/tot:+.0f}%, RANGE {100*rg/tot:+.0f}%.")
    print(f"    If almost all profit is TREND_UP and DOWN/RANGE are flat/negative,")
    print(f"    the edge is largely a 2025-2026 uptrend artifact and would suffer")
    print(f"    in a chop/bear regime. A balanced split = a more genuine edge.")
    print("=" * 80)


if __name__ == "__main__":
    main()
