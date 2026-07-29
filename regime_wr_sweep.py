"""
Regime / direction quality — where does WR actually break?

Tags portfolio trades (same path as live) by:
  - regime label (hybrid H1, or alternate detectors)
  - alignment: WITH-TREND / COUNTER / RANGE-NEUTRAL
  - H4 EMA bias at entry

Also sweeps hypothetical regime upgrades on the SAME trade list
(post-hoc filters — no re-run of signal detection).

  python regime_wr_sweep.py --days 30
  python regime_wr_sweep.py --days 30 --asset XAUUSD
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import strategy as S
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C
import floating_config as FC
import meta_gate as MG
import regime as RG
from portfolio_backtest import (
    fetch_portfolio_trades, prepare_portfolio_trades, simulate_portfolio,
    _build_regime_map, DEFAULT_BAL,
)

REGIMES = RG.REGIMES


def _rmap_variant(htf_ctx, name, htf_dfs):
    """Build alternate RegimeMap for comparison."""
    h1 = htf_dfs.get("H1")
    h4 = htf_dfs.get("H4")
    if h1 is None:
        return None
    if name == "hybrid":
        return S.build_regime_map(htf_ctx, tf="H1", detector="hybrid",
                                  win=40, er_hi=0.40, er_lo=0.25, confirm=2)
    if name == "hybrid_lo":
        return S.build_regime_map(htf_ctx, tf="H1", detector="hybrid",
                                  win=40, er_hi=0.32, er_lo=0.20, confirm=2)
    if name == "kaufman_c3":
        return S.build_regime_map(htf_ctx, tf="H1", detector="kaufman",
                                  win=40, er_trend=0.35, confirm=3)
    if name == "slope":
        return S.build_regime_map(htf_ctx, tf="H1", detector="slope",
                                  win=40, slope_k=0.0006)
    if name == "h4_hybrid":
        return S.build_regime_map(htf_ctx, tf="H4", detector="hybrid",
                                  win=30, er_hi=0.38, er_lo=0.22, confirm=2)
    if name == "mtf":
        return _build_mtf_map(h1, h4)
    return None


def _build_mtf_map(h1_df, h4_df):
    """H4 sets direction; H1 ER sets trend vs range (proposed upgrade)."""
    if h4_df is None or len(h4_df) < 50:
        return None
    h4_rm = RG.RegimeMap(h4_df, detector="hybrid", win=30,
                         er_hi=0.35, er_lo=0.20, confirm=2)
    h1_er = RG.efficiency_ratio(h4_df["close"].reindex(
        h1_df.index, method="ffill").to_numpy(), 40)
    # re-build on H1 timeline
    h1_close = h1_df["close"].to_numpy(dtype=float)
    h1_er = RG.efficiency_ratio(h1_close, 40)
    h4_close = h4_df["close"].to_numpy(dtype=float)
    h4_t = np.asarray(h4_df.index.values, dtype="datetime64[ns]")
    h1_t = np.asarray(h1_df.index.values, dtype="datetime64[ns]")
    h4_lab = h4_rm.labels
    n = len(h1_close)
    out = np.array(["RANGE"] * n, dtype=object)
    for i in range(n):
        j = int(np.searchsorted(h4_t, h1_t[i], side="right")) - 1
        if j < 0:
            continue
        h4_reg = h4_lab[j]
        er = h1_er[i]
        if np.isnan(er) or er < 0.28:
            out[i] = "RANGE"
        elif h4_reg == "TREND_UP":
            out[i] = "TREND_UP"
        elif h4_reg == "TREND_DOWN":
            out[i] = "TREND_DOWN"
        else:
            # H4 range but H1 momentum — bias only, still RANGE for meta-gate
            sl = RG.reg_slope(h1_close, 40)
            if not np.isnan(sl[i]) and er >= 0.32:
                out[i] = "TREND_UP" if sl[i] > 0 else "TREND_DOWN"
            else:
                out[i] = "RANGE"
    rm = RG.RegimeMap.__new__(RG.RegimeMap)
    rm.detector = "mtf"
    rm.win = 40
    rm.t = h1_t
    rm.labels = out
    return rm


def h4_ema_bias(htf_ctx, signal_time, ema_period=20):
    """+1 above H4 EMA, -1 below, 0 unknown."""
    ctx = htf_ctx.get("H4") if htf_ctx else None
    if ctx is None:
        return 0
    B = ctx if isinstance(ctx, S.Bars) else S.Bars(ctx)
    idx = S.htf_bar_index(B, signal_time)
    if idx < ema_period + 2:
        return 0
    ema = float(np.asarray(
        __import__("pandas").Series(B.c[:idx + 1]).ewm(
            span=ema_period, adjust=False).mean().iloc[-1]))
    px = B.c[idx]
    return 1 if px >= ema else -1


def alignment(regime, direction):
    d = direction.lower()
    r = str(regime).upper()
    if r == "TREND_UP":
        return "WITH-TREND" if d == "long" else "COUNTER"
    if r == "TREND_DOWN":
        return "WITH-TREND" if d == "short" else "COUNTER"
    return "RANGE"


def bucket_stats(trades, spread=0.0):
    if not trades:
        return dict(n=0, wr=0.0, avg_r=0.0, tot_r=0.0)
    Rs = np.array([t["R"] - spread / max(t.get("risk", 1), 1e-9) for t in trades])
    wr = 100.0 * np.mean(Rs > 0.05) if len(Rs) else 0.0
    return dict(n=len(Rs), wr=wr, avg_r=float(Rs.mean()), tot_r=float(Rs.sum()))


def print_bucket(title, buckets, order):
    print(f"\n  {title}")
    print(f"    {'bucket':<16}{'n':>5}{'WR%':>7}{'avgR':>8}{'totR':>8}")
    print("    " + "-" * 44)
    for b in order:
        st = bucket_stats(buckets.get(b, []))
        if st["n"]:
            print(f"    {b:<16}{st['n']:>5}{st['wr']:>7.1f}"
                  f"{st['avg_r']:>+8.2f}{st['tot_r']:>+8.1f}")


def filter_trades(trades, fn):
    return [t for t in trades if fn(t)]


def sim_wr(trades, spread):
    if not trades:
        return dict(n=0, wr=0.0, tot_r=0.0)
    st = bucket_stats(trades, spread)
    return dict(n=st["n"], wr=st["wr"], tot_r=st["tot_r"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--asset", default=None)
    args = ap.parse_args()

    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)
    oos_mg = FC.load_meta_gate(assets=assets)

    mt5 = M.connect()
    try:
        all_trades, wf, rmaps = fetch_portfolio_trades(args.days, mt5, oos_mg)
        trades = prepare_portfolio_trades(all_trades, rmaps, oos_mg)
        trades = [t for t in trades if t["_asset"] in assets]

        # attach H4 bias + alignment for current hybrid
        htf_cache = {}
        for asset in assets:
            _, prof = P.get_profile(asset)
            sym = P.resolve_symbol_for_profile(asset, mt5)
            _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5,
                                          tfs=("H4", "H1"))
            htf_cache[asset] = S.prepare_htf_context(htf_dfs)

        spread = 0.28
        for t in trades:
            htf = htf_cache.get(t["_asset"])
            t["_align"] = alignment(t.get("_regime", "RANGE"), t.get("dir", "long"))
            bias = h4_ema_bias(htf, t["time"])
            d = t.get("dir", "long").lower()
            if bias == 0:
                t["_h4_bias"] = "UNKNOWN"
            elif (bias > 0 and d == "long") or (bias < 0 and d == "short"):
                t["_h4_bias"] = "H4-WITH"
            else:
                t["_h4_bias"] = "H4-COUNTER"

        print("=" * 78)
        print(f"  REGIME / DIRECTION QUALITY  |  {args.days}d  |  trades={len(trades)}"
              f"  (wf {wf['before']}→{wf['after']})")
        print("=" * 78)

        # --- current hybrid breakdown ---
        by_reg = {r: [] for r in REGIMES}
        by_align = {"WITH-TREND": [], "COUNTER": [], "RANGE": []}
        by_h4 = {"H4-WITH": [], "H4-COUNTER": [], "UNKNOWN": []}
        for t in trades:
            by_reg[t.get("_regime", "RANGE")].append(t)
            by_align[t["_align"]].append(t)
            by_h4[t["_h4_bias"]].append(t)

        print_bucket("WR by regime (hybrid H1 — current)", by_reg, REGIMES)
        print_bucket("WR by trade vs regime direction", by_align,
                     ("WITH-TREND", "COUNTER", "RANGE"))
        print_bucket("WR by H4 EMA bias (independent of regime)", by_h4,
                     ("H4-WITH", "H4-COUNTER", "UNKNOWN"))

        # --- regime coverage on trade times ---
        rmap = rmaps.get(assets[0])
        if rmap:
            cov = {r: 0 for r in REGIMES}
            for t in trades:
                cov[t.get("_regime", "RANGE")] = cov.get(t.get("_regime"), 0) + 1
            tot = len(trades) or 1
            print(f"\n  Trade-time regime mix: "
                  + "  ".join(f"{r} {100*cov[r]/tot:.0f}%" for r in REGIMES))

        # --- detector sweep (same trades, relabel + filter) ---
        print(f"\n  {'─'*76}")
        print("  DETECTOR SWEEP — post-hoc: block COUNTER-trend entries per detector")
        print(f"  {'variant':<22}{'n':>5}{'WR%':>7}{'totR':>8}  (blocked counter)")
        print("  " + "-" * 52)

        asset = assets[0]
        _, prof = P.get_profile(asset)
        sym = P.resolve_symbol_for_profile(asset, mt5)
        _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=("H4", "H1"))
        htf_ctx = htf_cache[asset]

        variants = [
            ("baseline (no extra)", None),
            ("hybrid H1", "hybrid"),
            ("hybrid lo-thr", "hybrid_lo"),
            ("kaufman+confirm3", "kaufman_c3"),
            ("slope H1", "slope"),
            ("hybrid H4", "h4_hybrid"),
            ("MTF H4dir+H1er", "mtf"),
        ]
        for name, det in variants:
            if det is None:
                kept = trades
            else:
                rm = _rmap_variant(htf_ctx, det, htf_dfs)
                kept = []
                for t in trades:
                    reg = rm.at(t["time"]) if rm else "RANGE"
                    if alignment(reg, t.get("dir", "long")) == "COUNTER":
                        continue
                    kept.append(t)
            st = sim_wr(kept, spread)
            print(f"  {name:<22}{st['n']:>5}{st['wr']:>7.1f}{st['tot_r']:>+8.1f}")

        # --- proposed composite filters ---
        print(f"\n  {'─'*76}")
        print("  COMPOSITE FILTERS (proposed upgrades)")
        print(f"  {'filter':<32}{'n':>5}{'WR%':>7}{'totR':>8}")
        print("  " + "-" * 52)

        filters = [
            ("RANGE: require H4-WITH bias",
             lambda t: t["_align"] != "RANGE" or t["_h4_bias"] == "H4-WITH"),
            ("RANGE: block all (trend-only)",
             lambda t: t["_align"] != "RANGE"),
            ("COUNTER blocked + H4-WITH in RANGE",
             lambda t: t["_align"] != "COUNTER" and (
                 t["_align"] != "RANGE" or t["_h4_bias"] == "H4-WITH")),
            ("H4-WITH only (ignore regime)",
             lambda t: t["_h4_bias"] == "H4-WITH"),
            ("WITH-TREND or H4-WITH",
             lambda t: t["_align"] == "WITH-TREND" or t["_h4_bias"] == "H4-WITH"),
        ]
        for name, fn in filters:
            kept = filter_trades(trades, fn)
            st = sim_wr(kept, spread)
            print(f"  {name:<32}{st['n']:>5}{st['wr']:>7.1f}{st['tot_r']:>+8.1f}")

        # portfolio sim for best-looking filter
        res_base = simulate_portfolio(trades, DEFAULT_BAL, enforce_mc=True)
        filt = filter_trades(trades, lambda t: t["_align"] != "COUNTER" and (
            t["_align"] != "RANGE" or t["_h4_bias"] == "H4-WITH"))
        res_filt = simulate_portfolio(filt, DEFAULT_BAL, enforce_mc=True)
        print(f"\n  Portfolio sim (tiered MC):")
        print(f"    baseline   n={res_base['n_trades']} WR={res_base['wr']:.1f}%"
              f" ret={res_base['ret_pct']:+.1f}%")
        print(f"    composite  n={res_filt['n_trades']} WR={res_filt['wr']:.1f}%"
              f" ret={res_filt['ret_pct']:+.1f}%")

        print("\n" + "=" * 78)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
