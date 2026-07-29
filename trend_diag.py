"""
Hypothesis test: "the model loses because it trades AGAINST the trend / in no-trend
(low-ADX) conditions, not because of spread/slippage."

For each active asset we run the CURRENT backtest, then tag every trade by the
market state at its entry bar (computed causally, no lookahead):

    WITH-TREND   ADX >= adx_min AND trade side agrees with +DI/-DI
    COUNTER      ADX >= adx_min BUT trade side is against +DI/-DI
    NO-TREND     ADX <  adx_min (chop)

We also tag by the signal-TF trend (H1 slope) as a cross-check. Then we print
win-rate and net R for each bucket. If COUNTER and NO-TREND are the losers, the
fix is a momentum/direction gate (ADX>threshold, trade with +DI/-DI).

Read-only: changes NO live logic.

  python trend_diag.py --days 120
  python trend_diag.py --days 120 --adx-min 22
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import strategy as S
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C

META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)


def kwargs_for(prof, spread, min_sl, htf):
    o = P.apply_profile_to_settings(P.preset_for_profile(prof), prof, spread,
                                    min_sl_override=min_sl)
    if prof.get("rules_override"):
        o["enabled"] = list(prof["rules_override"])
    k = {x: o[x] for x in o if x not in META}
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    return k


def bucket_stats(trades, spread):
    if not trades:
        return (0, 0.0, 0.0)
    Rs = np.array([t["R"] - spread / t["risk"] for t in trades if t["risk"] > 0])
    wr = 100.0 * np.mean(Rs > 0.05) if len(Rs) else 0.0
    return (len(Rs), wr, float(Rs.sum()))


def line(name, st):
    n, wr, totr = st
    print(f"    {name:<14}{n:>5}{wr:>8.1f}{totr:>+10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--adx-min", type=float, default=22.0)
    ap.add_argument("--asset", default=None)
    args = ap.parse_args()
    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)

    print("=" * 78)
    print(f"  TREND / MOMENTUM DIAGNOSIS  |  broker {args.days}d  |  ADX_min={args.adx_min}")
    print("  Does the model lose on COUNTER-trend / NO-trend entries?")
    print("=" * 78)

    mt5 = M.connect()
    grand = {"WITH-TREND": [], "COUNTER": [], "NO-TREND": []}
    try:
        for asset in assets:
            _, prof = P.get_profile(asset)
            label = prof.get("label", asset)
            try:
                sym = P.resolve_symbol_for_profile(asset, mt5)
                spread = P.live_spread(sym, prof, mt5)
                _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
                _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5,
                                              tfs=("H4", "H1"))
            except Exception as e:
                print(f"\n  {label}: skip ({e})")
                continue
            B = S.Bars(d)
            ctx = S.M1Ctx(m1, d.index)
            htf = S.prepare_htf_context(htf_dfs)
            min_sl = S.__dict__  # placeholder to avoid unused warning
            import symbol_specs as SP
            min_sl = SP.calibrate_min_sl(B, cutoff, prof)
            adx_arr, pdi_arr, mdi_arr = S.compute_adx_series(B, period=14)

            k = kwargs_for(prof, spread, min_sl, htf)
            trades = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
                      if t["time"] >= cutoff]

            buckets = {"WITH-TREND": [], "COUNTER": [], "NO-TREND": []}
            for t in trades:
                i = t.get("idx")
                if i is None or i >= len(adx_arr):
                    continue
                a, pdi, mdi = adx_arr[i], pdi_arr[i], mdi_arr[i]
                if not np.isfinite(a) or a < args.adx_min:
                    b = "NO-TREND"
                else:
                    di_long = pdi > mdi
                    with_tr = (t["dir"] == "long" and di_long) or \
                              (t["dir"] == "short" and not di_long)
                    b = "WITH-TREND" if with_tr else "COUNTER"
                buckets[b].append(t)
                grand[b].append(t)

            print(f"\n  {label} ({asset})   {len(trades)} trades   spread={spread:.4g}")
            print(f"    {'bucket':<14}{'n':>5}{'WR%':>8}{'netR':>10}")
            print("    " + "-" * 37)
            for b in ("WITH-TREND", "COUNTER", "NO-TREND"):
                line(b, bucket_stats(buckets[b], spread))
    finally:
        M.shutdown(mt5)

    print("\n" + "=" * 78)
    print("  ALL ASSETS COMBINED")
    print(f"    {'bucket':<14}{'n':>5}{'WR%':>8}{'netR':>10}")
    print("    " + "-" * 37)
    for b in ("WITH-TREND", "COUNTER", "NO-TREND"):
        line(b, bucket_stats(grand[b], 0.0))
    print("=" * 78)


if __name__ == "__main__":
    main()
