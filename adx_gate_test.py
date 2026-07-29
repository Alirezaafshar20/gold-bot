"""
Test an ADX momentum gate on the current model, per asset, on broker data.

Variants:
  OFF        current model (no ADX gate)
  ADX>=k     only trade when ADX >= k  (remove low-momentum "chop" entries)
  ADX>=k+dir only trade when ADX >= k AND +DI/-DI agrees with the trade side

Reports trades / WR / netR / return% / DD for each, so we can see whether
filtering weak-momentum entries actually helps — and per asset (gold may prefer
counter-trend, BTC/Brent may prefer with-trend).

Read-only w.r.t. live config.

  python adx_gate_test.py --days 120 --adx-min 22
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import strategy as S
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C
import symbol_specs as SP

BAL = 1000.0
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


def stats(trades, risk, spread):
    if not trades:
        return None
    acc = S.simulate_account(trades, BAL, risk, spread)
    Rs = np.array([t["R"] - spread / t["risk"] for t in trades if t["risk"] > 0])
    wr = 100.0 * np.mean(Rs > 0.05) if len(Rs) else 0.0
    return dict(n=len(Rs), wr=wr, ret=acc["ret_pct"], dd=acc["max_dd"],
                totr=float(Rs.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--adx-min", type=float, default=22.0)
    ap.add_argument("--asset", default=None)
    args = ap.parse_args()
    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)

    print("=" * 82)
    print(f"  ADX GATE TEST  |  broker {args.days}d  |  ADX_min={args.adx_min}")
    print("=" * 82)

    mt5 = M.connect()
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
            min_sl = SP.calibrate_min_sl(B, cutoff, prof)
            risk = prof.get("risk_pct", 0.01)
            base = kwargs_for(prof, spread, min_sl, htf)

            variants = [
                ("OFF", {}),
                (f"ADX>={args.adx_min:g}", {"adx_gate": True, "adx_min": args.adx_min,
                                            "adx_require_dir": False}),
                (f"ADX>={args.adx_min:g}+dir", {"adx_gate": True, "adx_min": args.adx_min,
                                                "adx_require_dir": True}),
            ]
            print(f"\n  {label} ({asset})")
            print(f"    {'variant':<14}{'n':>5}{'WR%':>7}{'netR':>9}{'ret%':>8}{'DD%':>7}")
            print("    " + "-" * 50)
            for name, extra in variants:
                k = dict(base); k.update(extra)
                tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
                      if t["time"] >= cutoff]
                st = stats(tr, risk, spread)
                if st:
                    print(f"    {name:<14}{st['n']:>5}{st['wr']:>7.1f}"
                          f"{st['totr']:>+9.2f}{st['ret']:>+8.1f}{st['dd']:>7.1f}")
                else:
                    print(f"    {name:<14}    0      —        —       —      —")
    finally:
        M.shutdown(mt5)
    print("\n" + "=" * 82)


if __name__ == "__main__":
    main()
