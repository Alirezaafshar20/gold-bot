"""
Test whether an HTF trend/regime filter rescues performance in the current
(weak) regime. Runs each asset's CURRENT configured rules on recent broker data
three ways:
    OFF    - no trend filter (what we run now)
    H1     - long only above H1 EMA, short only below (regime gate)
    H4     - same gate on H4 (slower, stricter)

Also splits the OFF result into UP / DOWN / CHOP regimes so we can see where the
losses actually come from. Read-only; changes no live logic.

  python regime_test.py --days 120
  python regime_test.py --asset US30 --days 90
"""
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import strategy as S
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C

BAL = 1000.0
META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)


def base_kwargs(prof, spread, min_sl, htf):
    o = P.apply_profile_to_settings(P.preset_for_profile(prof), prof, spread,
                                    min_sl_override=min_sl)
    if prof.get("rules_override"):
        o["enabled"] = list(prof["rules_override"])
    k = {x: o[x] for x in o if x not in META}
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    return k


def regime_at(B, i, lookback=192):
    """Classify regime on the signal TF: UP / DOWN / CHOP via SMA position+slope.
    lookback 192 M15 bars ≈ 2 trading days of context."""
    j0 = max(0, i - lookback)
    seg = B.c[j0:i + 1]
    if len(seg) < 20:
        return "CHOP"
    sma_now = seg.mean()
    sma_prev = B.c[max(0, i - 2 * lookback):i - lookback + 1].mean() if i > lookback else sma_now
    px = B.c[i]
    band = 0.001 * px
    up = px > sma_now and sma_now > sma_prev + band
    dn = px < sma_now and sma_now < sma_prev - band
    return "UP" if up else "DOWN" if dn else "CHOP"


def stats(trades, risk, spread):
    if not trades:
        return None
    acc = S.simulate_account(trades, BAL, risk, spread)
    Rs = np.array([t["R"] - spread / t["risk"] for t in trades if t["risk"] > 0])
    w, l = Rs[Rs > 0], Rs[Rs <= 0]
    pf = float(w.sum() / (-l.sum())) if l.sum() < 0 else 99.99
    return dict(n=len(trades), wr=acc["wr"], pf=min(pf, 99.99),
                ret=acc["ret_pct"], dd=acc["max_dd"])


def run_variant(d, m1, B, ctx, htf, prof, spread, min_sl, trend, tf):
    k = base_kwargs(prof, spread, min_sl, htf)
    if trend:
        k["htf_trend"] = True
        k["htf_ema"] = 20
        k["htf_trend_tfs"] = [tf]
    return S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()
    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)

    print("=" * 90)
    print(f"  REGIME / TREND-FILTER TEST  |  broker data, last {args.days}d")
    print("  OFF = no filter (current)   H1/H4 = trade only with that HTF trend")
    print("=" * 90)

    mt5 = M.connect()
    try:
        for asset in assets:
            _, prof = P.get_profile(asset)
            label = prof.get("label", asset)
            try:
                sym = P.resolve_symbol_for_profile(asset, mt5)
                spread = P.live_spread(sym, prof, mt5)
                _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
                _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=("H4", "H1"))
            except Exception as e:
                print(f"\n  {label}: skip ({e})")
                continue
            B = S.Bars(d)
            ctx = S.M1Ctx(m1, d.index)
            htf = S.prepare_htf_context(htf_dfs)
            min_sl = __import__("symbol_specs").calibrate_min_sl(B, cutoff, prof)
            risk = prof.get("risk_pct", 0.01)

            print(f"\n  {label} ({asset})   rules={prof.get('rules_override')}")
            print(f"  {'variant':<10}{'n':>5}{'WR%':>7}{'PF':>7}{'ret%':>8}{'DD%':>7}")
            print("  " + "-" * 50)
            base = None
            for name, trend, tf in (("OFF", False, "H1"), ("H1", True, "H1"),
                                    ("H4", True, "H4")):
                tr = [t for t in run_variant(d, m1, B, ctx, htf, prof, spread,
                                             min_sl, trend, tf)
                      if t["time"] >= cutoff]
                st = stats(tr, risk, spread)
                if name == "OFF":
                    base = tr
                if st:
                    print(f"  {name:<10}{st['n']:>5}{st['wr']:>7.1f}{st['pf']:>7.2f}"
                          f"{st['ret']:>+8.1f}{st['dd']:>7.1f}")
                else:
                    print(f"  {name:<10}    0      —      —       —      —")

            # regime split of the OFF (current) trades
            if base:
                buckets = {"UP": [], "DOWN": [], "CHOP": []}
                for t in base:
                    i = t.get("idx", 0)
                    buckets[regime_at(B, i)].append(t)
                print(f"  {'— regime split (current rules) —':<40}")
                for reg in ("UP", "DOWN", "CHOP"):
                    st = stats(buckets[reg], risk, spread)
                    if st:
                        print(f"    {reg:<8}{st['n']:>5}{st['wr']:>7.1f}{st['pf']:>7.2f}"
                              f"{st['ret']:>+8.1f}")
                    else:
                        print(f"    {reg:<8}    0      —      —       —")
    finally:
        M.shutdown(mt5)
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
