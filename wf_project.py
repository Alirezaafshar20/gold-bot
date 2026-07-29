"""
Honest walk-forward projection for a $1000 account.

Trains (calibrates min-SL) on the first part of history, then trades the CURRENT
live rules OUT-OF-SAMPLE on the held-out tail. Reports the out-of-sample result
two ways and scales it to 90 days WITHOUT the compounding illusion.

Read-only. Nothing modified.

  python wf_project.py
  python wf_project.py --train-frac 0.5
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import pandas as pd

import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C
from multi_symbol_calibrate import _opt
from portfolio_backtest import simulate_portfolio
from walkforward import run_rules, split_trades, train_min_sl, acc_stats

BAL = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=68)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    mt5 = M.connect()
    oos = []
    oos_span_days = 0.0
    per_asset = {}
    for asset in C.PORTFOLIO_ASSETS:
        _, prof = P.get_profile(asset)
        risk = C.RISK_MAP.get(asset, X.profile_risk_pct(prof))
        mc = C.MAX_CONCURRENT_PER_ASSET
        try:
            sym = P.resolve_symbol_for_profile(asset, mt5)
            spread = P.live_spread(sym, prof, mt5)
            _, d, m1, _ = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
            htf_tfs = tuple(_opt(prof).get("htf_tfs", ("H4", "H1")))
            _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=htf_tfs)
        except RuntimeError as e:
            print(f"  skip {asset}: {e}")
            continue
        B = S.Bars(d); ctx = S.M1Ctx(m1, d.index); htf = S.prepare_htf_context(htf_dfs)
        lo, hi = d.index[0], d.index[-1]
        split_ts = lo + (hi - lo) * args.train_frac
        oos_span_days = max(oos_span_days, (hi - split_ts).total_seconds() / 86400.0)
        min_sl = train_min_sl(d, split_ts, prof)
        rules = list(prof.get("rules_override") or _opt(prof).get("enabled", []))
        spec = X.get_spec(sym, mt5)
        all_tr = run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, rules, mc)
        _, test = split_trades(all_tr, split_ts)
        for t in test:
            t["_asset"] = asset; t["_risk_pct"] = risk
            t["_spread"] = spread; t["_max_concurrent"] = mc; t["_spec"] = spec
        oos += test
        per_asset[asset] = {
            "label": prof.get("label", asset), "risk": risk,
            "spread": spread, "trades": test,
        }
    M.shutdown(mt5)

    if not oos:
        print("No out-of-sample trades.")
        return
    oos.sort(key=lambda t: (t["time"], t.get("exit_time")))

    comp = simulate_portfolio(oos, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK,
                              size_compound=True)
    flat = simulate_portfolio(oos, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK,
                              size_compound=False)

    span = max(oos_span_days, 1.0)
    n = comp["n"]
    trades_per_day = n / span
    flat_profit_per_day = (flat["final"] - BAL) / span
    flat_90 = BAL + flat_profit_per_day * 90
    # compound daily growth from the flat (linear) edge — realistic compounding
    daily_growth = (flat["final"] / BAL) ** (1.0 / span)
    comp_90 = BAL * (daily_growth ** 90)

    sep = "=" * 88
    print(sep)
    print(f"  WALK-FORWARD PROJECTION — $1000 account, your risk map")
    print(f"  Out-of-sample window: {span:.0f} days  |  {n} trades  "
          f"({trades_per_day:.1f}/day)  |  WR {comp['wr']:.1f}%  PF {comp['pf']:.2f}")
    print(sep)
    print(f"\n  OUT-OF-SAMPLE RESULT (held-out data the rules never saw):")
    print(f"    Fixed risk (no compounding):  ${BAL:,.0f} → ${flat['final']:,.0f}  "
          f"({flat['ret_pct']:+.1f}%)  DD {flat['max_dd']:.1f}%")
    print(f"    Compounding (snowball):       ${BAL:,.0f} → ${comp['final']:,.0f}  "
          f"({comp['ret_pct']:+.1f}%)  DD {comp['max_dd']:.1f}%")

    print(f"\n  SCALED TO 90 DAYS (honest estimate, same edge & trade rate):")
    print(f"    Fixed risk (linear):     ~${flat_90:,.0f}   ({(flat_90/BAL-1)*100:+.0f}%)")
    print(f"    Compounding (realistic): ~${comp_90:,.0f}   ({(comp_90/BAL-1)*100:+.0f}%)")

    print(f"\n  REALITY CHECK:")
    print(f"   • Use the FIXED-RISK number as your baseline expectation.")
    print(f"   • Compounding figures swing wildly with a few trades — treat as best-case.")
    print(f"   • This OOS window includes the recent weak regime; live fills differ.")
    print(f"   • Expect drawdowns around {max(comp['max_dd'], flat['max_dd']):.0f}%+ along the way.")
    print(sep)

    # ---- per-asset out-of-sample stats (isolated, fixed risk) ----
    print(f"\n  PER-SYMBOL OUT-OF-SAMPLE  (isolated, fixed ${BAL:,.0f}, your risk%)")
    print(f"  {'symbol':<11}{'risk%':>6}{'trades':>7}{'WR%':>7}{'PF':>7}"
          f"{'ret%':>8}{'DD%':>7}{'net$':>10}")
    print("  " + "-" * 70)
    for asset in C.PORTFOLIO_ASSETS:
        info = per_asset.get(asset)
        if not info:
            continue
        tr = info["trades"]
        st = acc_stats(tr, info["risk"], info["spread"])
        net = BAL * st["ret"] / 100.0
        print(f"  {info['label']:<11}{info['risk']*100:>5.1f}%{st['n']:>7}"
              f"{st['wr']:>7.1f}{st['pf']:>7.2f}{st['ret']:>+8.1f}{st['dd']:>7.1f}"
              f"{net:>+10.0f}")

    # ---- per-asset trade ledgers ----
    for asset in C.PORTFOLIO_ASSETS:
        info = per_asset.get(asset)
        if not info or not info["trades"]:
            continue
        tr = sorted(info["trades"], key=lambda t: t["time"])
        print(f"\n  --- {info['label']} OOS trades ({len(tr)}) ---")
        print(f"  {'#':>3}  {'entry time':<17}{'rule':<11}{'side':<6}"
              f"{'entry':>11}{'exit':>11}{'R':>7}  exit")
        print("  " + "-" * 78)
        spr = info["spread"]
        for i, t in enumerate(tr, 1):
            rule = t.get("setup") or t.get("rule", "?")
            side = "LONG" if t.get("dir") == "long" else "SHORT"
            entry = t.get("entry", 0.0)
            rpx = t.get("risk", 0.0)
            exit_px = entry + t["R"] * rpx if t.get("dir") == "long" else entry - t["R"] * rpx
            print(f"  {i:>3}  {str(t['time'])[:16]:<17}{rule:<11}{side:<6}"
                  f"{entry:>11.2f}{exit_px:>11.2f}{t['R']:>+7.2f}  {t.get('exit_reason','?')}")
    print(sep)


if __name__ == "__main__":
    main()
