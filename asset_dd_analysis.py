"""Per-asset SL/PF breakdown + max_concurrent comparison."""
import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
from collections import defaultdict

META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)

BAL = 1000.0


def run_backtest(d, m1, B, ctx, htf, prof, spread, eff_sl, rules, cutoff, max_concurrent=None):
    o = P.apply_profile_to_settings(
        S.dense_plus_settings(wide=True), prof, spread, min_sl_override=eff_sl)
    o["enabled"] = list(rules)
    if max_concurrent is not None:
        o["max_concurrent"] = max_concurrent
    k = {x: o[x] for x in o if x not in META}
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    return [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
            if t["time"] >= cutoff]


def analyze(trades, spread, risk, label):
    acc = S.simulate_account(trades, BAL, risk, spread)
    by = defaultdict(list)
    for t in trades:
        by[t.get("setup", "?")].append(t)

    print(f"\n=== {label} ===")
    print(f"  n={acc['n']}  WR={acc['wr']:.1f}%  PF={acc['pf']:.2f}  "
          f"ret={acc['ret_pct']:+.1f}%  DD={acc['max_dd']:.1f}%")
    print(f"  {'rule':<12}{'trades':>7}{'SL':>5}{'BE':>5}{'WR%':>7}{'PF':>7}{'totR':>8}")
    print("  " + "-" * 52)

    rows = []
    for rk, ts in by.items():
        Rs = np.array([x["R"] - spread / x["risk"] for x in ts])
        sl = sum(1 for x in ts if x.get("reason") == "sl" or x["R"] <= -0.99)
        be = sum(1 for x in ts if x.get("reason") == "be" or abs(x["R"]) < 0.01)
        w, l = Rs[Rs > 0], Rs[Rs <= 0]
        pf = float(w.sum() / (-l.sum())) if l.sum() < 0 else 999.0
        rows.append((sl, rk, len(ts), be, (Rs > 0).mean() * 100, pf, float(Rs.sum())))

    rows.sort(key=lambda x: (-x[0], -x[2]))
    for sl, rk, n, be, wr, pf, tot in rows:
        print(f"  {rk:<12}{n:>7}{sl:>5}{be:>5}{wr:>6.1f}%{pf:>7.2f}{tot:>+8.1f}")

    total_sl = sum(1 for t in trades if t.get("reason") == "sl" or t["R"] <= -0.99)
    print(f"  TOTAL SL exits: {total_sl}/{len(trades)}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=P.ASSET_KEYS)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--risk", type=float, default=None,
                    help="Override risk fraction (default: profile risk_pct)")
    args = ap.parse_args()

    _, prof = P.get_profile(args.asset)
    rules = tuple(prof["rules_override"])

    mt5 = M.connect()
    sym = P.resolve_symbol_for_profile(args.asset, mt5)
    spread = P.live_spread(sym, prof, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
    _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)

    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    eff_sl = X.calibrate_min_sl(B, cutoff, prof)
    risk = args.risk if args.risk is not None else X.profile_risk_pct(prof)

    print(f"{args.asset} {sym} | risk={risk*100:.2g}% | min_sl={eff_sl:.4f} | {args.days}d")

    tr = run_backtest(d, m1, B, ctx, htf, prof, spread, eff_sl, rules, cutoff)
    analyze(tr, spread, risk, f"Current rules {rules}")

    print("\n=== max_concurrent comparison ===")
    for mc in (2, 1):
        tr = run_backtest(d, m1, B, ctx, htf, prof, spread, eff_sl, rules, cutoff, max_concurrent=mc)
        acc = S.simulate_account(tr, BAL, risk, spread)
        sl = sum(1 for t in tr if t.get("reason") == "sl" or t["R"] <= -0.99)
        print(f"  concurrent={mc}: n={acc['n']} WR={acc['wr']:.1f}% PF={acc['pf']:.2f} "
              f"ret={acc['ret_pct']:+.1f}% DD={acc['max_dd']:.1f}% SL={sl}")


if __name__ == "__main__":
    main()
