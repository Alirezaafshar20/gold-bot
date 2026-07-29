"""
US500 — 90d backtest, gold method (VP off + tolerance + meta-gate + walk-forward).

  python us500_gold_backtest.py
  python us500_gold_backtest.py --fixed-risk
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse

import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
import weekly_adaptive as WA
from multi_symbol_calibrate import _run, THIN
from portfolio_backtest import simulate_portfolio, RISK_MAP, print_full_ledger

DAYS = 90
BAL = 1000.0
ASSET = "US500"

_GOLD_OPT = {
    "vp_mode": None,
    "fib_ob_only": True,
    "fib_demand_exempt": True,
    "session_start": 13,
    "session_end": 22,
    **FC.floating_opt_overrides(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-risk", action="store_true")
    ap.add_argument("--export", default="reports/us500_gold_trades.csv")
    args = ap.parse_args()

    risk = RISK_MAP[ASSET]
    _, prof = P.get_profile(ASSET)
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=[ASSET],
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )

    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile(ASSET, mt5)
        stats = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
                     max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                     meta_gate_override=oos_mg,
                     opt_extra=_GOLD_OPT, tol_override=dict(FC.TOLERANCE))

        raw = []
        for t in stats.get("trades") or []:
            tc = dict(t)
            tc["_asset"] = ASSET
            tc["_risk_pct"] = risk
            raw.append(tc)
        raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
        filtered, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(ASSET,))
        live = simulate_portfolio(filtered, BAL, size_compound=not args.fixed_risk)

        print("=" * 96)
        print(f"  US500 BACKTEST — {DAYS}d | gold method | {sym}")
        print(f"  VP off | tolerance {FC.TOLERANCE['entry_tol_atr']} | meta-gate | session 13-22")
        print(f"  Risk: {risk*100:.1f}% | Sizing: {'fixed-risk' if args.fixed_risk else 'compound'}")
        print("=" * 96)
        if oos_mg:
            print("  Meta-gate:")
            print(oos_mg.summary([ASSET]))
        print(f"\n  Walk-forward: {wf['before']} → {wf['after']} ({wf['before']-wf['after']} removed)")
        print(f"\n  RESULT: {live['n']} trades | WR {live['wr']:.1f}% | PF {live['pf']:.2f} | "
              f"DD {live['max_dd']:.1f}%")
        print(f"  ${BAL:,.2f} → ${live['final']:,.2f} | Profit ${live['profit']:+,.2f} ({live['ret_pct']:+.1f}%)")

        if live.get("monthly"):
            print("\n  Monthly P&L ($):")
            for m in sorted(live["monthly"]):
                print(f"    {m}: {live['monthly'][m]:>+10.2f}")

        print_full_ledger(live.get("ledger", []), title="US500 TRADES (gold method)")

        if args.export and live.get("ledger"):
            import csv
            import os
            os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
            fields = ("n", "entry_time", "exit_time", "asset", "rule", "side", "entry", "exit",
                      "R", "risk_usd", "net", "balance", "exit_reason")
            with open(args.export, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(fields)
                for i, r in enumerate(live["ledger"], 1):
                    w.writerow([
                        i, r["entry_time"], r["exit_time"], ASSET, r["rule"],
                        r["dir"], r["entry"], r["exit_px"], r["R"],
                        r["risk_usd"], r["net"], r["balance"], r["exit"],
                    ])
            print(f"\n  Saved: {args.export}")
        print("=" * 96)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
