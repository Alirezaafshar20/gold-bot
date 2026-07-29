"""
BRENT — 90d isolated backtest (current profile vs gold method).

  python brent_gold_backtest.py
  python brent_gold_backtest.py --fixed-risk
  python brent_gold_backtest.py --variant oos --fixed-risk
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

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
ASSET = "BRENT"

_OOS_RULES = ("FL", "BOS", "ADX_L", "EW_ABC", "PDC_RC_S")

_GOLD_OPT = {
    "vp_mode": None,
    "fib_ob_only": True,
    "fib_demand_exempt": True,
    "fib_nds_exempt": True,
    "session_start": 9,
    "session_end": 21,
    **FC.floating_opt_overrides(),
}


def _profile_for_variant(variant: str):
    _, prof = P.get_profile(ASSET)
    prof = dict(prof)
    if variant == "oos":
        prof["rules_override"] = _OOS_RULES
    if variant in ("gold", "oos"):
        prof["opt_overrides"] = {**prof.get("opt_overrides", {}), **_GOLD_OPT}
        prof["params"] = {**prof.get("params", {}), **FC.TOLERANCE}
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-risk", action="store_true")
    ap.add_argument("--variant", choices=("current", "gold", "oos"), default="gold",
                    help="current=profile rules | gold=3 rules+meta | oos=5 OOS rules+meta")
    ap.add_argument("--no-meta", action="store_true", help="VP off + tolerance only")
    ap.add_argument("--export", default="reports/brent_gold_trades.csv")
    args = ap.parse_args()

    risk = RISK_MAP[ASSET]
    prof = _profile_for_variant(args.variant)
    oos_mg = None
    if not args.no_meta and args.variant in ("gold", "oos"):
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
        opt_extra = _GOLD_OPT if args.variant in ("gold", "oos") else None
        tol = dict(FC.TOLERANCE) if args.variant in ("gold", "oos") else None
        if args.no_meta:
            opt_extra = {
                "vp_mode": None,
                "fib_ob_only": True,
                "fib_demand_exempt": True,
                "meta_gate": False,
                "session_start": 9,
                "session_end": 21,
            }
            tol = dict(FC.TOLERANCE)
            oos_mg = None

        stats = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
                     max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                     meta_gate_override=oos_mg,
                     opt_extra=opt_extra, tol_override=tol)

        raw = []
        for t in stats.get("trades") or []:
            tc = dict(t)
            tc["_asset"] = ASSET
            tc["_risk_pct"] = risk
            raw.append(tc)
        raw.sort(key=lambda t: (t["time"], t.get("exit_time")))

        use_wf = not args.no_meta and args.variant in ("gold", "oos")
        if use_wf:
            filtered, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(ASSET,))
        else:
            filtered, wf = raw, {"before": len(raw), "after": len(raw)}

        live = simulate_portfolio(filtered, BAL, size_compound=not args.fixed_risk)

        title = {
            "current": "پروفایل فعلی",
            "gold": "روش طلا (۳ rule + meta-gate)",
            "oos": "OOS کامل (۵ rule + meta-gate)",
        }[args.variant]
        if args.no_meta:
            title += " — بدون meta-gate"

        print("=" * 96)
        print(f"  BRENT BACKTEST — {DAYS}d | {title} | {sym}")
        print(f"  Rules: {', '.join(stats.get('enabled') or [])}")
        print(f"  Risk: {risk*100:.1f}% | Sizing: {'fixed-risk' if args.fixed_risk else 'compound'}")
        print("=" * 96)
        if oos_mg:
            print("  Meta-gate (OOS):")
            print(oos_mg.summary([ASSET]))
        print(f"\n  Raw trades: {len(raw)}")
        if use_wf:
            print(f"  Walk-forward: {wf['before']} → {wf['after']} ({wf['before']-wf['after']} removed)")
        print(f"\n  RESULT: {live['n']} trades | WR {live['wr']:.1f}% | PF {live['pf']:.2f} | "
              f"DD {live['max_dd']:.1f}%")
        print(f"  ${BAL:,.2f} → ${live['final']:,.2f} | Profit ${live['profit']:+,.2f} ({live['ret_pct']:+.1f}%)")

        if live.get("monthly"):
            print("\n  Monthly P&L ($):")
            for m in sorted(live["monthly"]):
                print(f"    {m}: {live['monthly'][m]:>+10.2f}")

        print_full_ledger(live.get("ledger", []), title="BRENT TRADES")

        if args.export and live.get("ledger"):
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
