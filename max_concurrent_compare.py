"""
Compare max concurrent open trades per symbol: 2 vs 4.

  python max_concurrent_compare.py
  python max_concurrent_compare.py --days 30
"""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
import weekly_adaptive as WA
from multi_symbol_calibrate import _run
from portfolio_backtest import simulate_portfolio, RISK_MAP

DAYS = 90
BAL = 1000.0


def fetch_trades(days, mt5, oos_mg):
    assets = list(C.PORTFOLIO_ASSETS)
    all_trades = []
    for asset in assets:
        _, prof = P.get_profile(asset)
        sym = P.resolve_symbol_for_profile(asset, mt5)
        risk = RISK_MAP[asset]
        stats = _run(sym, prof, mt5, days=days, asset_key=asset, risk_pct=risk,
                     max_concurrent=99,  # raw: no per-asset cap at signal gen
                     meta_gate_override=oos_mg)
        for t in stats.get("trades") or []:
            tc = dict(t)
            tc["_asset"] = asset
            tc["_risk_pct"] = risk
            all_trades.append(tc)
    all_trades.sort(key=lambda t: (t["time"], t.get("exit_time")))
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
        all_trades, wf = WA.apply_walkforward_filter(
            all_trades, mt5=mt5, assets=tuple(assets))
    else:
        wf = {"before": len(all_trades), "after": len(all_trades)}
    return all_trades, wf


def sim(trades, mc, cap=None, compound=False):
    for t in trades:
        t["_max_concurrent"] = mc
    return simulate_portfolio(
        trades, BAL,
        max_concurrent=mc,
        max_portfolio_risk_pct=cap,
        size_compound=compound,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    args = ap.parse_args()

    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=list(C.PORTFOLIO_ASSETS),
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )

    print(f"Fetching {args.days}d trades (walk-forward)...")
    mt5 = M.connect()
    try:
        trades, wf = fetch_trades(args.days, mt5, oos_mg)
    finally:
        M.shutdown(mt5)

    cap = C.MAX_PORTFOLIO_RISK
    print("=" * 88)
    print(f"  MAX CONCURRENT COMPARE  |  {args.days}d  |  cap {cap*100:.0f}% portfolio risk")
    print(f"  Walk-forward signals: {wf['before']} → {wf['after']}")
    print(f"  Risk: XAU {RISK_MAP['XAUUSD']*100:.1f}%  BTC {RISK_MAP['BTCUSD']*100:.1f}%")
    print("=" * 88)

    rows = []
    for mc in (2, 4):
        for label, compound in (("fixed-risk", False), ("compound", True)):
            r = sim(trades, mc, cap=cap, compound=compound)
            rows.append((mc, label, r))

    print(f"\n  {'max/symbol':<10} {'sizing':<12} {'trades':>6} {'skip cap':>8} {'scaled':>7} "
          f"{'WR%':>6} {'PF':>5} {'DD%':>6} {'return':>8} {'profit$':>10}")
    print("  " + "-" * 82)
    for mc, label, r in rows:
        print(f"  {mc:<10} {label:<12} {r['n']:>6} {r.get('skipped_cap', 0):>8} "
              f"{r.get('scaled', 0):>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"{r['max_dd']:>5.1f}% {r['ret_pct']:>+7.1f}% {r['profit']:>+10.0f}")

    r2f = sim(trades, 2, cap=cap, compound=False)
    r4f = sim(trades, 4, cap=cap, compound=False)
    r2c = sim(trades, 2, cap=cap, compound=True)
    r4c = sim(trades, 4, cap=cap, compound=True)

    print("\n  تفاوت 2 → 4 (با cap 15%):")
    print(f"    Fixed-risk : {r2f['n']} → {r4f['n']} trades  |  "
          f"{r2f['ret_pct']:+.1f}% → {r4f['ret_pct']:+.1f}%  |  "
          f"DD {r2f['max_dd']:.1f}% → {r4f['max_dd']:.1f}%")
    print(f"    Compound   : {r2c['n']} → {r4c['n']} trades  |  "
          f"{r2c['ret_pct']:+.1f}% → {r4c['ret_pct']:+.1f}%  |  "
          f"DD {r2c['max_dd']:.1f}% → {r4c['max_dd']:.1f}%")
    print(f"    Max open positions observed: {r2f.get('max_open', '?')} → {r4f.get('max_open', '?')}")

    # No cap scenario
    r4_nc = sim(trades, 4, cap=None, compound=False)
    print(f"\n  4/symbol بدون cap پورتفولیو (fixed):")
    print(f"    trades={r4_nc['n']}  ret={r4_nc['ret_pct']:+.1f}%  DD={r4_nc['max_dd']:.1f}%  "
          f"max_open={r4_nc.get('max_open', '?')}")

    print("\n  نکته: با cap 15%، حداکثر ریسک باز ≈ $150 روی $1000")
    print(f"        4× طلا (3.5%) = 14%  یا  2× طلا + 2× بیت ≈ 10%  — cap اغلب محدودکننده است")
    print("=" * 88)


if __name__ == "__main__":
    main()
