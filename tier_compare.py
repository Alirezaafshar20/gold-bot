"""Compare Gold tiered 2+2 vs flat max 2 (30d walk-forward)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import floating_config as FC
import meta_gate as MG
import mt5_data as M
import portfolio_config as C
from portfolio_backtest import (
    prepare_portfolio_trades, simulate_portfolio, fetch_portfolio_trades, THIN,
)

BAL = 1000.0

DAYS = 30


def run_mode(filtered, use_tier: bool, flat_mc: int | None = None):
    trades = []
    for t in filtered:
        tc = dict(t)
        if not use_tier:
            tc["_golden"] = False
            if flat_mc is not None:
                tc["_max_concurrent"] = flat_mc
        trades.append(tc)

    saved = dict(C.TIERED_CONCURRENT)
    if not use_tier:
        C.TIERED_CONCURRENT.clear()
    try:
        return simulate_portfolio(
            trades, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK, size_compound=False)
    finally:
        C.TIERED_CONCURRENT.clear()
        C.TIERED_CONCURRENT.update(saved)


def main():
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=list(C.PORTFOLIO_ASSETS),
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )
    mt5 = M.connect()
    try:
        trades, wf, rmaps = fetch_portfolio_trades(DAYS, mt5, oos_mg)
    finally:
        M.shutdown(mt5)

    mg = FC.load_meta_gate(assets=list(C.PORTFOLIO_ASSETS))
    filtered = prepare_portfolio_trades(trades, rmaps, mg)

    modes = [
        ("2base+2golden (فعلی)", True, None),
        ("فقط 2 همزمان", False, 2),
        ("4 همزمان بدون tier", False, 4),
    ]
    rows = []
    for label, tier, mc in modes:
        r = run_mode(filtered, tier, mc)
        avg_r = 0.0
        if r["ledger"]:
            avg_r = sum(float(x["R"]) for x in r["ledger"]) / len(r["ledger"])
        rows.append((label, r, avg_r))

    print("=" * 92)
    print(f"  GOLD MAX-CONCURRENT COMPARE  |  {DAYS}d fixed-risk  |  CH-REV فقط TREND_UP")
    print(f"  Filtered signals: {len(filtered)}  |  walk-forward {wf['before']}→{wf['after']}")
    print("=" * 92)
    print(f"  {'mode':<28} {'trades':>6} {'skip':>5} {'WR%':>6} {'PF':>6} {'DD%':>6} "
          f"{'return':>9} {'profit$':>10} {'avgR':>6}")
    print("  " + THIN)
    for label, r, avg_r in rows:
        print(f"  {label:<28} {r['n']:>6} {r.get('skipped_mc', 0):>5} {r['wr']:>5.1f}% "
              f"{r['pf']:>5.2f} {r['max_dd']:>5.1f}% {r['ret_pct']:>+8.1f}% "
              f"{r['profit']:>+10.2f} {avg_r:>+6.2f}")

    r_tiered = rows[0][1]
    print(f"\n  WR by rule (2base+2golden):")
    by_rule = {}
    for row in r_tiered["ledger"]:
        rule = row.get("rule", "?")
        by_rule.setdefault(rule, {"n": 0, "w": 0, "r": 0.0})
        by_rule[rule]["n"] += 1
        by_rule[rule]["r"] += float(row["R"])
        if float(row["R"]) > 0:
            by_rule[rule]["w"] += 1
    for rule, s in sorted(by_rule.items(), key=lambda x: -x[1]["n"]):
        wr = s["w"] / s["n"] * 100 if s["n"] else 0
        print(f"    {rule:<12} n={s['n']:>3}  WR={wr:>5.1f}%  avgR={s['r']/s['n']:>+6.2f}")

    t2, r2 = rows[1][1], rows[0][1]
    print(f"\n  خلاصه: 2+2 vs فقط 2  →  WR {r2['wr']:.1f}% vs {t2['wr']:.1f}%  |  "
          f"profit ${r2['profit']:+.0f} vs ${t2['profit']:+.0f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
