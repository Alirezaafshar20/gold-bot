"""Win rate by rule — XAUUSD + BTCUSD, last N days."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
import weekly_adaptive as WA
from multi_symbol_calibrate import _run
from portfolio_backtest import RISK_MAP

DAYS = 30


def rule_stats(trades):
    rows = []
    for t in trades:
        rows.append({
            "rule": t.get("rule", "?"),
            "win": 1 if t["R"] > 0 else 0,
            "R": t["R"],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby("rule").agg(
        n=("win", "count"),
        wins=("win", "sum"),
        wr=("win", "mean"),
        avg_r=("R", "mean"),
        tot_r=("R", "sum"),
    ).sort_values("wr", ascending=False)
    g["wr"] = g["wr"] * 100
    return g


def fetch_asset(asset, mt5, oos_mg):
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    risk = RISK_MAP[asset]
    stats = _run(sym, prof, mt5, days=DAYS, asset_key=asset, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=oos_mg)
    raw = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = asset
        raw.append(tc)
    filtered, _ = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(asset,))
    return filtered, stats.get("enabled", [])


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
        print("=" * 72)
        print(f"  WIN RATE BY RULE — last {DAYS} days | walk-forward | live profile")
        print("=" * 72)
        for asset in C.PORTFOLIO_ASSETS:
            trades, enabled = fetch_asset(asset, mt5, oos_mg)
            label = P.get_profile(asset)[1].get("label", asset)
            print(f"\n  {label} ({asset}) — {len(trades)} trades | rules: {', '.join(enabled)}")
            print(f"  {'rule':<12} {'n':>4} {'wins':>5} {'WR%':>6} {'avgR':>7} {'totR':>7}")
            print("  " + "-" * 44)
            g = rule_stats(trades)
            if g.empty:
                print("  (no trades)")
                continue
            for rule, row in g.iterrows():
                print(f"  {rule:<12} {int(row['n']):>4} {int(row['wins']):>5} "
                      f"{row['wr']:>5.0f}% {row['avg_r']:>+7.2f} {row['tot_r']:>+7.1f}")
            best = g[g["n"] >= 2].head(1)
            if not best.empty:
                r = best.index[0]
                print(f"\n  → بیشترین WR (n≥2): {r} — {best.iloc[0]['wr']:.0f}% ({int(best.iloc[0]['n'])} trades)")
            elif len(g):
                r = g.index[0]
                print(f"\n  → تنها/بهترین: {r} — {g.iloc[0]['wr']:.0f}% ({int(g.iloc[0]['n'])} trade)")
        print("\n" + "=" * 72)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
