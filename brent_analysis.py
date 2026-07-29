"""
BRENT isolated analysis — compare current profile vs gold/BTC method (90d, walk-forward).

  python brent_analysis.py
"""
from __future__ import annotations

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
from portfolio_backtest import simulate_portfolio, RISK_MAP

DAYS = 90
BAL = 1000.0
ASSET = "BRENT"
_OOS_RULES = ("FL", "BOS", "ADX_L", "EW_ABC", "PDC_RC_S")
_FLOATING = FC.floating_opt_overrides()
_GOLD_STYLE = {
    "vp_mode": None,
    "fib_ob_only": True,
    "fib_demand_exempt": True,
    "fib_nds_exempt": True,
    "session_start": 9,
    "session_end": 21,
    **_FLOATING,
}


def analyze_trades(trades, label):
    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)
    pnl_col = "net" if "net" in df.columns else "pnl_usd"
    n = len(df)
    wins = (df["R"] > 0).sum()
    gp = df.loc[df[pnl_col] > 0, pnl_col].sum()
    gl = abs(df.loc[df[pnl_col] < 0, pnl_col].sum())
    pf = gp / gl if gl else 999
    by_rule = df.groupby("rule").agg(
        n=("R", "count"),
        wr=("R", lambda x: (x > 0).mean() * 100),
        tot_r=("R", "sum"),
        pnl=(pnl_col, "sum"),
    ).sort_values("pnl", ascending=False)
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M").astype(str)
    by_month = df.groupby("month").agg(n=("R", "count"), pnl=(pnl_col, "sum"))
    return {
        "label": label, "n": n, "wr": wins / n * 100, "pf": pf,
        "avg_r": df["R"].mean(), "pnl": df[pnl_col].sum(),
        "by_rule": by_rule, "by_month": by_month,
    }


def run_variant(variant, mt5, oos_mg):
    risk = RISK_MAP.get(ASSET, 0.015)
    _, prof = P.get_profile(ASSET)
    prof = dict(prof)
    opt_extra = None
    tol_extra = None
    mg = None
    rules_note = list(prof.get("rules_override") or prof.get("rules", []))

    if variant == "oos_gold":
        prof["rules_override"] = _OOS_RULES
        opt_extra = dict(_GOLD_STYLE)
        tol_extra = dict(FC.TOLERANCE)
        mg = oos_mg
        rules_note = list(_OOS_RULES)
    elif variant == "gold_style":
        opt_extra = dict(_GOLD_STYLE)
        tol_extra = dict(FC.TOLERANCE)
        mg = oos_mg
    elif variant == "gold_no_meta":
        opt_extra = {
            "vp_mode": None,
            "fib_ob_only": True,
            "fib_demand_exempt": True,
            "meta_gate": False,
            "session_start": 9,
            "session_end": 21,
        }
        tol_extra = dict(FC.TOLERANCE)

    sym = P.resolve_symbol_for_profile(ASSET, mt5)
    stats = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=mg,
                 opt_extra=opt_extra, tol_override=tol_extra)
    raw = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = ASSET
        tc["_risk_pct"] = risk
        raw.append(tc)
    raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
    use_wf = variant in ("gold_style", "oos_gold")
    if use_wf:
        filtered, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(ASSET,))
    else:
        filtered, wf = raw, {"before": len(raw), "after": len(raw)}
    fix = simulate_portfolio(filtered, BAL, size_compound=False)
    comp = simulate_portfolio(filtered, BAL, size_compound=True)
    return {
        "sym": sym,
        "rules": rules_note,
        "raw_n": len(raw),
        "final_n": fix["n"],
        "wf_removed": wf["before"] - wf["after"],
        "fix": fix,
        "comp": comp,
        "enabled": stats.get("enabled", []),
    }


def print_report(results):
    print("\n" + "=" * 88)
    print(f"  BRENT — 90d isolated | risk {RISK_MAP.get(ASSET, 0)*100:.1f}%")
    print("=" * 88)
    titles = {
        "current": "پروفایل فعلی (FL, ADX_L, EW_ABC)",
        "gold_style": "روش طلا/بیت — ۳ rule + meta-gate + walk-forward",
        "oos_gold": "OOS کامل — ۵ rule + meta-gate + walk-forward",
        "gold_no_meta": "VP off + tolerance (بدون meta-gate)",
    }
    for key, title in titles.items():
        r = results.get(key, {})
        if r.get("error"):
            print(f"\n  [{title}] ERROR: {r['error']}")
            continue
        fix, comp = r["fix"], r["comp"]
        print(f"\n  --- {title} ---")
        print(f"  Symbol: {r['sym']} | Rules enabled: {', '.join(r.get('enabled', []))}")
        print(f"  Raw → final: {r['raw_n']} → {r['final_n']}  (walk-forward -{r['wf_removed']})")
        print(f"  Fixed-risk : {fix['ret_pct']:+.1f}%  WR {fix['wr']:.1f}%  PF {fix['pf']:.2f}  "
              f"DD {fix['max_dd']:.1f}%  (${fix['profit']:+,.0f})")
        print(f"  Compound   : {comp['ret_pct']:+.1f}%  WR {comp['wr']:.1f}%  PF {comp['pf']:.2f}  "
              f"DD {comp['max_dd']:.1f}%  (${comp['profit']:+,.0f})")
        if not fix.get("ledger"):
            print("  (no trades)")
            continue
        ana = analyze_trades(fix["ledger"], title)
        print("\n  Per-rule (fixed-risk $):")
        print(f"  {'rule':<12} {'n':>4} {'WR%':>6} {'totR':>7} {'pnl$':>9}")
        for rule, row in ana["by_rule"].iterrows():
            print(f"  {rule:<12} {int(row['n']):>4} {row['wr']:>5.0f}% {row['tot_r']:>+7.1f} {row['pnl']:>+9.0f}")
        print("\n  Monthly PnL ($):")
        for m, row in ana["by_month"].iterrows():
            print(f"    {m}: {int(row['n']):>3} trades  {row['pnl']:>+9.0f}")


def main():
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=[ASSET],
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )
    if oos_mg:
        print("Meta-gate BRENT (from oos_test.json):")
        print(oos_mg.summary([ASSET]))

    mt5 = M.connect()
    results = {}
    try:
        for variant in ("current", "gold_style", "oos_gold", "gold_no_meta"):
            try:
                results[variant] = run_variant(variant, mt5, oos_mg)
            except Exception as e:
                results[variant] = {"error": str(e)}
        print_report(results)
    finally:
        M.shutdown(mt5)

    print("\n" + "=" * 88)
    print("  خلاصه مقایسه BRENT")
    print(f"  {'variant':<14} {'final':>5} {'fix ret':>8} {'fix DD':>7} {'comp ret':>9} {'comp DD':>8}")
    print("  " + "-" * 58)
    labels = {
        "current": "فعلی",
        "gold_style": "طلا+meta",
        "oos_gold": "OOS+meta",
        "gold_no_meta": "VP+tol",
    }
    for variant, label in labels.items():
        r = results.get(variant, {})
        if r.get("error"):
            print(f"  {label:<14}  ERROR: {r['error'][:40]}")
            continue
        fix, comp = r.get("fix", {}), r.get("comp", {})
        print(f"  {label:<14} {r.get('final_n', 0):>5} "
              f"{fix.get('ret_pct', 0):>+7.1f}% {fix.get('max_dd', 0):>6.1f}% "
              f"{comp.get('ret_pct', 0):>+8.1f}% {comp.get('max_dd', 0):>7.1f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()
