"""
US500 + US30 isolated analysis (90d, walk-forward) — current vs gold-style method.
"""
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
_FLOATING = FC.floating_opt_overrides()
_GOLD_STYLE = {
    "vp_mode": None,
    "fib_ob_only": True,
    "fib_demand_exempt": True,
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


def run_asset(asset, mt5, variant, oos_mg):
    risk = RISK_MAP.get(asset, 0.015)
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    opt_extra = None
    tol_extra = None
    mg = None
    if variant == "gold_style":
        opt_extra = dict(_GOLD_STYLE)
        opt_extra["session_start"] = 13
        opt_extra["session_end"] = 22
        mg = oos_mg
        tol_extra = dict(FC.TOLERANCE)
    elif variant == "gold_no_meta":
        opt_extra = {
            "vp_mode": None,
            "fib_ob_only": True,
            "fib_demand_exempt": True,
            "meta_gate": False,
            "session_start": 13,
            "session_end": 22,
        }
        tol_extra = dict(FC.TOLERANCE)
    stats = _run(sym, prof, mt5, days=DAYS, asset_key=asset, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=mg,
                 opt_extra=opt_extra, tol_override=tol_extra)
    raw = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = asset
        tc["_risk_pct"] = risk
        raw.append(tc)
    raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
    filtered, wf = WA.apply_walkforward_filter(
        raw, mt5=mt5, assets=(asset,) if variant == "gold_style" else (asset,))
    fix = simulate_portfolio(filtered, BAL, size_compound=False)
    comp = simulate_portfolio(filtered, BAL, size_compound=True)
    return {
        "sym": sym,
        "raw_n": len(raw),
        "final_n": fix["n"],
        "wf_removed": wf["before"] - wf["after"],
        "fix": fix,
        "comp": comp,
        "ledger_fix": fix.get("ledger", []),
        "enabled": stats.get("enabled", []),
    }


def print_asset_report(asset, results):
    print("\n" + "=" * 88)
    print(f"  {asset} — 90d isolated | risk {RISK_MAP.get(asset, 0)*100:.1f}%")
    print("=" * 88)
    for key, title in [
        ("current", "پروفایل فعلی"),
        ("gold_style", "روش طلا/بیت (+ meta-gate OOS)"),
        ("gold_no_meta", "VP off + tolerance (بدون meta-gate)"),
    ]:
        r = results.get(key)
        if r is None:
            print(f"\n  [{title}] — خطا: {r}")
            continue
        if r.get("error"):
            print(f"\n  [{title}] ERROR: {r['error']}")
            continue
        fix, comp = r["fix"], r["comp"]
        print(f"\n  --- {title} ---")
        print(f"  Symbol: {r['sym']} | Rules: {', '.join(r.get('enabled', []))}")
        print(f"  Raw → final: {r['raw_n']} → {r['final_n']}  (walk-forward -{r['wf_removed']})")
        print(f"  Fixed-risk : {fix['ret_pct']:+.1f}%  WR {fix['wr']:.1f}%  PF {fix['pf']:.2f}  DD {fix['max_dd']:.1f}%  (${fix['profit']:+,.0f})")
        print(f"  Compound   : {comp['ret_pct']:+.1f}%  WR {comp['wr']:.1f}%  PF {comp['pf']:.2f}  DD {comp['max_dd']:.1f}%  (${comp['profit']:+,.0f})")
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
    assets = ["US500", "US30"]
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=assets,
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )
    if oos_mg:
        print("Meta-gate (US500 has OOS; US30 falls back to allow-all for unknown rules):")
        print(oos_mg.summary(assets))

    mt5 = M.connect()
    all_results = {}
    try:
        for asset in assets:
            all_results[asset] = {}
            for variant in ("current", "gold_style", "gold_no_meta"):
                try:
                    mg = oos_mg if variant == "gold_style" else None
                    all_results[asset][variant] = run_asset(asset, mt5, variant, mg)
                except Exception as e:
                    all_results[asset][variant] = {"error": str(e)}
            print_asset_report(asset, all_results[asset])
    finally:
        M.shutdown(mt5)

    print("\n" + "=" * 88)
    print("  خلاصه مقایسه")
    print(f"  {'asset':<8} {'variant':<12} {'final':>5} {'fix ret':>8} {'fix DD':>7} {'comp ret':>9} {'comp DD':>8}")
    print("  " + "-" * 60)
    for asset in assets:
        for variant, label in [
            ("current", "فعلی"),
            ("gold_style", "طلا+meta"),
            ("gold_no_meta", "VP+tol"),
        ]:
            r = all_results[asset].get(variant, {})
            if r.get("error"):
                print(f"  {asset:<8} {label:<12}  ERROR")
                continue
            fix, comp = r.get("fix", {}), r.get("comp", {})
            print(f"  {asset:<8} {label:<12} {r.get('final_n', 0):>5} "
                  f"{fix.get('ret_pct', 0):>+7.1f}% {fix.get('max_dd', 0):>6.1f}% "
                  f"{comp.get('ret_pct', 0):>+8.1f}% {comp.get('max_dd', 0):>7.1f}%")
    print("=" * 88)


if __name__ == "__main__":
    main()
