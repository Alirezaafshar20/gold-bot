"""BRENT 30d isolated snapshot — fixed + compound + weekly breakdown."""
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

DAYS = 30
BAL = 1000.0
ASSET = "BRENT"
_GOLD = {
    "vp_mode": None, "fib_ob_only": True, "fib_demand_exempt": True,
    "fib_nds_exempt": True, "session_start": 9, "session_end": 21,
    **FC.floating_opt_overrides(),
}


def main():
    mt5 = M.connect()
    try:
        _, prof = P.get_profile(ASSET)
        sym = P.resolve_symbol_for_profile(ASSET, mt5)
        risk = RISK_MAP[ASSET]
        mg = MG.load_meta_gate(
            FC.META.get("meta_gate_json", MG.DEFAULT_JSON), assets=[ASSET],
            min_regime_n=FC.META.get("meta_min_regime_n", 3),
            require_positive_r=FC.META.get("meta_require_positive_r", True),
            exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
        )
        stats = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
                     max_concurrent=C.MAX_CONCURRENT_PER_ASSET, meta_gate_override=mg,
                     opt_extra=_GOLD, tol_override=dict(FC.TOLERANCE))
        raw = [{**t, "_asset": ASSET, "_risk_pct": risk} for t in stats.get("trades") or []]
        raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
        filtered, _ = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(ASSET,))
        fix = simulate_portfolio(filtered, BAL, size_compound=False)
        comp = simulate_portfolio(filtered, BAL, size_compound=True)
        print(f"BRENT 30d isolated | {sym}")
        print(f"  Fixed   : {fix['n']}t  ret {fix['ret_pct']:+.1f}%  DD {fix['max_dd']:.1f}%  PF {fix['pf']:.2f}")
        print(f"  Compound: {comp['n']}t  ret {comp['ret_pct']:+.1f}%  DD {comp['max_dd']:.1f}%")
        if not fix.get("ledger"):
            return
        df = pd.DataFrame(fix["ledger"])
        print("\n  Per-rule ($ fixed-risk):")
        g = df.groupby("rule").agg(
            n=("R", "count"), wr=("R", lambda x: (x > 0).mean() * 100),
            tot_r=("R", "sum"), pnl=("net", "sum"),
        ).sort_values("pnl", ascending=False)
        for r, row in g.iterrows():
            print(f"    {r:<10} n={int(row.n):>2}  WR={row.wr:>4.0f}%  totR={row.tot_r:>+6.1f}  ${row.pnl:>+7.0f}")
        df["side"] = df["dir"]
        print("\n  FL by side:")
        fl = df[df["rule"].str.contains("FL", na=False)]
        if len(fl):
            for side, sg in fl.groupby("side"):
                print(f"    {side:<6} n={len(sg)}  WR={(sg['R']>0).mean()*100:.0f}%  totR={sg['R'].sum():+.1f}  ${sg['net'].sum():+.0f}")
        df["week"] = pd.to_datetime(df["entry_time"]).dt.strftime("%G-W%V")
        print("\n  Weekly:")
        for w, gw in df.groupby("week"):
            print(f"    {w}: {len(gw):>2}t  WR={(gw['R']>0).mean()*100:.0f}%  ${gw['net'].sum():>+7.0f}")
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
