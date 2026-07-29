"""XAUUSD session compare WITHOUT walk-forward — raw signals only."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
from multi_symbol_calibrate import _run
from portfolio_backtest import simulate_portfolio, RISK_MAP

_GOLD_OPT = {
    "vp_mode": None, "fib_ob_only": True, "fib_demand_exempt": True,
    **FC.floating_opt_overrides(),
}


def run(sym, prof, mt5, days, session):
    opt = dict(_GOLD_OPT)
    if session:
        opt["session_start"], opt["session_end"] = session
    else:
        opt["session_start"] = opt["session_end"] = None
    mg = MG.load_meta_gate(
        FC.META["meta_gate_json"], assets=["XAUUSD"],
        min_regime_n=FC.META["meta_min_regime_n"],
        require_positive_r=FC.META["meta_require_positive_r"],
        exclude_weak_rules=FC.META["meta_exclude_weak_rules"],
    )
    stats = _run(sym, prof, mt5, days=days, asset_key="XAUUSD",
                 risk_pct=RISK_MAP["XAUUSD"],
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=mg, opt_extra=opt,
                 tol_override=dict(FC.TOLERANCE))
    raw = [{**t, "_asset": "XAUUSD", "_risk_pct": RISK_MAP["XAUUSD"]}
           for t in stats.get("trades") or []]
    fix = simulate_portfolio(raw, 1000, size_compound=False)
    comp = simulate_portfolio(raw, 1000, size_compound=True)
    return fix, comp


if __name__ == "__main__":
    mt5 = M.connect()
    try:
        _, prof = P.get_profile("XAUUSD")
        sym = P.resolve_symbol_for_profile("XAUUSD", mt5)
        for days in (30, 90):
            print(f"\n=== XAU {days}d — بدون walk-forward (سیگنال خام) ===")
            for label, sess in [("سشن 9–21", (9, 21)), ("24/7", None)]:
                fix, comp = run(sym, prof, mt5, days, sess)
                print(f"  {label}: {fix['n']}t | fixed {fix['ret_pct']:+.1f}% DD {fix['max_dd']:.1f}% | "
                      f"compound {comp['ret_pct']:+.1f}% DD {comp['max_dd']:.1f}%")
    finally:
        M.shutdown(mt5)
