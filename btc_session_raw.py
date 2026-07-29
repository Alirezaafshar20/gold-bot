"""BTC session compare WITHOUT walk-forward — raw signals only."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
from multi_symbol_calibrate import _run
from portfolio_backtest import simulate_portfolio, RISK_MAP

_GOLD = {
    "vp_mode": None, "fib_ob_only": True, "fib_demand_exempt": True,
    "fib_nds_exempt": True, **FC.floating_opt_overrides(),
}


def run(days, session):
    _, prof = P.get_profile("BTCUSD")
    opt = dict(_GOLD)
    if session:
        opt["session_start"], opt["session_end"] = session
    else:
        opt["session_start"] = opt["session_end"] = None
    mg = MG.load_meta_gate(
        FC.META["meta_gate_json"], assets=["BTCUSD"],
        min_regime_n=FC.META["meta_min_regime_n"],
        require_positive_r=FC.META["meta_require_positive_r"],
        exclude_weak_rules=FC.META["meta_exclude_weak_rules"],
    )
    stats = _run(sym, prof, mt5, days=days, asset_key="BTCUSD",
                 risk_pct=RISK_MAP["BTCUSD"],
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=mg, opt_extra=opt,
                 tol_override=dict(FC.TOLERANCE))
    raw = [{**t, "_asset": "BTCUSD", "_risk_pct": RISK_MAP["BTCUSD"]}
           for t in stats.get("trades") or []]
    fix = simulate_portfolio(raw, 1000, size_compound=False)
    comp = simulate_portfolio(raw, 1000, size_compound=True)
    return fix, comp


if __name__ == "__main__":
    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile("BTCUSD", mt5)
        for days in (30, 90):
            print(f"\n=== BTC {days}d — بدون walk-forward (سیگنال خام) ===")
            for label, sess in [("سشن 9–21", (9, 21)), ("24/7", None)]:
                fix, comp = run(days, sess)
                print(f"  {label}: {fix['n']}t | fixed {fix['ret_pct']:+.1f}% DD {fix['max_dd']:.1f}% | "
                      f"compound {comp['ret_pct']:+.1f}% DD {comp['max_dd']:.1f}%")
    finally:
        M.shutdown(mt5)
