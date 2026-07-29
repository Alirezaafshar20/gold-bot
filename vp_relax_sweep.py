"""
VP filter relaxation sweep — XAUUSD 90d, walk-forward, fixed-risk.

  python vp_relax_sweep.py

vp_tol_atr: allow entry on wrong side of POC by N × ATR (softer edge).
vp_mode=None: VP fully off.
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
CUTOFF_LATE = pd.Timestamp("2026-06-23")


def run_variant(mt5, sym, prof, asset, oos_mg, *, opt_extra=None):
    risk = RISK_MAP[asset]
    stats = _run(sym, prof, mt5, days=DAYS, asset_key=asset, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=oos_mg, opt_extra=opt_extra)
    raw = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = asset
        tc["_risk_pct"] = risk
        raw.append(tc)
    raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
    filtered, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(asset,))
    live = simulate_portfolio(filtered, BAL, size_compound=False)
    late = [t for t in filtered if pd.Timestamp(t["time"]) >= CUTOFF_LATE]
    last_entry = max((pd.Timestamp(t["time"]) for t in filtered), default=None)
    return {
        "raw_n": len(raw),
        "final_n": live["n"],
        "removed_wf": wf["before"] - wf["after"],
        "late_n": len(late),
        "last_entry": str(last_entry)[:16] if last_entry else "—",
        "wr": live["wr"],
        "pf": live["pf"],
        "dd": live["max_dd"],
        "ret": live["ret_pct"],
    }


def main():
    asset = "XAUUSD"
    _, prof = P.get_profile(asset)
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=[asset],
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )

    variants = [
        ("VP سخت (tol=0)", {"vp_tol_atr": 0.0}),
        ("VP tol 0.25 ATR", {"vp_tol_atr": 0.25}),
        ("VP tol 0.5 ATR", {"vp_tol_atr": 0.5}),
        ("VP tol 1.0 ATR", {"vp_tol_atr": 1.0}),
        ("VP tol 1.5 ATR", {"vp_tol_atr": 1.5}),
        ("VP tol 2.0 ATR", {"vp_tol_atr": 2.0}),
        ("VP tol 3.0 ATR", {"vp_tol_atr": 3.0}),
        ("VP خاموش", {"vp_mode": None}),
        ("window 240 (tol=0)", {"vp_window": 240, "vp_tol_atr": 0.0}),
        ("window 720 (tol=0)", {"vp_window": 720, "vp_tol_atr": 0.0}),
        ("window 720 + tol 1.0", {"vp_window": 720, "vp_tol_atr": 1.0}),
    ]

    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile(asset, mt5)
        print("=" * 92)
        print(f"  VP RELAX SWEEP — {DAYS}d XAUUSD | fixed-risk | walk-forward")
        print("  vp_tol_atr = حداکثر فاصله از POC (سمت اشتباه) به واحد ATR")
        print("=" * 92)

        rows = []
        for name, opt_extra in variants:
            print(f"\n  Running: {name} ...", flush=True)
            r = run_variant(mt5, sym, prof, asset, oos_mg, opt_extra=opt_extra)
            r["name"] = name
            rows.append(r)

        base = rows[0]
        print("\n" + "=" * 92)
        print(f"  {'variant':<24} {'raw':>4} {'final':>5} {'wf-':>4} "
              f"{'Jun23+':>6} {'last entry':<17} {'WR':>5} {'PF':>5} {'DD':>5} {'ret%':>7}")
        print("  " + "-" * 88)
        for r in rows:
            delta = f" ({r['ret'] - base['ret']:+.1f})" if r["name"] != base["name"] else ""
            print(f"  {r['name']:<24} {r['raw_n']:>4} {r['final_n']:>5} {r['removed_wf']:>4} "
                  f"{r['late_n']:>6} {r['last_entry']:<17} {r['wr']:>4.0f}% "
                  f"{r['pf']:>5.2f} {r['dd']:>4.1f}% {r['ret']:>+6.1f}%{delta}")

        best = max(rows, key=lambda x: x["ret"])
        most = max(rows, key=lambda x: x["final_n"])
        print("\n  خلاصه:")
        print(f"  • بیشترین بازده: {best['name']} → {best['ret']:+.1f}% ({best['final_n']} trades)")
        print(f"  • بیشترین معامله: {most['name']} → {most['final_n']} trades ({most['ret']:+.1f}%)")
        print("=" * 92)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
