"""
Test each relaxation ONE AT A TIME vs baseline (90d XAUUSD, walk-forward).

  python relax_sweep.py
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


def run_variant(name, mt5, sym, prof, asset, oos_mg, *,
                opt_extra=None, tol_extra=None):
    risk = RISK_MAP[asset]
    stats = _run(sym, prof, mt5, days=DAYS, asset_key=asset, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=oos_mg,
                 opt_extra=opt_extra, tol_override=tol_extra)
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
        "name": name,
        "raw_n": len(raw),
        "final_n": live["n"],
        "removed_wf": wf["before"] - wf["after"],
        "late_n": len(late),
        "last_entry": str(last_entry)[:16] if last_entry else "—",
        "wr": live["wr"],
        "pf": live["pf"],
        "dd": live["max_dd"],
        "profit": live["profit"],
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
    tol = dict(FC.TOLERANCE)

    variants = [
        ("baseline (فعلی)", {}, None),
        ("1) session 8–22", {"session_start": 8, "session_end": 22}, None),
        ("2) بدون fib", {"require_fib": False}, None),
        ("3) entry_tol 0.48", None, {**tol, "entry_tol_atr": 0.48}),
        ("4) WAIT 64 bar", {"wait": 64}, None),
    ]

    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile(asset, mt5)
        print("=" * 88)
        print(f"  RELAX SWEEP — {DAYS}d XAUUSD | fixed-risk | walk-forward | هر تغییر جدا")
        print(f"  Tolerance base: entry={tol['entry_tol_atr']} detect={tol['detect_tol_atr']} "
              f"zone={tol['zone_tol_atr']} invalidate={tol['invalidate_tol_atr']}")
        print("=" * 88)

        rows = []
        for name, opt_extra, tol_extra in variants:
            print(f"\n  Running: {name} ...", flush=True)
            rows.append(run_variant(name, mt5, sym, prof, asset, oos_mg,
                                    opt_extra=opt_extra, tol_extra=tol_extra))

        base = rows[0]
        print("\n" + "=" * 88)
        print(f"  {'variant':<22} {'raw':>4} {'final':>5} {'wf-':>4} "
              f"{'Jun23+':>6} {'last entry':<17} {'WR':>5} {'PF':>5} {'DD':>5} {'ret%':>7}")
        print("  " + "-" * 84)
        for r in rows:
            delta = f" ({r['ret'] - base['ret']:+.1f})" if r["name"] != base["name"] else ""
            print(f"  {r['name']:<22} {r['raw_n']:>4} {r['final_n']:>5} {r['removed_wf']:>4} "
                  f"{r['late_n']:>6} {r['last_entry']:<17} {r['wr']:>4.0f}% "
                  f"{r['pf']:>5.2f} {r['dd']:>4.1f}% {r['ret']:>+6.1f}%{delta}")

        print("\n  تحلیل کوتاه:")
        for r in rows[1:]:
            d_raw = r["raw_n"] - base["raw_n"]
            d_fin = r["final_n"] - base["final_n"]
            d_late = r["late_n"] - base["late_n"]
            d_ret = r["ret"] - base["ret"]
            verdict = "✅" if d_fin > 0 and d_ret >= -5 else ("⚠️" if d_late > 0 else "❌")
            print(f"  {verdict} {r['name']}: raw{d_raw:+d} final{d_fin:+d} "
                  f"Jun23+{d_late:+d} ret{d_ret:+.1f}% last={r['last_entry']}")
        print("=" * 88)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
