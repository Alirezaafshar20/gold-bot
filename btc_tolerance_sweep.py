"""
BTC 90d tolerance sweep — tick fill (none confirm).

Wider entry_tol grid than ETH sweep (up to 1.5 ATR).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import floating_config as FC
import portfolio_backtest as PB
import portfolio_config as C
import mt5_data as M
import symbol_profiles as P
import meta_gate as MG
import weekly_adaptive as WA
import symbol_specs as X
from multi_symbol_calibrate import _run

DAYS = 90
BAL = 1000.0
ASSET = "BTCUSD"
BASE = dict(FC.TOLERANCE)
SEP = "=" * 100
THIN = "-" * 100

# Extended entry_tol grid (beyond ETH sweep)
ENTRY_SWEEP = (
    0.0, 0.08, 0.12, 0.15, 0.19, 0.22, 0.25, 0.30, 0.35, 0.38,
    0.42, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50,
)

SCALE_SWEEP = (0.4, 0.5, 0.65, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def _oos_gate():
    if not FC.META.get("meta_gate"):
        return None
    return MG.load_meta_gate(
        FC.META["meta_gate_json"],
        assets=[ASSET],
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )


def _portfolio_run(mt5, sym, prof, risk, rmap, oos_mg, tol_override, label, confirm="none"):
    stats = _run(
        sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
        max_concurrent=C.max_concurrent(ASSET),
        meta_gate_override=oos_mg,
        tol_override=tol_override,
        opt_extra={"entry_confirm_mode": confirm},
    )
    spec = X.get_spec(sym, mt5)
    trades = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = ASSET
        tc["_risk_pct"] = risk
        tc["_spread"] = stats["spread"]
        tc["_max_concurrent"] = C.max_concurrent(ASSET)
        tc["_spec"] = spec
        trades.append(tc)
    trades.sort(key=lambda x: (x["time"], x.get("exit_time")))
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
        trades, _ = WA.apply_walkforward_filter(trades, mt5=mt5, assets=(ASSET,))
    trades = PB.prepare_portfolio_trades(trades, {ASSET: rmap}, oos_mg)
    live = PB.simulate_portfolio(trades, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK)
    sl = sum(1 for t in trades if t["R"] <= -0.95)
    tol_n = sum(1 for t in trades if str(t.get("fill_kind", "")).lower() == "tol")
    exact_n = sum(1 for t in trades if str(t.get("fill_kind", "")).lower() == "exact")
    tot_r = sum(t["R"] for t in trades)
    return dict(
        label=label,
        confirm=confirm,
        entry=tol_override.get("entry_tol_atr", BASE["entry_tol_atr"]),
        detect=tol_override.get("detect_tol_atr", BASE["detect_tol_atr"]),
        zone=tol_override.get("zone_tol_atr", BASE["zone_tol_atr"]),
        inv=tol_override.get("invalidate_tol_atr", BASE["invalidate_tol_atr"]),
        n=live["n"],
        wr=live["wr"],
        pf=live["pf"],
        profit=live["profit"],
        ret=live["ret_pct"],
        dd=live["max_dd"],
        sl=sl,
        tol_n=tol_n,
        exact_n=exact_n,
        tot_r=tot_r,
        avg_r=tot_r / len(trades) if trades else 0.0,
    )


def _print_table(rows, title):
    print(f"\n  {title}")
    print(f"  {'label':<12} {'entry':>5} {'det':>5} {'zone':>5} {'inv':>5} "
          f"{'trades':>6} {'WR%':>6} {'PF':>5} {'avgR':>6} {'SL':>4} "
          f"{'ex':>4} {'tol':>4} {'profit':>9} {'DD%':>5}")
    print("  " + THIN)
    for r in rows:
        print(f"  {r['label']:<12} {r['entry']:>5.2f} {r['detect']:>5.2f} "
              f"{r['zone']:>5.2f} {r['inv']:>5.2f} {r['n']:>6} {r['wr']:>5.1f}% "
              f"{r['pf']:>5.2f} {r['avg_r']:>+6.2f} {r['sl']:>4} "
              f"{r['exact_n']:>4} {r['tol_n']:>4} {r['profit']:>+9.0f} {r['dd']:>5.1f}%")


def main():
    oos_mg = _oos_gate()
    mt5 = M.connect()
    try:
        _, prof = P.get_profile(ASSET)
        sym = P.resolve_symbol_for_profile(ASSET, mt5)
        risk = C.RISK_MAP[ASSET]
        rmap = PB._build_regime_map(sym, prof, mt5, DAYS)

        print(SEP)
        print(f"  BTC TOLERANCE SWEEP  |  {sym}  |  {DAYS}d  |  tick fill (none)")
        print(f"  base: entry={BASE['entry_tol_atr']} detect={BASE['detect_tol_atr']} "
              f"zone={BASE['zone_tol_atr']} inv={BASE['invalidate_tol_atr']} ATR")
        print(f"  meta-gate=ON  |  entry grid: {len(ENTRY_SWEEP)} points up to 1.50")
        print(SEP)

        entry_rows = []
        for et in ENTRY_SWEEP:
            tol = dict(BASE)
            tol["entry_tol_atr"] = et
            label = "exact" if et == 0.0 else f"e={et:g}"
            print(f"  entry_tol {label}...", flush=True)
            entry_rows.append(
                _portfolio_run(mt5, sym, prof, risk, rmap, oos_mg, tol, label))

        _print_table(entry_rows, "ENTRY_TOL_ATR sweep (detect/zone/inv = default)")

        scale_rows = []
        for sc in SCALE_SWEEP:
            tol = {k: round(v * sc, 3) for k, v in BASE.items()}
            label = f"s×{sc:g}"
            print(f"  scale {label}...", flush=True)
            scale_rows.append(
                _portfolio_run(mt5, sym, prof, risk, rmap, oos_mg, tol, label))

        _print_table(scale_rows, "FULL BUNDLE scale (all four tolerances × factor)")

        # Spot-check rejection at a few entry levels (BTC dislikes rejection at 0.38)
        rej_check = []
        for et in (0.38, 0.55, 0.80, 1.0):
            tol = dict(BASE)
            tol["entry_tol_atr"] = et
            label = f"rej e={et:g}"
            print(f"  rejection {label}...", flush=True)
            rej_check.append(
                _portfolio_run(mt5, sym, prof, risk, rmap, oos_mg, tol, label,
                               confirm="rejection"))

        _print_table(rej_check, "REJECTION confirm spot-check (for comparison)")

        best_entry = max(entry_rows, key=lambda x: x["profit"])
        best_pf = max(entry_rows, key=lambda x: x["pf"])
        best_wr = max(entry_rows, key=lambda x: x["wr"])
        base_row = next(r for r in entry_rows if abs(r["entry"] - BASE["entry_tol_atr"]) < 1e-6)
        best_scale = max(scale_rows, key=lambda x: x["profit"])

        print(f"\n  SUMMARY (tick / none confirm)")
        print(f"    Best profit     : {best_entry['label']}  entry={best_entry['entry']:.2f}  "
              f"${best_entry['profit']:+,.0f}  PF={best_entry['pf']:.2f}  "
              f"WR={best_entry['wr']:.1f}%  n={best_entry['n']}")
        print(f"    Best PF         : {best_pf['label']}  entry={best_pf['entry']:.2f}  "
              f"PF={best_pf['pf']:.2f}  profit=${best_pf['profit']:+,.0f}")
        print(f"    Best WR         : {best_wr['label']}  entry={best_wr['entry']:.2f}  "
              f"WR={best_wr['wr']:.1f}%  profit=${best_wr['profit']:+,.0f}")
        print(f"    Best scale      : {best_scale['label']}  "
              f"${best_scale['profit']:+,.0f}  PF={best_scale['pf']:.2f}  n={best_scale['n']}")
        print(f"    Default 0.38    : profit=${base_row['profit']:+,.0f}  "
              f"PF={base_row['pf']:.2f}  WR={base_row['wr']:.1f}%  n={base_row['n']}")

        low_dd = min(entry_rows, key=lambda x: x["dd"])
        print(f"    Lowest DD       : {low_dd['label']}  DD={low_dd['dd']:.1f}%  "
              f"profit=${low_dd['profit']:+,.0f}")

        print(SEP)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
