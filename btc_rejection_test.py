"""BTC 90d: none vs rejection entry confirm."""
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import portfolio_config as C
import portfolio_backtest as PB
import mt5_data as M
import symbol_profiles as P
import floating_config as FC
import meta_gate as MG
import weekly_adaptive as WA
import symbol_specs as X
from multi_symbol_calibrate import _run

DAYS = 90
BAL = 1000.0
ASSET = "BTCUSD"
MODES = ("none", "rejection")
SEP = "=" * 88


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


def _run_mode(mt5, sym, prof, risk, rmap, oos_mg, mode):
    stats = _run(
        sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=risk,
        max_concurrent=C.max_concurrent(ASSET),
        meta_gate_override=oos_mg,
        opt_extra={"entry_confirm_mode": mode},
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
    tot_r = sum(t["R"] for t in trades)
    by_rule = defaultdict(lambda: {"n": 0, "sl": 0, "R": 0.0})
    for t in trades:
        rk = str(t.get("setup", "?")).split("-")[0]
        by_rule[rk]["n"] += 1
        by_rule[rk]["R"] += t["R"]
        if t["R"] <= -0.95:
            by_rule[rk]["sl"] += 1
    return dict(
        mode=mode, raw=len(trades), n=live["n"], wr=live["wr"], pf=live["pf"],
        ret=live["ret_pct"], profit=live["profit"], dd=live["max_dd"],
        sl=sl, tot_r=tot_r, avg_r=tot_r / len(trades) if trades else 0,
        by_rule=dict(by_rule),
    )


def main():
    oos_mg = _oos_gate()
    mt5 = M.connect()
    try:
        _, prof = P.get_profile(ASSET)
        sym = P.resolve_symbol_for_profile(ASSET, mt5)
        risk = C.RISK_MAP[ASSET]
        rmap = PB._build_regime_map(sym, prof, mt5, DAYS)
        print(SEP)
        print(f"  BTC REJECTION TEST  |  {sym}  |  {DAYS}d  |  balance=${BAL:,.0f}")
        print(f"  rules: {prof.get('rules_override')}  risk={risk*100:.1f}%  spread~{prof.get('spread')}")
        print(SEP)

        rows = [_run_mode(mt5, sym, prof, risk, rmap, oos_mg, m) for m in MODES]

        print(f"  {'mode':<12} {'trades':>6} {'WR%':>6} {'PF':>6} {'avgR':>6} "
              f"{'SL':>4} {'return':>9} {'profit':>10} {'DD%':>6}")
        print("  " + "-" * 72)
        for r in rows:
            print(f"  {r['mode']:<12} {r['n']:>6} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['avg_r']:>+6.2f} {r['sl']:>4} {r['ret']:>+8.1f}% "
                  f"{r['profit']:>+10.0f} {r['dd']:>5.1f}%")

        base, rej = rows[0], rows[1]
        print(f"\n  DELTA rejection vs none:")
        print(f"    trades {rej['n'] - base['n']:+d}  WR {rej['wr'] - base['wr']:+.1f}pp  "
              f"PF {rej['pf'] - base['pf']:+.2f}  profit ${rej['profit'] - base['profit']:+,.0f}  "
              f"SL {rej['sl'] - base['sl']:+d}")

        for label, r in [("none", base), ("rejection", rej)]:
            print(f"\n  PER-RULE ({label})")
            for rk, v in sorted(r["by_rule"].items(), key=lambda x: -x[1]["n"]):
                sl_pct = v["sl"] / v["n"] * 100 if v["n"] else 0
                print(f"    {rk:<8} n={v['n']:>3}  SL={v['sl']:>2} ({sl_pct:.0f}%)  totR={v['R']:+.1f}")

        if rej["n"] == 0 and base["n"] == 0:
            print("\n  No BTC trades in period — check symbol / filters / meta-gate.")
        elif rej["profit"] > base["profit"]:
            print(f"\n  Verdict: rejection HELPS on BTC (+${rej['profit'] - base['profit']:,.0f})")
        elif rej["profit"] < base["profit"]:
            print(f"\n  Verdict: rejection HURTS on BTC (${rej['profit'] - base['profit']:+,.0f})")
        else:
            print("\n  Verdict: similar profit")
        print(SEP)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
