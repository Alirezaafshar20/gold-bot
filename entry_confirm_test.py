"""
Compare all entry-confirm modes on XAUUSD — full portfolio pipeline (90d).

Modes:
  none         — first wick in entry_tol band (legacy tick)
  m15_close    — touched + close inside band + directional candle
  rejection    — touched + close reclaims proximal (SMC rejection)
  rule_confirm — rejection only for CH_REV_L / WYCK_SOW; tick for others
"""
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
MODES = ("none", "m15_close", "rejection", "rule_confirm")
SEP = "=" * 92
THIN = "-" * 92


def _oos_gate():
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True) and FC.META.get("meta_gate"):
        return MG.load_meta_gate(
            FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
            assets=list(C.PORTFOLIO_ASSETS),
            min_regime_n=FC.META.get("meta_min_regime_n", 3),
            require_positive_r=FC.META.get("meta_require_positive_r", True),
            exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
        )
    return None


def _load_trades(mt5, mode, oos_mg):
    asset = "XAUUSD"
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    risk = C.RISK_MAP[asset]
    rmap = PB._build_regime_map(sym, prof, mt5, DAYS)
    extra = {"entry_confirm_mode": mode}
    if mode == "rule_confirm":
        extra["entry_confirm_rules"] = tuple(FC.ENTRY.get(
            "entry_confirm_rules", ("CH_REV_L", "WYCK_SOW")))
    stats = _run(
        sym, prof, mt5, days=DAYS, asset_key=asset, risk_pct=risk,
        max_concurrent=C.max_concurrent(asset),
        meta_gate_override=oos_mg,
        opt_extra=extra,
    )
    spec = X.get_spec(sym, mt5)
    trades = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = asset
        tc["_risk_pct"] = risk
        tc["_spread"] = stats["spread"]
        tc["_max_concurrent"] = C.max_concurrent(asset)
        tc["_spec"] = spec
        trades.append(tc)
    trades.sort(key=lambda x: (x["time"], x.get("exit_time")))
    wf_before = len(trades)
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
        trades, _ = WA.apply_walkforward_filter(trades, mt5=mt5, assets=(asset,))
    trades = PB.prepare_portfolio_trades(trades, {asset: rmap}, oos_mg)
    return trades, wf_before, stats


def _rule_bucket(t):
    setup = str(t.get("setup", "?"))
    return setup.split("-")[0].upper()


def _analyze(trades, live):
    sl = sum(1 for t in trades if t["R"] <= -0.95)
    wins = sum(1 for t in trades if t["R"] > 0.05)
    tot_r = sum(t["R"] for t in trades)
    by_rule = defaultdict(lambda: {"n": 0, "sl": 0, "R": 0.0})
    for t in trades:
        rk = _rule_bucket(t)
        by_rule[rk]["n"] += 1
        by_rule[rk]["R"] += t["R"]
        if t["R"] <= -0.95:
            by_rule[rk]["sl"] += 1
    fills = defaultdict(int)
    for t in trades:
        fills[str(t.get("fill_kind", "?")).lower()] += 1
    return dict(
        raw=len(trades),
        sl=sl,
        wins=wins,
        be=len(trades) - sl - wins,
        tot_r=tot_r,
        avg_r=tot_r / len(trades) if trades else 0.0,
        sl_pct=sl / len(trades) * 100 if trades else 0,
        by_rule=dict(by_rule),
        fills=dict(fills),
        n=live["n"],
        wr=live["wr"],
        pf=live["pf"],
        ret=live["ret_pct"],
        profit=live["profit"],
        dd=live["max_dd"],
    )


def _run_mode(mt5, mode, oos_mg):
    trades, wf_before, _ = _load_trades(mt5, mode, oos_mg)
    live = PB.simulate_portfolio(
        trades, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK)
    r = _analyze(trades, live)
    r["mode"] = mode
    r["wf_raw"] = wf_before
    return r


def _print_rule_table(rows, baseline):
    """Per-rule delta vs none mode."""
    base_rules = baseline.get("by_rule", {})
    print(f"\n  PER-RULE vs none (raw signals after filters)")
    print(f"  {'rule':<8} {'none':>5} {'noneSL':>7}  ", end="")
    for mode in MODES[1:]:
        print(f"{mode:>12}", end="")
    print()
    print("  " + THIN)
    all_rules = set(base_rules)
    for r in rows:
        all_rules |= set(r.get("by_rule", {}))
    for rk in sorted(all_rules, key=lambda k: -base_rules.get(k, {}).get("n", 0)):
        bn = base_rules.get(rk, {}).get("n", 0)
        bsl = base_rules.get(rk, {}).get("sl", 0)
        line = f"  {rk:<8} {bn:>5} {bsl:>6}  "
        for r in rows[1:]:
            rn = r["by_rule"].get(rk, {}).get("n", 0)
            line += f"{rn:>12}"
        print(line)


def main():
    oos_mg = _oos_gate()
    mt5 = M.connect()
    try:
        print(SEP)
        print(f"  ENTRY CONFIRM SWEEP  |  XAUUSD  |  {DAYS}d  |  balance=${BAL:,.0f}")
        print(f"  Pipeline: walk-forward + CH-REV macro filter + portfolio sim")
        print(SEP)

        rows = []
        for mode in MODES:
            print(f"  running {mode}...", flush=True)
            rows.append(_run_mode(mt5, mode, oos_mg))

        baseline = rows[0]
        print(f"\n{SEP}")
        print(f"  {'mode':<14} {'raw':>5} {'port':>5} {'WR%':>6} {'PF':>5} "
              f"{'avgR':>6} {'SL%':>6} {'return':>9} {'profit$':>11} {'DD%':>6}")
        print("  " + THIN)
        for r in rows:
            print(f"  {r['mode']:<14} {r['raw']:>5} {r['n']:>5} {r['wr']:>5.1f}% "
                  f"{r['pf']:>5.2f} {r['avg_r']:>+6.2f} {r['sl_pct']:>5.1f}% "
                  f"{r['ret']:>+8.1f}% {r['profit']:>+11.0f} {r['dd']:>5.1f}%")

        print(f"\n  FILL KIND COUNTS (raw signals)")
        print(f"  {'mode':<14}", end="")
        kinds = ("exact", "tol", "confirm", "reject")
        for k in kinds:
            print(f" {k:>8}", end="")
        print()
        for r in rows:
            print(f"  {r['mode']:<14}", end="")
            for k in kinds:
                print(f" {r['fills'].get(k, 0):>8}", end="")
            print()

        _print_rule_table(rows, baseline)

        print(f"\n  DELTA vs none (portfolio)")
        base = rows[0]
        for r in rows[1:]:
            print(f"  {r['mode']:<14} trades {r['n']-base['n']:>+4}  "
                  f"WR {r['wr']-base['wr']:>+5.1f}pp  PF {r['pf']-base['pf']:>+5.2f}  "
                  f"profit ${r['profit']-base['profit']:>+,.0f}  "
                  f"SL {r['sl']-base['sl']:>+3}")

        by_profit = max(rows, key=lambda x: x["profit"])
        by_pf = max(rows, key=lambda x: x["pf"])
        by_wr = max(rows, key=lambda x: x["wr"])
        print(f"\n  BEST BY METRIC")
        print(f"    profit : {by_profit['mode']} (${by_profit['profit']:+,.0f})")
        print(f"    PF     : {by_pf['mode']} ({by_pf['pf']:.2f})")
        print(f"    WR     : {by_wr['mode']} ({by_wr['wr']:.1f}%)")

        print(SEP)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
