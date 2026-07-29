"""
BTC — session (9–21) vs 24/7 (no time filter).

  python btc_session_compare.py
  python btc_session_compare.py --days 30
"""
from __future__ import annotations

import argparse
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

ASSET = "BTCUSD"
BAL = 1000.0
_GOLD = {
    "vp_mode": None,
    "fib_ob_only": True,
    "fib_demand_exempt": True,
    "fib_nds_exempt": True,
    **FC.floating_opt_overrides(),
}


def run_variant(label, mt5, oos_mg, days, session):
    risk = RISK_MAP[ASSET]
    _, prof = P.get_profile(ASSET)
    opt_extra = dict(_GOLD)
    if session is None:
        opt_extra["session_start"] = None
        opt_extra["session_end"] = None
    else:
        opt_extra["session_start"], opt_extra["session_end"] = session

    sym = P.resolve_symbol_for_profile(ASSET, mt5)
    stats = _run(sym, prof, mt5, days=days, asset_key=ASSET, risk_pct=risk,
                 max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                 meta_gate_override=oos_mg,
                 opt_extra=opt_extra, tol_override=dict(FC.TOLERANCE))
    raw = [{**t, "_asset": ASSET, "_risk_pct": risk} for t in stats.get("trades") or []]
    raw.sort(key=lambda t: (t["time"], t.get("exit_time")))
    filtered, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(ASSET,))
    fix = simulate_portfolio(filtered, BAL, size_compound=False)
    comp = simulate_portfolio(filtered, BAL, size_compound=True)
    return {
        "label": label,
        "session": session,
        "raw_n": len(raw),
        "final_n": fix["n"],
        "wf_removed": wf["before"] - wf["after"],
        "fix": fix,
        "comp": comp,
    }


def hour_breakdown(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["hour"] = pd.to_datetime(df["entry_time"]).dt.hour
    out = df.groupby("hour").agg(
        n=("R", "count"),
        wr=("R", lambda x: (x > 0).mean() * 100),
        tot_r=("R", "sum"),
        pnl=("net", "sum"),
    )
    return out.sort_values("pnl", ascending=False)


def print_result(r):
    fix, comp = r["fix"], r["comp"]
    sess = r["session"]
    sess_s = f"{sess[0]}–{sess[1]}" if sess else "24/7 (بدون محدودیت)"
    print(f"\n  --- {r['label']} | session {sess_s} ---")
    print(f"  Raw → final: {r['raw_n']} → {r['final_n']}  (walk-forward -{r['wf_removed']})")
    print(f"  Fixed-risk : {fix['ret_pct']:+.1f}%  WR {fix['wr']:.1f}%  PF {fix['pf']:.2f}  "
          f"DD {fix['max_dd']:.1f}%  (${fix['profit']:+,.0f})")
    print(f"  Compound   : {comp['ret_pct']:+.1f}%  WR {comp['wr']:.1f}%  PF {comp['pf']:.2f}  "
          f"DD {comp['max_dd']:.1f}%  (${comp['profit']:+,.0f})")
    if not fix.get("ledger"):
        return
    df = pd.DataFrame(fix["ledger"])
    g = df.groupby("rule").agg(
        n=("R", "count"), wr=("R", lambda x: (x > 0).mean() * 100),
        tot_r=("R", "sum"), pnl=("net", "sum"),
    ).sort_values("pnl", ascending=False)
    print("\n  Per-rule ($ fixed-risk):")
    print(f"  {'rule':<10} {'n':>4} {'WR%':>6} {'totR':>7} {'pnl$':>9}")
    for rule, row in g.iterrows():
        print(f"  {rule:<10} {int(row['n']):>4} {row['wr']:>5.0f}% {row['tot_r']:>+7.1f} {row['pnl']:>+9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()

    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=[ASSET],
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )

    mt5 = M.connect()
    results = {}
    try:
        results["current"] = run_variant("فعلی", mt5, oos_mg, args.days, (9, 21))
        results["247"] = run_variant("بدون محدودیت", mt5, oos_mg, args.days, None)
    finally:
        M.shutdown(mt5)

    print("=" * 88)
    print(f"  BTCUSD — session compare | {args.days}d | risk {RISK_MAP[ASSET]*100:.1f}%")
    print("=" * 88)
    for key in ("current", "247"):
        print_result(results[key])

    cur, full = results["current"]["fix"], results["247"]["fix"]
    extra = results["247"]["final_n"] - results["current"]["final_n"]
    print("\n" + "=" * 88)
    print("  خلاصه")
    print(f"  معاملات: {results['current']['final_n']} → {results['247']['final_n']}  (+{extra})")
    print(f"  Fixed ret: {cur['ret_pct']:+.1f}% → {full['ret_pct']:+.1f}%  "
          f"({full['ret_pct']-cur['ret_pct']:+.1f}pp)")
    print(f"  Fixed DD : {cur['max_dd']:.1f}% → {full['max_dd']:.1f}%")
    print(f"  Fixed PF : {cur['pf']:.2f} → {full['pf']:.2f}")

    # Extra trades outside 9–21
    ledger = results["247"]["fix"].get("ledger") or []
    if ledger:
        df = pd.DataFrame(ledger)
        df["hour"] = pd.to_datetime(df["entry_time"]).dt.hour
        outside = df[(df["hour"] < 9) | (df["hour"] >= 21)]
        inside = df[(df["hour"] >= 9) & (df["hour"] < 21)]
        print(f"\n  معاملات خارج سشن 9–21: {len(outside)} از {len(df)}")
        if len(outside):
            print(f"    WR {((outside['R']>0).mean()*100):.0f}%  totR {outside['R'].sum():+.1f}  "
                  f"PnL ${outside['net'].sum():+.0f}")
        print(f"  معاملات داخل سشن 9–21: {len(inside)}")
        if len(inside):
            print(f"    WR {((inside['R']>0).mean()*100):.0f}%  totR {inside['R'].sum():+.1f}  "
                  f"PnL ${inside['net'].sum():+.0f}")

        print("\n  ساعات پرسود (24/7 — fixed $):")
        hb = hour_breakdown(ledger)
        for h, row in hb.head(6).iterrows():
            flag = " *" if h < 9 or h >= 21 else ""
            print(f"    {h:02d}:00  n={int(row.n):>3}  WR={row.wr:>4.0f}%  ${row.pnl:>+7.0f}{flag}")
        print("\n  ساعات ضررده:")
        for h, row in hb.tail(4).iterrows():
            flag = " *" if h < 9 or h >= 21 else ""
            print(f"    {h:02d}:00  n={int(row.n):>3}  WR={row.wr:>4.0f}%  ${row.pnl:>+7.0f}{flag}")
        print("  (* = خارج سشن فعلی 9–21)")
    print("=" * 88)


if __name__ == "__main__":
    main()
