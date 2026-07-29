"""
Backtest every Wyckoff event individually with STABLE filters + loose filters.
Run: python wyckoff_research.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import wyckoff as WY

DAYS = 180
SPREAD = S.SPREAD_USD
BAL = 1000.0
RISK = 0.01


def stable_kw(htf):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus", "dense")
    o = S.stable_settings()
    k = {x: o[x] for x in o if x not in skip}
    k["htf_context"] = htf
    return k


def loose_kw():
    o = S.stable_settings()
    return dict(
        exit_mode=o["exit_mode"], fixed_tp_r=o["fixed_tp_r"], be_trigger=o["be_trigger"],
        tp_mode=o["tp_mode"], min_tp_r=o["min_tp_r"], max_tp_r=o["max_tp_r"],
        htf_lookback=o["htf_lookback"], htf_tfs=tuple(o["htf_tfs"]),
        min_risk_usd=o["min_risk_usd"], session_start=o["session_start"],
        session_end=o["session_end"], max_concurrent=1,
        require_fib=False, require_atr_regime=False, htf_trend=False,
        vp_mode=S.VP_MODE,
    )


def run_one(name, d, m1, B, ctx, htf, cutoff, kw):
    kw = dict(kw)
    kw["enabled"] = [name]
    kw["htf_context"] = htf if kw.get("htf_trend") else kw.get("htf_context")
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw)
          if t["time"] >= cutoff]
    if not tr:
        return None
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    Rs = np.array([t["R"] - SPREAD / t["risk"] for t in tr])
    w, l = Rs[Rs > 0], Rs[Rs <= 0]
    pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
    return dict(n=len(tr), wr=r["wr"], ret=r["ret_pct"], pf=pf, dd=r["max_dd"],
                tot_r=float(Rs.sum()), avg_r=float(Rs.mean()))


def raw_count(name, B):
    fn = WY.WYCK_DETECTORS[name]
    return sum(len(fn(B, i)) for i in range(3, B.n - 1))


def tier(row, min_n=3):
    if row["n"] < min_n:
        return "D — too few trades"
    if row["wr"] >= 70 and row["pf"] >= 2.0:
        return "A — strong"
    if row["wr"] >= 55 and row["pf"] >= 1.3:
        return "B — acceptable"
    if row["wr"] >= 45:
        return "C — weak"
    return "F — avoid"


def main():
    print("=" * 80)
    print("  WYCKOFF METHOD — full event ranking  |  XAUUSD M15  |  180 days")
    print("=" * 80)

    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    kw_s = stable_kw(htf)
    kw_l = loose_kw()

    print("\n  --- RAW SETUP COUNTS (before any filter) ---")
    for name in WY.WYCK_ALL_RULES:
        print(f"    {name:<14} raw={raw_count(name, B):>4}  |  {WY.WYCK_THEORY_TIER.get(name, '')}")

    rows_s, rows_l = [], []
    for name in WY.WYCK_ALL_RULES:
        rs = run_one(name, d, m1, B, ctx, htf, cutoff, kw_s)
        rl = run_one(name, d, m1, B, ctx, htf, cutoff, kw_l)
        if rs:
            rs["rule"] = name
            rs["filter"] = "STABLE"
            rs["tier"] = tier(rs)
            rs["cat"] = WY.WYCK_RULE_META[name]["category"]
            rows_s.append(rs)
        if rl:
            rl["rule"] = name
            rl["filter"] = "LOOSE"
            rl["tier"] = tier(rl)
            rows_l.append(rl)

    print("\n  --- STABLE FILTERS (fib + ATR + HTF trend + session + minSL $3) ---")
    print(f"  {'RULE':<14} {'raw':>4} {'n':>4} {'WR%':>6} {'PF':>6} {'totR':>7} {'ret%':>7}  TIER")
    print("  " + "-" * 76)
    for name in WY.WYCK_ALL_RULES:
        raw = raw_count(name, B)
        rs = next((x for x in rows_s if x["rule"] == name), None)
        if rs:
            print(f"  {name:<14} {raw:>4} {rs['n']:>4} {rs['wr']:>5.1f}% {rs['pf']:>6.2f} "
                  f"{rs['tot_r']:>+7.1f} {rs['ret']:>+6.1f}%  {rs['tier']}")
        else:
            print(f"  {name:<14} {raw:>4}    0      —      —       —      —   (no trades)")

    print("\n  --- LOOSE FILTERS (session + minSL only) ---")
    print(f"  {'RULE':<14} {'n':>4} {'WR%':>6} {'PF':>6} {'ret%':>7}  TIER")
    print("  " + "-" * 50)
    rows_l.sort(key=lambda x: -x["wr"])
    for r in rows_l:
        print(f"  {r['rule']:<14} {r['n']:>4} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{r['ret']:>+6.1f}%  {r['tier']}")

    print("\n  --- BY WYCKOFF PHASE (STABLE, n>=3) ---")
    for cat in ("phase_c", "phase_d", "phase_a", "legacy"):
        items = [r for r in rows_s if WY.WYCK_RULE_META[r["rule"]]["category"] == cat and r["n"] >= 3]
        if not items:
            continue
        label = WY.WYCK_CATEGORIES[cat]["label"]
        avg = np.mean([x["wr"] for x in items])
        print(f"\n  [{label}]  avg WR={avg:.1f}%")
        for r in sorted(items, key=lambda x: -x["wr"]):
            print(f"    {r['rule']:<14} n={r['n']:>3} WR={r['wr']:>5.1f}% PF={r['pf']:>5.2f} "
                  f"ret={r['ret']:+.1f}%")

    # NDS-WYCK nested
    print("\n  --- NDS-WYCK (nested spring inside HTF zone) ---")
    kw_n = dict(kw_s)
    kw_n["enabled"] = ["NDS_WYCK"]
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw_n) if t["time"] >= cutoff]
    if tr:
        r = S.simulate_account(tr, BAL, RISK, SPREAD)
        print(f"    n={r['n']} WR={r['wr']:.1f}% ret={r['ret_pct']:+.1f}%")
    else:
        print("    no trades")

    # Best combo: STABLE + top wyck rules
    good = [r["rule"] for r in rows_s if r["tier"].startswith("A") or r["tier"].startswith("B")]
    if good:
        combo = list(S.OPT_STABLE_RULES) + good[:3]
        kw_c = dict(kw_s)
        kw_c["enabled"] = list(dict.fromkeys(combo))
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw_c) if t["time"] >= cutoff]
        r = S.simulate_account(tr, BAL, RISK, SPREAD)
        print(f"\n  --- COMBO STABLE + best Wyck ({', '.join(good[:3])}) ---")
        print(f"    n={r['n']} WR={r['wr']:.1f}% ret={r['ret_pct']:+.1f}% DD={r['max_dd']:.1f}%")

    print("\n  --- LEGACY vs NEW SPRING ---")
    leg = next((x for x in rows_s if x["rule"] == "WYCK"), None)
    spr = next((x for x in rows_s if x["rule"] == "WYCK_SPRING"), None)
    if leg:
        print(f"    WYCK (legacy, high vol):  n={leg['n']} WR={leg['wr']:.1f}%")
    if spr:
        print(f"    WYCK_SPRING (classic):    n={spr['n']} WR={spr['wr']:.1f}%")

    print("\n  Wyckoff theory: Spring/TEST/LPS = Tier A entries.")
    print("  See wyckoff.py for full trigger logic.")
    print("=" * 80)


if __name__ == "__main__":
    main()
