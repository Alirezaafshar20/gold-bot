"""
Backtest every Al Brooks + existing SMC rule individually with STABLE filters.
Ranks by win-rate, profit factor, and total R. Run: python al_brooks_research.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import al_brooks as AB

DAYS = 180
SPREAD = S.SPREAD_USD
BAL = 1000.0
RISK = 0.01
MIN_TRADES_TIER = 3


def stable_kw(htf):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus", "dense")
    o = S.stable_settings()
    k = {x: o[x] for x in o if x not in skip}
    k["htf_context"] = htf
    k["enabled"] = None  # set per rule
    return k


def test_rule(name, d, m1, B, ctx, htf, cutoff, kw_base):
    kw = dict(kw_base)
    kw["enabled"] = [name]
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw)
          if t["time"] >= cutoff]
    if not tr:
        return None
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    Rs = np.array([t["R"] - SPREAD / t["risk"] for t in tr])
    w = Rs[Rs > 0]
    l = Rs[Rs <= 0]
    pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
    return {
        "rule": name,
        "n": len(tr),
        "wr": r["wr"],
        "ret": r["ret_pct"],
        "pf": pf,
        "dd": r["max_dd"],
        "tot_r": float(Rs.sum()),
        "avg_r": float(Rs.mean()),
    }


def tier(row):
    if row["n"] < MIN_TRADES_TIER:
        return "D — too few trades"
    if row["wr"] >= 70 and row["pf"] >= 2.0:
        return "A — strong"
    if row["wr"] >= 55 and row["pf"] >= 1.3:
        return "B — acceptable"
    if row["wr"] >= 45:
        return "C — weak"
    return "F — avoid"


def main():
    print("=" * 78)
    print("  AL BROOKS + SMC RULE RANKING  |  STABLE filters  |  XAUUSD M15  |  180d")
    print("=" * 78)

    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    kw = stable_kw(htf)

    all_rules = (
        [("SMC / ICT", k) for k in ("OB", "NDS", "DEMAND", "FVG", "BOS", "WYCK", "QM", "FL")]
        + [("Al Brooks", k) for k in AB.AB_ALL_RULES]
    )

    rows = []
    print("\n  Testing each rule in isolation (STABLE filters: fib, ATR, HTF trend, session)...")
    for family, name in all_rules:
        try:
            row = test_rule(name, d, m1, B, ctx, htf, cutoff, kw)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        if row:
            row["family"] = family
            meta = AB.AB_RULE_META.get(name, {})
            row["category"] = meta.get("category", "smc")
            row["theory"] = AB.AB_BROOKS_THEORY_TIER.get(name, "—")
            row["tier"] = tier(row)
            rows.append(row)

    rows.sort(key=lambda x: (-x["wr"] if x["n"] >= MIN_TRADES_TIER else -1,
                             -x["pf"], -x["n"]))

    print(f"\n  {'RULE':<18} {'fam':<10} {'n':>4} {'WR%':>6} {'PF':>6} {'totR':>7} {'ret%':>7}  TIER")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['rule']:<18} {r['family'][:8]:<10} {r['n']:>4} {r['wr']:>5.1f}% "
              f"{r['pf']:>6.2f} {r['tot_r']:>+7.1f} {r['ret']:>+6.1f}%  {r['tier']}")

    # By Brooks category
    print("\n  --- BY AL BROOKS CATEGORY (AB rules only, n>=%d) ---" % MIN_TRADES_TIER)
    cats = {}
    for r in rows:
        if r["family"] != "Al Brooks" or r["n"] < MIN_TRADES_TIER:
            continue
        c = r["category"]
        cats.setdefault(c, []).append(r)
    for cat_key in ("trend_continuation", "breakout", "reversal", "trading_range"):
        items = cats.get(cat_key, [])
        if not items:
            continue
        label = AB.AB_CATEGORIES.get(cat_key, {}).get("label", cat_key)
        avg_wr = np.mean([x["wr"] for x in items])
        print(f"\n  [{label}]  avg WR={avg_wr:.1f}%  ({len(items)} rules)")
        for r in sorted(items, key=lambda x: -x["wr"]):
            print(f"    {r['rule']:<16} n={r['n']:>3} WR={r['wr']:>5.1f}% PF={r['pf']:>5.2f}  "
                  f"{r['theory']}")

    a_rules = [r for r in rows if r["tier"].startswith("A")]
    print("\n  --- TIER A (WR>=70%%, PF>=2, n>=3) — add to live consideration ---")
    if a_rules:
        for r in a_rules:
            print(f"    {r['rule']:<16} n={r['n']} WR={r['wr']:.1f}% ret={r['ret']:+.1f}%")
    else:
        print("    (none met all criteria on XAUUSD 180d with STABLE filters)")

    # Combo test: STABLE + all tier A AB rules
    ab_a = [r["rule"] for r in a_rules if r["family"] == "Al Brooks"]
    if ab_a:
        combo = list(S.OPT_STABLE_RULES) + ab_a
        kw2 = dict(kw)
        kw2["enabled"] = combo
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw2)
              if t["time"] >= cutoff]
        rc = S.simulate_account(tr, BAL, RISK, SPREAD)
        print(f"\n  --- COMBO: STABLE + Tier-A AB ({', '.join(ab_a)}) ---")
        print(f"    n={rc['n']} WR={rc['wr']:.1f}% ret={rc['ret_pct']:+.1f}% DD={rc['max_dd']:.1f}%")

    print("\n  --- THEORY vs REALITY (Brooks teaching vs this backtest) ---")
    print("  Brooks ranks H2/L2 and BO-PB as highest probability with-trend.")
    print("  Encyclopedia has 600+ sections — only codifiable OHLC subset tested here.")
    print("=" * 78)


if __name__ == "__main__":
    main()
