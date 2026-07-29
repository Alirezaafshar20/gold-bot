"""
Rank extended strategies (ICT, classic PA, systematic, CVD proxy).
Run: python extended_strategies_research.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import extended_strategies as EX

DAYS = 180
SPREAD = 0.28
BAL = 1000.0
RISK = 0.01
MIN_N = 3
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")


def kw_from(opt, htf):
    k = {x: opt[x] for x in opt if x not in META}
    k["htf_context"] = htf
    k["enabled"] = None
    return k


def test(name, d, m1, B, ctx, htf, cutoff, opt):
    k = kw_from(opt, htf)
    k["enabled"] = [name]
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
          if t["time"] >= cutoff]
    if not tr:
        return None
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    return dict(n=len(tr), wr=r["wr"], ret=r["ret_pct"], pf=r["pf"], dd=r["max_dd"])


def raw(name, B):
    fn = EX.EXT_DETECTORS.get(name)
    if not fn:
        return 0
    return sum(len(fn(B, i, EX.EXT_DEFAULT)) for i in range(80, B.n - 1))


def tier(row):
    if row["n"] < MIN_N:
        return "D"
    if row["wr"] >= 70 and row["pf"] >= 2:
        return "A"
    if row["wr"] >= 55 and row["pf"] >= 1.3:
        return "B"
    if row["wr"] >= 45:
        return "C"
    return "F"


def run_filter(filter_name, preset, d, m1, B, ctx, htf, cutoff):
    rows = []
    rules = EX.EXT_ALL_RULES
    n = len(rules)
    for idx, name in enumerate(rules):
        print(f"  [{filter_name}] {idx + 1}/{n} {name}...", end="\r", flush=True)
        row = test(name, d, m1, B, ctx, htf, cutoff, preset)
        rows.append((name, EX.EXT_RULE_META[name]["label"], row, raw(name, B)))
    print(" " * 60)
    return rows


def print_family(title, rows, filter_name):
    print(f"\n  === {title} ({filter_name}) ===")
    print(f"  {'RULE':<14} {'raw':>5} {'n':>4} {'WR%':>6} {'PF':>6} {'ret%':>7} tier")
    print("  " + "-" * 58)
    sub = [(n, fam, r, rc) for n, fam, r, rc in rows if fam == title]
    sub.sort(key=lambda x: (-(x[2]["wr"] if x[2] else 0), -(x[2]["n"] if x[2] else 0)))
    for name, fam, row, rc in sub:
        if row:
            print(f"  {name:<14} {rc:>5} {row['n']:>4} {row['wr']:>5.1f}% "
                  f"{row['pf']:>6.2f} {row['ret']:>+6.1f}%  {tier(row)}")
        else:
            print(f"  {name:<14} {rc:>5}    0      —      —       —   —")


def main():
    print("=" * 72)
    print("  EXTENDED STRATEGIES RESEARCH — ICT / Classic / Systematic / CVD")
    print(f"  XAUUSD M15 | ~70d effective (M1 cap) | STABLE + DENSE filters")
    print("=" * 72)

    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    all_rows = {}
    for filter_name, preset in [("STABLE", S.stable_settings()), ("DENSE", S.dense_settings())]:
        all_rows[filter_name] = run_filter(filter_name, preset, d, m1, B, ctx, htf, cutoff)
        for fam in EX.EXT_FAMILIES.values():
            print_family(fam["label"], all_rows[filter_name], filter_name)

    print("\n  === TOP PICKS (DENSE, tier A or B, n>=" + str(MIN_N) + ") ===")
    picks = []
    for name, fam, row, rc in all_rows["DENSE"]:
        if row and row["n"] >= MIN_N and tier(row) in ("A", "B"):
            picks.append((name, fam, row))
    picks.sort(key=lambda x: (-x[2]["wr"], -x[2]["ret"]))
    if picks:
        for name, fam, row in picks[:15]:
            print(f"    {name:<14} {row['n']:>3} trades  WR={row['wr']:.1f}%  ret={row['ret']:+.1f}%  ({fam})")
    else:
        print("    None reached tier A/B on DENSE — filters very strict for these rules.")

    ref = test("OB", d, m1, B, ctx, htf, cutoff, S.dense_settings())
    if ref:
        print(f"\n  REFERENCE DENSE OB: n={ref['n']} WR={ref['wr']:.1f}% ret={ref['ret']:+.1f}%")
    print("\n  Note: Footprint/Delta = CVD proxy via tick volume only (not true order flow).")
    print("  ICT killzones use broker server hours — adjust EXT_DEFAULT if needed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
