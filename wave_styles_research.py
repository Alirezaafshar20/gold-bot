"""
Backtest wave styles: Elliott, NeoWave, Wolfe, Harmonic, Wyckoff.
Each rule alone with STABLE + DENSE filters. Run: python wave_styles_research.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import wyckoff as WY
import wave_styles as WS

DAYS = 180
SPREAD = 0.28
BAL = 1000.0
RISK = 0.01
MIN_N = 3


def kw_from(opt, htf):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
            "dense", "medium", "dense_plus", "dense_wide")
    k = {x: opt[x] for x in opt if x not in skip}
    k["htf_context"] = htf
    k["enabled"] = None
    return k


def test(name, d, m1, B, ctx, htf, cutoff, opt):
    k = kw_from(opt, htf)
    k["enabled"] = [name]
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k) if t["time"] >= cutoff]
    if not tr:
        return None
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    Rs = np.array([t["R"] - SPREAD / t["risk"] for t in tr])
    w, l = Rs[Rs > 0], Rs[Rs <= 0]
    pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
    return dict(n=len(tr), wr=r["wr"], ret=r["ret_pct"], pf=pf, dd=r["max_dd"], tot_r=float(Rs.sum()))


def raw(name, B):
    fn = WS.STYLE_DETECTORS.get(name) or WY.WYCK_DETECTORS.get(name)
    if not fn:
        return 0
    p = WS.STYLE_DEFAULT if name in WS.STYLE_DETECTORS else WY.WYCK_DEFAULT
    return sum(len(fn(B, i, p)) for i in range(80, B.n - 1))


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


def print_block(title, rows, filter_name):
    print(f"\n  === {title} ({filter_name}) ===", flush=True)
    print(f"  {'RULE':<16} {'raw':>4} {'n':>4} {'WR%':>6} {'PF':>6} {'ret%':>7}  tier", flush=True)
    print("  " + "-" * 58, flush=True)
    for name, fam, row, rc in rows:
        if row:
            print(f"  {name:<16} {rc:>4} {row['n']:>4} {row['wr']:>5.1f}% "
                  f"{row['pf']:>6.2f} {row['ret']:>+6.1f}%  {tier(row)}", flush=True)
        else:
            print(f"  {name:<16} {rc:>4}    0      —      —       —   —", flush=True)


def main():
    print("=" * 72)
    print("  WAVE STYLES RESEARCH — Elliott / NeoWave / Wolfe / Harmonic / Wyckoff")
    print(f"  XAUUSD M15 | {DAYS}d | filters: STABLE and DENSE")
    print("=" * 72)

    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    all_rules = []
    for fam_key, fam in WS.STYLE_FAMILIES.items():
        for rk in fam["rules"]:
            all_rules.append((rk, fam["label"]))
    for fam_key, fam in WY.WYCK_CATEGORIES.items():
        for rk in fam["rules"]:
            if rk not in [x[0] for x in all_rules]:
                all_rules.append((rk, fam["label"]))

    # key Wyckoff only (not all 13 to save time) — best + spring + lps
    wyck_pick = ["WYCK_SPRING", "WYCK_TEST", "WYCK_LPS", "WYCK_SOS", "WYCK_SOW", "WYCK"]
    all_rules = [(r, WS.STYLE_RULE_META.get(r, WY.WYCK_RULE_META.get(r, {})).get("label", "?"))
                 for r in WS.STYLE_ALL_RULES]
    all_rules += [(r, "Wyckoff") for r in wyck_pick]

    for filter_name, preset in [("STABLE", S.stable_settings()), ("DENSE", S.dense_settings())]:
        rows = []
        for name, fam in all_rules:
            print(f"  testing {name} @ {filter_name}...", end="\r", flush=True)
            rc = raw(name, B)
            row = test(name, d, m1, B, ctx, htf, cutoff, preset)
            rows.append((name, fam, row, rc))
        rows.sort(key=lambda x: (-(x[2]["wr"] if x[2] else 0), -(x[2]["n"] if x[2] else 0)))

        for fam_key, fam in WS.STYLE_FAMILIES.items():
            print_block(fam["label"], [(n, f, r, rc) for n, f, r, rc in rows if f == fam["label"]], filter_name)
        print_block("Wyckoff (selected rules)", [(n, "Wyckoff", r, rc) for n, f, r, rc in rows if f == "Wyckoff"], filter_name)

    # Reference: core SMC on DENSE
    ref = test("OB", d, m1, B, ctx, htf, cutoff, S.dense_settings())
    ref2 = test("DEMAND", d, m1, B, ctx, htf, cutoff, S.dense_settings())
    print("\n  === REFERENCE (DENSE filter) ===")
    for label, name in [("OB", "OB"), ("DEMAND", "DEMAND")]:
        r = test(name, d, m1, B, ctx, htf, cutoff, S.dense_settings())
        if r:
            print(f"    {label}: n={r['n']} WR={r['wr']:.1f}% ret={r['ret']:+.1f}%")

    print("\n  Note: full Elliott/NeoWave needs discretionary wave count; OHLC rules are approximations.")
    print("=" * 72)


if __name__ == "__main__":
    main()
