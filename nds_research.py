"""
NDS family research — test each nested trigger vs STABLE baseline (180d).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

SPREAD = 0.28
RISK = 0.01
DAYS = 180


def load():
    mt5 = M.connect()
    try:
        sym = M.resolve_symbol("XAUUSD@", mt5)
        _, d, m1, cutoff = M.fetch_pair(sym, "M15", days=DAYS, mt5=mt5)
        _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    finally:
        M.shutdown(mt5)
    return d, m1, cutoff, S.prepare_htf_context(htf_dfs)


def run(d, m1, htf, enabled, label=""):
    opt = S.stable_settings()
    skip = {"spread_usd", "nds_mode", "nds_extended"}
    kw = {k: v for k, v in {**opt, "htf_context": htf, "tf_min": 15,
                            "enabled": list(enabled)}.items() if k not in skip}
    tr = S.run_backtest(d, m1, **kw)
    return tr


def stats(tr, cutoff, spread=SPREAD):
    if cutoff is not None:
        tr = [t for t in tr if t["time"] >= cutoff]
    if not tr:
        return None
    acc = S.simulate_account(tr, risk_pct=RISK, spread_price=spread)
    pos = sum(1 for v in acc["monthly"].values() if v > 0)
    neg = sum(1 for v in acc["monthly"].values() if v < 0)
    avg_r = float(np.mean([t["R"] for t in tr]))
    return {"n": acc["n"], "wr": acc["wr"], "ret": acc["ret_pct"],
            "pf": acc["pf"], "dd": acc["max_dd"], "avg_r": avg_r,
            "+mo": pos, "-mo": neg, "trades": tr}


def count_raw_signals(d, htf):
    B = S.Bars(d)
    counts = {k: 0 for k in S.NDS_FAMILY}
    for i in range(3, B.n - 1):
        for name, fn in S.NDS_FAMILY.items():
            counts[name] += len(fn(B, i, htf_context=htf, signal_time=B.t[i]))
    return counts


def per_rule(trades):
    from collections import defaultdict
    by = defaultdict(list)
    for t in trades:
        by[t.get("setup", "?")].append(t)
    rows = []
    for tag, ts in sorted(by.items()):
        R = np.array([x["R"] for x in ts])
        w, l = R[R > 0], R[R <= 0]
        pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
        rows.append((tag, len(R), (R > 0).mean() * 100, pf, R.sum()))
    return rows


def main():
    d, m1, cutoff, htf = load()
    print("=" * 72)
    print("  NDS FAMILY RESEARCH  |  XAUUSD@ M15  |  180 days  |  STABLE filters")
    print("=" * 72)

    print("\n  Raw nested signals (before filters/fills):")
    raw = count_raw_signals(d, htf)
    for k, v in raw.items():
        print(f"    {k.replace('_', '-'):10s}  {v:4d} bars")

    cases = [
        ("STABLE (OB+NDS+DEM)", S.OPT_STABLE_RULES),
        ("NDS EXTENDED (all triggers)", S.OPT_NDS_EXTENDED),
        ("NDS family only", S.NDS_FAMILY_RULES),
    ]
    # Each NDS variant alone (with OB for context in live-style stack)
    for rule in S.NDS_FAMILY_RULES:
        cases.append((f"OB + {rule} only", ("OB", rule)))

    print("\n  Backtest results (with fib/ATR/session/HTF filters):")
    print(f"  {'Preset':<32s} {'n':>4s} {'WR%':>6s} {'ret%':>7s} {'PF':>6s} "
          f"{'DD%':>5s} {'avgR':>6s} {'+mo':>4s}")
    print("  " + "-" * 68)

    best = None
    baseline = None
    extended = None
    for label, rules in cases:
        tr = run(d, m1, htf, rules)
        s = stats(tr, cutoff)
        if s is None:
            print(f"  {label:<32s}    — no trades")
            continue
        print(f"  {label:<32s} {s['n']:4d} {s['wr']:5.1f}% {s['ret']:+6.1f}% "
              f"{s['pf']:6.2f} {s['dd']:5.1f}% {s['avg_r']:+5.2f} {s['+mo']:4d}")
        if label.startswith("STABLE"):
            baseline = s
        if "EXTENDED" in label:
            extended = s
        if best is None or s["ret"] > best[1]["ret"]:
            best = (label, s)

    if extended and baseline:
        print("\n  STABLE vs NDS EXTENDED:")
        print(f"    Trades:  {baseline['n']} -> {extended['n']}")
        print(f"    Return:  {baseline['ret']:+.1f}% -> {extended['ret']:+.1f}% "
              f"({extended['ret'] - baseline['ret']:+.1f}%)")
        print(f"    WR:      {baseline['wr']:.1f}% -> {extended['wr']:.1f}%")
        print(f"    Avg R:   {baseline['avg_r']:+.2f} -> {extended['avg_r']:+.2f}")

    if extended:
        print("\n  Per-rule breakdown (NDS EXTENDED run):")
        print(f"  {'RULE':<12s}{'trades':>7s}{'WR%':>8s}{'PF':>7s}{'totR':>9s}")
        for tag, n, wr, pf, tot in per_rule(extended["trades"]):
            print(f"  {tag:<12s}{n:7d}{wr:7.1f}%{pf:7.2f}{tot:+9.1f}")

    print("\n  Recommendation:")
    if extended and baseline and extended["ret"] > baseline["ret"] + 2:
        print(f"    -> Use --nds-extended (best combined: {best[0] if best else '?'})")
    elif extended and baseline:
        print("    -> Keep STABLE (OB+NDS+DEM); extended triggers did not beat baseline")
        print("       Best single add-on above — check OB+NDS_* rows")
    print("=" * 72)


if __name__ == "__main__":
    main()
