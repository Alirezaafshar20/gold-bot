"""Sweep: relax DENSE filters one-by-one and in combos. Find more trades, keep WR high."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

DAYS = 180
SPREAD = 0.28
BAL = 1000.0
RISK = 0.01


def run_backtest(d, m1, B, ctx, htf, cutoff, opt):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus", "dense", "medium")
    k = {x: opt[x] for x in opt if x not in skip}
    k["htf_context"] = htf
    if opt.get("htf_trend_tfs") is not None:
        k["htf_trend_tfs"] = opt["htf_trend_tfs"]
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k) if t["time"] >= cutoff]
    if not tr:
        return dict(n=0, wr=0, ret=0, pf=0, dd=0)
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    return dict(n=r["n"], wr=r["wr"], ret=r["ret_pct"], pf=r["pf"], dd=r["max_dd"])


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    base = S.dense_settings()
    s0 = run_backtest(d, m1, B, ctx, htf, cutoff, base)

    print("=" * 78)
    print("  DENSE RELAX SWEEP — XAUUSD M15 180d")
    print(f"  BASELINE DENSE: n={s0['n']} WR={s0['wr']:.1f}% ret={s0['ret']:+.1f}% PF={s0['pf']:.2f} DD={s0['dd']:.1f}%")
    print("=" * 78)
    print(f"  {'label':<38} {'n':>4} {'WR':>6} {'ret':>7} {'PF':>6} {'DD':>5}")
    print("  " + "-" * 68)

    rows = [("DENSE baseline", s0)]

    def add(label, **kw):
        o = dict(base)
        o.update(kw)
        s = run_backtest(d, m1, B, ctx, htf, cutoff, o)
        rows.append((label, s))
        flag = " ***" if s["n"] > s0["n"] and s["wr"] >= 75 else ""
        print(f"  {label:<38} {s['n']:>4} {s['wr']:>5.1f}% {s['ret']:>+6.1f}% "
              f"{s['pf']:>6.2f} {s['dd']:>4.1f}%{flag}")

    add("DENSE baseline")

    # Session
    add("session 9-21", session_start=9, session_end=21)
    add("session 8-22 (=MEDIUM)", session_start=8, session_end=22)
    add("session 7-22", session_start=7, session_end=22)

    # ATR regime
    add("ATR ratio 1.6", atr_max_ratio=1.6)
    add("ATR ratio 1.7", atr_max_ratio=1.7)
    add("ATR ratio 1.8", atr_max_ratio=1.8)
    add("no ATR regime", require_atr_regime=False)

    # HTF trend
    add("HTF trend H1 only", htf_trend_tfs=("H1",))
    add("no HTF trend", htf_trend=False)

    # VP filter
    add("no VP filter", vp_mode=None)

    # Fib
    add("no fib", require_fib=False)

    # min SL
    add("minSL $2.5", min_risk_usd=2.5)
    add("minSL $2.0", min_risk_usd=2.0)

    # concurrent
    add("max_concurrent=3", max_concurrent=3)

    # NDS ratio wider
    add("nds ratio 0.65", nds_max_ratio=0.65)
    add("nds ratio 0.70", nds_max_ratio=0.70)

    # Extra rule
    add("+ NDS_FVG", enabled=["OB", "NDS", "NDS_FVG", "DEMAND"])

    # Best combos from prior research
    add("sess8-22 + H1 trend", session_start=8, session_end=22, htf_trend_tfs=("H1",))
    add("sess8-22 + atr1.7", session_start=8, session_end=22, atr_max_ratio=1.7)
    add("sess8-22 + no VP", session_start=8, session_end=22, vp_mode=None)
    add("H1 trend + atr1.7", htf_trend_tfs=("H1",), atr_max_ratio=1.7)
    add("sess8-22 + H1 + atr1.7", session_start=8, session_end=22,
        htf_trend_tfs=("H1",), atr_max_ratio=1.7)
    add("sess8-22 + nds0.65", session_start=8, session_end=22, nds_max_ratio=0.65)
    add("sess8-22 + minSL2.5", session_start=8, session_end=22, min_risk_usd=2.5)

    print("\n  --- TOP PICKS (more trades than baseline, WR >= 75%) ---")
    good = [(l, s) for l, s in rows if s["n"] > s0["n"] and s["wr"] >= 75]
    good.sort(key=lambda x: (-x[1]["n"], -x[1]["wr"], -x[1]["ret"]))
    if not good:
        print("  None at WR>=75%. Showing WR>=70%:")
        good = [(l, s) for l, s in rows if s["n"] > s0["n"] and s["wr"] >= 70]
        good.sort(key=lambda x: (-x[1]["n"], -x[1]["wr"], -x[1]["ret"]))
    for label, s in good[:8]:
        print(f"    {label}: n={s['n']} WR={s['wr']:.1f}% ret={s['ret']:+.1f}% PF={s['pf']:.2f}")

    print("=" * 78)


if __name__ == "__main__":
    main()
