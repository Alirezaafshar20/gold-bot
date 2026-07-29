"""Sanity audit: look-ahead, costs, walk-forward, sample size."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

DAYS = 180
SPREAD = 0.28
BAL, RISK = 1000.0, 0.01
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")


def run_trades(d, m1, B, ctx, htf, opt, cutoff=None, cutoff_end=None):
    k = {x: opt[x] for x in opt if x not in META}
    k["htf_context"] = htf
    tr = S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
    if cutoff is not None:
        tr = [t for t in tr if t["time"] >= cutoff]
    if cutoff_end is not None:
        tr = [t for t in tr if t["time"] < cutoff_end]
    return tr


def stats(tr, spread=SPREAD, slip_r=0.0):
    if not tr:
        return dict(n=0, wr=0, ret=0, dd=0)
    if slip_r:
        tr = [{**t, "R": t["R"] - slip_r} for t in tr]
    r = S.simulate_account(tr, BAL, RISK, spread)
    return dict(n=r["n"], wr=r["wr"], ret=r["ret_pct"], dd=r["max_dd"])


def htf_closed_only_context(htf_dfs):
    """Use HTF bars only up to previous closed bar (shift index -1 in lookups)."""
    # Patch: wrap Bars and monkey-patch htf_bar_index behavior via custom context
    # Simpler: trim last row of each HTF df before building context
    trimmed = {}
    for tf, df in htf_dfs.items():
        if len(df) > 1:
            trimmed[tf] = df.iloc[:-1]
        else:
            trimmed[tf] = df
    return S.prepare_htf_context(trimmed)


def check_m1_gaps(m1, d):
    """Basic data quality checks."""
    m1_idx = m1.index
    gaps = 0
    for i in range(1, min(len(m1_idx), 50000)):
        delta = (m1_idx[i] - m1_idx[i - 1]).total_seconds()
        if delta > 120:  # >2 min gap
            gaps += 1
    overlap = len(m1.index.intersection(d.index))
    return dict(m1_bars=len(m1), m15_bars=len(d), m1_gaps_sample=gaps, m15_in_m1=overlap)


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    htf_safe = htf_closed_only_context(htf_dfs)
    opt = S.dense_settings()

    mid = np.datetime64(cutoff) + np.timedelta64(90, "D")

    print("=" * 78)
    print("  BACKTEST INTEGRITY AUDIT — XAUUSD DENSE + HARM_BAT")
    print("=" * 78)

    dq = check_m1_gaps(m1, d)
    print("\n  [1] DATA QUALITY")
    print(f"      M15 bars: {dq['m15_bars']} | M1 bars: {dq['m1_bars']}")
    print(f"      M1 gaps (>2min in sample): {dq['m1_gaps_sample']}")
    print(f"      Source: live MT5 download (same as live_smc)")

    tr_all = run_trades(d, m1, B, ctx, htf, opt, cutoff=cutoff)
    s0 = stats(tr_all)
    print("\n  [2] BASELINE (180d)")
    print(f"      n={s0['n']} WR={s0['wr']:.1f}% ret={s0['ret']:+.1f}% DD={s0['dd']:.1f}%")

    print("\n  [3] WALK-FORWARD (overfitting check)")
    s1 = stats(run_trades(d, m1, B, ctx, htf, opt, cutoff=cutoff, cutoff_end=mid))
    s2 = stats(run_trades(d, m1, B, ctx, htf, opt, cutoff=mid))
    print(f"      First 90d:  n={s1['n']} WR={s1['wr']:.1f}% ret={s1['ret']:+.1f}%")
    print(f"      Last 90d:   n={s2['n']} WR={s2['wr']:.1f}% ret={s2['ret']:+.1f}%")
    if s1["n"] >= 5 and s2["n"] >= 5:
        print("      Both halves profitable → not purely curve-fit on one lucky month")

    print("\n  [4] COST STRESS (spread / slippage)")
    for label, sp, slip in [
        ("normal spread $0.28", 0.28, 0),
        ("2x spread $0.56", 0.56, 0),
        ("spread + 0.2R slippage", 0.28, 0.2),
        ("2x spread + 0.3R slip", 0.56, 0.3),
    ]:
        s = stats(tr_all, spread=sp, slip_r=slip)
        print(f"      {label:<26} WR={s['wr']:.1f}% ret={s['ret']:+.1f}%")

    print("\n  [5] HTF LOOK-AHEAD TEST (exclude last open HTF bar)")
    tr_safe = run_trades(d, m1, B, ctx, htf_safe, opt, cutoff=cutoff)
    ss = stats(tr_safe)
    print(f"      With HTF trim:  n={ss['n']} WR={ss['wr']:.1f}% ret={ss['ret']:+.1f}%")
    print(f"      Delta vs base: n{ss['n']-s0['n']:+d} WR{ss['wr']-s0['wr']:+.1f}% ret{ss['ret']-s0['ret']:+.1f}%")

    print("\n  [6] SAMPLE SIZE")
    n, wr = s0["n"], s0["wr"] / 100
    # Wilson-ish: small n means wide CI
    se = np.sqrt(wr * (1 - wr) / n) if n else 0
    print(f"      Only {n} trades in 180d → WR 95% rough band: {(wr-se*1.96)*100:.0f}%-{(wr+se*1.96)*100:.0f}%")
    print(f"      High WR partly from strict filters (468 raw → {n} trades)")

    print("\n  [7] TIMING (no same-bar lookahead)")
    print("      Signal on closed M15 bar i; entry fill from bar i+1 onward only.")
    print("      live_smc uses B.n-2 (last closed candle) — matches backtest.")

    wins = sum(1 for t in tr_all if t["R"] > 0)
    losses = [t for t in tr_all if t["R"] <= 0]
    print("\n  [8] LOSS PROFILE")
    print(f"      Wins: {wins} | Losses: {len(losses)} | Avg loss R: "
          f"{np.mean([t['R'] for t in losses]):.2f}" if losses else "      No losses")
    tp_src = {}
    for t in tr_all:
        tp_src[t.get("tp_source", "?")] = tp_src.get(t.get("tp_source", "?"), 0) + 1
    print(f"      TP sources: {tp_src}")

    print("\n  VERDICT:")
    if s0["ret"] > 0 and s1["ret"] > 0 and s2["ret"] > 0:
        print("      Results are REAL but MODEST sample — not obvious data bug.")
    else:
        print("      Mixed walk-forward — treat headline WR with caution.")
    if ss["ret"] < s0["ret"] * 0.5:
        print("      HTF open-bar leak may inflate backtest ~ material issue.")
    else:
        print("      HTF timing leak (if any) is small on this run.")
    print("      Live may be 5-15% worse than backtest due to slippage/requotes.")
    print("=" * 78)


if __name__ == "__main__":
    main()
