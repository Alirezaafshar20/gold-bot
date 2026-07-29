"""Analyze model behavior around gold spike / high-volatility periods."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import strategy as S
import mt5_data as M

DAYS = 180
WINDOW_BARS = 32  # ~8h each side on M15


def spike_score(B, i, lb=100):
    cur = B.atr[i] if B.atr[i] > 0 else B.rng[i]
    med = float(np.median(B.atr[max(0, i - lb):i])) or 1e-6
    return cur / med, cur, B.rng[i]


def count_window(B, ctx, htf, opt, i0, i1, trades):
    """Setups and ATR skips in bar range [i0, i1]."""
    enabled = list(opt["enabled"])
    atr_skip = raw = vp = fib = htf_rej = session_rej = 0
    trade_in = 0
    t0, t1 = B.t[i0], B.t[i1]
    for tr in trades:
        ti = int(np.searchsorted(B.t, np.datetime64(tr["time"])))
        if i0 <= ti <= i1:
            trade_in += 1
    for i in range(max(3, i0), min(B.n - 1, i1 + 1)):
        if opt.get("require_atr_regime") and not S.atr_regime_pass(
                B, i, lookback=opt.get("atr_lookback", 100),
                max_ratio=opt.get("atr_max_ratio", 1.5)):
            atr_skip += 1
            continue
        for name in enabled:
            for direction, proximal, distal, tag in S.iter_rule_setups(
                    name, B, i, S.DEFAULT_PARAMS, htf_context=htf,
                    signal_time=B.t[i], htf_tfs=tuple(opt["htf_tfs"])):
                raw += 1
                if not S.vp_pass(B, i, direction, proximal, opt.get("vp_mode")):
                    vp += 1
                    continue
                _sl, _entry, risk = S.compute_entry_sl_risk(B, i, direction, proximal, distal)
                if risk <= 0:
                    continue
                nested = S.is_nds_setup(
                    B, i, direction, proximal, distal, htf, B.t[i],
                    tuple(opt["htf_tfs"]), opt.get("nds_max_ratio", 0.6))
                if opt.get("require_fib") and not (
                        opt.get("fib_nds_exempt") and nested):
                    if not S.fib_retrace_pass(B, i, direction, _entry):
                        fib += 1
                        continue
                if opt.get("session_start") is not None:
                    if not S.session_pass(B.t[i], opt["session_start"], opt["session_end"]):
                        session_rej += 1
                        continue
                if opt.get("htf_trend") and htf:
                    ok = all(S.htf_trend_pass(
                        htf, B.t[i], direction, tf=tf, ema_period=opt.get("htf_ema", 20))
                        for tf in (opt.get("htf_trend_tfs") or tuple(opt["htf_tfs"])))
                    if not ok:
                        htf_rej += 1
    return dict(
        t0=t0, t1=t1, atr_skip=atr_skip, raw=raw, vp=vp, fib=fib,
        htf=htf_rej, session=session_rej, trades=trade_in)


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    opt = S.dense_settings()
    meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
            "dense", "medium", "dense_plus", "dense_wide")
    k = {x: opt[x] for x in opt if x not in meta}
    k["htf_context"] = htf
    trades = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]

    lb = 100
    scores = []
    for i in range(lb + 5, B.n - 1):
        if B.t[i] < cutoff:
            continue
        ratio, atr, rng = spike_score(B, i, lb)
        scores.append((ratio, rng, i, B.t[i]))
    scores.sort(reverse=True)
    top = scores[:12]

    # Global stats in test window
    test_i0 = int(np.searchsorted(B.t, np.datetime64(cutoff)))
    total_bars = B.n - 1 - test_i0
    atr_bars = sum(
        1 for i in range(test_i0, B.n - 1)
        if opt.get("require_atr_regime") and not S.atr_regime_pass(B, i, 100, 1.5))
    high_vol = sum(1 for i in range(test_i0, B.n - 1) if spike_score(B, i)[0] > 1.5)

    print("=" * 78)
    print("  SPIKE / HIGH-VOL ANALYSIS — DENSE + HARM_BAT | XAUUSD M15 | 180d")
    print("=" * 78)
    print(f"  Test bars: {total_bars}  |  ATR-filter skipped bars: {atr_bars} "
          f"({100*atr_bars/total_bars:.1f}%)")
    print(f"  Bars with ATR > 1.5x median: {high_vol} ({100*high_vol/total_bars:.1f}%)")
    print(f"  Total trades in window: {len(trades)}")
    print()
    print("  Top volatility spikes (ATR vs 100-bar median):")
    print(f"  {'time':<20} {'ATR ratio':>10} {'M15 range$':>10}  window stats")
    print("  " + "-" * 72)

    for ratio, rng, i, t in top:
        i0 = max(test_i0, i - WINDOW_BARS)
        i1 = min(B.n - 2, i + WINDOW_BARS)
        w = count_window(B, ctx, htf, opt, i0, i1, trades)
        print(f"  {str(t)[:19]:<20} {ratio:>9.2f}x ${rng:>8.1f}  "
              f"ATRskip={w['atr_skip']:>2} raw={w['raw']:>3} fib={w['fib']:>2} "
              f"VP={w['vp']:>2} trades={w['trades']}")

    print()
    print("  --- WHY no trades around spikes ---")
    print("  1) ATR regime: when ATR > 1.5x median, ENTIRE bar is skipped (by design).")
    print("  2) Before spike: vol often rising → fib/VP/HTF reject; limit needs pullback.")
    print("  3) Model does NOT predict spikes — it waits for OB/DEM/HARM after structure.")
    print("  4) Trades appear AFTER spike when vol calms and pullback to zone fills.")
    print()

    # Example: Mar 30 2026 big drop area from backtest
    spike_dates = ["2026-03-30", "2026-04-06", "2026-05-12"]
    print("  Known big-move days vs nearby trades:")
    for ds in spike_dates:
        day = pd.Timestamp(ds)
        day_tr = [t for t in trades if pd.Timestamp(t["time"]).date() == day.date()]
        day_bars = sum(1 for i in range(B.n) if B.t[i].date() == day.date())
        day_atr_skip = sum(
            1 for i in range(B.n)
            if B.t[i].date() == day.date()
            and opt.get("require_atr_regime")
            and not S.atr_regime_pass(B, i, 100, 1.5))
        print(f"    {ds}: M15 bars={day_bars} ATR-skipped={day_atr_skip} trades={len(day_tr)}")
        for t in day_tr:
            print(f"      -> {t['time']} {t.get('setup', t.get('rule'))} {t['direction']}")

    opt2 = S.dense_plus_settings(False)
    k2 = {x: opt2[x] for x in opt2 if x not in meta}
    k2["htf_context"] = htf
    trades2 = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k2)
               if t["time"] >= cutoff]
    print()
    print(f"  DENSE+ (no ATR filter): {len(trades2)} trades vs DENSE {len(trades)} "
          f"(+{len(trades2)-len(trades)} in spike-heavy periods)")
    print("=" * 78)


if __name__ == "__main__":
    main()
