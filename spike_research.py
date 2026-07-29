"""Compare DENSE-WIDE vs DENSE-WIDE+SPIKE on gold — spike-day coverage."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

DAYS = 90
SPIKE_DATES = ["2026-03-30", "2026-04-06", "2026-05-12", "2026-05-04"]


def run(opt, d, m1, cutoff, htf):
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
            "dense", "medium", "dense_plus", "dense_wide")
    k = {x: opt[x] for x in opt if x not in meta}
    k["htf_context"] = htf
    if opt.get("spike_mode"):
        k["spike_mode"] = True
        k["fib_spike_exempt"] = opt.get("fib_spike_exempt", True)
        k["spike_params"] = opt.get("spike_params")
    return [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
            if t["time"] >= cutoff]


def spike_trades(trades):
    return [t for t in trades if str(t.get("setup", "")).startswith("SPIKE")]


def day_trades(trades, ds):
    from pandas import Timestamp
    d = Timestamp(ds).date()
    return [t for t in trades if Timestamp(t["time"]).date() == d]


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    htf = S.prepare_htf_context(htf_dfs)
    M.shutdown(mt5)

    opt_base = S.dense_plus_settings(wide=True)
    opt_spike = S.dense_spike_settings(wide=True)
    meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
            "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
            "fib_spike_exempt", "spike_params")

    tr_base = run(opt_base, d, m1, cutoff, htf)
    tr_spike = run(opt_spike, d, m1, cutoff, htf)
    sp = spike_trades(tr_spike)

    print("=" * 78)
    print("  SPIKE RESEARCH — XAUUSD DENSE-WIDE vs +SPIKE  |  90d")
    print("=" * 78)
    for label, tr in [("DENSE-WIDE", tr_base), ("+ SPIKE rules", tr_spike)]:
        r = S.simulate_account(tr, 1000, 0.01, S.SPREAD_USD)
        print(f"  {label:<16} trades={r['n']:>3}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  "
              f"ret={r['ret_pct']:+.1f}%  DD={r['max_dd']:.1f}%")
    print(f"\n  SPIKE-rule entries: {len(sp)}")
    for t in sp:
        print(f"    {t['time']}  {t.get('setup')}  {t.get('dir', t.get('direction'))}  R={t['R']:+.2f}  "
              f"{t.get('exit_reason')}")

    print("\n  Known big-move days:")
    for ds in SPIKE_DATES:
        b = day_trades(tr_base, ds)
        s = day_trades(tr_spike, ds)
        bs = day_trades(sp, ds)
        print(f"    {ds}: base={len(b)}  +spike={len(s)}  (spike-rule={len(bs)})")
        for t in bs:
            print(f"      SPIKE -> {t['time']} {t.get('setup')} {t.get('dir')} R={t['R']:+.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
