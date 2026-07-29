"""Quick: slightly looser DENSE+HARM_BAT variants."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M

DAYS, SPREAD, BAL, RISK = 180, 0.28, 1000, 0.01
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")


def run(d, m1, B, ctx, htf, cutoff, label, opt):
    k = {x: opt[x] for x in opt if x not in META}
    k["htf_context"] = htf
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k) if t["time"] >= cutoff]
    if not tr:
        print(f"  {label:<32} n= 0")
        return
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    rules = {}
    for t in tr:
        rules[t.get("rule", "?")] = rules.get(t.get("rule", "?"), 0) + 1
    print(f"  {label:<32} n={r['n']:>2}  WR={r['wr']:>5.1f}%  ret={r['ret_pct']:>+6.1f}%  "
          f"DD={r['max_dd']:.1f}%  {rules}")


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
    print("DENSE + HARM_BAT — relax ladder (180d M15)")
    print("-" * 90)
    run(d, m1, B, ctx, htf, cutoff, "DENSE (current)", base)

    o = dict(base)
    o["session_start"], o["session_end"] = 8, 22
    run(d, m1, B, ctx, htf, cutoff, "+ session 8-22", o)

    o = dict(base)
    o["atr_max_ratio"] = 1.7
    run(d, m1, B, ctx, htf, cutoff, "+ ATR 1.7x", o)

    o = dict(base)
    o["session_start"], o["session_end"] = 9, 21
    run(d, m1, B, ctx, htf, cutoff, "+ session 9-21", o)

    run(d, m1, B, ctx, htf, cutoff, "DENSE+ (no ATR)", S.dense_plus_settings(False))
    run(d, m1, B, ctx, htf, cutoff, "DENSE-WIDE", S.dense_plus_settings(True))

    o = dict(base)
    o["session_start"], o["session_end"] = 8, 22
    o["atr_max_ratio"] = 1.7
    run(d, m1, B, ctx, htf, cutoff, "sess8-22 + ATR1.7", o)

    o = dict(base)
    o["require_fib"] = False
    run(d, m1, B, ctx, htf, cutoff, "no fib (risky)", o)


if __name__ == "__main__":
    main()
