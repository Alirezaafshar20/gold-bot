import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M

META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")
opt = S.dense_settings()
spread = 0.28
days = 180

mt5 = M.connect()
sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
print("DENSE + HARM_BAT by signal TF (180d)")
print("-" * 58)
for tf in ["M5", "M15", "M30", "H1"]:
    try:
        _, d, m1, cutoff = M.fetch_pair(sym, tf, mt5=mt5, days=days)
        _, htf_dfs = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=("H4", "H1"))
        B = S.Bars(d)
        ctx = S.M1Ctx(m1, d.index)
        htf = S.prepare_htf_context(htf_dfs)
        k = {x: opt[x] for x in opt if x not in META}
        k["htf_context"] = htf
        tmin = M.tf_minutes(tf)
        tr = [x for x in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=tmin, **k)
              if x["time"] >= cutoff]
        if tr:
            r = S.simulate_account(tr, 1000, 0.01, spread)
            print(f"  {tf:4}  bars={len(d):5}  n={r['n']:3}  WR={r['wr']:5.1f}%  ret={r['ret_pct']:+.1f}%")
        else:
            print(f"  {tf:4}  bars={len(d):5}  n=  0")
    except Exception as e:
        print(f"  {tf:4}  ERROR: {e}")
M.shutdown(mt5)
print("\nM1 = exit engine only (not signal TF)")
