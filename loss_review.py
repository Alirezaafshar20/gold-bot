"""
Deep-dive: why the recent consecutive losses (Jul 9-10)?

1. Market character around the losing trades (ATR spike, swing size vs SL).
2. Regime detector output vs realized market (lag / misclassification).
3. Trade-by-trade autopsy from reports/portfolio_trades.csv.
4. Loss-streak probability check from OOS history (is this streak "normal"?).
"""
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mt5_data as M
import strategy as S
import symbol_profiles as P
import floating_config as FC

mt5 = M.connect()
SYM = P.resolve_symbol_for_profile("XAUUSD", mt5)
print(f"symbol = {SYM}")

# ── 1. recent market character ──────────────────────────────────────────
def bars(tf, n):
    df = M.get_bars_df(SYM, tf, n) if hasattr(M, "get_bars_df") else None
    if df is None:
        import MetaTrader5 as mt5t
        tfmap = {"M15": mt5t.TIMEFRAME_M15, "H1": mt5t.TIMEFRAME_H1,
                 "H4": mt5t.TIMEFRAME_H4, "D1": mt5t.TIMEFRAME_D1}
        r = mt5t.copy_rates_from_pos(SYM, tfmap[tf], 0, n)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
    return df

d1 = bars("D1", 30)
print("\n== D1 last 12 days ==")
d1 = d1.tail(12)
d1_atr = (d1["high"] - d1["low"]).rolling(14, min_periods=5).mean()
for t, row in d1.iterrows():
    rng = row["high"] - row["low"]
    body = row["close"] - row["open"]
    print(f"  {str(t)[:10]}  O={row['open']:.1f} H={row['high']:.1f} "
          f"L={row['low']:.1f} C={row['close']:.1f}  range={rng:.1f} body={body:+.1f}")

h1 = bars("H1", 400)
h1["tr"] = np.maximum(h1["high"] - h1["low"],
                      np.maximum(abs(h1["high"] - h1["close"].shift()),
                                 abs(h1["low"] - h1["close"].shift())))
h1["atr"] = h1["tr"].rolling(14).mean()
base_atr = h1["atr"].iloc[:-120].median()
print(f"\n== H1 ATR ==  baseline(median older)={base_atr:.2f}")
recent = h1.tail(120)
by_day = recent.groupby(recent.index.date)["atr"].mean()
for d, v in by_day.items():
    print(f"  {d}  meanATR={v:.2f}  x{v/base_atr:.2f} of baseline")

# ── 2. regime map vs price over the losing window ──────────────────────
htf_dfs = {"H1": bars("H1", 400), "H4": bars("H4", 400) if True else None}
try:
    htf_dfs["H4"] = bars("H4", 400)
except Exception:
    pass
ctx = S.prepare_htf_context({k: v for k, v in htf_dfs.items() if v is not None})
opt = FC.REGIME
rmap = S.build_regime_map(
    ctx, tf=opt.get("regime_tf", "H1"), detector=opt.get("regime_detector", "mtf"),
    win=opt.get("regime_win", 40), er_trend=opt.get("regime_er_trend", 0.35),
    slope_k=opt.get("regime_slope_k", 0.0006), er_hi=opt.get("regime_er_hi", 0.32),
    er_lo=opt.get("regime_er_lo", 0.20), confirm=opt.get("regime_confirm", 2),
    h4_win=opt.get("regime_h4_win", 30), h4_lookback=opt.get("regime_h4_lookback", 60))

print("\n== regime vs H1 close (Jul 6 → now, every 4h) ==")
sel = h1[h1.index >= "2026-07-06"]
for t in sel.index[::4]:
    reg = rmap.at(t)
    macro = rmap.macro_at(t) if hasattr(rmap, "macro_at") else "?"
    c = float(sel.loc[t, "close"])
    print(f"  {t}  close={c:.1f}  regime={reg}  macro={macro}")

# ── 3. trade autopsy ────────────────────────────────────────────────────
print("\n== journal autopsy (reports/portfolio_trades.csv) ==")
j = pd.read_csv("reports/portfolio_trades.csv")
j["entry_time"] = pd.to_datetime(j["entry_time"])
for _, r in j.iterrows():
    reg = rmap.at(pd.Timestamp(r["entry_time"]))
    print(f"  #{int(r['n']):>2} {str(r['entry_time'])[:16]} {r['rule']:<7} {r['side']:<5} "
          f"entry={r['entry']:.1f} R={r['R']:+.2f} risk={r['use_risk_pct']*100:.1f}% "
          f"pnl={r['pnl_usd']:+.0f} fill={r['fill_kind']} exit={r['exit_reason']} regime@entry={reg}")

wins = (j["R"] > 0.1).sum(); losses = (j["R"] < -0.5).sum()
print(f"\n  total={len(j)}  wins={wins}  SL={losses}  netR={j['R'].sum():+.2f}  "
      f"pnl={j['pnl_usd'].sum():+.0f}")

# longs vs shorts after Jul 9
late = j[j["entry_time"] >= "2026-07-09"]
print(f"  Jul9+: n={len(late)} netR={late['R'].sum():+.2f} "
      f"(long n={len(late[late['side']=='long'])} R={late[late['side']=='long']['R'].sum():+.1f} | "
      f"short n={len(late[late['side']=='short'])} R={late[late['side']=='short']['R'].sum():+.1f})")

# ── 4. streak statistics from OOS history ───────────────────────────────
print("\n== loss-streak expectation (OOS history) ==")
try:
    o = pd.read_csv("reports/oos_trades.csv")
    rcol = "R" if "R" in o.columns else ("r" if "r" in o.columns else None)
    if rcol:
        g = o[o.get("asset", "XAUUSD").astype(str).str.contains("XAU", na=True)] if "asset" in o.columns else o
        rs = g[rcol].values
        wr = (rs > 0.1).mean()
        # longest losing streaks
        streak = best = 0
        for r in rs:
            streak = streak + 1 if r < -0.5 else 0
            best = max(best, streak)
        print(f"  OOS n={len(rs)} WR={wr*100:.1f}%  longest SL streak in OOS: {best}")
        print(f"  P(6 straight SL somewhere in 100 trades) with WR={wr*100:.0f}% ≈ "
              f"{100*(1-(1-(1-wr)**6)**94):.0f}% (rough)")
except Exception as e:
    print(f"  skipped ({e})")

print("\ndone")
