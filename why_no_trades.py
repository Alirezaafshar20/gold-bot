"""
Why is the bot not trading TODAY? Walk the entry funnel gate by gate.

For every H1 slot today: regime label, macro bias, which rules the effective
(weekly) meta-gate allows, and the direction filter. Then count today's raw
M15 setups per rule BEFORE gating, so we can see what was detected vs blocked.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import datetime as dt
import pandas as pd

import mt5_data as M
import strategy as S
import symbol_profiles as P
import floating_config as FC

ASSET = "XAUUSD"
mt5 = M.connect()
sym = P.resolve_symbol_for_profile(ASSET, mt5)
_, prof = P.get_profile(ASSET)
opt = P.resolve_live_opt(prof)
opt = P.apply_profile_to_settings(opt, prof, P.live_spread(sym, prof, mt5))

# regime map exactly as live builds it
_, htf_dfs = M.fetch_htf_bars(sym, days=45, mt5=mt5, tfs=("H4", "H1"))
ctx = S.prepare_htf_context(htf_dfs)
rmap = S.build_regime_map(
    ctx, tf=opt.get("regime_tf", "H1"), detector=opt.get("regime_detector", "mtf"),
    win=opt.get("regime_win", 40), er_trend=opt.get("regime_er_trend", 0.35),
    slope_k=opt.get("regime_slope_k", 0.0006), er_hi=opt.get("regime_er_hi", 0.32),
    er_lo=opt.get("regime_er_lo", 0.20), confirm=opt.get("regime_confirm", 2),
    h4_win=opt.get("regime_h4_win", 30), h4_lookback=opt.get("regime_h4_lookback", 60))

mg = FC.load_meta_gate(assets=[ASSET])
rules = FC.OOS_RULES[ASSET]

print(f"symbol={sym}  rules={rules}")
print(f"meta-gate source: {FC.meta_gate_source()}")
if mg:
    print(mg.summary([ASSET]))

h1 = ctx.get("H1")
h1B = h1 if isinstance(h1, S.Bars) else S.Bars(h1)
today = pd.Timestamp(dt.date.today())
idx_today = [i for i in range(h1B.n) if pd.Timestamp(h1B.t[i]) >= today]

print(f"\n== TODAY hour by hour (H1) ==")
print(f"  {'time':<17} {'close':>8} {'regime':<11} {'macro':>5} "
      f"{'rules allowed by meta-gate':<38} dir-filter")
for i in idx_today:
    t = h1B.t[i]
    reg = rmap.at(t)
    macro = rmap.macro_at(t) if hasattr(rmap, "macro_at") else 0
    allowed = [r for r in rules if mg is None or mg.allows(ASSET, r, reg)]
    if reg == "TREND_UP":
        dirf = "long only"
    elif reg == "TREND_DOWN":
        dirf = "short only"
    else:
        dirf = "short only (macro<0)" if macro < 0 else (
            "long only (macro>0)" if macro > 0 else "both")
    print(f"  {str(pd.Timestamp(t))[:16]:<17} {h1B.c[i]:>8.1f} {reg:<11} {macro:>+5} "
          f"{','.join(allowed) or '— NONE (skip all)':<38} {dirf}")

# raw setups today BEFORE any gate (detector level)
print(f"\n== raw M15 setups detected today (before gates) ==")
d = M.fetch_bars(sym, "M15", count=1200, mt5=mt5)
B = S.Bars(d)
det = P.merge_params(S.DEFAULT_PARAMS, prof)
n_raw = 0
for i in range(3, B.n - 1):
    if pd.Timestamp(B.t[i]) < today:
        continue
    for name in rules:
        for (direction, proximal, distal, tag) in S.iter_rule_setups(
                name, B, i, det, htf_context=ctx, signal_time=B.t[i],
                htf_tfs=tuple(opt.get("htf_tfs", ("H4", "H1")))):
            reg = rmap.at(B.t[i])
            gate_ok = mg is None or mg.allows(ASSET, name, reg)
            dir_ok = S.regime_allows(rmap, B.t[i], direction)
            n_raw += 1
            print(f"  {str(pd.Timestamp(B.t[i]))[:16]}  {name:<10} {direction:<5} "
                  f"prox={proximal:.1f}  regime={reg}  "
                  f"meta={'PASS' if gate_ok else 'BLOCK'}  dir={'PASS' if dir_ok else 'BLOCK'}")
if n_raw == 0:
    print("  (none — detectors found no qualifying pattern today)")
print("\ndone")
