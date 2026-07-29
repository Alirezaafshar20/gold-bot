"""
Parity probe: does the regime label at today's key moments depend on how many
H1/H4 bars the map was built from?

live_portfolio.register_zones -> get_bars(sym, htf, 400)   (400 H1 + 400 H4)
portfolio_backtest            -> fetch_htf_bars(days=N, warmup=120)

Same detector, same params — different history length => potentially different
debounce path / macro pivots => different labels => different trades.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import mt5_data as M
import strategy as S
import symbol_profiles as P
import floating_config as FC

ASSET = "XAUUSD"
mt5 = M.connect()
sym = P.resolve_symbol_for_profile(ASSET, mt5)
opt = FC.REGIME

# today's decision moments (backtest trades on Jul 16)
CHECKS = ["2026-07-16 04:00", "2026-07-16 04:45", "2026-07-16 15:15",
          "2026-07-16 15:45"]


def build(h1_bars, h4_bars, label):
    h1 = M.fetch_bars(sym, "H1", count=h1_bars, mt5=mt5)
    h4 = M.fetch_bars(sym, "H4", count=h4_bars, mt5=mt5)
    ctx = S.prepare_htf_context({"H1": h1, "H4": h4})
    rmap = S.build_regime_map(
        ctx, tf="H1", detector=opt["regime_detector"], win=opt["regime_win"],
        er_hi=opt["regime_er_hi"], er_lo=opt["regime_er_lo"],
        confirm=opt["regime_confirm"], h4_win=opt["regime_h4_win"],
        h4_lookback=opt["regime_h4_lookback"])
    print(f"\n  [{label}]  H1 bars={len(h1)}  H4 bars={len(h4)}")
    for c in CHECKS:
        t = pd.Timestamp(c)
        print(f"    {c}  regime={rmap.at(t):<11} macro={rmap.macro_at(t):+d}")
    return rmap


# exactly like LIVE (register_zones fetches 400 bars of each HTF)
build(400, 400, "LIVE 400+400")

# exactly like BACKTEST --days 30: bars_for_period + 120 warmup
d30_h1 = 30 * 24 + 120
d30_h4 = 30 * 6 + 120
build(d30_h1, d30_h4, "BACKTEST 30d (+120 warmup)")

# exactly like BACKTEST --days 4
d4_h1 = 4 * 24 + 120
d4_h4 = 4 * 6 + 120
build(d4_h1, d4_h4, "BACKTEST 4d (+120 warmup)")

print("\nIf the labels differ between LIVE and BACKTEST rows, the parity gap")
print("is the regime-map history length — not the strategy logic.")
