"""
Diagnose today's live trades: was each position sized to the INTENDED risk?
Compares the bot's assumed loss/lot (calc_lot model) to the broker's real specs,
and to the actual realized P&L from the screenshot.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import mt5_data as M
import symbol_specs as X
import portfolio_config as C
import symbol_profiles as P

# (key, lot, entry, sl, closed_price, observed_pnl, intended_risk_pct, bal_at_entry)
LIVE = [
    ("BRENT", 0.03, 90.60, 91.57, 91.59, -29.70, 0.03, 930),
    ("BRENT", 0.05, 91.17, 91.63, 91.63, -23.00, 0.03, 900),
    ("US500", 4.40, 7332.57, 7346.13, 7346.20, -56.57, 0.07, 877),
    ("BTCUSD", 0.21, 61341.60, 61764.65, 61809.01, -281.40, 0.10, 900),
]


def main():
    mt5 = M.connect()
    print("=" * 96)
    print("  TODAY'S LIVE TRADES — risk sizing audit")
    print("=" * 96)
    print(f"  {'sym':<8}{'lot':>6}{'SLdist':>10}{'$/lot(bot)':>12}"
          f"{'risk$(bot)':>11}{'real loss':>11}{'$/lot(real)':>13}{'over-risk':>10}")
    print("  " + "-" * 92)
    for key, lot, entry, sl, closed, pnl, rp, bal in LIVE:
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError:
            sym = key
        spec = X.get_spec(sym, mt5)
        sl_dist = abs(sl - entry)
        closed_dist = abs(closed - entry)
        per_lot_bot = X.loss_per_lot_usd(spec, sl_dist)
        risk_bot = lot * per_lot_bot
        # back out the broker's real $/lot from observed pnl & actual move
        per_lot_real = abs(pnl) / lot / closed_dist if (lot and closed_dist) else 0
        per_lot_real_full = per_lot_real * sl_dist  # scaled to SL distance
        over = abs(pnl) / risk_bot if risk_bot else 0
        print(f"  {key:<8}{lot:>6.2f}{sl_dist:>10.2f}{per_lot_bot:>12.2f}"
              f"{risk_bot:>11.2f}{abs(pnl):>11.2f}{per_lot_real_full:>13.2f}{over:>9.2f}x")
    print("  " + "-" * 92)
    print("  $/lot(bot) = what calc_lot assumed | $/lot(real) = implied by realized P&L")
    print("  over-risk = real loss ÷ bot-intended risk  (≈1.0 = correct, >1 = oversized)")

    print("\n  BTC contract spec (live broker):")
    sym = P.resolve_symbol_for_profile("BTCUSD", mt5)
    spec = X.get_spec(sym, mt5)
    for k in ("symbol", "contract_size", "tick_size", "tick_value", "point",
              "vol_min", "vol_step", "vol_max"):
        print(f"    {k:<15} = {spec.get(k)}")
    M.shutdown(mt5)
    print("=" * 96)


if __name__ == "__main__":
    main()
