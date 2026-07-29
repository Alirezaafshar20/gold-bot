"""
RUN THIS ON THE LIVE MT5 TERMINAL (the same machine the bot trades on).

It compares, per symbol:
  • OLD sizing  = tick_value model (what the bot used before)
  • NEW sizing  = broker order_calc_profit (the real P&L the broker applies)

If NEW differs from OLD, the old code was mis-sizing on THIS broker.
After the fix, the bot uses NEW automatically.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import mt5_data as M
import symbol_specs as X
import symbol_profiles as P
import portfolio_config as C

SAMPLE_BAL = 1000.0


def tick_value_only(spec, sl_distance):
    """Reproduce the OLD model (tick_value, ignoring order_calc_profit)."""
    tick_size = spec["tick_size"]
    tick_value = spec["tick_value"]
    if tick_value > 0 and tick_size > 0:
        linear = (sl_distance / tick_size) * tick_value
        cs = spec.get("contract_size") or 1.0
        if cs == 1.0:
            return max(linear, sl_distance * cs)
        return linear
    cs = spec.get("contract_size") or 1.0
    return sl_distance * cs


def main():
    mt5 = M.connect()
    print(f"[MT5] {mt5.terminal_info().company}  |  account {mt5.account_info().login}")
    print("=" * 100)
    print("  LIVE BROKER SIZING CHECK  (run on the trading terminal)")
    print("=" * 100)
    print(f"  {'sym':<8}{'price':>12}{'risk%':>7}{'SLdist':>10}"
          f"{'$/lot OLD':>12}{'$/lot REAL':>12}{'lot OLD':>9}{'lot NEW':>9}{'ratio':>8}")
    print("  " + "-" * 96)

    for key in C.PORTFOLIO_ASSETS:
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError as e:
            print(f"  {key:<8}  NOT FOUND ({e})")
            continue
        spec = X.get_spec(sym, mt5)
        risk = C.RISK_MAP.get(key, 0.05)
        tick = mt5.symbol_info_tick(sym)
        price = tick.ask if tick else 0.0
        # representative SL distance = 0.3% of price (just for the comparison)
        sl_dist = max(price * 0.003, spec["tick_size"] * 10)

        per_old = tick_value_only(spec, sl_dist)
        per_real = X.loss_per_lot_usd(spec, sl_dist, mt5=mt5, symbol=sym, ref_price=price)
        lot_old = max(spec["vol_min"],
                      round((SAMPLE_BAL * risk / per_old) / spec["vol_step"]) * spec["vol_step"]) \
            if per_old > 0 else spec["vol_min"]
        lot_old = min(lot_old, spec["vol_max"])
        lot_new = X.calc_lot(sym, SAMPLE_BAL, risk, sl_dist, mt5)
        ratio = per_real / per_old if per_old > 0 else 0
        flag = "  <-- MISMATCH" if abs(ratio - 1) > 0.05 else ""
        print(f"  {key:<8}{price:>12.2f}{risk*100:>6.0f}%{sl_dist:>10.2f}"
              f"{per_old:>12.2f}{per_real:>12.2f}{lot_old:>9.2f}{lot_new:>9.2f}"
              f"{ratio:>7.2f}x{flag}")

    print("  " + "-" * 96)
    print("  ratio = REAL ÷ OLD.  1.00 = old code was fine.  >1 = old code OVER-sized (too risky).")
    print("  lot NEW is what the fixed bot will now send. Verify BTC lot dropped accordingly.")
    M.shutdown(mt5)
    print("=" * 100)


if __name__ == "__main__":
    main()
