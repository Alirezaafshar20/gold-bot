"""
Audit: does live_portfolio.py size/filter trades EXACTLY like portfolio_backtest.py?

Checks per symbol:
  1. max_concurrent: profile (backtest source) vs live (2)
  2. min SL: 90-day window (backtest) vs ~600-bar window (live)
  3. lot rounding: intended risk$ vs ACTUAL risk$ after vol_min/vol_step clamp
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C

BALANCES = [930.0, 5000.0, 50000.0]


def live_window_min_sl(B, profile):
    if profile.get("min_sl_mode") != "atr":
        return profile.get("min_sl", 0.0)
    idx = max(0, B.n - min(B.n - 10, 96 * 30))
    return X.calibrate_min_sl(B, B.t[idx], profile)


def main():
    mt5 = M.connect()
    print("=" * 100)
    print("  AUDIT: live_portfolio vs portfolio_backtest")
    print("=" * 100)

    for key in C.PORTFOLIO_ASSETS:
        _, prof = P.get_profile(key)
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError as e:
            print(f"\n{key}: NOT FOUND ({e})")
            continue
        spec = X.get_spec(sym, mt5)
        prof_mc = (prof.get("opt_overrides") or {}).get("max_concurrent", S.MAX_CONCURRENT)

        # backtest-style 90d window
        _, d90, m1, cut90 = M.fetch_pair(sym, "M15", mt5=mt5, days=90)
        B90 = S.Bars(d90)
        sl_bt = X.calibrate_min_sl(B90, cut90, prof)

        # live-style 600-bar window (what live actually loads)
        d_live = M.fetch_bars(sym, "M15", count=600, mt5=mt5)
        Bl = S.Bars(d_live)
        sl_live = live_window_min_sl(Bl, prof)

        print(f"\n{'─'*100}")
        print(f"  {prof.get('label', key)}  ({sym})")
        print(f"  contract_size={spec['contract_size']}  tick_size={spec['tick_size']}  "
              f"tick_value={spec['tick_value']}  vol_min={spec['vol_min']}  vol_step={spec['vol_step']}")
        print(f"  max_concurrent:  backtest(profile)={prof_mc}   live={C.MAX_CONCURRENT_PER_ASSET}   "
              f"{'MISMATCH' if prof_mc != C.MAX_CONCURRENT_PER_ASSET else 'ok'}")
        print(f"  min SL filter:   backtest(90d)={sl_bt:.5g}   live(600bar)={sl_live:.5g}   "
              f"diff={100*(sl_live-sl_bt)/sl_bt:+.1f}%")

        risk = prof.get("risk_pct", 0.01) if key not in C.RISK_MAP else C.RISK_MAP[key]
        print(f"  risk%={risk*100:.0f}   |  lot rounding (SL distance = {sl_bt:.5g}):")
        print(f"     {'balance':>9}  {'intended$':>10}  {'raw lot':>9}  {'final lot':>9}  "
              f"{'actual$':>9}  {'error':>7}")
        for bal in BALANCES:
            intended = bal * risk
            loss_per_lot = X.loss_per_lot_usd(spec, sl_bt)
            raw_lot = intended / loss_per_lot if loss_per_lot > 0 else 0
            final_lot = X.calc_lot(sym, bal, risk, sl_bt, mt5)
            actual = final_lot * loss_per_lot
            err = 100 * (actual - intended) / intended if intended else 0
            print(f"     {bal:>9,.0f}  {intended:>10.2f}  {raw_lot:>9.4f}  {final_lot:>9.2f}  "
                  f"{actual:>9.2f}  {err:>+6.1f}%")

    M.shutdown(mt5)
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
