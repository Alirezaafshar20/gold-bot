"""
Diagnose calibration consistency between backtest windows and live AFTER fix.
Uses the SAME path the backtest now uses (_calibrate_min_sl_90d).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C
from multi_symbol_calibrate import _calibrate_min_sl_90d


def bt_min_sl(sym, prof, days, mt5):
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=days)
    B = S.Bars(d)
    return _calibrate_min_sl_90d(sym, prof, B, cutoff, mt5)


def live_min_sl(sym, prof, mt5):
    n = int(90 * 1440 / 15) + 60
    df = M.fetch_bars(sym, "M15", count=n, mt5=mt5)
    B = S.Bars(df)
    return X.calibrate_min_sl(B, B.t[0], prof)


def main():
    mt5 = M.connect()
    print("=" * 88)
    print("  CALIBRATION CONSISTENCY AFTER FIX  (min-SL price distance)")
    print("=" * 88)
    print(f"  {'asset':<8}{'bt --days10':>13}{'bt --days30':>13}{'bt --days90':>13}"
          f"{'live 90d':>12}")
    print("  " + "-" * 84)
    for key in C.PORTFOLIO_ASSETS:
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError as e:
            print(f"  {key:<8} ERR {e}")
            continue
        _, prof = P.get_profile(key)
        s10 = bt_min_sl(sym, prof, 10, mt5)
        s30 = bt_min_sl(sym, prof, 30, mt5)
        s90 = bt_min_sl(sym, prof, 90, mt5)
        sl = live_min_sl(sym, prof, mt5)
        print(f"  {key:<8}{s10:>13.4g}{s30:>13.4g}{s90:>13.4g}{sl:>12.4g}")
    print("  " + "-" * 84)
    print("  All four columns should now be EQUAL per asset → backtest == live threshold.")
    M.shutdown(mt5)
    print("=" * 88)


if __name__ == "__main__":
    main()
