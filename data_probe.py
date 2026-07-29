"""
Probe how far back the broker actually serves each timeframe, per symbol.
Tells us what is feasible for a 2022→now calibration and what TF we can use
for accurate intrabar exits.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import datetime as dt
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C

try:
    import MetaTrader5 as mt5
except ImportError:
    print("pip install MetaTrader5"); sys.exit(1)

TFS = [("M1", mt5.TIMEFRAME_M1), ("M5", mt5.TIMEFRAME_M5),
       ("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1)]


FROM_2022 = dt.datetime(2022, 4, 1)


def earliest_pos(sym, tf_const):
    """Furthest back a single max request reaches (per-request cap ~99999)."""
    rates = mt5.copy_rates_from_pos(sym, tf_const, 0, 99999)
    if rates is None or len(rates) == 0:
        return None, 0
    first = dt.datetime.utcfromtimestamp(int(rates[0]["time"]))
    return first, len(rates)


def range_2022(sym, tf_const):
    """Does the broker actually serve data back to 2022-04-01 via range query?"""
    rates = mt5.copy_rates_range(sym, tf_const, FROM_2022, dt.datetime.now())
    if rates is None or len(rates) == 0:
        return None, 0
    first = dt.datetime.utcfromtimestamp(int(rates[0]["time"]))
    return first, len(rates)


def main():
    M.connect()
    print("=" * 100)
    print(f"  BROKER DATA AVAILABILITY  |  {mt5.terminal_info().company}")
    print("=" * 100)
    print(f"  {'symbol':<12}{'TF':>5}{'1-req bars':>11}{'1-req earliest':>18}"
          f"{'range bars':>12}{'range earliest':>18}")
    print("  " + "-" * 84)
    for key in C.PORTFOLIO_ASSETS:
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError as e:
            print(f"  {key:<12} NOT FOUND ({e})")
            continue
        for tf_name, tf_const in TFS:
            f1, n1 = earliest_pos(sym, tf_const)
            fr, nr = range_2022(sym, tf_const)
            s1 = f1.strftime("%Y-%m-%d") if f1 else "—"
            sr = fr.strftime("%Y-%m-%d") if fr else "—"
            print(f"  {sym:<12}{tf_name:>5}{n1:>11}{s1:>18}{nr:>12}{sr:>18}")
        print()
    mt5.shutdown()
    print("=" * 100)
    print("  'range earliest' = how far back the broker REALLY stores this TF.")
    print("  If range earliest > 2022-04, the broker simply doesn't have older data.")


if __name__ == "__main__":
    main()
