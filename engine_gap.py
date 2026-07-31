"""Measure the calibration engine's optimism per timeframe.

Two engines decide entries in this project:

  strategy.run_backtest   entry granted when a SIGNAL-TF bar wick enters the
                          entry band (mid prices, no spread haircut, no M1
                          path ordering). Used by multi_symbol_calibrate._run,
                          calibrate_long and oos_test — i.e. by rule selection.

  live_portfolio.process_once  entry granted only when the M1 path actually
                          reaches the resting limit under limit_fill_mode, with
                          the spread haircut applied. Used by live and
                          live_replay.

If the gap between them is driven by bar granularity, it must shrink as the
signal timeframe shrinks: an M5 wick is a much better proxy for an M1 touch
than an M15 wick is. This script measures the calibration side on both
timeframes over the same window so the two can be compared against the
live_replay numbers.

Usage: python engine_gap.py [days]
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import floating_config as FC
import mt5_data as M
import portfolio_config as C
import symbol_profiles as P
from multi_symbol_calibrate import _run

ASSET = "XAUUSD"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

mt5 = M.connect()
RISK = C.RISK_MAP[ASSET]

# live_replay on the same 60-day window, --flat 0.02, from reports/rp_*.log
TRUTH = {"M15": (61, 32.8, 1.00, 0.09), "M5": (39, 38.5, 2.10, 17.57)}

VARIANTS = [
    ("slot at fill  (current)", {"arm_blocks_same_dir": False}),
    ("slot at arm   (like live)", {"arm_blocks_same_dir": True}),
]

print("=" * 100)
print(f"  CALIBRATION ENGINE (strategy.run_backtest) vs live_replay — "
      f"{ASSET}, {DAYS}d, risk={RISK*100:.1f}%")
print("=" * 100)

for tf in ("M15", "M5"):
    FC.apply_tf_paths(tf)
    rules = FC.OOS_RULES["XAUUSD"]
    _, prof = P.get_profile(ASSET)
    sym = P.resolve_symbol_for_profile(ASSET, mt5)
    print(f"\n  {tf}   rules: {','.join(rules)}")
    print(f"  {'variant':<28} {'n':>4} {'WR%':>6} {'PF':>6} {'netR':>8} "
          f"{'R/trade':>9}")
    print("  " + "-" * 66)
    for label, extra in VARIANTS:
        r = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET, risk_pct=RISK,
                 max_concurrent=C.max_concurrent(ASSET), tf=tf,
                 opt_extra=dict(extra))
        tr = r.get("trades") or []
        net = sum(t["R"] for t in tr)
        per = net / len(tr) if tr else 0.0
        print(f"  {label:<28} {r['n']:>4} {r['wr']:>6.1f} "
              f"{r['pf']:>6.2f} {net:>+8.2f} {per:>+9.3f}")
    n, wr, pf, net = TRUTH[tf]
    print(f"  {'live_replay (ground truth)':<28} {n:>4} {wr:>6.1f} "
          f"{pf:>6.2f} {net:>+8.2f} {net/n:>+9.3f}")

print()
print("  The 'slot at arm' row is the corrected engine. How close it lands to")
print("  the live_replay row is the measure of whether the armed-zone slot was")
print("  the whole gap or only part of it.")
