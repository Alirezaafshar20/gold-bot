"""
Diagnose backtest fills vs live limit reality for recent trades.

For each backtest trade checks:
  - EXACT: did M15 wick touch the limit price?
  - TOL_ONLY: only within entry_tol_atr (backtest fills, live limit won't)
  - NEVER: price never came close within wait window
  - LIVE_SKIP: at signal bar, live would skip placing order (entry vs cur_price)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import strategy as S
import mt5_data as M
import symbol_profiles as P
import floating_config as FC
from multi_symbol_calibrate import _run

DAYS = 4
WAIT = S.WAIT_BARS


def main():
    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile("XAUUSD", mt5)
        _, prof = P.get_profile("XAUUSD")
        det = P.merge_params(S.DEFAULT_PARAMS, prof)
        etol_mult = det.get("entry_tol_atr", 0.38)
        _, d, m1, _ = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS + 1)
        B = S.Bars(d)
        stats = _run(sym, prof, mt5, days=DAYS, asset_key="XAUUSD",
                     risk_pct=0.035, max_concurrent=2)
        trades = stats["trades"]

        print("=" * 90)
        print(f"  FILL PARITY DIAG  |  XAUUSD  |  {DAYS}d  |  entry_tol_atr={etol_mult}")
        print("=" * 90)
        print(f"  Trades from backtest: {len(trades)}\n")
        hdr = (f"  {'time':<17} {'rule':<8} {'side':<5} {'entry':>8} "
               f"{'exact':>5} {'tol':>5} {'never':>5} {'live_skip':>9} "
               f"{'min_gap':>8}  R")
        print(hdr)
        print("  " + "-" * 86)

        counts = {"exact": 0, "tol_only": 0, "never": 0, "live_skip": 0}
        for t in trades:
            ts = pd.Timestamp(t["time"])
            idx = int(np.searchsorted(B.t, np.datetime64(ts), side="left"))
            idx = min(max(idx, 0), B.n - 1)
            entry = float(t["entry"])
            direction = t["dir"]
            atr_i = B.atr[idx] if B.atr[idx] > 0 else B.rng[idx]
            etol = etol_mult * atr_i
            sig_close = B.c[idx]
            exact = False
            tol_only = False
            min_gap = 999.0

            if direction == "long":
                live_skip = entry > sig_close + etol
                for j in range(idx + 1, min(idx + 1 + WAIT, B.n)):
                    gap = B.l[j] - entry
                    min_gap = min(min_gap, gap)
                    if B.l[j] <= entry:
                        exact = True
                        break
                    if B.l[j] <= entry + etol:
                        tol_only = True
            else:
                live_skip = entry < sig_close - etol
                for j in range(idx + 1, min(idx + 1 + WAIT, B.n)):
                    gap = entry - B.h[j]
                    min_gap = min(min_gap, gap)
                    if B.h[j] >= entry:
                        exact = True
                        break
                    if B.h[j] >= entry - etol:
                        tol_only = True

            never = not exact and not tol_only
            if exact:
                counts["exact"] += 1
            elif tol_only:
                counts["tol_only"] += 1
            if never:
                counts["never"] += 1
            if live_skip:
                counts["live_skip"] += 1

            side = direction.upper()[:5]
            print(f"  {str(ts)[:16]:<17} {str(t.get('rule','?')):<8} {side:<5} "
                  f"{entry:>8.2f} {'Y' if exact else 'N':>5} "
                  f"{'Y' if tol_only and not exact else 'N':>5} "
                  f"{'Y' if never else 'N':>5} "
                  f"{'Y' if live_skip else 'N':>9} "
                  f"{min_gap:>+8.2f}  {t['R']:+.2f}")

        print("\n  SUMMARY")
        print(f"    Exact limit touch (live CAN fill):     {counts['exact']}")
        print(f"    Tolerance-only (backtest yes, live no): {counts['tol_only']}")
        print(f"    Never reached (phantom if in backtest): {counts['never']}")
        print(f"    Live would skip at signal bar:          {counts['live_skip']}")
        print(f"    Typical etol on M15: ~{etol:.2f}$ (0.38 x ATR)")
        print("=" * 90)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
