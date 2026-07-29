"""
Download FULL available history per symbol/timeframe to local CSV, paging past
the 99999-bars-per-request MT5 limit. Run this ONCE on the strong machine; the
calibration/backtest then reads these files (fast, reproducible, offline).

  python data_cache.py                      # all portfolio symbols, M15 + M1
  python data_cache.py --tfs M15,M5,M1
  python data_cache.py --from 2022-04-01    # stop paging once we reach this date

Files: data/cache/{SYMBOL}_{TF}.csv   (UTC time index, OHLCV)
Nothing about the strategy or live logic is touched.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import os
import datetime as dt
import pandas as pd

import mt5_data as M
import symbol_profiles as P
import portfolio_config as C

try:
    import MetaTrader5 as mt5
except ImportError:
    print("pip install MetaTrader5"); sys.exit(1)

CACHE_DIR = "data/cache"
PAGE = 99999  # MT5 per-request hard cap


def download_all(sym, tf_str, stop_before=None, max_pages=200):
    """Page copy_rates_from_pos backwards until the broker runs out of bars."""
    tf_const = getattr(mt5, M.TF_MAP[tf_str.upper()])
    if not mt5.symbol_select(sym, True):
        raise RuntimeError(f"cannot select {sym}")
    frames = []
    pos = 0
    for _ in range(max_pages):
        chunk = mt5.copy_rates_from_pos(sym, tf_const, pos, PAGE)
        if chunk is None or len(chunk) == 0:
            break
        df = pd.DataFrame(chunk)
        frames.append(df)
        got = len(df)
        oldest = dt.datetime.utcfromtimestamp(int(df["time"].iloc[0]))
        print(f"    {sym} {tf_str}: page pos={pos} got={got} oldest={oldest:%Y-%m-%d}",
              flush=True)
        pos += got
        if got < PAGE:
            break
        if stop_before is not None and oldest <= stop_before:
            break
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    out["time"] = pd.to_datetime(out["time"], unit="s")
    out = out.set_index("time").sort_index()
    out = out.rename(columns={"tick_volume": "volume"})
    return out[["open", "high", "low", "close", "volume"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(C.PORTFOLIO_ASSETS))
    ap.add_argument("--tfs", default="M15,M1")
    ap.add_argument("--from", dest="from_date", default=None,
                    help="Stop paging once a page reaches this date (YYYY-MM-DD)")
    args = ap.parse_args()
    stop = dt.datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None
    tfs = [t.strip().upper() for t in args.tfs.split(",") if t.strip()]
    keys = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    M.connect()
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("=" * 80)
    print(f"  DATA CACHE  |  {mt5.terminal_info().company}  |  tfs={tfs}")
    if stop:
        print(f"  paging back until {stop:%Y-%m-%d}")
    print("=" * 80)

    for key in keys:
        try:
            sym = P.resolve_symbol_for_profile(key, mt5)
        except RuntimeError as e:
            print(f"  {key}: skip ({e})"); continue
        for tf in tfs:
            try:
                df = download_all(sym, tf, stop_before=stop)
            except RuntimeError as e:
                print(f"  {sym} {tf}: ERR {e}"); continue
            if df is None or df.empty:
                print(f"  {sym} {tf}: no data"); continue
            path = os.path.join(CACHE_DIR, f"{key}_{tf}.csv")
            df.to_csv(path)
            print(f"  SAVED {path}  rows={len(df)}  "
                  f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    mt5.shutdown()
    print("=" * 80)
    print("  Done. Calibration can now read data/cache/*.csv offline.")


if __name__ == "__main__":
    main()
