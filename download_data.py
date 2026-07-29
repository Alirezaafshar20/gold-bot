"""
دریافت داده‌های قیمتی طلا برای آموزش مدل
==========================================

روش‌های پشتیبانی‌شده:
  1. Yahoo Finance (رایگان) — طلای روزانه
  2. MetaTrader 5 (رایگان) — XAUUSD با هر تایم‌فریم (M1, M5, M15, M30, H1, H4, D1)

تایم‌فریم پیشنهادی برای آموزش:
  - H1 یا M15: تعادل خوب بین سیگنال تمیز و داده کافی (پیشنهاد اصلی)
  - M5: فقط برای آزمایش؛ نویز بیشتر
  - M1: معمولاً توصیه نمی‌شود (نویز زیاد، overfit)
  - H4 / D1: برای روند بلندمدت؛ در مدل به‌صورت چندتایم‌فریم خودکار استفاده می‌شود

استفاده:
  python download_data.py --source mt5 --timeframe H1
  python download_data.py --source mt5 --timeframe M15 --output data/XAUUSD_M15.csv
  python download_data.py --source yahoo --years 8
"""

import argparse
import os
import sys
import struct

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

DATA_PATH = "data/XAUUSD_M15.csv"   # default output path

# نگاشت تایم‌فریم متاتریدر ۵
MT5_TIMEFRAMES = {
    "M1": ("TIMEFRAME_M1", 1),
    "M5": ("TIMEFRAME_M5", 5),
    "M15": ("TIMEFRAME_M15", 15),
    "M30": ("TIMEFRAME_M30", 30),
    "H1": ("TIMEFRAME_H1", 60),
    "H4": ("TIMEFRAME_H4", 240),
    "D1": ("TIMEFRAME_D1", 1440),
}


def _yahoo_download_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """نرمال‌سازی ستون‌های Yahoo Finance DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]
    rename_map = {"open": "open", "high": "high", "low": "low",
                  "close": "close", "volume": "volume", "adj close": "close"}
    df = df.rename(columns=rename_map)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index.name = "datetime"
    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def download_yahoo_h1(save_path: str = "data/XAUUSD_H1.csv") -> str:
    """
    دانلود داده ساعتی طلا از Yahoo Finance (GC=F) — تا 730 روز (2 سال).
    این روش برای ترین مدل وقتی بروکر تاریخچه کافی ندارد استفاده می‌شود.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[Error] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print("[Yahoo-H1] Downloading gold hourly data (up to 2 years)...")
    df = yf.download("GC=F", period="730d", interval="1h", progress=False, auto_adjust=True)

    if df is None or df.empty or len(df) < 200:
        print("[Error] Not enough data. Check internet connection.")
        sys.exit(1)

    df = _yahoo_download_normalize(df)
    df = df.dropna()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Yahoo-H1] Bars: {len(df)}")
    print(f"[Yahoo-H1] Range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"[Yahoo-H1] Saved: {save_path}")
    return save_path


def download_yahoo(years: int = 8, save_path: str = "data/XAUUSD_D1.csv"):
    """
    دانلود داده طلا از Yahoo Finance (GC=F = Gold Futures).
    تایم‌فریم: روزانه (برای تاریخچه طولانی؛ داده 1ساعته در Yahoo محدود است).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[Error] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print(f"[Yahoo] Downloading gold (GC=F) daily data, last {years} years...")

    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)

    df = yf.download(
        "GC=F",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if df.empty or len(df) < 100:
        print("[Error] Not enough data downloaded. Check internet or try different period.")
        sys.exit(1)

    df = _yahoo_download_normalize(df)
    df = df.dropna()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[Yahoo] Saved {len(df)} candles to {save_path}")
    print(f"[Yahoo] Range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return save_path


def download_mt5(
    timeframe: str = "H1",
    symbol: str = None,
    save_path: str = None,
    bars: int = 100000,
    years: int = 5,
):
    """
    Download from MetaTrader 5 and save to CSV (optional archive only).
    Live trading and backtest.py read MT5 directly — CSV is NOT required.
    """
    import mt5_data as M

    tf_upper = timeframe.upper()
    if tf_upper not in MT5_TIMEFRAMES:
        print(f"[Error] Unknown timeframe: {timeframe}")
        print(f"  Allowed: {', '.join(MT5_TIMEFRAMES.keys())}")
        sys.exit(1)

    mt5 = M.connect()
    try:
        sym = M.resolve_symbol(symbol or M.DEFAULT_SYMBOL, mt5)
        print(f"[MT5] symbol: {sym}")
        print(f"[MT5] downloading {bars} {tf_upper} bars...")
        df = M.fetch_bars(sym, tf_upper, count=bars, mt5=mt5)
    finally:
        M.shutdown(mt5)

    if save_path is None:
        safe = sym.replace("@", "").replace("/", "_")
        save_path = f"data/{safe}_{tf_upper}.csv"

    out = df.reset_index().rename(columns={"time": "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out.to_csv(save_path, index=False)

    print(f"[MT5] Symbol: {sym} | Timeframe: {tf_upper} | Bars: {len(out)}")
    print(f"[MT5] Saved: {save_path}")
    print(f"[MT5] Range: {out['datetime'].iloc[0]} to {out['datetime'].iloc[-1]}")
    print("[Tip] backtest.py reads MT5 directly — CSV save is optional.")
    return save_path


def extend_with_yahoo15m(existing_csv: str = "data/XAUUSD_M15.csv") -> str:
    """
    داده‌های موجود M15 را با آخرین 60 روز از Yahoo Finance (15m) تکمیل می‌کند.
    داده قدیمی حفظ می‌شود و ردیف‌های تکراری حذف می‌شوند.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[Error] yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print("[Yahoo-15m] Downloading last 60 days of XAUUSD 15m data...")
    df_new = yf.download("GC=F", period="60d", interval="15m", progress=False, auto_adjust=True)

    if df_new.empty:
        print("[Error] No data returned from Yahoo Finance.")
        sys.exit(1)

    # flatten MultiIndex columns if present
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = [c[0].lower() for c in df_new.columns]
    else:
        df_new.columns = [c.lower() for c in df_new.columns]

    df_new = df_new.rename(columns={"open": "open", "high": "high",
                                     "low": "low", "close": "close", "volume": "volume"})
    df_new = df_new[["open", "high", "low", "close", "volume"]].copy()
    df_new.index.name = "datetime"
    df_new = df_new.reset_index()
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.tz_localize(None)
    df_new["datetime"] = df_new["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Yahoo-15m] New bars: {len(df_new)}")
    print(f"[Yahoo-15m] New range: {df_new['datetime'].iloc[0]} to {df_new['datetime'].iloc[-1]}")

    if os.path.exists(existing_csv):
        df_old = pd.read_csv(existing_csv)
        print(f"[Yahoo-15m] Existing rows: {len(df_old)} (last: {df_old['datetime'].iloc[-1]})")
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        print(f"[Yahoo-15m] No existing file found; saving new data only.")
        df_combined = df_new

    df_combined["datetime"] = pd.to_datetime(df_combined["datetime"])
    df_combined = df_combined.drop_duplicates(subset="datetime")
    df_combined = df_combined.sort_values("datetime")
    df_combined["datetime"] = df_combined["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_combined = df_combined.dropna()

    os.makedirs(os.path.dirname(existing_csv) or ".", exist_ok=True)
    df_combined.to_csv(existing_csv, index=False)
    print(f"[Yahoo-15m] Updated file: {existing_csv}")
    print(f"[Yahoo-15m] Total rows: {len(df_combined)}")
    print(f"[Yahoo-15m] Full range: {df_combined['datetime'].iloc[0]} to {df_combined['datetime'].iloc[-1]}")
    return existing_csv


def _find_mt4_hst_files(period: int = 15) -> list:
    """جستجوی فایل‌های .hst متاتریدر 4 در AppData."""
    appdata = os.environ.get("APPDATA", "")
    terminal_root = os.path.join(appdata, "MetaQuotes", "Terminal")
    if not os.path.isdir(terminal_root):
        return []

    symbol_variants = ["XAUUSD", "XAUUSDm", "XAUUSD@", "XAUUSD.", "GOLD"]
    filenames = {f"{s}{period}.hst": s for s in symbol_variants}

    found = []
    for terminal_hash in os.listdir(terminal_root):
        history_dir = os.path.join(terminal_root, terminal_hash, "history")
        if not os.path.isdir(history_dir):
            continue
        for server in os.listdir(history_dir):
            server_dir = os.path.join(history_dir, server)
            if not os.path.isdir(server_dir):
                continue
            for fname, sym in filenames.items():
                fpath = os.path.join(server_dir, fname)
                if os.path.exists(fpath) and os.path.getsize(fpath) > 200:
                    found.append((fpath, sym, server))
    return found


def _parse_mt4_hst(filepath: str) -> pd.DataFrame | None:
    """فایل باینری .hst را پارس کرده و DataFrame برمی‌گرداند."""
    with open(filepath, "rb") as f:
        raw = f.read()

    if len(raw) < 152:
        return None

    version = struct.unpack_from("<i", raw, 0)[0]

    if version == 400:
        # Header: 148 bytes | Record: time(i4) open(d8) low(d8) high(d8) close(d8) vol(i4) = 44 bytes
        header_size = 148
        rec_fmt  = "<iddddI"
        rec_size = struct.calcsize(rec_fmt)   # 4+8+8+8+8+4 = 40
        col_order = ("time", "open", "low", "high", "close", "volume")
    elif version == 401:
        # Header: 148 bytes | Record: time(q8) open(d8) high(d8) low(d8) close(d8) vol(q8) spread(i4) real_vol(q8) = 60 bytes
        header_size = 148
        rec_fmt  = "<qddddqiq"
        rec_size = struct.calcsize(rec_fmt)   # 8+8+8+8+8+8+4+8 = 60
        col_order = ("time", "open", "high", "low", "close", "volume", "spread", "real_vol")
    else:
        print(f"[MT4-HST] Unknown version: {version} in {filepath}")
        return None

    n_records = (len(raw) - header_size) // rec_size
    if n_records <= 0:
        return None

    records = []
    offset = header_size
    for _ in range(n_records):
        if offset + rec_size > len(raw):
            break
        row = struct.unpack_from(rec_fmt, raw, offset)
        records.append(row)
        offset += rec_size

    if not records:
        return None

    df = pd.DataFrame(records, columns=col_order)

    # یکسان‌سازی ستون‌ها — اطمینان از اینکه high >= low
    if "high" in df.columns and "low" in df.columns:
        mask = df["high"] < df["low"]
        if mask.any():
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    df = df[df["close"] > 0]
    return df


def import_from_mt4(save_path: str = "data/XAUUSD_M15.csv") -> str:
    """
    مستقیماً فایل .hst متاتریدر 4 را می‌خواند — بدون نیاز به EA یا اسکریپت.
    اگر MT4 باز باشد فایل را flush می‌کند؛ بهتر است MT4 بسته باشد.
    """
    print("[MT4-HST] Searching for MT4 history files...")
    candidates = _find_mt4_hst_files(period=15)

    # همچنین CSV خروجی EA را هم چک کن (اگر قبلاً export کرده بود)
    common_root = os.path.join(
        os.environ.get("APPDATA", ""),
        "MetaQuotes", "Terminal", "Common", "Files", "python_bridge"
    )
    csv_candidates = [
        os.path.join(common_root, "XAUUSD_M15_history.csv"),
        os.path.join(common_root, "XAUUSDm_M15_history.csv"),
        os.path.join(common_root, "XAUUSD@_M15_history.csv"),
    ]

    df = None
    source_info = ""

    # اول HST فایل‌ها رو امتحان کن
    if candidates:
        # بزرگترین فایل رو انتخاب کن
        candidates.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)
        for fpath, sym, server in candidates:
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"[MT4-HST] Found: {fpath} ({size_mb:.1f} MB, server={server})")
            df_tmp = _parse_mt4_hst(fpath)
            if df_tmp is not None and len(df_tmp) > 100:
                df = df_tmp
                source_info = f"{fpath} ({len(df)} bars)"
                break

    # اگر HST کار نکرد، CSV export رو چک کن
    if df is None:
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                print(f"[MT4] Found CSV export: {csv_path}")
                df_tmp = pd.read_csv(csv_path)
                df_tmp.columns = [c.lower().strip() for c in df_tmp.columns]
                date_col = next((c for c in df_tmp.columns if "time" in c or "date" in c), df_tmp.columns[0])
                df_tmp = df_tmp.rename(columns={date_col: "datetime"})
                df_tmp["datetime"] = pd.to_datetime(df_tmp["datetime"])
                df_tmp = df_tmp[["datetime", "open", "high", "low", "close", "volume"]].dropna()
                if len(df_tmp) > 100:
                    df = df_tmp
                    source_info = csv_path
                    break

    if df is None:
        print("\n[Error] Could not read MT4 history data.")
        print("  MT4 history files searched in:")
        print(f"  {os.path.join(os.environ.get('APPDATA',''), 'MetaQuotes', 'Terminal', '*', 'history', '*', 'XAUUSD15.hst')}")
        print("\n  Possible fixes:")
        print("  1. Make sure MT4 is installed (WM Markets terminal)")
        print("  2. Open a XAUUSD M15 chart in MT4 and scroll back (Page Up) to load history")
        print("  3. Then close MT4 and run this command again")
        print("  4. OR use: python download_data.py --source yahoo15m")
        sys.exit(1)

    # ادغام با داده موجود (حفظ تاریخچه قدیمی)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        old_rows = len(df_old)
        df = pd.concat([df_old, df], ignore_index=True)
        print(f"[MT4] Merging with existing data ({old_rows} rows)...")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\n[MT4] Source : {source_info}")
    print(f"[MT4] Bars   : {len(df)}")
    print(f"[MT4] Range  : {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"[MT4] Saved  : {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Download gold market data")
    parser.add_argument("--source", choices=["yahoo", "yahooh1", "yahoo15m", "mt4", "mt5"], default="mt5",
                       help="yahoo=daily | yahoo15m=15m last 60 days (extend existing) | mt5=MetaTrader5")
    parser.add_argument("--timeframe", type=str, default="H1",
                       help="MT5 only: M1, M5, M15, M30, H1, H4, D1 (default: H1)")
    parser.add_argument("--symbol", type=str, default="XAUUSD@",
                       help="MT5 symbol (default: XAUUSD@ — WM Markets etc.)")
    parser.add_argument("--years", type=int, default=5,
                       help="Years of history (Yahoo and MT5; default 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV path (default: data/XAUUSD_<TF>.csv)")
    parser.add_argument("--bars", type=int, default=100000,
                       help="Number of bars to download from MT5 (default 100000)")

    args = parser.parse_args()

    if args.source == "mt4":
        path = args.output or DATA_PATH
        import_from_mt4(save_path=path)
    elif args.source == "yahooh1":
        path = args.output or "data/XAUUSD_H1.csv"
        download_yahoo_h1(save_path=path)
    elif args.source == "yahoo15m":
        existing = args.output or DATA_PATH
        extend_with_yahoo15m(existing_csv=existing)
    elif args.source == "yahoo":
        path = args.output or "data/XAUUSD_D1.csv"
        download_yahoo(years=args.years, save_path=path)
    else:
        path = args.output or f"data/{args.symbol}_{args.timeframe.upper()}.csv"
        download_mt5(
            timeframe=args.timeframe,
            symbol=args.symbol,
            save_path=path,
            bars=args.bars,
            years=args.years,
        )
        print(f"\n[OK] Saved to {path}")
        print("[Tip] backtest.py reads MT5 directly:  python backtest.py --symbol XAUUSD@")


if __name__ == "__main__":
    main()
