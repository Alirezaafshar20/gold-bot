"""
Load Dukascopy 1-minute history and shape it for the existing backtest engine.

Why Dukascopy: the live broker (WM Markets) only keeps ~100 days of M1, so a
2022-onward calibration cannot use broker M1 for intrabar exits. Dukascopy
serves free M1 back many years. We use ONE consistent feed: resample its M1
to M15 or M5 for SIGNALS (--tf), and use the raw M1 for EXIT resolution.

Expected files (one per symbol), UTC, produced by dukascopy-node:
    data/dukascopy/XAUUSD_M1.csv
    data/dukascopy/US500_M1.csv
    data/dukascopy/US30_M1.csv
    data/dukascopy/BRENT_M1.csv
    data/dukascopy/BTCUSD_M1.csv

Accepted CSV headers (auto-detected):
    timestamp/time/date , open , high , low , close , volume
timestamp may be ms-epoch, s-epoch, or ISO string.
"""
import glob
import os
import pandas as pd

DUKA_DIR = "data/dukascopy"

# Project key -> dukascopy-node instrument id (for the download command).
DUKA_INSTRUMENT = {
    "XAUUSD": "xauusd",
    "US500":  "usa500idxusd",
    "US30":   "usa30idxusd",
    "BRENT":  "brentcmdusd",
    "BTCUSD": "btcusd",
}


def _read_csv_any(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[k] for k in ("timestamp", "time", "date", "datetime", "gmt time")
                 if k in cols), None)
    if tcol is None:
        # dukascopy-node sometimes writes the time as the first unnamed column
        tcol = df.columns[0]
    o = cols.get("open"); h = cols.get("high")
    lo = cols.get("low"); c = cols.get("close")
    v = cols.get("volume") or cols.get("vol")
    if None in (o, h, lo, c):
        raise ValueError(f"{path}: missing OHLC columns (got {list(df.columns)})")

    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        # epoch: decide ms vs s by magnitude
        unit = "ms" if ts.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True, errors="coerce")
    out = pd.DataFrame({
        "open": df[o].astype(float),
        "high": df[h].astype(float),
        "low": df[lo].astype(float),
        "close": df[c].astype(float),
        "volume": (df[v].astype(float) if v else 0.0),
    })
    out.index = idx.dt.tz_convert(None)  # store as naive UTC (matches MT5 feed)
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_m1(key, data_dir=DUKA_DIR):
    """Read {KEY}_M1.csv, or merge yearly chunks {KEY}_M1_YYYY.csv if present."""
    single = os.path.join(data_dir, f"{key}_M1.csv")
    chunks = sorted(glob.glob(os.path.join(data_dir, f"{key}_M1_*.csv")))
    paths = ([single] if os.path.exists(single) else []) + chunks
    if not paths:
        raise FileNotFoundError(
            f"No data for {key} in {data_dir} (looked for {key}_M1.csv and "
            f"{key}_M1_YYYY.csv). Download from Dukascopy first "
            f"(instrument '{DUKA_INSTRUMENT.get(key, '?')}', 1-minute, from 2022-04-01).")
    frames = []
    for p in paths:
        try:
            if os.path.getsize(p) == 0:
                continue                      # skip 0-byte failed downloads
            df = _read_csv_any(p)
            if len(df):
                frames.append(df)
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"  [dukascopy_loader] skipping unreadable {os.path.basename(p)}: {e}")
    if not frames:
        raise FileNotFoundError(
            f"No usable rows for {key} in {data_dir} (all matching files were "
            f"empty/unreadable).")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def to_signal(m1, tf="M15"):
    """Resample M1 -> signal TF the same way MT5 stamps bars (open time, left-closed)."""
    t = str(tf).upper().replace(" ", "")
    if t in ("M5", "5M", "5"):
        rule = "5min"
    elif t in ("M15", "15M", "15", ""):
        rule = "15min"
    else:
        raise ValueError(f"Unsupported signal tf {tf!r}; use M5 or M15")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = m1.resample(rule, label="left", closed="left", origin="epoch").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def to_m15(m1):
    """Resample M1 -> M15 (legacy alias)."""
    return to_signal(m1, "M15")


def load_pair(key, start=None, end=None, data_dir=DUKA_DIR, tf="M15"):
    """Return (signal_df, m1_df) for a symbol, optionally clipped to a window.

    `tf` is the signal timeframe (M5 or M15). Exits always use raw M1.
    """
    m1 = load_m1(key, data_dir)
    if start is not None:
        m1 = m1[m1.index >= pd.Timestamp(start)]
    if end is not None:
        m1 = m1[m1.index <= pd.Timestamp(end)]
    if m1.empty:
        raise ValueError(f"{key}: no M1 rows in window {start}..{end}")
    sig = to_signal(m1, tf)
    return sig, m1


def coverage_report(data_dir=DUKA_DIR):
    print("=" * 78)
    print("  DUKASCOPY M1 COVERAGE")
    print("=" * 78)
    print(f"  {'key':<10}{'rows':>12}{'start':>14}{'end':>14}{'gaps>1h':>9}")
    print("  " + "-" * 60)
    for key in DUKA_INSTRUMENT:
        try:
            m1 = load_m1(key, data_dir)
        except FileNotFoundError:
            print(f"  {key:<10}  (missing)")
            continue
        gaps = (m1.index.to_series().diff() > pd.Timedelta("1h")).sum()
        print(f"  {key:<10}{len(m1):>12}{str(m1.index[0]):>14.10}"
              f"{str(m1.index[-1]):>14.10}{gaps:>9}")
    print("=" * 78)


if __name__ == "__main__":
    coverage_report()
