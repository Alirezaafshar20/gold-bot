"""
MetaTrader 5 data access — single source for live + backtest.
No CSV files required. Works with broker-specific symbols (e.g. XAUUSD@).
"""
import numpy as np
import pandas as pd

import market_clock as _clock

TF_MAP = {
    "M1":  "TIMEFRAME_M1",
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1":  "TIMEFRAME_H1",
    "H4":  "TIMEFRAME_H4",
    "D1":  "TIMEFRAME_D1",
}

TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}

DEFAULT_SYMBOL = "XAUUSD@"
FALLBACK_SYMBOLS = ["XAUUSD@", "XAUUSD", "XAUUSDm", "XAUUSD.", "GOLD"]
WARMUP_BARS = 520       # VP window = 480; extra bars for indicator warm-up


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("MetaTrader5 not installed. Run: pip install MetaTrader5")
    return mt5


def connect():
    """Initialize MT5. Returns the mt5 module. MT5 terminal must be running."""
    mt5 = _mt5()
    if not mt5.initialize():
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {err}. Is MetaTrader 5 open and logged in?")
    return mt5


def shutdown(mt5=None):
    mt5 = mt5 or _mt5()
    mt5.shutdown()


def resolve_symbol(preferred=None, mt5=None):
    """
    Find a tradable gold symbol on this broker.
    Tries the preferred name first, then common variants, then any symbol containing XAU.
    """
    mt5 = mt5 or connect()
    tried = []
    candidates = []
    if preferred:
        candidates.append(preferred)
    for s in FALLBACK_SYMBOLS:
        if s not in candidates:
            candidates.append(s)

    for sym in candidates:
        tried.append(sym)
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        if not info.visible and not mt5.symbol_select(sym, True):
            continue
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            continue
        return sym

    all_syms = mt5.symbols_get() or []
    for info in sorted(all_syms, key=lambda x: x.name):
        if "XAU" in info.name.upper():
            if mt5.symbol_select(info.name, True):
                tick = mt5.symbol_info_tick(info.name)
                if tick is not None:
                    return info.name

    raise RuntimeError(
        f"Could not find a gold symbol. Tried: {tried}. "
        "Pass --symbol exactly as shown in MT5 Market Watch (e.g. XAUUSD@)."
    )


def fetch_bars(symbol, tf_str, count=100000, mt5=None):
    """
    Download the most recent `count` bars from MT5.
    Returns a DataFrame indexed by time with columns open/high/low/close/volume.

    The index is normalised to UTC. MT5 hands back the broker's SERVER wall
    clock (measured UTC+3 here) dressed up as an epoch, while the historical
    CSVs are true UTC. Leaving that gap open shifted every session-gated rule by
    three hours and moved the day boundary, so "yesterday's high" only matched
    the backtest on half of all bars.
    """
    mt5 = mt5 or connect()
    tf_str = tf_str.upper()
    if tf_str not in TF_MAP:
        raise ValueError(f"Unknown timeframe {tf_str}. Use: {', '.join(TF_MAP)}")

    tf_value = getattr(mt5, TF_MAP[tf_str])
    count = min(int(count), 99999)          # MT5 hard limit per request

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol '{symbol}' in Market Watch")

    rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, count)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"No {tf_str} data for {symbol}. MT5 error: {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df.index = _clock.to_utc(df.index, _clock.server_utc_offset(mt5))
    df.index.name = "time"
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def tf_minutes(tf_str):
    tf_str = tf_str.upper()
    if tf_str not in TF_MINUTES:
        raise ValueError(f"Unknown timeframe {tf_str}. Use: {', '.join(TF_MINUTES)}")
    return TF_MINUTES[tf_str]


def _bars_for_period(tf_min, days, warmup=WARMUP_BARS):
    """How many signal-TF bars to request for `days` calendar days + warm-up."""
    test_bars = int(np.ceil(days * 24 * 60 / tf_min))
    return min(99999, test_bars + warmup)


def _m1_bars_for_period(tf_min, days, warmup=WARMUP_BARS):
    """M1 bars covering the same wall-clock span (+ warm-up in minutes)."""
    return min(99999, int(days * 24 * 60) + warmup * tf_min)


def fetch_pair(symbol, signal_tf, m1_count=None, signal_count=None, mt5=None,
               days=None, date_from=None, date_to=None):
    """
    Fetch signal-timeframe + M1 bars for honest back-testing.

    Period filter (optional — only TRADES inside the window are reported):
      days=N       last N calendar days up to the latest bar
      date_from/to explicit range (YYYY-MM-DD or YYYY-MM-DD HH:MM)

    Extra warm-up bars are fetched before the window so VP/indicators are valid.
    Returns: (symbol, signal_df, m1_df, cutoff_ts)
      cutoff_ts = only count trades with entry time >= cutoff_ts (None = all)
    """
    mt5 = mt5 or connect()
    sym = resolve_symbol(symbol, mt5)
    tf = signal_tf.upper()
    tf_min = tf_minutes(tf)

    if days is not None:
        signal_count = signal_count or _bars_for_period(tf_min, days)
        m1_count = m1_count or _m1_bars_for_period(tf_min, days)
    else:
        signal_count = signal_count or 100000
        m1_count = m1_count or 100000

    m1 = fetch_bars(sym, "M1", count=m1_count, mt5=mt5)
    sig = fetch_bars(sym, tf, count=signal_count, mt5=mt5)

    lo, hi = m1.index[0], m1.index[-1]
    sig = sig[(sig.index >= lo) & (sig.index <= hi)]
    if len(sig) < 60:
        raise RuntimeError(
            f"Not enough overlapping {tf} bars ({len(sig)}). "
            f"Increase --days or --bars (broker may have limited history)."
        )

    cutoff = None
    if date_to is not None:
        hi_cut = pd.Timestamp(date_to)
    else:
        hi_cut = sig.index.max()
    if date_from is not None:
        cutoff = pd.Timestamp(date_from)
    elif days is not None:
        cutoff = hi_cut - pd.Timedelta(days=int(days))

    return sym, sig, m1, cutoff


def fetch_htf_bars(symbol, days=None, signal_count=None, mt5=None,
                   tfs=("H4", "H1")):
    """Download higher-TF bars aligned with backtest period (for HTF take-profit)."""
    mt5 = mt5 or connect()
    sym = resolve_symbol(symbol, mt5)
    out = {}
    for tf in tfs:
        tf = tf.upper()
        tf_min = tf_minutes(tf)
        if days is not None:
            count = _bars_for_period(tf_min, days, warmup=120)
        else:
            count = signal_count or 100000
        out[tf] = fetch_bars(sym, tf, count=count, mt5=mt5)
    return sym, out
