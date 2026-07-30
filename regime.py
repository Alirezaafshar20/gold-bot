"""
Market-regime detection — lower lag than a plain moving average.

A moving average is just a lagging mean, so it confirms turns late (exactly where
we bleed). This module offers several detectors and classifies every HTF bar into
one of three regimes using ONLY past data (no lookahead):

    TREND_UP    strong directional up   move
    TREND_DOWN  strong directional down move
    RANGE       choppy / sideways (low directional efficiency)

Detectors
---------
  kaufman   Efficiency Ratio (net move / path length) for strength + slope sign
            for direction. Low lag, no smoothing. RECOMMENDED.
  slope     Normalised linear-regression slope over the window (least squares).
            Direction = slope sign, strength = |slope| vs threshold.
  donchian  Position inside the N-bar high/low channel. Breakout/structure based,
            essentially no lag on new extremes.
  ma        Plain EMA position (the laggy baseline, kept for comparison).
  hybrid    ER (strength) + regression-slope (direction) run through a
            HYSTERESIS band + a few-bar CONFIRMATION. Low lag like ER, but does
            not whipsaw at the threshold like a single cut-off. RECOMMENDED when
            the plain kaufman flips too often. See _classify_hybrid.
  mtf       H4 swing-structure macro bias + H1 ER local state. Direction from
            H4 (HH/HL vs LH/LL + EMA fallback); strength/chop from H1 ER.
            In RANGE, macro_at() exposes bullish/bearish bias for rule routing.

Anti-whipsaw (works with any detector via post-processing)
----------------------------------------------------------
  hysteresis : enter a trend at `er_hi`, leave it only below `er_lo` (er_hi>er_lo)
               so the state is "sticky" between the two bands.
  confirm    : a new state must persist `confirm` bars before it commits.
  Both are OFF by default so existing calibration output is unchanged; turn them
  on with detector="hybrid" (built in) or hysteresis=True / confirm>1.

Usage
-----
    import regime
    rm = regime.RegimeMap(htf_bars_df, detector="kaufman", win=40)
    reg = rm.at(signal_time)          # -> "TREND_UP" / "TREND_DOWN" / "RANGE"

`htf_bars_df` must have a DatetimeIndex and a 'close' column (H1 or H4 bars).
"""
import numpy as np
import pandas as pd

REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE")


def efficiency_ratio(close, win):
    """Kaufman ER in [0,1]; 1 = perfectly efficient trend, 0 = pure chop."""
    s = pd.Series(np.asarray(close, dtype=float))
    change = (s - s.shift(win)).abs()
    path = s.diff().abs().rolling(win).sum()
    er = change / path.replace(0.0, np.nan)
    return er.to_numpy()


def reg_slope(close, win):
    """Least-squares slope per bar, normalised by price (per-bar % move)."""
    s = np.asarray(close, dtype=float)
    n = len(s)
    out = np.full(n, np.nan)
    x = np.arange(win, dtype=float)
    x -= x.mean()
    denom = (x * x).sum()
    for i in range(win, n):
        y = s[i - win + 1:i + 1]
        slope = float((x * (y - y.mean())).sum() / denom)
        out[i] = slope / s[i] if s[i] else 0.0
    return out


def _to_ohlc_df(src):
    """Normalise Bars / DataFrame to OHLC with DatetimeIndex."""
    if hasattr(src, "c") and hasattr(src, "t"):
        idx = pd.DatetimeIndex(src.t)
        return pd.DataFrame({
            "high": np.asarray(src.h, dtype=float),
            "low": np.asarray(src.l, dtype=float),
            "close": np.asarray(src.c, dtype=float),
        }, index=idx)
    df = src if isinstance(src, pd.DataFrame) else pd.DataFrame(src)
    out = df.copy()
    if "close" not in out.columns:
        out["close"] = out.iloc[:, -1]
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]
    return out


def _pivot_points(high, low, left=2, right=2):
    """Return lists of (index, value) for swing highs and lows.

    Vectorised over sliding windows: the per-bar Python loop this replaces was
    the single hottest path in a replay, since the regime map rebuilds it for
    every bar of every higher-timeframe candle.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    w = left + right + 1
    if n < w:
        return [], []
    idx = np.arange(left, n - right)
    prev = np.maximum(idx - 1, 0)
    m = len(idx)
    # Fold the window with w slice-wise max/min passes. For the tiny windows
    # this runs on (w=5), that beats building a strided view per call.
    hmax = high[0:m].copy()
    lmin = low[0:m].copy()
    for k in range(1, w):
        np.maximum(hmax, high[k:k + m], out=hmax)
        np.minimum(lmin, low[k:k + m], out=lmin)
    is_ph = (high[idx] >= hmax) & (high[idx] > high[prev])
    is_pl = (low[idx] <= lmin) & (low[idx] < low[prev])
    ph = [(int(i), float(high[i])) for i in idx[is_ph]]
    pl = [(int(i), float(low[i])) for i in idx[is_pl]]
    return ph, pl


def macro_structure_bias(high, low, close, i, lookback=60, pivot=2, ema_period=20):
    """+1 bull / -1 bear / 0 neutral from H4 swing structure (causal)."""
    j0 = max(0, i - lookback)
    h = high[j0:i + 1]
    l = low[j0:i + 1]
    c = close[j0:i + 1]
    if len(c) < pivot * 2 + 5:
        return 0
    ph, pl = _pivot_points(h, l, left=pivot, right=pivot)
    bias = 0
    if len(ph) >= 2 and len(pl) >= 2:
        hh = ph[-1][1] > ph[-2][1]
        hl = pl[-1][1] > pl[-2][1]
        lh = ph[-1][1] < ph[-2][1]
        ll = pl[-1][1] < pl[-2][1]
        if hh and hl:
            bias = 1
        elif lh and ll:
            bias = -1
        elif hh and not ll:
            bias = 1
        elif ll and not hh:
            bias = -1
    if bias == 0 and len(c) >= ema_period + 3:
        ema = pd.Series(c).ewm(span=ema_period, adjust=False).mean().to_numpy()
        slope = ema[-1] - ema[max(0, len(ema) - 6)]
        px = c[-1]
        if px >= ema[-1] and slope >= 0:
            bias = 1
        elif px <= ema[-1] and slope <= 0:
            bias = -1
        elif px > ema[-1]:
            bias = 1
        elif px < ema[-1]:
            bias = -1
    return bias


class MtfRegimeMap:
    """H4 macro direction + H1 local ER; labels on the H1 timeline.

    Optional V-shape / shock extensions (all causal):
      structure_break  H1 close breaking the brk_win-bar high/low overrides the
                       (slower) H4 macro label immediately — catches fast
                       reversals half a day before H4 pivots confirm them.
      shock_at(t)      realized-volatility ratio: TR sum of the last shock_win
                       H1 bars vs its rolling median baseline. >1 = hotter than
                       normal. Detects violent whipsaw days that ER calls
                       "RANGE" because net displacement is small.
      macro_age_at(t)  hours since the H4 macro bias last flipped — a fresh
                       flip means the new direction is unproven.
    """

    def __init__(self, h1_src, h4_src, h4_win=30, h1_win=40,
                 er_hi=0.32, er_lo=0.20, confirm=2, h4_lookback=60,
                 structure_break=False, brk_win=40,
                 shock_win=24, shock_baseline=720):
        self.detector = "mtf"
        self.win = h1_win
        self.er_hi = er_hi
        self.er_lo = er_lo
        self.confirm = confirm
        h1 = _to_ohlc_df(h1_src)
        h4 = _to_ohlc_df(h4_src)
        self.t = np.asarray(h1.index.values, dtype="datetime64[ns]")
        h1_close = h1["close"].to_numpy(dtype=float)
        h1_high = h1["high"].to_numpy(dtype=float)
        h1_low = h1["low"].to_numpy(dtype=float)
        h4_high = h4["high"].to_numpy(dtype=float)
        h4_low = h4["low"].to_numpy(dtype=float)
        h4_close = h4["close"].to_numpy(dtype=float)
        h4_t = np.asarray(h4.index.values, dtype="datetime64[ns]")
        n = len(h1_close)
        h1_er = efficiency_ratio(h1_close, h1_win)
        h1_sl = reg_slope(h1_close, h1_win)
        macro = np.zeros(n, dtype=int)
        raw = np.array(["RANGE"] * n, dtype=object)
        # Four H1 bars share one H4 bar, so the macro bias for a given H4 index
        # is the same answer computed four times. The arrays are fixed for the
        # whole loop, so memoising it is exact, not an approximation.
        macro_by_h4: dict[int, int] = {}
        for i in range(n):
            j = int(np.searchsorted(h4_t, self.t[i], side="right")) - 1
            if j < 0:
                continue
            bias = macro_by_h4.get(j)
            if bias is None:
                bias = macro_structure_bias(
                    h4_high, h4_low, h4_close, j, lookback=h4_lookback)
                macro_by_h4[j] = bias
            macro[i] = bias
            if structure_break and i >= brk_win:
                if h1_close[i] > h1_high[i - brk_win:i].max():
                    raw[i] = "TREND_UP"
                    macro[i] = 1
                    continue
                if h1_close[i] < h1_low[i - brk_win:i].min():
                    raw[i] = "TREND_DOWN"
                    macro[i] = -1
                    continue
            m = macro[i]
            er = h1_er[i]
            sl = h1_sl[i]
            if np.isnan(er):
                continue
            chop = er < er_lo
            if m > 0:
                raw[i] = "RANGE" if chop else "TREND_UP"
            elif m < 0:
                raw[i] = "RANGE" if chop else "TREND_DOWN"
            elif er >= er_hi and not np.isnan(sl):
                raw[i] = "TREND_UP" if sl > 0 else "TREND_DOWN"
        self.macro = macro
        self.labels = RegimeMap._debounce(raw, max(confirm, 1))

        # shock ratio: rolling shock_win-bar RANGE vs its rolling median.
        # NOTE: bar-size metrics (ATR / TR sums) miss V-days — hourly bars stay
        # normal while the day sweeps $100+. Windowed range catches exactly that.
        roll_hi = pd.Series(h1_high).rolling(shock_win).max()
        roll_lo = pd.Series(h1_low).rolling(shock_win).min()
        rng = roll_hi - roll_lo
        base = rng.shift(1).rolling(shock_baseline,
                                    min_periods=max(shock_win * 4, 60)).median()
        self._shock = (rng / base.replace(0.0, np.nan)).to_numpy()

        # hours since the macro bias last flipped sign (inf until first flip)
        self._macro_age_h = np.full(n, np.inf)
        last_flip = None
        for i in range(1, n):
            if macro[i] != macro[i - 1] and macro[i] != 0:
                last_flip = i
            if last_flip is not None:
                self._macro_age_h[i] = float(
                    (self.t[i] - self.t[last_flip]) / np.timedelta64(1, "h"))

    def _idx(self, signal_time):
        ts = np.datetime64(pd.Timestamp(signal_time), "ns")
        return int(np.searchsorted(self.t, ts, side="right")) - 1

    def at(self, signal_time):
        idx = self._idx(signal_time)
        if idx < 0 or idx >= len(self.labels):
            return "RANGE"
        return self.labels[idx]

    def macro_at(self, signal_time):
        idx = self._idx(signal_time)
        if idx < 0 or idx >= len(self.macro):
            return 0
        return int(self.macro[idx])

    def shock_at(self, signal_time):
        """Realized-vol ratio vs baseline (1.0 = normal; nan-safe)."""
        idx = self._idx(signal_time)
        if idx < 0 or idx >= len(self._shock):
            return 1.0
        v = self._shock[idx]
        return float(v) if np.isfinite(v) else 1.0

    def macro_age_at(self, signal_time):
        """Hours since the last macro flip (inf if it never flipped)."""
        idx = self._idx(signal_time)
        if idx < 0 or idx >= len(self._macro_age_h):
            return float("inf")
        return float(self._macro_age_h[idx])

    def coverage(self):
        from collections import Counter
        c = Counter(self.labels[self.win:])
        tot = sum(c.values()) or 1
        return {r: 100.0 * c.get(r, 0) / tot for r in REGIMES}

    def macro_coverage(self):
        m = self.macro[self.win:]
        tot = len(m) or 1
        return {
            "BULL": 100.0 * np.sum(m > 0) / tot,
            "BEAR": 100.0 * np.sum(m < 0) / tot,
            "NEUTRAL": 100.0 * np.sum(m == 0) / tot,
        }


def build_mtf_map(htf_context, h4_win=30, h1_win=40,
                  er_hi=0.32, er_lo=0.20, confirm=2, h4_lookback=60,
                  structure_break=False, brk_win=40,
                  shock_win=24, shock_baseline=720):
    """Build MtfRegimeMap from strategy HTF context (needs H1 + H4)."""
    if not htf_context:
        return None
    h1 = htf_context.get("H1")
    h4 = htf_context.get("H4")
    if h1 is None or h4 is None:
        return None
    try:
        return MtfRegimeMap(h1, h4, h4_win=h4_win, h1_win=h1_win,
                            er_hi=er_hi, er_lo=er_lo, confirm=confirm,
                            h4_lookback=h4_lookback,
                            structure_break=structure_break, brk_win=brk_win,
                            shock_win=shock_win, shock_baseline=shock_baseline)
    except Exception as exc:
        # Returning None here silently degrades every caller to "RANGE always",
        # so make the failure loud rather than losing the regime gate quietly.
        print(f"[regime] build_mtf_map FAILED ({type(exc).__name__}: {exc}) — "
              f"caller will fall back to RANGE for every bar")
        return None


class RegimeMap:
    """Pre-computes a regime label for every HTF bar, then maps any time to it."""

    def __init__(self, htf_df, detector="kaufman", win=40,
                 er_trend=0.35, slope_k=0.0006, ema_period=20, chan_band=0.25,
                 er_hi=0.40, er_lo=0.25, confirm=1, hysteresis=False):
        self.detector = detector
        self.win = win
        self.er_hi = er_hi
        self.er_lo = er_lo
        self.confirm = confirm
        self.hysteresis = hysteresis
        # Accept a pandas DataFrame, a strategy.Bars object, or a dict.
        if hasattr(htf_df, "c") and hasattr(htf_df, "t"):
            close = np.asarray(htf_df.c, dtype=float)
            self.t = np.asarray(htf_df.t, dtype="datetime64[ns]")
            df = pd.DataFrame({"close": close,
                               "high": np.asarray(htf_df.h, dtype=float),
                               "low": np.asarray(htf_df.l, dtype=float)})
        else:
            df = htf_df if isinstance(htf_df, pd.DataFrame) else pd.DataFrame(htf_df)
            col = "close" if "close" in df.columns else df.columns[-1]
            close = df[col].to_numpy(dtype=float)
            self.t = np.asarray(df.index.values, dtype="datetime64[ns]")
        if detector == "hybrid":
            self.labels = self._classify_hybrid(close, win, er_hi, er_lo,
                                                 max(confirm, 2), slope_k)
        else:
            lab = self._classify(df, close, detector, win, er_trend,
                                  slope_k, ema_period, chan_band)
            if hysteresis or confirm > 1:
                lab = self._debounce(lab, confirm)
            self.labels = lab

    @staticmethod
    def _debounce(lab, confirm):
        """Commit a new regime only after it persists `confirm` bars (anti-noise)."""
        if confirm <= 1:
            return lab
        out = lab.copy()
        state = lab[0]
        pend = None
        cnt = 0
        for i in range(len(lab)):
            d = lab[i]
            if d == state:
                pend, cnt = None, 0
            else:
                if d == pend:
                    cnt += 1
                else:
                    pend, cnt = d, 1
                if cnt >= confirm:
                    state, pend, cnt = d, None, 0
            out[i] = state
        return out

    @staticmethod
    def _classify_hybrid(close, win, er_hi, er_lo, confirm, slope_k):
        """ER strength + slope direction, with a hysteresis band and confirmation.

        - strength from Kaufman ER: enter a trend at er>=er_hi, drop back to
          RANGE only when er<er_lo (sticky between the bands -> no whipsaw).
        - direction from the regression slope sign, cross-checked with the
          net win-bar momentum sign (agreement preferred, slope as fallback).
        - a state change must survive `confirm` consecutive bars.
        """
        n = len(close)
        er = efficiency_ratio(close, win)
        sl = reg_slope(close, win)
        mom = close - np.concatenate([np.full(win, np.nan), close[:-win]])
        out = np.array(["RANGE"] * n, dtype=object)
        state = "RANGE"
        pend, cnt = None, 0
        for i in range(n):
            if np.isnan(er[i]) or np.isnan(sl[i]):
                out[i] = state
                continue
            strong = er[i] >= er_hi
            weak = er[i] < er_lo
            if sl[i] > 0 and (np.isnan(mom[i]) or mom[i] >= 0):
                dirsign = 1
            elif sl[i] < 0 and (np.isnan(mom[i]) or mom[i] <= 0):
                dirsign = -1
            else:
                dirsign = 1 if sl[i] > 0 else -1
            if state == "RANGE":
                desired = ("TREND_UP" if dirsign > 0 else "TREND_DOWN") \
                    if strong else "RANGE"
            else:
                cur = 1 if state == "TREND_UP" else -1
                if weak:
                    desired = "RANGE"
                elif dirsign == -cur and strong:
                    desired = "TREND_UP" if dirsign > 0 else "TREND_DOWN"
                else:
                    desired = state
            if desired == state:
                pend, cnt = None, 0
            else:
                if desired == pend:
                    cnt += 1
                else:
                    pend, cnt = desired, 1
                if cnt >= confirm:
                    state, pend, cnt = desired, None, 0
            out[i] = state
        return out

    @staticmethod
    def _classify(df, close, detector, win, er_trend, slope_k, ema_period, chan_band):
        n = len(close)
        lab = np.array(["RANGE"] * n, dtype=object)

        if detector == "kaufman":
            er = efficiency_ratio(close, win)
            mom = close - np.concatenate([np.full(win, np.nan), close[:-win]])
            for i in range(n):
                if np.isnan(er[i]) or er[i] < er_trend:
                    continue
                lab[i] = "TREND_UP" if mom[i] > 0 else "TREND_DOWN"

        elif detector == "slope":
            sl = reg_slope(close, win)
            for i in range(n):
                if np.isnan(sl[i]) or abs(sl[i]) < slope_k:
                    continue
                lab[i] = "TREND_UP" if sl[i] > 0 else "TREND_DOWN"

        elif detector == "donchian":
            hi = df.get("high", df[df.columns[-1]]).to_numpy(dtype=float)
            lo = df.get("low", df[df.columns[-1]]).to_numpy(dtype=float)
            for i in range(win, n):
                ch = hi[i - win:i].max()
                cl = lo[i - win:i].min()
                rng = ch - cl
                if rng <= 0:
                    continue
                pos = (close[i] - cl) / rng
                if pos >= 1.0 - chan_band:
                    lab[i] = "TREND_UP"
                elif pos <= chan_band:
                    lab[i] = "TREND_DOWN"

        elif detector == "ma":
            ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().to_numpy()
            for i in range(ema_period, n):
                lab[i] = "TREND_UP" if close[i] >= ema[i] else "TREND_DOWN"

        else:
            raise ValueError(f"unknown detector: {detector}")
        return lab

    def at(self, signal_time):
        ts = np.datetime64(pd.Timestamp(signal_time), "ns")
        idx = int(np.searchsorted(self.t, ts, side="right")) - 1
        if idx < 0 or idx >= len(self.labels):
            return "RANGE"
        return self.labels[idx]

    def coverage(self):
        from collections import Counter
        c = Counter(self.labels[self.win:])
        tot = sum(c.values()) or 1
        return {r: 100.0 * c.get(r, 0) / tot for r in REGIMES}
