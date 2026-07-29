"""
Extended strategy detectors: ICT advanced, classic PA, systematic, order-flow proxy.
Each returns list of (direction, proximal, distal, tag).
"""
import numpy as np
import pandas as pd

import market_clock as _clock

EXT_DEFAULT = {
    "swing_lb": 20,
    "range_lb": 40,
    "vwap_lb": 48,
    "vp_lb": 48,
    "vp_pct": 0.70,
    "fib_ote_lo": 0.618,
    "fib_ote_hi": 0.786,
    "donchian_lb": 96,
    "bb_period": 20,
    "bb_std": 2.0,
    "kc_period": 20,
    "kc_mult": 1.5,
    "rsi_period": 14,
    "div_lb": 20,
    "ema_fast": 9,
    "ema_slow": 21,
    "adx_period": 14,
    "adx_min": 25,
    "cvd_lb": 30,
    # regression channel (draw a channel, trade its top/bottom)
    "ch_lb": 80,          # bars used to fit the channel (~20h on M15)
    "ch_k": 2.0,          # channel half-width in residual std devs
    # prior day/week levels (does price react to yesterday / last week?)
    "lvl_buf_atr": 0.15,  # how far beyond the swept level the stop sits
    # human-like tolerance (ATR multiples — 0 = exact legacy)
    "entry_tol_atr": 0.0,
    "detect_tol_atr": 0.0,
    "zone_tol_atr": 0.0,
    "invalidate_tol_atr": 0.0,
    # Session windows on the EXCHANGE clock (New York, per CLOCK.session_tz).
    # _hour() converts each bar's UTC label before comparing, so DST moves the
    # windows with the exchange instead of drifting an hour twice a year.
    "sb_hours": ((10, 11), (14, 15)),   # ICT silver bullet: AM and PM sessions
    "judas_hours": (2, 5),              # London-open false move
    "ny_open_hours": (8, 11),           # NY open killzone
    "asian_end_hour": 4,                # Asian range closes at London open
}


def _memo(B):
    """Derived-series cache that lives ON the Bars object.

    A module-level dict keyed on id(B) is unsafe here: live rebuilds Bars every
    cycle with a fixed length, CPython hands the new object the freed address,
    and the key silently matches the previous cycle's data. Hanging the memo off
    the instance ties it to the data it was computed from and frees it with the
    object instead of growing forever.
    """
    m = getattr(B, "_ext_memo", None)
    if m is None:
        m = {}
        try:
            B._ext_memo = m
        except AttributeError:
            return {}          # Bars-like object without a __dict__: skip memo
    return m


def _atr(B, i):
    return B.atr[i] if B.atr[i] > 0 else B.rng[i] + 1e-6


def _hour(B, i):
    """Hour on the exchange clock the session windows are written in.

    Bar labels are UTC; the windows below mean New York time, so the two have
    to be reconciled here or every session rule fires in the wrong session.
    """
    return _clock.session_hour(B.t[i])


def _in_hours(h, ranges):
    if isinstance(ranges[0], int):
        return ranges[0] <= h < ranges[1]
    return any(a <= h < b for a, b in ranges)


def _swing_low(B, i, lb):
    lo = max(0, i - lb)
    return float(B.l[lo:i + 1].min())


def _swing_high(B, i, lb):
    lo = max(0, i - lb)
    return float(B.h[lo:i + 1].max())


def _vwap(B, i, lb):
    lo = max(0, i - lb + 1)
    tp = B.tp[lo:i + 1]
    vol = B.vol[lo:i + 1]
    v = vol.sum()
    if v <= 0:
        return float(B.c[i])
    return float((tp * vol).sum() / v)


def _value_area(B, i, lb, pct=0.70):
    lo = max(0, i - lb + 1)
    tp = B.tp[lo:i + 1]
    vol = B.vol[lo:i + 1]
    if len(tp) < 10 or vol.sum() <= 0:
        return None
    hist, edges = np.histogram(tp, bins=30, weights=vol)
    order = np.argsort(hist)[::-1]
    total = hist.sum()
    acc = 0
    mask = np.zeros(len(hist), dtype=bool)
    for j in order:
        mask[j] = True
        acc += hist[j]
        if acc >= total * pct:
            break
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    lo_px = float(edges[idx.min()])
    hi_px = float(edges[idx.max() + 1])
    poc_i = int(order[0])
    poc = float((edges[poc_i] + edges[poc_i + 1]) / 2)
    return {"val": lo_px, "vah": hi_px, "poc": poc}


def _ema_series(B, period):
    m = _memo(B)
    key = (period, "ema")
    if key not in m:
        m[key] = pd.Series(B.c).ewm(span=period, adjust=False).mean().values
    return m[key]


def _rsi_series(B, period=14):
    m = _memo(B)
    key = (period, "rsi")
    if key not in m:
        c = pd.Series(B.c)
        d = c.diff()
        up = d.clip(lower=0).rolling(period).mean()
        dn = (-d.clip(upper=0)).rolling(period).mean()
        rs = up / dn.replace(0, np.nan)
        m[key] = (100 - 100 / (1 + rs)).fillna(50).values
    return m[key]


def _macd_series(B):
    m = _memo(B)
    if "macd" not in m:
        c = pd.Series(B.c)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        m["macd"] = (ema12 - ema26).values
    return m["macd"]


def _bb_kc(B, i, p):
    per = p.get("bb_period", 20)
    if i < per + 2:
        return None
    sl = B.c[i - per + 1:i + 1]
    mid = sl.mean()
    std = sl.std()
    if std <= 0:
        return None
    bb_u, bb_l = mid + p.get("bb_std", 2) * std, mid - p.get("bb_std", 2) * std
    tr = np.maximum(B.h[i - per + 1:i + 1] - B.l[i - per + 1:i + 1],
                    np.maximum(np.abs(B.h[i - per + 1:i + 1] - B.c[i - per:i]),
                               np.abs(B.l[i - per + 1:i + 1] - B.c[i - per:i])))
    atr = tr.mean()
    kc_u = mid + p.get("kc_mult", 1.5) * atr
    kc_l = mid - p.get("kc_mult", 1.5) * atr
    return bb_u, bb_l, kc_u, kc_l, mid


def _pivot_levels(B, i):
    """Prior session proxy: last 96 M15 bars (~24h)."""
    lb = 96
    if i < lb + 1:
        return None
    seg = slice(i - lb, i)
    h, l, c = B.h[seg].max(), B.l[seg].min(), B.c[i - 1]
    p = (h + l + c) / 3
    r1 = 2 * p - l
    s1 = 2 * p - h
    rng = h - l
    h3 = c + rng * 1.1 / 4
    l3 = c - rng * 1.1 / 4
    return {"p": p, "r1": r1, "s1": s1, "h3": h3, "l3": l3}


def _cvd_slope(B, i, lb):
    lo = max(1, i - lb + 1)
    delta = 0.0
    for j in range(lo, i + 1):
        if B.c[j] >= B.c[j - 1]:
            delta += B.vol[j]
        else:
            delta -= B.vol[j]
    return delta


def _reg_channel(B, lb, k):
    """Vectorised rolling linear-regression channel (mid / upper / lower).

    For each bar the last `lb` closes are fit with least squares; the channel
    is the fitted value at the current bar +/- k * std(residuals). No lookahead:
    bar i only uses closes up to and including i. Cached per (Bars, lb, k).
    """
    m = _memo(B)
    key = (lb, round(float(k), 3), "regchan")
    if key in m:
        return m[key]
    c = B.c.astype(float)
    n = len(c)
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    if n >= lb and lb >= 3:
        x = np.arange(lb, dtype=float)
        xbar = x.mean()
        sxx = float(((x - xbar) ** 2).sum())
        cs = pd.Series(c)
        sy = cs.rolling(lb).sum().values
        sy2 = (cs * cs).rolling(lb).sum().values
        # sum_j (j * y_j) inside each window, oldest weight 0 .. newest lb-1
        corr = np.correlate(c, x, mode="valid")          # index m -> window end m+lb-1
        sum_xy = np.full(n, np.nan)
        sum_xy[lb - 1:] = corr
        sxy = sum_xy - xbar * sy
        slope = sxy / sxx
        ybar = sy / lb
        pred = ybar + slope * ((lb - 1) - xbar)           # fitted value at current bar
        syy = sy2 - (sy * sy) / lb
        ss_res = np.maximum(syy - (sxy * sxy) / sxx, 0.0)
        resid = np.sqrt(ss_res / lb)
        mid = pred
        up = pred + k * resid
        lo = pred - k * resid
    m[key] = (mid, up, lo)
    return mid, up, lo


def _prior_levels(B):
    """Previous-DAY and previous-WEEK O/H/L/C aligned to every bar (no lookahead).

    At any bar the returned levels come from the LAST FULLY COMPLETED day/week,
    so they are known before the current bar forms. Cached per Bars.

    The day boundary comes from market_clock, not from UTC midnight: the level
    traders actually defend is the high of the session that ended at the New
    York close, and the backtest and the live bot have to agree on it.
    """
    m = _memo(B)
    if "priorlvl" in m:
        return m["priorlvl"]
    idx = pd.DatetimeIndex(B.t)
    df = pd.DataFrame({"o": B.o, "h": B.h, "l": B.l, "c": B.c}, index=idx)
    agg = {"o": "first", "h": "max", "l": "min", "c": "last"}

    dkey = _clock.day_key(idx)
    daily = df.groupby(dkey).agg(**{k: (k, v) for k, v in agg.items()})
    pday = daily.shift(1)
    wkey = _clock.week_key(idx)
    weekly = df.groupby(wkey).agg(**{k: (k, v) for k, v in agg.items()})
    pweek = weekly.shift(1)

    out = {
        "pdo": pday["o"].reindex(dkey).to_numpy(),
        "pdh": pday["h"].reindex(dkey).to_numpy(),
        "pdl": pday["l"].reindex(dkey).to_numpy(),
        "pdc": pday["c"].reindex(dkey).to_numpy(),
        "pwh": pweek["h"].reindex(wkey).to_numpy(),
        "pwl": pweek["l"].reindex(wkey).to_numpy(),
        "pwo": pweek["o"].reindex(wkey).to_numpy(),
        "pwc": pweek["c"].reindex(wkey).to_numpy(),
    }
    m["priorlvl"] = out
    return out


# ── ICT advanced ──────────────────────────────────────────────────────────

def detect_ict_breaker_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 5:
        return s
    for j in range(i - 15, i - 2):
        if B.body[j] >= 0:
            continue
        ob_lo, ob_hi = B.l[j], B.h[j]
        broke = any(B.c[k] < ob_lo for k in range(j + 1, i))
        if not broke:
            continue
        if B.l[i] <= ob_hi and B.c[i] > ob_lo and B.c[i] > B.o[i]:
            s.append(("long", ob_hi, ob_lo, "ICT-BRK"))
            return s
    return s


def detect_ict_breaker_bear(B, i, p=EXT_DEFAULT):
    s = []
    if i < 25:
        return s
    for j in range(i - 15, i - 2):
        if B.body[j] <= 0:
            continue
        ob_lo, ob_hi = B.l[j], B.h[j]
        broke = any(B.c[k] > ob_hi for k in range(j + 1, i))
        if not broke:
            continue
        if B.h[i] >= ob_lo and B.c[i] < ob_hi and B.c[i] < B.o[i]:
            s.append(("short", ob_lo, ob_hi, "ICT-BRK"))
            return s
    return s


def detect_ict_mitigation_bull(B, i, p=EXT_DEFAULT):
    s = []
    if i < 5:
        return s
    for j in range(i - 12, i - 2):
        if not (B.l[j] > B.h[j - 2]):
            continue
        gap_lo, gap_hi = B.h[j - 2], B.l[j]
        if any(B.l[k] < gap_lo for k in range(j + 1, i)):
            if gap_lo <= B.l[i] <= gap_hi and B.c[i] > B.o[i]:
                s.append(("long", gap_lo, B.h[j - 2], "ICT-MIT"))
                return s
    return s


def detect_ict_mitigation_bear(B, i, p=EXT_DEFAULT):
    s = []
    if i < 5:
        return s
    for j in range(i - 12, i - 2):
        if not (B.h[j] < B.l[j - 2]):
            continue
        gap_hi, gap_lo = B.l[j - 2], B.h[j]
        if any(B.h[k] > gap_hi for k in range(j + 1, i)):
            if gap_lo <= B.h[i] <= gap_hi and B.c[i] < B.o[i]:
                s.append(("short", gap_hi, B.l[j - 2], "ICT-MIT"))
                return s
    return s


def detect_ict_silver_bullet_bull(B, i, p=EXT_DEFAULT):
    s = []
    if not _in_hours(_hour(B, i), p.get("sb_hours", ((10, 11), (14, 15)))):
        return s
    atr = _atr(B, i)
    if B.body[i] < 0.6 * atr or B.c[i] <= B.o[i]:
        return s
    if B.l[i] > B.h[i - 2]:
        s.append(("long", B.l[i], B.h[i - 2], "ICT-SB"))
    elif i >= 2 and B.body[i - 1] < 0 and B.body[i] > 0.5 * atr:
        s.append(("long", B.l[i - 1], B.l[i - 1], "ICT-SB"))
    return s


def detect_ict_silver_bullet_bear(B, i, p=EXT_DEFAULT):
    s = []
    if not _in_hours(_hour(B, i), p.get("sb_hours", ((10, 11), (14, 15)))):
        return s
    atr = _atr(B, i)
    if -B.body[i] < 0.6 * atr or B.c[i] >= B.o[i]:
        return s
    if B.h[i] < B.l[i - 2]:
        s.append(("short", B.h[i], B.l[i - 2], "ICT-SB"))
    elif i >= 2 and B.body[i - 1] > 0 and -B.body[i] > 0.5 * atr:
        s.append(("short", B.h[i - 1], B.h[i - 1], "ICT-SB"))
    return s


def detect_ict_ote_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 3:
        return s
    lo_i = i - lb
    sh = B.h[lo_i:i].max()
    sl = B.l[lo_i:i].min()
    if sh <= sl:
        return s
    sh_i = lo_i + int(np.argmax(B.h[lo_i:i]))
    if sh_i >= i - 2:
        return s
    leg = sh - B.l[sh_i]
    if leg <= 0:
        return s
    retr = (sh - B.l[i]) / leg
    lo_f, hi_f = p.get("fib_ote_lo", 0.618), p.get("fib_ote_hi", 0.786)
    if lo_f * 0.95 <= retr <= hi_f * 1.05 and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], sl, "ICT-OTE"))
    return s


def detect_ict_ote_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 3:
        return s
    lo_i = i - lb
    sh = B.h[lo_i:i].max()
    sl = B.l[lo_i:i].min()
    if sh <= sl:
        return s
    sl_i = lo_i + int(np.argmin(B.l[lo_i:i]))
    if sl_i >= i - 2:
        return s
    leg = B.h[sl_i] - sl
    if leg <= 0:
        return s
    retr = (B.h[i] - sl) / leg
    lo_f, hi_f = p.get("fib_ote_lo", 0.618), p.get("fib_ote_hi", 0.786)
    if lo_f * 0.95 <= retr <= hi_f * 1.05 and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], sh, "ICT-OTE"))
    return s


def detect_ict_amd_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("range_lb", 40)
    if i < lb + 5:
        return s
    lo_i = i - lb
    rh, rl = B.h[lo_i:i - 5].max(), B.l[lo_i:i - 5].min()
    width = rh - rl
    atr = _atr(B, i)
    if width > 3 * atr or width < 0.8 * atr:
        return s
    sweep = B.l[i - 3:i].min() < rl - 0.1 * atr
    if sweep and B.c[i] > rh * 0.5 + rl * 0.5 and B.c[i] > B.o[i] and B.body[i] > 0.4 * atr:
        s.append(("long", B.l[i], B.l[i - 3:i].min(), "ICT-AMD"))
    return s


def detect_ict_amd_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("range_lb", 40)
    if i < lb + 5:
        return s
    lo_i = i - lb
    rh, rl = B.h[lo_i:i - 5].max(), B.l[lo_i:i - 5].min()
    width = rh - rl
    atr = _atr(B, i)
    if width > 3 * atr or width < 0.8 * atr:
        return s
    sweep = B.h[i - 3:i].max() > rh + 0.1 * atr
    if sweep and B.c[i] < rh * 0.5 + rl * 0.5 and B.c[i] < B.o[i] and -B.body[i] > 0.4 * atr:
        s.append(("short", B.h[i], B.h[i - 3:i].max(), "ICT-AMD"))
    return s


def detect_ict_sweep_mss_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 3:
        return s
    prior = _swing_low(B, i - 1, lb)
    if B.l[i] < prior and B.c[i] > prior:
        if B.l[i] > B.l[i - 1] and B.c[i] > B.h[i - 1]:
            s.append(("long", B.l[i], B.l[i], "ICT-MSS"))
    return s


def detect_ict_sweep_mss_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 3:
        return s
    prior = _swing_high(B, i - 1, lb)
    if B.h[i] > prior and B.c[i] < prior:
        if B.h[i] < B.h[i - 1] and B.c[i] < B.l[i - 1]:
            s.append(("short", B.h[i], B.h[i], "ICT-MSS"))
    return s


def detect_ict_judas_bull(B, i, p=EXT_DEFAULT):
    s = []
    h0, h1 = p.get("judas_hours", (7, 9))
    if not (h0 <= _hour(B, i) < h1):
        return s
    lb = 32
    if i < lb:
        return s
    asian_hi = B.h[i - lb:i - 4].max()
    asian_lo = B.l[i - lb:i - 4].min()
    if B.l[i - 2] < asian_lo and B.c[i] > asian_lo and B.c[i] > B.o[i]:
        s.append(("long", asian_lo, B.l[i - 2], "ICT-JUD"))
    return s


def detect_ict_judas_bear(B, i, p=EXT_DEFAULT):
    s = []
    h0, h1 = p.get("judas_hours", (7, 9))
    if not (h0 <= _hour(B, i) < h1):
        return s
    lb = 32
    if i < lb:
        return s
    asian_hi = B.h[i - lb:i - 4].max()
    asian_lo = B.l[i - lb:i - 4].min()
    if B.h[i - 2] > asian_hi and B.c[i] < asian_hi and B.c[i] < B.o[i]:
        s.append(("short", asian_hi, B.h[i - 2], "ICT-JUD"))
    return s


def detect_ict_ny_open_bull(B, i, p=EXT_DEFAULT):
    s = []
    h0, h1 = p.get("ny_open_hours", (13, 15))
    if not (h0 <= _hour(B, i) < h1):
        return s
    if i < 8:
        return s
    or_hi = B.h[i - 4:i].max()
    or_lo = B.l[i - 4:i].min()
    if B.c[i - 1] > or_hi and or_lo <= B.l[i] <= or_hi and B.c[i] > B.o[i]:
        s.append(("long", or_hi, or_lo, "ICT-NYO"))
    return s


def detect_ict_ny_open_bear(B, i, p=EXT_DEFAULT):
    s = []
    h0, h1 = p.get("ny_open_hours", (13, 15))
    if not (h0 <= _hour(B, i) < h1):
        return s
    if i < 8:
        return s
    or_hi = B.h[i - 4:i].max()
    or_lo = B.l[i - 4:i].min()
    if B.c[i - 1] < or_lo and or_lo <= B.h[i] <= or_hi and B.c[i] < B.o[i]:
        s.append(("short", or_lo, or_hi, "ICT-NYO"))
    return s


# ── Classic PA ────────────────────────────────────────────────────────────

def detect_mp_val_long(B, i, p=EXT_DEFAULT):
    s = []
    va = _value_area(B, i, p.get("vp_lb", 48), p.get("vp_pct", 0.70))
    if va is None:
        return s
    if va["val"] <= B.l[i] <= va["val"] + _atr(B, i) * 0.3 and B.c[i] > B.o[i]:
        s.append(("long", va["val"], va["val"] - _atr(B, i), "MP-VAL"))
    return s


def detect_mp_vah_short(B, i, p=EXT_DEFAULT):
    s = []
    va = _value_area(B, i, p.get("vp_lb", 48), p.get("vp_pct", 0.70))
    if va is None:
        return s
    if va["vah"] - _atr(B, i) * 0.3 <= B.h[i] <= va["vah"] and B.c[i] < B.o[i]:
        s.append(("short", va["vah"], va["vah"] + _atr(B, i), "MP-VAH"))
    return s


def detect_vwap_long(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("vwap_lb", 48)
    if i < lb + 2:
        return s
    vw = _vwap(B, i, lb)
    vw_p = _vwap(B, i - 1, lb)
    zt = p.get("zone_tol_atr", 0) * _atr(B, i)
    if (B.c[i - 1] < vw_p + zt and B.l[i] <= vw + zt and B.h[i] >= vw - zt
            and B.c[i] > vw - zt and B.c[i] > B.o[i]):
        s.append(("long", vw, vw - _atr(B, i) * 0.5, "VWAP"))
    return s


def detect_vwap_short(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("vwap_lb", 48)
    if i < lb + 2:
        return s
    vw = _vwap(B, i, lb)
    vw_p = _vwap(B, i - 1, lb)
    zt = p.get("zone_tol_atr", 0) * _atr(B, i)
    if (B.c[i - 1] > vw_p - zt and B.l[i] <= vw + zt and B.h[i] >= vw - zt
            and B.c[i] < vw + zt and B.c[i] < B.o[i]):
        s.append(("short", vw, vw + _atr(B, i) * 0.5, "VWAP"))
    return s


def detect_vsa_stopping_vol_bull(B, i, p=EXT_DEFAULT):
    s = []
    if i < 20:
        return s
    vol_ma = B.vol[i - 20:i].mean()
    if vol_ma <= 0:
        return s
    spread = B.rng[i]
    if B.vol[i] > 1.8 * vol_ma and spread > 1.2 * _atr(B, i):
        if B.c[i] < B.o[i] and (B.c[i] - B.l[i]) / spread > 0.6:
            s.append(("long", B.c[i], B.l[i], "VSA-SV"))
    return s


def detect_vsa_upthrust_bear(B, i, p=EXT_DEFAULT):
    s = []
    if i < 20:
        return s
    vol_ma = B.vol[i - 20:i].mean()
    if vol_ma <= 0:
        return s
    spread = B.rng[i]
    if B.vol[i] > 1.8 * vol_ma and spread > 1.2 * _atr(B, i):
        if B.c[i] > B.o[i] and (B.h[i] - B.c[i]) / spread > 0.6:
            s.append(("short", B.c[i], B.h[i], "VSA-UT"))
    return s


def detect_bk_hs_short(B, i, p=EXT_DEFAULT):
    s = []
    lb = 40
    if i < lb:
        return s
    seg = B.h[i - lb:i]
    peaks = []
    for j in range(2, len(seg) - 2):
        if seg[j] >= seg[j - 1] and seg[j] >= seg[j - 2] and seg[j] >= seg[j + 1]:
            peaks.append((j, seg[j]))
    if len(peaks) < 2:
        return s
    p1, p2 = peaks[-2], peaks[-1]
    if abs(p1[1] - p2[1]) > 0.5 * _atr(B, i):
        return s
    neck = B.l[i - lb + p1[0]:i - lb + p2[0]].min()
    if B.c[i] < neck and B.c[i] < B.o[i]:
        s.append(("short", neck, p2[1], "BK-HS"))
    return s


def detect_bk_triangle_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = 30
    if i < lb:
        return s
    hs = B.h[i - lb:i - 5]
    ls = B.l[i - lb:i - 5]
    if hs[-1] >= hs[0] or ls[-1] <= ls[0]:
        return s
    res = hs.max()
    if B.c[i] > res and B.body[i] > 0.4 * _atr(B, i):
        s.append(("long", res, ls.min(), "BK-TRI"))
    return s


def detect_bk_flag_bull(B, i, p=EXT_DEFAULT):
    s = []
    if i < 25:
        return s
    pole = B.c[i - 15] - B.c[i - 25]
    if pole < 2 * _atr(B, i):
        return s
    fh, fl = B.h[i - 10:i].max(), B.l[i - 10:i].min()
    if fh - fl > 1.5 * _atr(B, i):
        return s
    if B.c[i] > fh and B.c[i] > B.o[i]:
        s.append(("long", fh, fl, "BK-FLAG"))
    return s


def detect_turtle_long(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("donchian_lb", 96)
    if i < lb + 1:
        return s
    hi = B.h[i - lb:i].max()
    if B.c[i] > hi and B.c[i - 1] <= hi:
        s.append(("long", hi, B.l[i - lb:i].min(), "TURT"))
    return s


def detect_turtle_short(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("donchian_lb", 96)
    if i < lb + 1:
        return s
    lo = B.l[i - lb:i].min()
    if B.c[i] < lo and B.c[i - 1] >= lo:
        s.append(("short", lo, B.h[i - lb:i].max(), "TURT"))
    return s


def detect_nf_big_shadow_bull(B, i, p=EXT_DEFAULT):
    s = []
    r = B.rng[i]
    if r <= 0:
        return s
    body = abs(B.body[i])
    lower = B.c[i] - B.l[i] if B.c[i] >= B.o[i] else B.o[i] - B.l[i]
    if body / r < 0.35 and lower / r > 0.55 and lower > 2 * body:
        if B.l[i] <= _swing_low(B, i - 1, 15) + 0.2 * _atr(B, i):
            s.append(("long", B.c[i], B.l[i], "NF-BS"))
    return s


def detect_nf_big_shadow_bear(B, i, p=EXT_DEFAULT):
    s = []
    r = B.rng[i]
    if r <= 0:
        return s
    body = abs(B.body[i])
    upper = B.h[i] - B.c[i] if B.c[i] <= B.o[i] else B.h[i] - B.o[i]
    if body / r < 0.35 and upper / r > 0.55 and upper > 2 * body:
        if B.h[i] >= _swing_high(B, i - 1, 15) - 0.2 * _atr(B, i):
            s.append(("short", B.c[i], B.h[i], "NF-BS"))
    return s


# ── Systematic ────────────────────────────────────────────────────────────

def detect_ichi_kumo_bull(B, i, p=EXT_DEFAULT):
    s = []
    if i < 52:
        return s
    c = pd.Series(B.c[: i + 1])
    tenkan = (c.rolling(9).max() + c.rolling(9).min()) / 2
    kijun = (c.rolling(26).max() + c.rolling(26).min()) / 2
    sa = ((c.rolling(9).max() + c.rolling(9).min()) / 2).shift(26)
    sb = ((c.rolling(52).max() + c.rolling(52).min()) / 2).shift(26)
    cloud_top = np.maximum(sa.iloc[-1], sb.iloc[-1])
    if B.c[i] > cloud_top and tenkan.iloc[-2] <= kijun.iloc[-2] and tenkan.iloc[-1] > kijun.iloc[-1]:
        s.append(("long", B.l[i], min(sa.iloc[-1], sb.iloc[-1]), "ICHI"))
    return s


def detect_ichi_kumo_bear(B, i, p=EXT_DEFAULT):
    s = []
    if i < 52:
        return s
    c = pd.Series(B.c[: i + 1])
    tenkan = (c.rolling(9).max() + c.rolling(9).min()) / 2
    kijun = (c.rolling(26).max() + c.rolling(26).min()) / 2
    sa = ((c.rolling(9).max() + c.rolling(9).min()) / 2).shift(26)
    sb = ((c.rolling(52).max() + c.rolling(52).min()) / 2).shift(26)
    cloud_bot = np.minimum(sa.iloc[-1], sb.iloc[-1])
    if B.c[i] < cloud_bot and tenkan.iloc[-2] >= kijun.iloc[-2] and tenkan.iloc[-1] < kijun.iloc[-1]:
        s.append(("short", B.h[i], max(sa.iloc[-1], sb.iloc[-1]), "ICHI"))
    return s


def detect_squeeze_bull(B, i, p=EXT_DEFAULT):
    s = []
    bk = _bb_kc(B, i, p)
    if bk is None:
        return s
    bb_u, bb_l, kc_u, kc_l, mid = bk
    if not (bb_u < kc_u and bb_l > kc_l):
        return s
    if B.c[i] > bb_u and B.c[i] > B.o[i]:
        s.append(("long", B.c[i], mid, "SQZ"))
    return s


def detect_squeeze_bear(B, i, p=EXT_DEFAULT):
    s = []
    bk = _bb_kc(B, i, p)
    if bk is None:
        return s
    bb_u, bb_l, kc_u, kc_l, mid = bk
    if not (bb_u < kc_u and bb_l > kc_l):
        return s
    if B.c[i] < bb_l and B.c[i] < B.o[i]:
        s.append(("short", B.c[i], mid, "SQZ"))
    return s


def detect_div_rsi_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("div_lb", 20)
    if i < lb + 5:
        return s
    rsi = _rsi_series(B, p.get("rsi_period", 14))
    lo_i = i - lb
    p1, p2 = lo_i + int(np.argmin(B.l[lo_i:i - 3])), i
    if B.l[p2] < B.l[p1] and rsi[p2] > rsi[p1] and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], B.l[p2], "DIV-RSI"))
    return s


def detect_div_rsi_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("div_lb", 20)
    if i < lb + 5:
        return s
    rsi = _rsi_series(B, p.get("rsi_period", 14))
    lo_i = i - lb
    p1, p2 = lo_i + int(np.argmax(B.h[lo_i:i - 3])), i
    if B.h[p2] > B.h[p1] and rsi[p2] < rsi[p1] and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], B.h[p2], "DIV-RSI"))
    return s


def detect_div_macd_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("div_lb", 20)
    if i < lb + 5:
        return s
    macd = _macd_series(B)
    lo_i = i - lb
    p1 = lo_i + int(np.argmin(B.l[lo_i:i - 3]))
    if B.l[i] < B.l[p1] and macd[i] > macd[p1] and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], B.l[i], "DIV-MACD"))
    return s


def detect_div_macd_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("div_lb", 20)
    if i < lb + 5:
        return s
    macd = _macd_series(B)
    lo_i = i - lb
    p1 = lo_i + int(np.argmax(B.h[lo_i:i - 3]))
    if B.h[i] > B.h[p1] and macd[i] < macd[p1] and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], B.h[i], "DIV-MACD"))
    return s


def detect_pivot_long(B, i, p=EXT_DEFAULT):
    s = []
    piv = _pivot_levels(B, i)
    if piv is None:
        return s
    if abs(B.l[i] - piv["s1"]) < 0.3 * _atr(B, i) and B.c[i] > B.o[i]:
        s.append(("long", piv["s1"], piv["s1"] - _atr(B, i), "PIVOT"))
    return s


def detect_pivot_short(B, i, p=EXT_DEFAULT):
    s = []
    piv = _pivot_levels(B, i)
    if piv is None:
        return s
    if abs(B.h[i] - piv["r1"]) < 0.3 * _atr(B, i) and B.c[i] < B.o[i]:
        s.append(("short", piv["r1"], piv["r1"] + _atr(B, i), "PIVOT"))
    return s


def detect_cam_long(B, i, p=EXT_DEFAULT):
    s = []
    piv = _pivot_levels(B, i)
    if piv is None:
        return s
    if abs(B.l[i] - piv["l3"]) < 0.25 * _atr(B, i) and B.c[i] > B.o[i]:
        s.append(("long", piv["l3"], piv["l3"] - _atr(B, i), "CAM"))
    return s


def detect_cam_short(B, i, p=EXT_DEFAULT):
    s = []
    piv = _pivot_levels(B, i)
    if piv is None:
        return s
    if abs(B.h[i] - piv["h3"]) < 0.25 * _atr(B, i) and B.c[i] < B.o[i]:
        s.append(("short", piv["h3"], piv["h3"] + _atr(B, i), "CAM"))
    return s


def detect_ma_cross_bull(B, i, p=EXT_DEFAULT):
    s = []
    if i < p.get("ema_slow", 21) + 2:
        return s
    ef = _ema_series(B, p.get("ema_fast", 9))
    es = _ema_series(B, p.get("ema_slow", 21))
    if ef[i - 1] <= es[i - 1] and ef[i] > es[i] and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], es[i], "MA-X"))
    return s


def detect_ma_cross_bear(B, i, p=EXT_DEFAULT):
    s = []
    if i < p.get("ema_slow", 21) + 2:
        return s
    ef = _ema_series(B, p.get("ema_fast", 9))
    es = _ema_series(B, p.get("ema_slow", 21))
    if ef[i - 1] >= es[i - 1] and ef[i] < es[i] and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], es[i], "MA-X"))
    return s


def detect_adx_trend_bull(B, i, p=EXT_DEFAULT):
    s = []
    per = p.get("adx_period", 14)
    if i < per * 3:
        return s
    h, l, c = pd.Series(B.h[: i + 1]), pd.Series(B.l[: i + 1]), pd.Series(B.c[: i + 1])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.rolling(per).mean()
    pdi = 100 * pd.Series(pdm).rolling(per).mean() / atr
    mdi = 100 * pd.Series(mdm).rolling(per).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.rolling(per).mean().iloc[-1]
    if adx >= p.get("adx_min", 25) and pdi.iloc[-1] > mdi.iloc[-1] and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], B.l[i] - _atr(B, i), "ADX"))
    return s


def detect_adx_trend_bear(B, i, p=EXT_DEFAULT):
    s = []
    per = p.get("adx_period", 14)
    if i < per * 3:
        return s
    h, l, c = pd.Series(B.h[: i + 1]), pd.Series(B.l[: i + 1]), pd.Series(B.c[: i + 1])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.rolling(per).mean()
    pdi = 100 * pd.Series(pdm).rolling(per).mean() / atr
    mdi = 100 * pd.Series(mdm).rolling(per).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.rolling(per).mean().iloc[-1]
    if adx >= p.get("adx_min", 25) and mdi.iloc[-1] > pdi.iloc[-1] and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], B.h[i] + _atr(B, i), "ADX"))
    return s


# ── Order flow proxy (tick volume) ────────────────────────────────────────

def detect_cvd_div_bull(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("cvd_lb", 30)
    if i < lb + 5:
        return s
    lo_i = i - lb
    p1 = lo_i + int(np.argmin(B.l[lo_i:i - 3]))
    cvd1 = _cvd_slope(B, p1, lb // 2)
    cvd2 = _cvd_slope(B, i, lb // 2)
    if B.l[i] < B.l[p1] and cvd2 > cvd1 and B.c[i] > B.o[i]:
        s.append(("long", B.l[i], B.l[p1], "CVD"))
    return s


def detect_cvd_div_bear(B, i, p=EXT_DEFAULT):
    s = []
    lb = p.get("cvd_lb", 30)
    if i < lb + 5:
        return s
    lo_i = i - lb
    p1 = lo_i + int(np.argmax(B.h[lo_i:i - 3]))
    cvd1 = _cvd_slope(B, p1, lb // 2)
    cvd2 = _cvd_slope(B, i, lb // 2)
    if B.h[i] > B.h[p1] and cvd2 < cvd1 and B.c[i] < B.o[i]:
        s.append(("short", B.h[i], B.h[p1], "CVD"))
    return s


# ── Regression channel (trade the top / bottom of a drawn channel) ─────────

def detect_channel_rev_bull(B, i, p=EXT_DEFAULT):
    """Fade the LOWER band: price pokes the channel bottom and rejects up."""
    s = []
    lb = p.get("ch_lb", 80)
    if i < lb:
        return s
    mid, up, lo = _reg_channel(B, lb, p.get("ch_k", 2.0))
    band_lo, band_mid = lo[i], mid[i]
    if not np.isfinite(band_lo) or not np.isfinite(band_mid):
        return s
    dt = p.get("detect_tol_atr", 0) * _atr(B, i)
    if B.l[i] <= band_lo + dt and B.c[i] > band_lo - dt and B.c[i] > B.o[i]:
        s.append(("long", band_lo, band_lo - 0.75 * _atr(B, i), "CH-REV"))
    return s


def detect_channel_rev_bear(B, i, p=EXT_DEFAULT):
    """Fade the UPPER band: price pokes the channel top and rejects down."""
    s = []
    lb = p.get("ch_lb", 80)
    if i < lb:
        return s
    mid, up, lo = _reg_channel(B, lb, p.get("ch_k", 2.0))
    band_up, band_mid = up[i], mid[i]
    if not np.isfinite(band_up) or not np.isfinite(band_mid):
        return s
    dt = p.get("detect_tol_atr", 0) * _atr(B, i)
    if B.h[i] >= band_up - dt and B.c[i] < band_up + dt and B.c[i] < B.o[i]:
        s.append(("short", band_up, band_up + 0.75 * _atr(B, i), "CH-REV"))
    return s


def detect_channel_bo_bull(B, i, p=EXT_DEFAULT):
    """Break ABOVE the channel top; enter on a retest, stop back at mid-line."""
    s = []
    lb = p.get("ch_lb", 80)
    if i < lb + 1:
        return s
    mid, up, lo = _reg_channel(B, lb, p.get("ch_k", 2.0))
    if not (np.isfinite(up[i]) and np.isfinite(up[i - 1]) and np.isfinite(mid[i])):
        return s
    if B.c[i] > up[i] and B.c[i - 1] <= up[i - 1]:
        s.append(("long", up[i], mid[i], "CH-BO"))
    return s


def detect_channel_bo_bear(B, i, p=EXT_DEFAULT):
    """Break BELOW the channel bottom; enter on a retest, stop back at mid-line."""
    s = []
    lb = p.get("ch_lb", 80)
    if i < lb + 1:
        return s
    mid, up, lo = _reg_channel(B, lb, p.get("ch_k", 2.0))
    if not (np.isfinite(lo[i]) and np.isfinite(lo[i - 1]) and np.isfinite(mid[i])):
        return s
    if B.c[i] < lo[i] and B.c[i - 1] >= lo[i - 1]:
        s.append(("short", lo[i], mid[i], "CH-BO"))
    return s


# ── Prior day / week levels (does price react to yesterday / last week?) ────

def detect_pd_low_sweep_bull(B, i, p=EXT_DEFAULT):
    """Sweep BELOW yesterday's low, then reclaim it -> long the reaction."""
    s = []
    lv = _prior_levels(B)
    pdl = lv["pdl"][i]
    if not np.isfinite(pdl):
        return s
    buf = p.get("lvl_buf_atr", 0.15) * _atr(B, i)
    if B.l[i] < pdl and B.c[i] > pdl and B.c[i] > B.o[i]:
        s.append(("long", pdl, B.l[i] - buf, "PD-SWEEP"))
    return s


def detect_pd_high_sweep_bear(B, i, p=EXT_DEFAULT):
    """Sweep ABOVE yesterday's high, then reject back below -> short."""
    s = []
    lv = _prior_levels(B)
    pdh = lv["pdh"][i]
    if not np.isfinite(pdh):
        return s
    buf = p.get("lvl_buf_atr", 0.15) * _atr(B, i)
    if B.h[i] > pdh and B.c[i] < pdh and B.c[i] < B.o[i]:
        s.append(("short", pdh, B.h[i] + buf, "PD-SWEEP"))
    return s


def detect_pw_low_sweep_bull(B, i, p=EXT_DEFAULT):
    """Sweep BELOW last week's low, then reclaim it -> long the reaction."""
    s = []
    lv = _prior_levels(B)
    pwl = lv["pwl"][i]
    if not np.isfinite(pwl):
        return s
    buf = p.get("lvl_buf_atr", 0.15) * _atr(B, i)
    if B.l[i] < pwl and B.c[i] > pwl and B.c[i] > B.o[i]:
        s.append(("long", pwl, B.l[i] - buf, "PW-SWEEP"))
    return s


def detect_pw_high_sweep_bear(B, i, p=EXT_DEFAULT):
    """Sweep ABOVE last week's high, then reject back below -> short."""
    s = []
    lv = _prior_levels(B)
    pwh = lv["pwh"][i]
    if not np.isfinite(pwh):
        return s
    buf = p.get("lvl_buf_atr", 0.15) * _atr(B, i)
    if B.h[i] > pwh and B.c[i] < pwh and B.c[i] < B.o[i]:
        s.append(("short", pwh, B.h[i] + buf, "PW-SWEEP"))
    return s


def detect_pd_close_reclaim_bull(B, i, p=EXT_DEFAULT):
    """Cross back ABOVE yesterday's close -> reaction long (retest entry)."""
    s = []
    lv = _prior_levels(B)
    pdc = lv["pdc"][i]
    if not np.isfinite(pdc):
        return s
    if B.c[i] > pdc and B.c[i - 1] <= pdc and B.c[i] > B.o[i]:
        s.append(("long", pdc, pdc - 0.75 * _atr(B, i), "PD-CLOSE"))
    return s


def detect_pd_close_reclaim_bear(B, i, p=EXT_DEFAULT):
    """Cross back BELOW yesterday's close -> reaction short (retest entry)."""
    s = []
    lv = _prior_levels(B)
    pdc = lv["pdc"][i]
    if not np.isfinite(pdc):
        return s
    if B.c[i] < pdc and B.c[i - 1] >= pdc and B.c[i] < B.o[i]:
        s.append(("short", pdc, pdc + 0.75 * _atr(B, i), "PD-CLOSE"))
    return s


EXT_FAMILIES = {
    "ict": {"label": "ICT / SMC Advanced", "rules": {
        "ICT_BRK_L": detect_ict_breaker_bull, "ICT_BRK_S": detect_ict_breaker_bear,
        "ICT_MIT_L": detect_ict_mitigation_bull, "ICT_MIT_S": detect_ict_mitigation_bear,
        "ICT_SB_L": detect_ict_silver_bullet_bull, "ICT_SB_S": detect_ict_silver_bullet_bear,
        "ICT_OTE_L": detect_ict_ote_bull, "ICT_OTE_S": detect_ict_ote_bear,
        "ICT_AMD_L": detect_ict_amd_bull, "ICT_AMD_S": detect_ict_amd_bear,
        "ICT_MSS_L": detect_ict_sweep_mss_bull, "ICT_MSS_S": detect_ict_sweep_mss_bear,
        "ICT_JUD_L": detect_ict_judas_bull, "ICT_JUD_S": detect_ict_judas_bear,
        "ICT_NYO_L": detect_ict_ny_open_bull, "ICT_NYO_S": detect_ict_ny_open_bear,
    }},
    "classic": {"label": "Classic Price Action", "rules": {
        "MP_VAL_L": detect_mp_val_long, "MP_VAH_S": detect_mp_vah_short,
        "VWAP_L": detect_vwap_long, "VWAP_S": detect_vwap_short,
        "VSA_SV_L": detect_vsa_stopping_vol_bull, "VSA_UT_S": detect_vsa_upthrust_bear,
        "BK_HS_S": detect_bk_hs_short,
        "BK_TRI_L": detect_bk_triangle_bull, "BK_FLAG_L": detect_bk_flag_bull,
        "TURT_L": detect_turtle_long, "TURT_S": detect_turtle_short,
        "NF_BS_L": detect_nf_big_shadow_bull, "NF_BS_S": detect_nf_big_shadow_bear,
    }},
    "systematic": {"label": "Systematic / Indicators", "rules": {
        "ICHI_L": detect_ichi_kumo_bull, "ICHI_S": detect_ichi_kumo_bear,
        "SQZ_L": detect_squeeze_bull, "SQZ_S": detect_squeeze_bear,
        "DIV_RSI_L": detect_div_rsi_bull, "DIV_RSI_S": detect_div_rsi_bear,
        "DIV_MACD_L": detect_div_rsi_bull, "DIV_MACD_S": detect_div_rsi_bear,
        "PIVOT_L": detect_pivot_long, "PIVOT_S": detect_pivot_short,
        "CAM_L": detect_cam_long, "CAM_S": detect_cam_short,
        "MA_X_L": detect_ma_cross_bull, "MA_X_S": detect_ma_cross_bear,
        "ADX_L": detect_adx_trend_bull, "ADX_S": detect_adx_trend_bear,
    }},
    "orderflow": {"label": "Order Flow Proxy (tick vol)", "rules": {
        "CVD_L": detect_cvd_div_bull, "CVD_S": detect_cvd_div_bear,
    }},
    "channel": {"label": "Regression Channel", "rules": {
        "CH_REV_L": detect_channel_rev_bull, "CH_REV_S": detect_channel_rev_bear,
        "CH_BO_L": detect_channel_bo_bull, "CH_BO_S": detect_channel_bo_bear,
    }},
    "levels": {"label": "Prior Day/Week Levels", "rules": {
        "PDL_SW_L": detect_pd_low_sweep_bull, "PDH_SW_S": detect_pd_high_sweep_bear,
        "PWL_SW_L": detect_pw_low_sweep_bull, "PWH_SW_S": detect_pw_high_sweep_bear,
        "PDC_RC_L": detect_pd_close_reclaim_bull, "PDC_RC_S": detect_pd_close_reclaim_bear,
    }},
}

# fix MACD div mapping
EXT_FAMILIES["systematic"]["rules"]["DIV_MACD_L"] = detect_div_macd_bull
EXT_FAMILIES["systematic"]["rules"]["DIV_MACD_S"] = detect_div_macd_bear

EXT_DETECTORS = {}
EXT_RULE_META = {}
for fam_key, fam in EXT_FAMILIES.items():
    for rule_key, fn in fam["rules"].items():
        EXT_DETECTORS[rule_key] = fn
        EXT_RULE_META[rule_key] = {"family": fam_key, "label": fam["label"]}

EXT_ALL_RULES = list(EXT_DETECTORS.keys())
