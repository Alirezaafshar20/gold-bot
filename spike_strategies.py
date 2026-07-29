"""
Pre-spike / volatility-expansion detectors.

Goal: enter on compression → expansion BEFORE or AS a spike starts,
while the main SMC rules wait for post-spike pullbacks.

Patterns:
  SPIKE-SQS  — BB inside KC squeeze, then expansion break (momentum release)
  SPIKE-BRK  — ATR compression + range break + impulse candle
  SPIKE-VOL  — volume surge + range expansion (when tick volume available)
"""
import numpy as np
import pandas as pd

SPIKE_DEFAULT = {
    "compress_lb": 24,       # ~6h on M15
    "atr_lb": 100,
    "max_compress_ratio": 0.82,   # recent ATR vs long median — must be quiet
    "expand_mult": 1.35,          # current range vs compressed ATR
    "body_mult": 0.75,            # min body size vs ATR on impulse bar
    "range_lb": 20,               # range box for breakout
    "vol_lb": 30,
    "vol_surge": 1.6,             # volume vs median
    "bb_period": 20,
    "bb_std": 2.0,
    "kc_period": 20,
    "kc_mult": 1.5,
}


def _atr(B, i):
    return B.atr[i] if B.atr[i] > 0 else B.rng[i] + 1e-6


def _compressed(B, i, p):
    lb = p.get("compress_lb", 24)
    atr_lb = p.get("atr_lb", 100)
    if i < lb + 5:
        return False, 0.0
    short_med = float(np.median(B.atr[max(0, i - lb):i]))
    long_med = float(np.median(B.atr[max(0, i - atr_lb):i])) or short_med
    if long_med <= 0:
        return False, short_med
    ratio = short_med / long_med
    return ratio <= p.get("max_compress_ratio", 0.82), short_med


def _expansion_bar(B, i, ref_atr, p, bullish=True):
    rng = B.rng[i]
    if rng < p.get("expand_mult", 1.35) * ref_atr:
        return False
    body = B.body[i]
    atr_i = _atr(B, i)
    if abs(body) < p.get("body_mult", 0.75) * atr_i:
        return False
    if bullish and body <= 0:
        return False
    if not bullish and body >= 0:
        return False
    return True


def _bb_kc(B, i, p):
    per = p.get("bb_period", 20)
    if i < per + 2:
        return None
    c = B.c[max(0, i - per + 1):i + 1]
    mid = float(np.mean(c))
    std = float(np.std(c)) or 1e-6
    bb_u = mid + p.get("bb_std", 2.0) * std
    bb_l = mid - p.get("bb_std", 2.0) * std
    tr = []
    for j in range(max(1, i - per + 1), i + 1):
        tr.append(max(B.h[j] - B.l[j], abs(B.h[j] - B.c[j - 1]), abs(B.l[j] - B.c[j - 1])))
    atr = float(np.mean(tr)) if tr else _atr(B, i)
    kc_u = mid + p.get("kc_mult", 1.5) * atr
    kc_l = mid - p.get("kc_mult", 1.5) * atr
    return bb_u, bb_l, kc_u, kc_l, mid


def _was_squeezed(B, i, p, lookback=8):
    """True if any of last `lookback` bars had BB inside KC."""
    for j in range(max(p.get("bb_period", 20) + 2, i - lookback), i):
        bk = _bb_kc(B, j, p)
        if bk is None:
            continue
        bb_u, bb_l, kc_u, kc_l, _ = bk
        if bb_u < kc_u and bb_l > kc_l:
            return True
    return False


def detect_spike_sqs_bull(B, i, p=SPIKE_DEFAULT):
    s = []
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _was_squeezed(B, i, p):
        return s
    bk = _bb_kc(B, i, p)
    if bk is None:
        return s
    bb_u, bb_l, kc_u, kc_l, mid = bk
    if not _expansion_bar(B, i, ref, p, bullish=True):
        return s
    if B.c[i] <= bb_u:
        return s
    s.append(("long", B.c[i], mid, "SPIKE-SQS"))
    return s


def detect_spike_sqs_bear(B, i, p=SPIKE_DEFAULT):
    s = []
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _was_squeezed(B, i, p):
        return s
    bk = _bb_kc(B, i, p)
    if bk is None:
        return s
    bb_u, bb_l, kc_u, kc_l, mid = bk
    if not _expansion_bar(B, i, ref, p, bullish=False):
        return s
    if B.c[i] >= bb_l:
        return s
    s.append(("short", B.c[i], mid, "SPIKE-SQS"))
    return s


def detect_spike_brk_bull(B, i, p=SPIKE_DEFAULT):
    s = []
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _expansion_bar(B, i, ref, p, bullish=True):
        return s
    lb = p.get("range_lb", 20)
    if i < lb + 2:
        return s
    box_hi = float(B.h[i - lb:i].max())
    if B.c[i] <= box_hi:
        return s
    recent_lo = float(B.l[i - lb:i + 1].min())
    s.append(("long", box_hi, recent_lo, "SPIKE-BRK"))
    return s


def detect_spike_brk_bear(B, i, p=SPIKE_DEFAULT):
    s = []
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _expansion_bar(B, i, ref, p, bullish=False):
        return s
    lb = p.get("range_lb", 20)
    if i < lb + 2:
        return s
    box_lo = float(B.l[i - lb:i].min())
    if B.c[i] >= box_lo:
        return s
    recent_hi = float(B.h[i - lb:i + 1].max())
    s.append(("short", box_lo, recent_hi, "SPIKE-BRK"))
    return s


def detect_spike_vol_bull(B, i, p=SPIKE_DEFAULT):
    s = []
    if B.vol[i] <= 0:
        return s
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _expansion_bar(B, i, ref, p, bullish=True):
        return s
    vl = p.get("vol_lb", 30)
    med_v = float(np.median(B.vol[max(0, i - vl):i])) or 1.0
    if B.vol[i] < p.get("vol_surge", 1.6) * med_v:
        return s
    s.append(("long", B.c[i], B.l[i], "SPIKE-VOL"))
    return s


def detect_spike_vol_bear(B, i, p=SPIKE_DEFAULT):
    s = []
    if B.vol[i] <= 0:
        return s
    ok, ref = _compressed(B, i, p)
    if not ok:
        return s
    if not _expansion_bar(B, i, ref, p, bullish=False):
        return s
    vl = p.get("vol_lb", 30)
    med_v = float(np.median(B.vol[max(0, i - vl):i])) or 1.0
    if B.vol[i] < p.get("vol_surge", 1.6) * med_v:
        return s
    s.append(("short", B.c[i], B.h[i], "SPIKE-VOL"))
    return s


SPIKE_DETECTORS = {
    "SPIKE_SQS_L": detect_spike_sqs_bull,
    "SPIKE_SQS_S": detect_spike_sqs_bear,
    "SPIKE_BRK_L": detect_spike_brk_bull,
    "SPIKE_BRK_S": detect_spike_brk_bear,
    "SPIKE_VOL_L": detect_spike_vol_bull,
    "SPIKE_VOL_S": detect_spike_vol_bear,
}

SPIKE_RULE_META = {
    "SPIKE_SQS_L": {"tag": "SPIKE-SQS", "label": "Spike squeeze release (long)"},
    "SPIKE_SQS_S": {"tag": "SPIKE-SQS", "label": "Spike squeeze release (short)"},
    "SPIKE_BRK_L": {"tag": "SPIKE-BRK", "label": "Vol compression range break (long)"},
    "SPIKE_BRK_S": {"tag": "SPIKE-BRK", "label": "Vol compression range break (short)"},
    "SPIKE_VOL_L": {"tag": "SPIKE-VOL", "label": "Volume surge expansion (long)"},
    "SPIKE_VOL_S": {"tag": "SPIKE-VOL", "label": "Volume surge expansion (short)"},
}

SPIKE_RULE_KEYS = frozenset(SPIKE_DETECTORS.keys())
