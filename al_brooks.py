"""
Al Brooks price action pattern detectors (codified from his trilogy + encyclopedia).

Each detector returns list of (direction, proximal, distal, tag).
Patterns are grouped by Brooks market phase: trend continuation, reversal,
breakout, trading range. Not every encyclopedia slide is codifiable from OHLC;
this module implements the core, rule-based subset.
"""
import numpy as np
import pandas as pd

AB_DEFAULT = {
    "ema_period": 20,
    "swing_lb": 20,
    "range_lb": 40,
    "pullback_lb": 15,
    "min_body_ratio": 0.45,
    "dt_tolerance": 0.35,   # ATR fraction for double top/bottom equality
    "wedge_push_lb": 8,
}


def _atr(B, i):
    return B.atr[i] if B.atr[i] > 0 else B.rng[i] + 1e-6


def _ema(B, i, period=20):
    if i < 2:
        return float(B.c[i])
    return float(pd.Series(B.c[: i + 1]).ewm(span=period, adjust=False).mean().iloc[-1])


def _bull_trend(B, i, p):
    lb = p.get("swing_lb", 20)
    if i < lb + 5:
        return False
    ema = _ema(B, i, p.get("ema_period", 20))
    lo_i = i - lb
    return B.c[i] >= ema and B.l[i] > B.l[lo_i : i - 5].min()


def _bear_trend(B, i, p):
    lb = p.get("swing_lb", 20)
    if i < lb + 5:
        return False
    ema = _ema(B, i, p.get("ema_period", 20))
    lo_i = i - lb
    return B.c[i] <= ema and B.h[i] < B.h[lo_i : i - 5].max()


def _trading_range(B, i, p):
    lb = p.get("range_lb", 40)
    if i < lb + 5:
        return False
    lo_i = i - lb
    rh = B.h[lo_i:i].max()
    rl = B.l[lo_i:i].min()
    width = rh - rl
    atr_i = _atr(B, i)
    if width < 1.5 * atr_i or width > 8 * atr_i:
        return False
    # price oscillating — not at trend extreme
    pos = (B.c[i] - rl) / width if width > 0 else 0.5
    return 0.15 < pos < 0.85


def _signal_bull(B, i, p):
    r = B.rng[i]
    if r <= 0:
        return False
    br = p.get("min_body_ratio", 0.45)
    return B.body[i] > 0 and B.body[i] / r >= br and (B.c[i] - B.l[i]) / r >= 0.55


def _signal_bear(B, i, p):
    r = B.rng[i]
    if r <= 0:
        return False
    br = p.get("min_body_ratio", 0.45)
    return B.body[i] < 0 and (-B.body[i]) / r >= br and (B.h[i] - B.c[i]) / r >= 0.55


def _swing_lows(B, lo, hi):
    idx = []
    for j in range(lo + 1, hi):
        if B.l[j] <= B.l[j - 1] and B.l[j] <= B.l[j + 1]:
            idx.append(j)
    return idx


def _swing_highs(B, lo, hi):
    idx = []
    for j in range(lo + 1, hi):
        if B.h[j] >= B.h[j - 1] and B.h[j] >= B.h[j + 1]:
            idx.append(j)
    return idx


# ── Bar counting: H1/H2/L1/L2 (Brooks Ch.17) ─────────────────────────────

def detect_ab_h1(B, i, p=AB_DEFAULT):
    """High 1: first bar in bull flag with high above prior bar."""
    s = []
    if i < 8 or not _bull_trend(B, i, p):
        return s
    lb = p.get("pullback_lb", 15)
    lo = max(3, i - lb)
    bear_bars = sum(1 for j in range(lo, i) if B.body[j] < 0)
    if bear_bars < 2:
        return s
    if B.h[i] > B.h[i - 1] and _signal_bull(B, i, p):
        entry = B.l[i]
        sl_ref = min(B.l[lo:i + 1])
        s.append(("long", entry, sl_ref, "AB-H1"))
    return s


def detect_ab_h2(B, i, p=AB_DEFAULT):
    """High 2: second push up in bull flag — Brooks' preferred with-trend entry."""
    s = []
    if i < 12 or not _bull_trend(B, i, p):
        return s
    lb = p.get("pullback_lb", 15)
    lo = max(3, i - lb)
    h1 = None
    for j in range(lo + 1, i):
        if B.h[j] > B.h[j - 1] and B.body[j] > 0:
            h1 = j
            break
    if h1 is None:
        return s
    # second leg down after H1
    if B.l[h1:i].min() >= B.l[h1]:
        return s
    if B.h[i] > B.h[i - 1] and _signal_bull(B, i, p):
        s.append(("long", B.l[i], min(B.l[lo:i + 1]), "AB-H2"))
    return s


def detect_ab_l1(B, i, p=AB_DEFAULT):
    s = []
    if i < 8 or not _bear_trend(B, i, p):
        return s
    lb = p.get("pullback_lb", 15)
    lo = max(3, i - lb)
    bull_bars = sum(1 for j in range(lo, i) if B.body[j] > 0)
    if bull_bars < 2:
        return s
    if B.l[i] < B.l[i - 1] and _signal_bear(B, i, p):
        s.append(("short", B.h[i], max(B.h[lo:i + 1]), "AB-L1"))
    return s


def detect_ab_l2(B, i, p=AB_DEFAULT):
    """Low 2: second push down in bear flag."""
    s = []
    if i < 12 or not _bear_trend(B, i, p):
        return s
    lb = p.get("pullback_lb", 15)
    lo = max(3, i - lb)
    l1 = None
    for j in range(lo + 1, i):
        if B.l[j] < B.l[j - 1] and B.body[j] < 0:
            l1 = j
            break
    if l1 is None:
        return s
    if B.h[l1:i].max() <= B.h[l1]:
        return s
    if B.l[i] < B.l[i - 1] and _signal_bear(B, i, p):
        s.append(("short", B.h[i], max(B.h[lo:i + 1]), "AB-L2"))
    return s


# ── Wedges & three-push (Ch.18) ────────────────────────────────────────────

def detect_ab_wedge_flag_bull(B, i, p=AB_DEFAULT):
    """Wedge bull flag: 3 pushes down in bull trend, buy reversal."""
    s = []
    if i < 20 or not _bull_trend(B, i, p):
        return s
    lo = i - p.get("wedge_push_lb", 8) * 3
    lows = _swing_lows(B, max(3, lo), i)
    if len(lows) < 3:
        return s
    l3 = lows[-3:]
    if not (B.l[l3[1]] < B.l[l3[0]] and B.l[l3[2]] < B.l[l3[1]]):
        return s
    if _signal_bull(B, i, p):
        s.append(("long", B.l[i], B.l[l3[0]], "AB-WEDGE-BF"))
    return s


def detect_ab_wedge_flag_bear(B, i, p=AB_DEFAULT):
    s = []
    if i < 20 or not _bear_trend(B, i, p):
        return s
    lo = i - p.get("wedge_push_lb", 8) * 3
    highs = _swing_highs(B, max(3, lo), i)
    if len(highs) < 3:
        return s
    h3 = highs[-3:]
    if not (B.h[h3[1]] > B.h[h3[0]] and B.h[h3[2]] > B.h[h3[1]]):
        return s
    if _signal_bear(B, i, p):
        s.append(("short", B.h[i], B.h[h3[0]], "AB-WEDGE-BF"))
    return s


def detect_ab_wedge_rev_bull(B, i, p=AB_DEFAULT):
    """Wedge bottom reversal: 3 pushes down at lows."""
    s = []
    if i < 25:
        return s
    lo = i - 30
    lows = _swing_lows(B, max(3, lo), i)
    if len(lows) < 3:
        return s
    l3 = lows[-3:]
    if not (B.l[l3[1]] <= B.l[l3[0]] and B.l[l3[2]] <= B.l[l3[1]]):
        return s
    if not _bear_trend(B, l3[0], p) and B.c[i] > _ema(B, i, p.get("ema_period", 20)):
        return s
    if _signal_bull(B, i, p):
        s.append(("long", B.l[i], B.l[l3[2]], "AB-WEDGE-BOT"))
    return s


def detect_ab_wedge_rev_bear(B, i, p=AB_DEFAULT):
    s = []
    if i < 25:
        return s
    lo = i - 30
    highs = _swing_highs(B, max(3, lo), i)
    if len(highs) < 3:
        return s
    h3 = highs[-3:]
    if not (B.h[h3[1]] >= B.h[h3[0]] and B.h[h3[2]] >= B.h[h3[1]]):
        return s
    if _signal_bear(B, i, p):
        s.append(("short", B.h[i], B.h[h3[2]], "AB-WEDGE-TOP"))
    return s


# ── Two-legged pullback (ABC) ──────────────────────────────────────────────

def detect_ab_two_leg_bull(B, i, p=AB_DEFAULT):
    """Two-legged pullback in bull trend then signal bar."""
    s = []
    if i < 20 or not _bull_trend(B, i, p):
        return s
    lo = i - 25
    lows = _swing_lows(B, max(3, lo), i - 1)
    if len(lows) < 2:
        return s
    leg1, leg2 = lows[-2], lows[-1]
    if leg2 <= leg1:
        return s
    if B.l[leg2] < B.l[leg1] and _signal_bull(B, i, p):
        s.append(("long", B.l[i], B.l[leg2], "AB-2LEG"))
    return s


def detect_ab_two_leg_bear(B, i, p=AB_DEFAULT):
    s = []
    if i < 20 or not _bear_trend(B, i, p):
        return s
    lo = i - 25
    highs = _swing_highs(B, max(3, lo), i - 1)
    if len(highs) < 2:
        return s
    leg1, leg2 = highs[-2], highs[-1]
    if leg2 >= leg1:
        return s
    if B.h[leg2] > B.h[leg1] and _signal_bear(B, i, p):
        s.append(("short", B.h[i], B.h[leg2], "AB-2LEG"))
    return s


# ── Breakout & failed breakout (Ch.5) ──────────────────────────────────────

def detect_ab_bo_pullback(B, i, p=AB_DEFAULT):
    """Breakout pullback: break swing then retest within few bars."""
    s = []
    lb = p.get("swing_lb", 20)
    if i < lb + 8:
        return s
    prior_high = B.h[i - lb : i - 3].max()
    prior_low = B.l[i - lb : i - 3].min()
    atr_i = _atr(B, i)
    # bull BO pullback
    for j in range(max(3, i - 8), i - 2):
        if B.c[j] > prior_high and B.body[j] > 0:
            for k in range(j + 1, i):
                if B.l[k] <= prior_high + 0.3 * atr_i and B.l[k] >= prior_high - 0.5 * atr_i:
                    if _signal_bull(B, i, p):
                        s.append(("long", B.l[i], min(B.l[j:k + 1]), "AB-BO-PB"))
                    return s
    # bear BO pullback
    for j in range(max(3, i - 8), i - 2):
        if B.c[j] < prior_low and B.body[j] < 0:
            for k in range(j + 1, i):
                if B.h[k] >= prior_low - 0.3 * atr_i and B.h[k] <= prior_low + 0.5 * atr_i:
                    if _signal_bear(B, i, p):
                        s.append(("short", B.h[i], max(B.h[j:k + 1]), "AB-BO-PB"))
                    return s
    return s


def detect_ab_failed_bo(B, i, p=AB_DEFAULT):
    """Failed breakout: break range then close back inside (fade)."""
    s = []
    if not _trading_range(B, i, p):
        return s
    lb = p.get("range_lb", 40)
    lo = i - lb
    rh = B.h[lo:i].max()
    rl = B.l[lo:i].min()
    atr_i = _atr(B, i)
    tol = 0.15 * atr_i
    # failed break above
    if any(B.h[j] > rh + tol for j in range(max(lo, i - 5), i)):
        if B.c[i] < rh and _signal_bear(B, i, p):
            s.append(("short", B.h[i], max(B.h[max(lo, i - 5):i + 1]), "AB-FAIL-BO"))
    # failed break below
    if any(B.l[j] < rl - tol for j in range(max(lo, i - 5), i)):
        if B.c[i] > rl and _signal_bull(B, i, p):
            s.append(("long", B.l[i], min(B.l[max(lo, i - 5):i + 1]), "AB-FAIL-BO"))
    return s


# ── Double top/bottom (Ch.20) ──────────────────────────────────────────────

def detect_ab_double_top(B, i, p=AB_DEFAULT):
    s = []
    if i < 25:
        return s
    lb = 30
    lo = i - lb
    highs = _swing_highs(B, lo, i - 1)
    if len(highs) < 2:
        return s
    h1, h2 = highs[-2], highs[-1]
    atr_i = _atr(B, i)
    tol = p.get("dt_tolerance", 0.35) * atr_i
    if abs(B.h[h1] - B.h[h2]) > tol:
        return s
    neck = B.l[h1:h2 + 1].min()
    if B.c[i] < neck and _signal_bear(B, i, p):
        s.append(("short", B.h[i], max(B.h[h1], B.h[h2]), "AB-DTOP"))
    return s


def detect_ab_double_bottom(B, i, p=AB_DEFAULT):
    s = []
    if i < 25:
        return s
    lb = 30
    lo = i - lb
    lows = _swing_lows(B, lo, i - 1)
    if len(lows) < 2:
        return s
    l1, l2 = lows[-2], lows[-1]
    atr_i = _atr(B, i)
    tol = p.get("dt_tolerance", 0.35) * atr_i
    if abs(B.l[l1] - B.l[l2]) > tol:
        return s
    neck = B.h[l1:l2 + 1].max()
    if B.c[i] > neck and _signal_bull(B, i, p):
        s.append(("long", B.l[i], min(B.l[l1], B.l[l2]), "AB-DBOT"))
    return s


# ── Outside / inside bar ───────────────────────────────────────────────────

def detect_ab_outside_bar(B, i, p=AB_DEFAULT):
    """Outside bar reversal at trend extreme."""
    s = []
    if i < 10:
        return s
    if not (B.h[i] > B.h[i - 1] and B.l[i] < B.l[i - 1]):
        return s
    lb = p.get("swing_lb", 20)
    if _bull_trend(B, i - 1, p) and B.c[i] < B.o[i] and _signal_bear(B, i, p):
        if B.h[i - 1] >= B.h[i - lb:i - 1].max() * 0.998:
            s.append(("short", B.h[i], B.h[i], "AB-OUT-REV"))
    if _bear_trend(B, i - 1, p) and B.c[i] > B.o[i] and _signal_bull(B, i, p):
        if B.l[i - 1] <= B.l[i - lb:i - 1].min() * 1.002:
            s.append(("long", B.l[i], B.l[i], "AB-OUT-REV"))
    return s


def detect_ab_inside_bar_bo(B, i, p=AB_DEFAULT):
    """Inside bar breakout in trend direction."""
    s = []
    if i < 8:
        return s
    if not (B.h[i - 1] < B.h[i - 2] and B.l[i - 1] > B.l[i - 2]):
        return s
    if _bull_trend(B, i, p) and B.c[i] > B.h[i - 1] and B.body[i] > 0:
        s.append(("long", B.h[i - 1], B.l[i - 1], "AB-IB-BO"))
    if _bear_trend(B, i, p) and B.c[i] < B.l[i - 1] and B.body[i] < 0:
        s.append(("short", B.l[i - 1], B.h[i - 1], "AB-IB-BO"))
    return s


# ── Micro channel & climax ────────────────────────────────────────────────

def detect_ab_micro_channel(B, i, p=AB_DEFAULT):
    """Micro channel pullback: 4+ bull/bear bars then first pullback bar."""
    s = []
    if i < 8:
        return s
    streak = 0
    for j in range(i - 1, max(2, i - 8), -1):
        if B.body[j] > 0 and B.c[j] > B.c[j - 1]:
            streak += 1
        else:
            break
    if streak >= 4 and B.body[i] < 0 and _bull_trend(B, i, p):
        if _signal_bull(B, i, p) or B.body[i - 1] < 0:
            s.append(("long", B.l[i], min(B.l[i - streak:i + 1]), "AB-MICRO"))
    streak = 0
    for j in range(i - 1, max(2, i - 8), -1):
        if B.body[j] < 0 and B.c[j] < B.c[j - 1]:
            streak += 1
        else:
            break
    if streak >= 4 and B.body[i] > 0 and _bear_trend(B, i, p):
        if _signal_bear(B, i, p) or B.body[i - 1] > 0:
            s.append(("short", B.h[i], max(B.h[i - streak:i + 1]), "AB-MICRO"))
    return s


def detect_ab_climax_rev(B, i, p=AB_DEFAULT):
    """Climactic reversal: 3+ large trend bars then opposite signal bar."""
    s = []
    if i < 10:
        return s
    atr_i = _atr(B, i)
    bull_run = all(B.body[i - k] > 0.8 * atr_i for k in range(1, 4))
    bear_run = all(-B.body[i - k] > 0.8 * atr_i for k in range(1, 4))
    if bull_run and _signal_bear(B, i, p):
        s.append(("short", B.h[i], B.h[i - 3], "AB-CLIMAX"))
    if bear_run and _signal_bull(B, i, p):
        s.append(("long", B.l[i], B.l[i - 3], "AB-CLIMAX"))
    return s


# ── Final flag & TR fade ───────────────────────────────────────────────────

def detect_ab_final_flag_bull(B, i, p=AB_DEFAULT):
    """Tight bull flag after extended rally — buy breakout or signal bar."""
    s = []
    if i < 30 or not _bull_trend(B, i, p):
        return s
    lo = i - 12
    width = B.h[lo:i].max() - B.l[lo:i].min()
    atr_i = _atr(B, i)
    if width > 2.5 * atr_i:
        return s
    rally = B.c[i] - B.c[i - 30]
    if rally < 3 * atr_i:
        return s
    if _signal_bull(B, i, p):
        s.append(("long", B.l[i], B.l[lo:i].min(), "AB-FLAG"))
    return s


def detect_ab_final_flag_bear(B, i, p=AB_DEFAULT):
    s = []
    if i < 30 or not _bear_trend(B, i, p):
        return s
    lo = i - 12
    width = B.h[lo:i].max() - B.l[lo:i].min()
    atr_i = _atr(B, i)
    if width > 2.5 * atr_i:
        return s
    drop = B.c[i - 30] - B.c[i]
    if drop < 3 * atr_i:
        return s
    if _signal_bear(B, i, p):
        s.append(("short", B.h[i], B.h[lo:i].max(), "AB-FLAG"))
    return s


def detect_ab_tr_fade(B, i, p=AB_DEFAULT):
    """Fade to opposite side of trading range."""
    s = []
    if not _trading_range(B, i, p):
        return s
    lb = p.get("range_lb", 40)
    lo = i - lb
    rh = B.h[lo:i].max()
    rl = B.l[lo:i].min()
    width = rh - rl
    if width <= 0:
        return s
    pos = (B.c[i] - rl) / width
    if pos >= 0.85 and _signal_bear(B, i, p):
        s.append(("short", B.h[i], rh, "AB-TR-FADE"))
    if pos <= 0.15 and _signal_bull(B, i, p):
        s.append(("long", B.l[i], rl, "AB-TR-FADE"))
    return s


def detect_ab_mtr(B, i, p=AB_DEFAULT):
    """Major Trend Reversal: strong counter-trend bar after extended trend."""
    s = []
    if i < 40:
        return s
    atr_i = _atr(B, i)
    lo = i - 40
    up_move = B.c[i - 5] - B.c[lo]
    dn_move = B.c[lo] - B.c[i - 5]
    if up_move > 5 * atr_i and _signal_bear(B, i, p) and B.c[i] < B.l[i - 5:i].min():
        s.append(("short", B.h[i], B.h[i], "AB-MTR"))
    if dn_move > 5 * atr_i and _signal_bull(B, i, p) and B.c[i] > B.h[i - 5:i].max():
        s.append(("long", B.l[i], B.l[i], "AB-MTR"))
    return s


def detect_ab_measuring_gap(B, i, p=AB_DEFAULT):
    """Body gap in trend (Brooks measuring gap) — enter on first pullback."""
    s = []
    if i < 10:
        return s
    # bull body gap: low > prior body top
    if i >= 2:
        body_top_prev = max(B.o[i - 1], B.c[i - 1])
        body_bot_curr = min(B.o[i], B.c[i])
        if _bull_trend(B, i, p) and B.l[i] > body_top_prev and B.body[i - 1] > 0:
            if B.l[i] <= body_top_prev + 0.5 * _atr(B, i):
                s.append(("long", B.l[i], B.l[i - 2], "AB-GAP"))
        body_bot_prev = min(B.o[i - 1], B.c[i - 1])
        if _bear_trend(B, i, p) and B.h[i] < body_bot_prev and B.body[i - 1] < 0:
            if B.h[i] >= body_bot_prev - 0.5 * _atr(B, i):
                s.append(("short", B.h[i], B.h[i - 2], "AB-GAP"))
    return s


# ── Registry with Brooks categories ───────────────────────────────────────

AB_CATEGORIES = {
    "trend_continuation": {
        "label": "Trend continuation (Brooks: with-trend entries)",
        "rules": {
            "AB_H1": detect_ab_h1,
            "AB_H2": detect_ab_h2,
            "AB_L1": detect_ab_l1,
            "AB_L2": detect_ab_l2,
            "AB_2LEG": detect_ab_two_leg_bull,  # bear via tag in combined
            "AB_MICRO": detect_ab_micro_channel,
            "AB_FLAG": detect_ab_final_flag_bull,
            "AB_WEDGE_BF": detect_ab_wedge_flag_bull,
            "AB_IB_BO": detect_ab_inside_bar_bo,
            "AB_GAP": detect_ab_measuring_gap,
        },
    },
    "breakout": {
        "label": "Breakout & breakout pullback",
        "rules": {
            "AB_BO_PB": detect_ab_bo_pullback,
        },
    },
    "reversal": {
        "label": "Reversal (counter-trend / major turns)",
        "rules": {
            "AB_WEDGE_BOT": detect_ab_wedge_rev_bull,
            "AB_WEDGE_TOP": detect_ab_wedge_rev_bear,
            "AB_DTOP": detect_ab_double_top,
            "AB_DBOT": detect_ab_double_bottom,
            "AB_OUT_REV": detect_ab_outside_bar,
            "AB_CLIMAX": detect_ab_climax_rev,
            "AB_MTR": detect_ab_mtr,
        },
    },
    "trading_range": {
        "label": "Trading range (fade & failed BO)",
        "rules": {
            "AB_FAIL_BO": detect_ab_failed_bo,
            "AB_TR_FADE": detect_ab_tr_fade,
        },
    },
}

# Flat detector map for backtest (bear variants share keys where combined)
AB_DETECTORS = {}
AB_RULE_META = {}

for cat_key, cat in AB_CATEGORIES.items():
    for rule_key, fn in cat["rules"].items():
        AB_DETECTORS[rule_key] = fn
        AB_RULE_META[rule_key] = {"category": cat_key, "label": cat["label"]}

# Separate bear-only detectors registered under distinct keys
AB_DETECTORS["AB_2LEG_BEAR"] = detect_ab_two_leg_bear
AB_DETECTORS["AB_WEDGE_BF_BEAR"] = detect_ab_wedge_flag_bear
AB_DETECTORS["AB_FLAG_BEAR"] = detect_ab_final_flag_bear
AB_RULE_META["AB_2LEG_BEAR"] = {"category": "trend_continuation", "label": "Two-leg bear pullback"}
AB_RULE_META["AB_WEDGE_BF_BEAR"] = {"category": "trend_continuation", "label": "Wedge bear flag"}
AB_RULE_META["AB_FLAG_BEAR"] = {"category": "trend_continuation", "label": "Final bear flag"}

AB_ALL_RULES = list(AB_DETECTORS.keys())

# Brooks theoretical reliability (from his teaching — for reference column)
AB_BROOKS_THEORY_TIER = {
    "AB_H2": "A — best with-trend entry",
    "AB_L2": "A — best with-trend entry",
    "AB_H1": "B — early, lower probability",
    "AB_L1": "B — early, lower probability",
    "AB_BO_PB": "A — high when trend strong",
    "AB_2LEG": "A — classic ABC pullback",
    "AB_WEDGE_BF": "B — wedge flag (3 pushes)",
    "AB_MICRO": "B — micro channel first pullback",
    "AB_FAIL_BO": "A — in trading range",
    "AB_TR_FADE": "B — range fade",
    "AB_DTOP": "B — needs strong bear signal",
    "AB_DBOT": "B — needs strong bull signal",
    "AB_WEDGE_BOT": "B — wedge reversal",
    "AB_WEDGE_TOP": "B — wedge reversal",
    "AB_CLIMAX": "C — climactic, risky",
    "AB_MTR": "C — major reversal, low WR alone",
    "AB_OUT_REV": "B — outside bar at extreme",
    "AB_IB_BO": "B — inside bar BO",
    "AB_GAP": "A — measuring gap in strong trend",
    "AB_FLAG": "B — final flag",
}
