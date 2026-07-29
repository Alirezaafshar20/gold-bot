"""
Classic wave / structure pattern detectors:
  Elliott Wave, NeoWave, Wolfe Wave, Harmonic (Gartley/Bat/Butterfly/Crab)

Each returns list of (direction, proximal, distal, tag).
Wyckoff lives in wyckoff.py — included in style_research only.
"""
import numpy as np

STYLE_DEFAULT = {
    "swing_lb": 3,
    "scan_lb": 80,
    "ratio_tol": 0.08,
    "wolfe_tol": 0.15,
    "neowave_time_eq": 0.35,
    # Butterfly and Crab complete BEYOND X (AD of 1.27-1.618), so D itself is
    # the extreme of the pattern and there is no older pivot to hide a stop
    # behind. Their invalidation is a decisive break past D, placed here as a
    # fraction of the pattern's own CD leg so it scales with the structure
    # instead of an arbitrary ATR.
    "harm_d_buffer": 0.15,
}


def _atr(B, i):
    return B.atr[i] if B.atr[i] > 0 else B.rng[i] + 1e-6


def _all_swings(B, lb=3):
    """Every fractal pivot in the series, memoised on the Bars object itself.

    Keying a module dict on id(B) was unsafe: live rebuilds Bars each cycle at a
    fixed length, the freed address gets handed to the new object, and the old
    pivot list comes back for new data.
    """
    m = getattr(B, "_wave_memo", None)
    if m is None:
        m = {}
        try:
            B._wave_memo = m
        except AttributeError:
            m = {}
    if lb not in m:
        lows, highs = [], []
        for j in range(lb, B.n - lb - 1):
            if B.l[j] == B.l[j - lb:j + lb + 1].min():
                lows.append(j)
            if B.h[j] == B.h[j - lb:j + lb + 1].max():
                highs.append(j)
        m[lb] = (lows, highs)
    return m[lb]


def _swing_indices(B, lo, hi, kind="low", lb=3):
    """Pivots that are already CONFIRMED as of bar `hi`.

    A fractal pivot at j is only known once lb further bars have printed, so a
    scan standing at bar i may not use pivots newer than i-lb. Without that cut
    the pattern search reads bars that do not exist yet at decision time, and
    the setup cannot be reproduced live — measured at 13-32% of signals across
    the Elliott / NeoWave families.
    """
    lows, highs = _all_swings(B, lb)
    pool = lows if kind == "low" else highs
    confirmed = hi - lb
    return [j for j in pool if lo <= j <= confirmed]


def _line_y(x0, y0, x1, y1, x):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def detect_ew_wave2(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 80)
    tol = p.get("ratio_tol", 0.08)
    if i < lb:
        return s
    lo = i - lb
    for w0 in _swing_indices(B, lo, i, "low"):
        for w1 in _swing_indices(B, w0, i, "high"):
            if w1 <= w0 or w1 >= i:
                continue
            up = B.h[w1] - B.l[w0]
            if up <= 0:
                continue
            retr = (B.h[w1] - B.l[i]) / up
            if 0.382 * (1 - tol) <= retr <= 0.786 * (1 + tol) and B.c[i] > B.o[i]:
                s.append(("long", B.l[i], min(B.l[w0], B.l[i]), "EW-W2"))
                return s
    return s


def detect_ew_wave4(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 80)
    tol = p.get("ratio_tol", 0.08)
    if i < lb:
        return s
    lo = i - lb
    for w0 in _swing_indices(B, lo, i, "low"):
        for w1 in _swing_indices(B, w0, i, "high"):
            if w1 <= w0:
                continue
            for w3 in _swing_indices(B, w1, i, "high"):
                if w3 <= w1 or B.l[i] <= B.l[w0]:
                    continue
                w3_len = B.h[w3] - B.l[w1]
                if w3_len <= 0:
                    continue
                retr = (B.h[w3] - B.l[i]) / w3_len
                if 0.236 * (1 - tol) <= retr <= 0.618 * (1 + tol) and i > w3:
                    s.append(("long", B.l[i], B.l[w0], "EW-W4"))
                    return s
    return s


def detect_ew_abc_bull(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 80)
    if i < lb:
        return s
    lo = i - lb
    for a in _swing_indices(B, lo, i, "high"):
        for b in _swing_indices(B, a, i, "low"):
            if b <= a:
                continue
            if i <= b:
                continue
            ab = B.h[a] - B.l[b]
            if ab <= 0:
                continue
            bc = B.l[b] - B.l[i]
            if 0.618 * 0.85 <= bc / ab <= 1.618 * 1.15 and B.c[i] > B.o[i]:
                s.append(("long", B.l[i], B.l[i], "EW-ABC"))
                return s
    return s


def detect_ew_impulse5(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 100)
    if i < lb:
        return s
    lo = i - lb
    pts = sorted(
        [(j, "L", B.l[j]) for j in _swing_indices(B, lo, i, "low")] +
        [(j, "H", B.h[j]) for j in _swing_indices(B, lo, i, "high")],
        key=lambda x: x[0])
    for k in range(len(pts) - 4):
        seq = pts[k:k + 5]
        if [x[1] for x in seq] != ["L", "H", "L", "H", "L"]:
            continue
        w0, w1, w2, w3, w4 = [seq[m][0] for m in range(5)]
        if not (B.l[w2] > B.l[w0] and B.h[w3] > B.h[w1] and B.h[w1] > B.l[w0]):
            continue
        if i == w4 and B.c[i] > B.o[i]:
            s.append(("long", B.l[i], B.l[w0], "EW-IMP5"))
            return s
    return s


def detect_nw_wave2(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 80)
    if i < lb:
        return s
    lo = i - lb
    for w0 in _swing_indices(B, lo, i, "low"):
        for w1 in _swing_indices(B, w0, i, "high"):
            if w1 <= w0 or w1 >= i:
                continue
            t1, t2 = w1 - w0, i - w1
            if t2 > t1 * (1 + p.get("neowave_time_eq", 0.35)):
                continue
            up = B.h[w1] - B.l[w0]
            if up <= 0:
                continue
            retr = (B.h[w1] - B.l[i]) / up
            if 0.50 <= retr <= 0.618 * 1.05 and B.c[i] > B.o[i]:
                s.append(("long", B.l[i], B.l[w0], "NW-W2"))
                return s
    return s


def detect_nw_wave4(B, i, p=STYLE_DEFAULT):
    s = []
    lb = p.get("scan_lb", 80)
    if i < lb:
        return s
    lo = i - lb
    for w0 in _swing_indices(B, lo, i, "low"):
        for w1 in _swing_indices(B, w0, i, "high"):
            if w1 <= w0:
                continue
            for w3 in _swing_indices(B, w1, i, "high"):
                if w3 <= w1 or B.l[i] <= B.l[w0]:
                    continue
                w3_len = B.h[w3] - B.l[w1]
                if w3_len <= 0:
                    continue
                retr = (B.h[w3] - B.l[i]) / w3_len
                if 0.382 <= retr <= 0.55 and i > w3:
                    s.append(("long", B.l[i], B.l[w0], "NW-W4"))
                    return s
    return s


def _detect_wolfe(B, i, p, bullish=True):
    s = []
    lb = p.get("scan_lb", 80)
    tol = p.get("wolfe_tol", 0.15) * _atr(B, i)
    if i < lb:
        return s
    lo = i - lb
    if bullish:
        for p1 in _swing_indices(B, lo, i, "low"):
            for p2 in _swing_indices(B, p1, i, "high"):
                if p2 <= p1 or B.h[p2] <= B.l[p1]:
                    continue
                for p3 in _swing_indices(B, p2, i, "low"):
                    if p3 <= p2 or B.l[p3] >= B.l[p1]:
                        continue
                    for p4 in _swing_indices(B, p3, i, "high"):
                        if p4 <= p3 or B.h[p4] >= B.h[p2] or B.h[p4] <= B.l[p1]:
                            continue
                        if p4 >= i - 5 and i >= p4:
                            if B.l[i] >= B.l[p3]:
                                continue
                            line14 = _line_y(p1, B.l[p1], p4, B.h[p4], i)
                            if abs(B.l[i] - line14) <= tol * 2:
                                s.append(("long", B.l[i], B.l[p3], "WOLFE-B"))
                                return s
    else:
        for p1 in _swing_indices(B, lo, i, "high"):
            for p2 in _swing_indices(B, p1, i, "low"):
                if p2 <= p1 or B.l[p2] >= B.h[p1]:
                    continue
                for p3 in _swing_indices(B, p2, i, "high"):
                    if p3 <= p2 or B.h[p3] <= B.h[p1]:
                        continue
                    for p4 in _swing_indices(B, p3, i, "low"):
                        if p4 <= p3 or B.l[p4] <= B.l[p2] or B.l[p4] >= B.h[p1]:
                            continue
                        if p4 >= i - 5 and i >= p4:
                            if B.h[i] <= B.h[p3]:
                                continue
                            line14 = _line_y(p1, B.h[p1], p4, B.l[p4], i)
                            if abs(B.h[i] - line14) <= tol * 2:
                                s.append(("short", B.h[i], B.h[p3], "WOLFE-S"))
                                return s
    return s


def detect_wolfe_bull(B, i, p=STYLE_DEFAULT):
    return _detect_wolfe(B, i, p, True)


def detect_wolfe_bear(B, i, p=STYLE_DEFAULT):
    return _detect_wolfe(B, i, p, False)


def _harmonic_xabcd(B, x, a, b, c, d, bullish, ratios, tol):
    if bullish:
        xa = B.h[a] - B.l[x]
        ab = B.h[a] - B.l[b]
        bc = B.h[b] - B.l[c]
        ad = B.h[a] - B.l[d]
    else:
        xa = B.h[x] - B.l[a]
        ab = B.h[b] - B.l[a]
        bc = B.h[c] - B.l[b]
        ad = B.h[d] - B.l[a]
    if min(xa, ab, bc) <= 0:
        return False
    r_ab, r_bc, r_ad = ab / xa, bc / ab, ad / xa
    for key, (lo, hi) in ratios.items():
        val = {"AB": r_ab, "BC": r_bc, "AD": r_ad}[key]
        if not (lo * (1 - tol) <= val <= hi * (1 + tol)):
            return False
    return True


def _scan_harmonic(B, i, p, name, ratios, bullish=True):
    """Scan for an XABCD completion at bar i.

    Entry is D; the stop goes beyond whichever of X and D is further from it.

    Pinning the stop to X only works while D stays inside XA — true for
    Gartley, Bat and Cypher. Butterfly and Crab complete at an AD of
    1.27-1.618, so D overshoots X: an X-based stop sits on the PROFIT side of
    the entry, and clamping it back to D makes proximal == distal, which is
    worse. A zero-risk setup still books trades, but its R is then measured
    against whatever generic swing buffer the sizing code happens to apply, so
    a 0.5 ATR stop turns every winner into a huge multiple and the rule ranks
    for the wrong reason. Both failures were present on 100% of Butterfly and
    Crab signals.

    So when D is the extreme, the invalidation is a break past D by a fraction
    of the pattern's own CD leg.
    """
    s = []
    lb = p.get("scan_lb", 100)
    tol = p.get("ratio_tol", 0.08)
    dbuf = p.get("harm_d_buffer", 0.15)
    if i < lb:
        return s
    lo = i - lb
    if bullish:
        for x in _swing_indices(B, lo, i, "low"):
            for a in _swing_indices(B, x, i, "high"):
                if a <= x:
                    continue
                for b in _swing_indices(B, a, i, "low"):
                    if b <= a:
                        continue
                    for c in _swing_indices(B, b, i - 1, "high"):
                        if c <= b:
                            continue
                        if _harmonic_xabcd(B, x, a, b, c, i, True, ratios, tol):
                            beyond_d = B.l[i] - dbuf * max(B.h[c] - B.l[i], 0.0)
                            s.append(("long", B.l[i],
                                      min(B.l[x], beyond_d), name))
                            return s
    else:
        for x in _swing_indices(B, lo, i, "high"):
            for a in _swing_indices(B, x, i, "low"):
                if a <= x:
                    continue
                for b in _swing_indices(B, a, i, "high"):
                    if b <= a:
                        continue
                    for c in _swing_indices(B, b, i - 1, "low"):
                        if c <= b:
                            continue
                        if _harmonic_xabcd(B, x, a, b, c, i, False, ratios, tol):
                            beyond_d = B.h[i] + dbuf * max(B.h[i] - B.l[c], 0.0)
                            s.append(("short", B.h[i],
                                      max(B.h[x], beyond_d), name))
                            return s
    return s


def detect_harm_gartley(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-GART",
                          {"AB": (0.618, 0.618), "BC": (0.382, 0.886), "AD": (0.786, 0.786)}, True)


def detect_harm_gartley_bear(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-GART",
                          {"AB": (0.618, 0.618), "BC": (0.382, 0.886), "AD": (0.786, 0.786)}, False)


def detect_harm_bat(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-BAT",
                          {"AB": (0.382, 0.50), "BC": (0.382, 0.886), "AD": (0.886, 0.886)}, True)


def detect_harm_bat_bear(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-BAT",
                          {"AB": (0.382, 0.50), "BC": (0.382, 0.886), "AD": (0.886, 0.886)}, False)


def detect_harm_butterfly(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-BFLY",
                          {"AB": (0.786, 0.786), "BC": (0.382, 0.886), "AD": (1.27, 1.618)}, True)


def detect_harm_butterfly_bear(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-BFLY",
                          {"AB": (0.786, 0.786), "BC": (0.382, 0.886), "AD": (1.27, 1.618)}, False)


def detect_harm_crab(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-CRAB",
                          {"AB": (0.382, 0.618), "BC": (0.382, 0.886), "AD": (1.618, 1.618)}, True)


def detect_harm_crab_bear(B, i, p=STYLE_DEFAULT):
    return _scan_harmonic(B, i, p, "HARM-CRAB",
                          {"AB": (0.382, 0.618), "BC": (0.382, 0.886), "AD": (1.618, 1.618)}, False)


def detect_harm_cypher(B, i, p=STYLE_DEFAULT):
    s = []
    lb, tol = p.get("scan_lb", 100), p.get("ratio_tol", 0.08)
    if i < lb:
        return s
    lo = i - lb
    for x in _swing_indices(B, lo, i, "low"):
        for a in _swing_indices(B, x, i, "high"):
            if a <= x:
                continue
            xa = B.h[a] - B.l[x]
            if xa <= 0:
                continue
            for b in _swing_indices(B, a, i, "low"):
                if b <= a:
                    continue
                ab = B.h[a] - B.l[b]
                if not (0.382 * (1 - tol) <= ab / xa <= 0.618 * (1 + tol)):
                    continue
                for c in _swing_indices(B, b, i - 1, "high"):
                    if c <= b:
                        continue
                    bc = B.h[c] - B.l[b]
                    if bc <= 0 or not (1.13 <= bc / ab <= 1.414 * (1 + tol)):
                        continue
                    ad = B.h[a] - B.l[i]
                    if 0.786 * (1 - tol) <= ad / xa <= 0.786 * (1 + tol):
                        s.append(("long", B.l[i], B.l[x], "HARM-CYP"))
                        return s
    return s


STYLE_FAMILIES = {
    "elliott": {"label": "Elliott Wave", "rules": {
        "EW_W2": detect_ew_wave2, "EW_W4": detect_ew_wave4,
        "EW_ABC": detect_ew_abc_bull, "EW_IMP5": detect_ew_impulse5}},
    "neowave": {"label": "NeoWave", "rules": {
        "NW_W2": detect_nw_wave2, "NW_W4": detect_nw_wave4}},
    "wolfe": {"label": "Wolfe Wave", "rules": {
        "WOLFE_B": detect_wolfe_bull, "WOLFE_S": detect_wolfe_bear}},
    "harmonic": {"label": "Harmonic Patterns", "rules": {
        "HARM_GART": detect_harm_gartley, "HARM_GART_S": detect_harm_gartley_bear,
        "HARM_BAT": detect_harm_bat, "HARM_BAT_S": detect_harm_bat_bear,
        "HARM_BFLY": detect_harm_butterfly, "HARM_BFLY_S": detect_harm_butterfly_bear,
        "HARM_CRAB": detect_harm_crab, "HARM_CRAB_S": detect_harm_crab_bear,
        "HARM_CYP": detect_harm_cypher}},
}

STYLE_DETECTORS = {}
STYLE_RULE_META = {}
for fam_key, fam in STYLE_FAMILIES.items():
    for rule_key, fn in fam["rules"].items():
        STYLE_DETECTORS[rule_key] = fn
        STYLE_RULE_META[rule_key] = {"family": fam_key, "label": fam["label"]}

STYLE_ALL_RULES = list(STYLE_DETECTORS.keys())
