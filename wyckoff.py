"""
Wyckoff Method — codified event detectors (accumulation + distribution).

Based on Wyckoff schematics Phase A–E:
  Accumulation: SC, ST, Spring, Spring-Test, SOS, LPS, BUEC
  Distribution: BC, UT, UTAD, SOW, LPSY

Each returns list of (direction, proximal, distal, tag).
"""
import numpy as np

WYCK_DEFAULT = {
    "wyck_lb": 40,          # trading range lookback bars
    "wyck_min_w": 1.5,      # min TR width × ATR
    "wyck_max_w": 8.0,      # max TR width × ATR
    "wyck_vol_hi": 1.5,     # climax / SOS volume threshold
    "wyck_vol_lo": 0.85,    # spring test / LPS volume threshold
    "wyck_trend_lb": 20,    # prior trend context
    "spring_tol": 0.08,     # ATR fraction below support for spring
    "ut_tol": 0.08,         # ATR fraction above resistance for UT
    "sos_pct": 0.85,        # SOS must close above this fraction of TR
    "event_window": 20,     # bars to link spring→test, SOS→LPS
}


def _atr(B, i):
    return B.atr[i] if B.atr[i] > 0 else B.rng[i] + 1e-6


def _tr(B, i, p):
    """Trading range bounds over lookback ending at bar i-1 (exclude signal bar)."""
    W = p.get("wyck_lb", 40)
    if i < W + 5:
        return None
    lo = i - W
    hi = i  # exclusive — range built from prior bars
    rh = B.h[lo:hi].max()
    rl = B.l[lo:hi].min()
    width = rh - rl
    atr_i = _atr(B, i)
    if width <= 0:
        return None
    wmin = p.get("wyck_min_w", 1.5) * atr_i
    wmax = p.get("wyck_max_w", 8.0) * atr_i
    if width < wmin or width > wmax:
        return None
    mid = (rh + rl) / 2.0
    return dict(lo=lo, hi=hi, rh=rh, rl=rl, mid=mid, width=width, atr=atr_i, W=W)


def _vol_ratio(B, i, lo, hi):
    seg = B.vol[max(lo, 0):hi]
    if len(seg) == 0:
        return 1.0
    m = float(np.mean(seg))
    return float(B.vol[i] / m) if m > 0 else 1.0


def _prior_downtrend(B, i, p):
    """Context for accumulation: decline into the TR."""
    lb = p.get("wyck_trend_lb", 20)
    W = p.get("wyck_lb", 40)
    lo = max(0, i - W - lb)
    mid = i - W // 2
    if mid <= lo:
        return True
    return B.c[lo] > B.c[mid] and B.c[mid] >= B.c[i - W // 2]


def _prior_uptrend(B, i, p):
    """Context for distribution: rally into the TR."""
    lb = p.get("wyck_trend_lb", 20)
    W = p.get("wyck_lb", 40)
    lo = max(0, i - W - lb)
    mid = i - W // 2
    if mid <= lo:
        return True
    return B.c[lo] < B.c[mid] and B.c[mid] <= B.c[i - W // 2]


def _find_recent_spring(B, i, tr, p):
    """Bar index of recent spring (false break below rl) within window."""
    win = p.get("event_window", 20)
    tol = p.get("spring_tol", 0.08) * tr["atr"]
    for j in range(max(tr["lo"], i - win), i):
        if B.l[j] < tr["rl"] - tol * 0.3 and B.c[j] > tr["rl"]:
            return j, B.l[j]
    return None, None


def _find_recent_ut(B, i, tr, p):
    win = p.get("event_window", 20)
    tol = p.get("ut_tol", 0.08) * tr["atr"]
    for j in range(max(tr["lo"], i - win), i):
        if B.h[j] > tr["rh"] + tol * 0.3 and B.c[j] < tr["rh"]:
            return j, B.h[j]
    return None, None


def _find_recent_sos(B, i, tr, p):
    """Recent sign of strength: close above TR resistance."""
    win = p.get("event_window", 20)
    sos_pct = p.get("sos_pct", 0.85)
    level = tr["rl"] + sos_pct * tr["width"]
    for j in range(max(tr["lo"], i - win), i):
        if B.c[j] > level and B.body[j] > 0 and _vol_ratio(B, j, tr["lo"], j) >= 1.1:
            return j
    return None


def _find_recent_sow(B, i, tr, p):
    win = p.get("event_window", 20)
    level = tr["rl"] + (1 - p.get("sos_pct", 0.85)) * tr["width"]
    for j in range(max(tr["lo"], i - win), i):
        if B.c[j] < level and B.body[j] < 0 and _vol_ratio(B, j, tr["lo"], j) >= 1.1:
            return j
    return None


# ── Phase A: Climax & Secondary Test ───────────────────────────────────────

def detect_wyck_sc(B, i, p=WYCK_DEFAULT):
    """Selling Climax: wide bear bar + high volume at/near TR low after decline."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_downtrend(B, i, p):
        return s
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    near_low = B.l[i] <= tr["rl"] + 0.25 * tr["width"]
    wide = B.rng[i] >= 1.2 * tr["atr"]
    bear = B.body[i] < -0.5 * tr["atr"]
    if near_low and wide and bear and vr >= p.get("wyck_vol_hi", 1.5):
        s.append(("long", tr["rl"], B.l[i], "WYCK-SC"))
    return s


def detect_wyck_st(B, i, p=WYCK_DEFAULT):
    """Secondary Test: retest SC low with lower volume (selling exhaustion)."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_downtrend(B, i, p):
        return s
    # find prior SC-like bar in range
    sc_low = tr["rl"]
    sc_i = None
    for j in range(tr["lo"], i):
        vr = _vol_ratio(B, j, tr["lo"], j)
        if B.l[j] <= tr["rl"] + 0.15 * tr["width"] and vr >= p.get("wyck_vol_hi", 1.5):
            sc_i = j
            sc_low = B.l[j]
    if sc_i is None or i - sc_i > 25:
        return s
    near = abs(B.l[i] - sc_low) <= 0.3 * tr["atr"]
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    if near and vr <= p.get("wyck_vol_hi", 1.5) and B.c[i] >= B.l[i] + 0.4 * B.rng[i]:
        s.append(("long", sc_low, B.l[i], "WYCK-ST"))
    return s


def detect_wyck_bc(B, i, p=WYCK_DEFAULT):
    """Buying Climax: wide bull bar + high volume at TR high after uptrend."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_uptrend(B, i, p):
        return s
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    near_high = B.h[i] >= tr["rh"] - 0.25 * tr["width"]
    wide = B.rng[i] >= 1.2 * tr["atr"]
    bull = B.body[i] > 0.5 * tr["atr"]
    if near_high and wide and bull and vr >= p.get("wyck_vol_hi", 1.5):
        s.append(("short", tr["rh"], B.h[i], "WYCK-BC"))
    return s


# ── Phase C: Spring / Upthrust ─────────────────────────────────────────────

def detect_wyck_spring(B, i, p=WYCK_DEFAULT):
    """
    Spring (Phase C): false break below TR support, close back inside.
    Wyckoff: volume on spring can vary; recovery into range is key.
    """
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_downtrend(B, i, p):
        return s
    tol = p.get("spring_tol", 0.08) * tr["atr"]
    pierce = B.l[i] < tr["rl"] - tol * 0.2
    recovery = B.c[i] > tr["rl"]
    bull_close = B.c[i] > B.o[i] or B.c[i] > tr["mid"]
    if pierce and recovery and bull_close:
        s.append(("long", tr["rl"], B.l[i], "WYCK-SPRING"))
    return s


def detect_wyck_spring_test(B, i, p=WYCK_DEFAULT):
    """Spring Test: low-volume retest of spring low after spring event."""
    s = []
    win = p.get("event_window", 20)
    if i < win + 5:
        return s
    spring_j = None
    spring_low = None
    for j in range(i - win, i):
        tr_j = _tr(B, j, p)
        if tr_j is None:
            continue
        tol = p.get("spring_tol", 0.08) * tr_j["atr"]
        if B.l[j] < tr_j["rl"] - tol * 0.2 and B.c[j] > tr_j["rl"]:
            spring_j = j
            spring_low = B.l[j]
    if spring_j is None:
        return s
    tr = _tr(B, i, p)
    if tr is None:
        return s
    near = abs(B.l[i] - spring_low) <= 0.35 * tr["atr"]
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    if near and vr <= p.get("wyck_vol_lo", 0.85) and B.c[i] > spring_low:
        s.append(("long", spring_low, B.l[i], "WYCK-TEST"))
    return s


def detect_wyck_ut(B, i, p=WYCK_DEFAULT):
    """Upthrust (UT): false break above TR resistance, close back inside."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_uptrend(B, i, p):
        return s
    tol = p.get("ut_tol", 0.08) * tr["atr"]
    pierce = B.h[i] > tr["rh"] + tol * 0.2
    recovery = B.c[i] < tr["rh"]
    bear_close = B.c[i] < B.o[i] or B.c[i] < tr["mid"]
    if pierce and recovery and bear_close:
        s.append(("short", tr["rh"], B.h[i], "WYCK-UT"))
    return s


def detect_wyck_utad(B, i, p=WYCK_DEFAULT):
    """UTAD: second upthrust after prior UT — distribution trap."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_uptrend(B, i, p):
        return s
    ut_i, _ = _find_recent_ut(B, i, tr, p)
    if ut_i is None:
        return s
    tol = p.get("ut_tol", 0.08) * tr["atr"]
    pierce = B.h[i] > tr["rh"] + tol * 0.1
    recovery = B.c[i] < tr["rh"]
    if pierce and recovery and i > ut_i + 2:
        s.append(("short", tr["rh"], B.h[i], "WYCK-UTAD"))
    return s


# ── Phase D: SOS / SOW / LPS / LPSY ───────────────────────────────────────

def detect_wyck_sos(B, i, p=WYCK_DEFAULT):
    """Sign of Strength: strong rally toward/above TR top on expanding spread+volume."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_downtrend(B, i, p):
        return s
    level = tr["rl"] + p.get("sos_pct", 0.85) * tr["width"]
    sj, _ = _find_recent_spring(B, i, tr, p)
    if sj is None and B.c[i] < tr["mid"]:
        return s  # prefer SOS after spring context
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    strong = B.body[i] > 0.6 * tr["atr"] and B.c[i] > level
    if strong and vr >= 1.15:
        s.append(("long", tr["mid"], min(B.l[i], tr["rl"]), "WYCK-SOS"))
    return s


def detect_wyck_sow(B, i, p=WYCK_DEFAULT):
    """Sign of Weakness: strong drop toward/below TR bottom after UT."""
    s = []
    tr = _tr(B, i, p)
    if tr is None or not _prior_uptrend(B, i, p):
        return s
    level = tr["rl"] + (1 - p.get("sos_pct", 0.85)) * tr["width"]
    ut_i, _ = _find_recent_ut(B, i, tr, p)
    if ut_i is None and B.c[i] > tr["mid"]:
        return s
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    strong = B.body[i] < -0.6 * tr["atr"] and B.c[i] < level
    if strong and vr >= 1.15:
        s.append(("short", tr["mid"], max(B.h[i], tr["rh"]), "WYCK-SOW"))
    return s


def detect_wyck_lps(B, i, p=WYCK_DEFAULT):
    """Last Point of Support: pullback after SOS on diminished volume."""
    s = []
    tr = _tr(B, i, p)
    if tr is None:
        return s
    sos_i = _find_recent_sos(B, i, tr, p)
    if sos_i is None or i <= sos_i:
        return s
    hold = B.l[i] >= tr["mid"] and B.c[i] > tr["rl"]
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    small = B.rng[i] <= 1.1 * tr["atr"]
    if hold and vr <= p.get("wyck_vol_lo", 0.85) * 1.1 and small and B.body[i] <= 0:
        s.append(("long", B.l[i], B.l[i], "WYCK-LPS"))
    return s


def detect_wyck_lpsy(B, i, p=WYCK_DEFAULT):
    """Last Point of Supply: pullback after SOW on diminished volume."""
    s = []
    tr = _tr(B, i, p)
    if tr is None:
        return s
    sow_i = _find_recent_sow(B, i, tr, p)
    if sow_i is None or i <= sow_i:
        return s
    hold = B.h[i] <= tr["mid"] and B.c[i] < tr["rh"]
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    small = B.rng[i] <= 1.1 * tr["atr"]
    if hold and vr <= p.get("wyck_vol_lo", 0.85) * 1.1 and small and B.body[i] >= 0:
        s.append(("short", B.h[i], B.h[i], "WYCK-LPSY"))
    return s


def detect_wyck_buec(B, i, p=WYCK_DEFAULT):
    """Backup to edge of creek: retest TR top (resistance→support) after SOS breakout."""
    s = []
    tr = _tr(B, i, p)
    if tr is None:
        return s
    sos_i = _find_recent_sos(B, i, tr, p)
    if sos_i is None:
        return s
    tol = 0.25 * tr["atr"]
    touch = B.l[i] <= tr["rh"] + tol and B.l[i] >= tr["rh"] - tol
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    if touch and vr <= 1.0 and B.c[i] > tr["rh"]:
        s.append(("long", tr["rh"], B.l[i], "WYCK-BUEC"))
    return s


# ── Legacy detector (original high-volume spring/UT) ───────────────────────

def detect_wyckoff_legacy(B, i, p=WYCK_DEFAULT):
    """Original WYCK rule: spring/UT with mandatory high volume (strict)."""
    s = []
    tr = _tr(B, i, p)
    if tr is None:
        return s
    vr = _vol_ratio(B, i, tr["lo"], tr["hi"])
    if vr < p.get("wyck_vol_hi", 1.5) * 0.8:
        return s
    if B.l[i] < tr["rl"] and B.c[i] > tr["rl"]:
        s.append(("long", tr["rl"], B.l[i], "WYCK"))
    if B.h[i] > tr["rh"] and B.c[i] < tr["rh"]:
        s.append(("short", tr["rh"], B.h[i], "WYCK"))
    return s


# ── Registry ───────────────────────────────────────────────────────────────

WYCK_CATEGORIES = {
    "phase_a": {
        "label": "Phase A — Climax & Secondary Test",
        "rules": {
            "WYCK_SC": detect_wyck_sc,
            "WYCK_ST": detect_wyck_st,
            "WYCK_BC": detect_wyck_bc,
        },
    },
    "phase_c": {
        "label": "Phase C — Spring / Upthrust (key Wyckoff entries)",
        "rules": {
            "WYCK_SPRING": detect_wyck_spring,
            "WYCK_TEST": detect_wyck_spring_test,
            "WYCK_UT": detect_wyck_ut,
            "WYCK_UTAD": detect_wyck_utad,
        },
    },
    "phase_d": {
        "label": "Phase D — SOS/SOW & LPS (confirmation entries)",
        "rules": {
            "WYCK_SOS": detect_wyck_sos,
            "WYCK_SOW": detect_wyck_sow,
            "WYCK_LPS": detect_wyck_lps,
            "WYCK_LPSY": detect_wyck_lpsy,
            "WYCK_BUEC": detect_wyck_buec,
        },
    },
    "legacy": {
        "label": "Legacy (original code)",
        "rules": {
            "WYCK": detect_wyckoff_legacy,
        },
    },
}

WYCK_DETECTORS = {}
WYCK_RULE_META = {}
WYCK_THEORY_TIER = {
    "WYCK_SPRING": "A — Phase C key entry (Wyckoff)",
    "WYCK_TEST": "A — best confirmation after spring",
    "WYCK_LPS": "A — classic Phase D entry",
    "WYCK_BUEC": "A — conservative post-SOS entry",
    "WYCK_SOS": "B — momentum, may be late",
    "WYCK_SOW": "B — momentum short",
    "WYCK_UT": "A — Phase C distribution trap",
    "WYCK_UTAD": "A — UT after distribution",
    "WYCK_LPSY": "A — Phase D short entry",
    "WYCK_SC": "C — catching falling knife",
    "WYCK_ST": "B — early accumulation",
    "WYCK_BC": "C — early distribution",
    "WYCK": "B — legacy spring/UT + high vol",
}

for cat_key, cat in WYCK_CATEGORIES.items():
    for rule_key, fn in cat["rules"].items():
        WYCK_DETECTORS[rule_key] = fn
        WYCK_RULE_META[rule_key] = {"category": cat_key, "label": cat["label"]}

WYCK_ALL_RULES = list(WYCK_DETECTORS.keys())
