"""
================================================================================
  SMC/ICT + RTM + Wyckoff  —  Rule-Based Price-Action Strategy (CORE)
================================================================================

This single file is the WHOLE model. It is 100% rule-based and deterministic:
there is NO machine-learning model and therefore NOTHING to train or retrain.
Every trade comes from a transparent price-action rule you can read below.

Rules (each one is a small, causal detector):
  WYCK   - Wyckoff Spring / Upthrust (fake break of a range, then reversal)
  FVG    - Fair Value Gap (price imbalance that gets re-filled)
  OB     - Order Block (last opposite candle before a strong move)
  BOS    - Break of Structure (trend continuation on the retest)
  DEMAND - Demand zone (strong bullish move out of a tight base)   [RTM]
  SUPPLY - Supply zone (strong bearish move out of a tight base)   [RTM]
  FL     - Flag Limit (last opposite candle right before an impulse)
  QM     - Quasimodo (reversal after a liquidity sweep)

Exit modes (backtest.py / live --optimized):
  trail          — SL + trailing 1R after +1R
  be1r-fixed-tp  — BE at +1R + fixed TP (OPTIMIZED preset uses 2R)

OPTIMIZED preset (--optimized): STABLE — OB+DEM, fib pullback, ATR regime filter,
  HTF trend H4+H1, BE@1.5R + htf_blend TP (2R floor), min SL $3, session 10-20.
  Legacy aggressive: --aggressive | more signals: --more-trades

Honest back-testing reads M5/M15 + M1 bars directly from MetaTrader 5
(via mt5_data.py) to resolve exactly how each candle was formed.
See backtest.py — no CSV files required.
================================================================================
"""
import numpy as np
import pandas as pd

import al_brooks as _ab
AB_DETECTORS = _ab.AB_DETECTORS
AB_RULE_META = _ab.AB_RULE_META

import wyckoff as _wy
WYCK_DETECTORS = _wy.WYCK_DETECTORS
WYCK_RULE_META = _wy.WYCK_RULE_META

import wave_styles as _ws
STYLE_DETECTORS = _ws.STYLE_DETECTORS
STYLE_RULE_META = _ws.STYLE_RULE_META

import extended_strategies as _ext
EXT_DETECTORS = _ext.EXT_DETECTORS
EXT_RULE_META = _ext.EXT_RULE_META

import spike_strategies as _spike
SPIKE_DETECTORS = _spike.SPIKE_DETECTORS
SPIKE_RULE_KEYS = _spike.SPIKE_RULE_KEYS

import regime as _regime


# ─────────────────────────── configuration ────────────────────────────────
TRAIL_TRIGGER = 1.0     # activate trailing after +1R in profit
TRAIL_DIST    = 1.0     # trail the stop 1R behind the best price
BREAKEVEN     = True    # move stop to entry when trailing activates
WAIT_BARS     = 48      # how many bars to wait for the limit entry to fill
MAX_HOLD      = 96      # max bars to hold a trade
MAX_CONCURRENT = 2      # max simultaneous open trades
VP_MODE       = "poc_inv"   # volume-profile momentum filter (long above POC)
SPREAD_USD    = 0.28    # WM Markets gold spread (price units)
RISK_PCT      = 0.01    # risk 1% of balance per trade
START_BALANCE = 1000.0

# ── STABLE preset (default --optimized): tuned for monthly consistency ─────
OPT_STABLE_RULES      = ("OB", "NDS", "DEMAND")  # NDS before DEMAND (nested first)
OPT_STABLE_MIN_SL     = 3.0     # wider SL — survives gold spikes better
OPT_STABLE_TP_R       = 2.0     # reachable TP → higher WR
OPT_STABLE_TP_MODE    = "htf_blend"  # max(2R, nearest H4/H1 zone)
OPT_STABLE_BE         = 1.5
OPT_STABLE_ATR_REGIME = True    # skip when ATR > 1.5× median (crash/rally chaos)
OPT_STABLE_ATR_RATIO  = 1.5
OPT_STABLE_ATR_LB     = 100
OPT_STABLE_FIB        = True
# BALANCED preset: STABLE filters + 2 concurrent positions (more trades, same quality)
OPT_BALANCED_CONCURRENT = 2
OPT_BALANCED_RULES    = ("OB", "NDS", "NDS_FVG", "DEMAND")  # +nested FVG vs STABLE
OPT_BALANCED_SESSION  = (8, 22)   # optional wider session in --balanced-plus
# DENSE preset: STABLE + 2 concurrent + fib exempt for nested (HTF) setups
OPT_DENSE_CONCURRENT  = 2
OPT_DENSE_FIB_NDS_EXEMPT = True   # NDS already has HTF confluence — skip fib retrace
OPT_DENSE_RULES       = (
    "OB", "NDS", "DEMAND", "HARM_BAT",           # core + harmonic Bat
    "MA_X_S", "SQZ_S", "ICT_SB_L", "ICT_MIT_S",  # extended research tier B
)
# Pre-spike expansion rules (compression → breakout); use --spike or dense-spike preset
OPT_SPIKE_RULES       = ("SPIKE_BRK_L", "SPIKE_BRK_S", "SPIKE_SQS_L", "SPIKE_SQS_S")
OPT_DENSE_SPIKE_RULES = OPT_DENSE_RULES + OPT_SPIKE_RULES
OPT_DENSE_PLUS_ATR = False        # dense-plus: skip ATR regime bar filter
# MEDIUM preset: STABLE + 2 concurrent + fib filter on OB only (DEM/NDS skip fib)
OPT_MEDIUM_CONCURRENT = 2
OPT_MEDIUM_FIB_OB_ONLY = True

# ── Legacy / optional presets (--more-trades, --high-wr, etc.) ─────────────
OPT_RULES         = ("OB", "NDS", "DEMAND", "FVG")
OPT_RULES_HIGH_WR = ("OB", "DEMAND")
OPT_MAX_CONCURRENT = 1
OPT_MIN_RISK_USD  = 2.5
OPT_FIXED_TP_R    = 2.5
OPT_TP_MODE       = "fixed"
OPT_BE_TRIGGER    = 2.0     # BE only after +2R — fewer $0 exits, more TP/SL
OPT_HTF_TREND     = True    # long only above H1 EMA, short only below
OPT_HTF_EMA       = 20
OPT_HTF_TFS       = ("H4", "H1")
OPT_MIN_TP_R      = 2.0
OPT_MAX_TP_R      = 10.0
OPT_HTF_LOOKBACK  = 100
OPT_SESSION_START = 10      # broker server hour (London/NY overlap approx.)
OPT_SESSION_END   = 20
OPT_MORE_SESSION  = (8, 22) # wider window for --more-trades
OPT_MORE_HTF_TFS  = ("H1",) # H1 trend only (less strict than H4+H1)
OPT_MORE_MIN_SL   = 2.0
OPT_FIB_LOOKBACK  = 48      # bars for swing high/low
OPT_FIB_LO        = 0.382   # retrace zone (ICT discount/premium)
OPT_FIB_HI        = 0.786
OPT_NDS_MAX_RATIO = 0.6     # LTF zone width <= 60% of HTF parent (0.4 = very strict)
OPT_NDS_HTF_LB    = 40      # HTF bars to scan for parent zones
# Extended NDS family (nested LTF pattern inside HTF demand/supply)
NDS_FAMILY_RULES  = ("NDS", "NDS_OB", "NDS_FVG", "NDS_BOS", "NDS_WYCK", "NDS_QM", "NDS_FRESH")
OPT_NDS_EXTENDED  = ("OB",) + NDS_FAMILY_RULES + ("DEMAND",)
OPT_EXIT_MODE     = "be1r-fixed-tp"

DEFAULT_PARAMS = {"impulse_mult": 1.2, "fl_mult": 1.0, "qm_lookback": 20,
                  "base_max": 0.5,
                  "wyck_lb": 40, "wyck_max_w": 8.0, "wyck_min_w": 1.5,
                  "wyck_vol": 1.2,
                  # human-like tolerance (multiples of ATR at signal bar)
                  "entry_tol_atr": 0.0,       # fill limit if wick within this of proximal
                  "detect_tol_atr": 0.0,      # softer pattern detection (FVG gap, etc.)
                  "zone_tol_atr": 0.0,        # VWAP / level "close enough" touch
                  "invalidate_tol_atr": 0.0}  # extra room before zone invalidation

# Simple English explanations printed when a trade opens (PowerShell-friendly).
RULE_EXPLANATIONS = {
    "WYCK":   "Wyckoff Spring/Upthrust: price faked a break of the range and snapped back.",
    "FVG":    "Fair Value Gap: price left an imbalance and is returning to fill it.",
    "OB":     "Order Block: entering at the last opposite candle before a strong move.",
    "BOS":    "Break of Structure: trend continued; entering on the retest of the broken level.",
    "DEMAND": "Demand zone: strong bullish move out of a tight base; buying the base.",
    "SUPPLY": "Supply zone: strong bearish move out of a tight base; selling the base.",
    "FL":     "Flag Limit: last opposite candle right before an impulsive move.",
    "QM":     "Quasimodo: reversal after a liquidity sweep of a prior swing.",
    "NDS":    "Nested Demand/Supply: M15 zone inside H4/H1 zone (same direction).",
    "NDS-OB": "Nested OB: order block inside HTF demand/supply zone.",
    "NDS-FVG":"Nested FVG: fair value gap inside HTF zone (imbalance fill).",
    "NDS-BOS":"Nested BOS: structure break while inside HTF zone.",
    "NDS-WYCK":"Nested Wyckoff: spring/upthrust inside HTF zone (liquidity sweep).",
    "NDS-QM": "Nested Quasimodo: sweep + reversal inside HTF zone.",
    "NDS-FRESH": "Nested zone + first retest of HTF zone (untested).",
    "SPIKE-SQS": "Vol squeeze release: quiet ATR + BB/KC squeeze then expansion break.",
    "SPIKE-BRK": "Pre-spike range break: ATR compression then impulse through the box.",
    "SPIKE-VOL": "Volume surge + range expansion (tick volume spike).",
}


# ─────────────────────────── data loading ─────────────────────────────────
def load_csv(filepath):
    """Load an OHLCV CSV with a datetime column into a time-indexed DataFrame."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    date_cols = [c for c in df.columns if c in ("datetime", "date", "time", "timestamp")]
    if date_cols:
        col = df[date_cols[0]].astype(str).str.strip()
        df["datetime"] = pd.to_datetime(col, format="mixed", errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Found: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df.dropna(subset=["open", "high", "low", "close"])


class Bars:
    """Fast numpy view of OHLCV + derived ATR/body/range arrays."""
    def __init__(self, df):
        self.o = df["open"].values.astype(float)
        self.h = df["high"].values.astype(float)
        self.l = df["low"].values.astype(float)
        self.c = df["close"].values.astype(float)
        self.vol = df["volume"].values.astype(float) if "volume" in df.columns \
            else np.ones(len(df))
        self.tp = (self.h + self.l + self.c) / 3.0
        self.t = df.index
        self.n = len(df)
        tr = np.maximum(self.h[1:] - self.l[1:],
                        np.maximum(np.abs(self.h[1:] - self.c[:-1]),
                                   np.abs(self.l[1:] - self.c[:-1])))
        self.atr = np.concatenate([[self.h[0] - self.l[0]],
                                   pd.Series(tr).rolling(14).mean().bfill().values])
        self.body = self.c - self.o
        self.rng = (self.h - self.l)


# ─────────────────────────── detectors (rules) ────────────────────────────
# Each returns a list of (direction, proximal, distal, tag).
#   direction: "long" / "short"
#   proximal : limit entry price (price must return here to fill)
#   distal   : invalidation side (used to place the structural stop)

def _atr_tol(atr_i, p, key):
    """ATR-scaled tolerance from detector params (0 = exact legacy behavior)."""
    return float(p.get(key, 0.0) or 0.0) * atr_i


def detect_fvg(B, i, p=DEFAULT_PARAMS):
    s = []
    atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)
    dt = _atr_tol(atr_i, p, "detect_tol_atr")
    if B.l[i] > B.h[i - 2] - dt:
        s.append(("long", B.l[i], B.h[i - 2], "FVG"))
    if B.h[i] < B.l[i - 2] + dt:
        s.append(("short", B.h[i], B.l[i - 2], "FVG"))
    return s


def detect_demand_supply(B, i, p=DEFAULT_PARAMS):
    s = []
    atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)
    mult = p.get("impulse_mult", 1.2)
    dt = _atr_tol(atr_i, p, "detect_tol_atr")
    base_small = abs(B.body[i - 1]) < (p.get("base_max", 0.5) * atr_i + dt)
    if base_small and (B.body[i] > mult * atr_i - dt) and (B.c[i] > B.h[i - 1] - dt):
        s.append(("long", B.h[i - 1], B.l[i - 1], "DEMAND"))
    if base_small and (-B.body[i] > mult * atr_i - dt) and (B.c[i] < B.l[i - 1] + dt):
        s.append(("short", B.l[i - 1], B.h[i - 1], "SUPPLY"))
    return s


def zone_lo_hi(proximal, distal):
    return min(proximal, distal), max(proximal, distal)


def nested_zone_inside(ltf_lo, ltf_hi, htf_lo, htf_hi, max_width_ratio=OPT_NDS_MAX_RATIO):
    """LTF zone fully inside HTF zone and narrow enough (Nested Zones rule)."""
    htf_w = htf_hi - htf_lo
    ltf_w = ltf_hi - ltf_lo
    if htf_w <= 0 or ltf_w <= 0:
        return False
    inside = ltf_lo >= htf_lo and ltf_hi <= htf_hi
    return inside and ltf_w <= max_width_ratio * htf_w


def htf_demand_supply_zones(hB, end_idx, lookback=OPT_NDS_HTF_LB, p=DEFAULT_PARAMS):
    zones = []
    lo = max(3, end_idx - lookback)
    for j in range(lo, end_idx + 1):
        for direction, prox, dist, tag in detect_demand_supply(hB, j, p):
            zlo, zhi = zone_lo_hi(prox, dist)
            zones.append((direction, zlo, zhi, tag))
    return zones


def find_htf_parent_zone(ltf_lo, ltf_hi, direction, htf_context, signal_time,
                         htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                         htf_lookback=OPT_NDS_HTF_LB):
    """Return (hB, parent_bar, hlo, hhi) for the HTF zone nesting this LTF zone."""
    if htf_context is None or signal_time is None:
        return None
    for tf in htf_tfs:
        ctx = htf_context.get(tf)
        if ctx is None:
            continue
        hB = ctx if isinstance(ctx, Bars) else Bars(ctx)
        idx = htf_bar_index(hB, signal_time)
        for j in range(max(3, idx - htf_lookback), idx + 1):
            for d, prox, dist, _tag in detect_demand_supply(hB, j):
                if d != direction:
                    continue
                hlo, hhi = zone_lo_hi(prox, dist)
                if nested_zone_inside(ltf_lo, ltf_hi, hlo, hhi, max_ratio):
                    return hB, j, hlo, hhi
    return None


def is_htf_zone_fresh(hB, parent_bar, signal_bar, direction, hlo, hhi):
    """True if HTF zone had no touch between formation and signal (first retest)."""
    for k in range(parent_bar + 1, signal_bar):
        if hB.l[k] <= hhi and hB.h[k] >= hlo:
            return False
    return True


def is_nds_setup(B, bar_i, direction, proximal, distal, htf_context, signal_time,
                 htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                 htf_lookback=OPT_NDS_HTF_LB, require_fresh=False):
    """True when this zone is nested inside an HTF demand/supply zone."""
    if htf_context is None or signal_time is None:
        return False
    ltf_lo, ltf_hi = zone_lo_hi(proximal, distal)
    parent = find_htf_parent_zone(ltf_lo, ltf_hi, direction, htf_context, signal_time,
                                  htf_tfs, max_ratio, htf_lookback)
    if parent is None:
        return False
    if not require_fresh:
        return True
    hB, pj, hlo, hhi = parent
    idx = htf_bar_index(hB, signal_time)
    return is_htf_zone_fresh(hB, pj, idx, direction, hlo, hhi)


def resolve_trade_tag(name, tag, B, bar_i, direction, proximal, distal,
                      htf_context, signal_time, htf_tfs=OPT_HTF_TFS,
                      nds_max_ratio=OPT_NDS_MAX_RATIO, nds_enabled=False):
    """Rule label for trade list + per-rule stats (never raw SUPPLY)."""
    if tag.startswith("NDS") or name.startswith("NDS"):
        return tag if tag.startswith("NDS") else name.replace("_", "-")
    if name == "NDS" or tag == "NDS":
        return "NDS"
    if tag in ("SUPPLY", "DEMAND"):
        if is_nds_setup(
                B, bar_i, direction, proximal, distal, htf_context, signal_time,
                htf_tfs, nds_max_ratio):
            return "NDS"
        if nds_enabled and name == "DEMAND":
            return "NDS"
        return "DEMAND"
    return tag


def trades_for_enabled_rule(rule_name, trades):
    """
    Match trades to an enabled rule using the same label shown in the trade list
    (setup / display tag). Nested DEMAND inside HTF zones displays as NDS.
    """
    def disp(t):
        return t.get("setup") or t.get("rule", "?")

    if rule_name == "NDS":
        return [t for t in trades
                if disp(t) == "NDS" or str(disp(t)).startswith("NDS-")]
    if rule_name == "DEMAND":
        return [t for t in trades if disp(t) == "DEMAND"]
    if rule_name == "OB":
        return [t for t in trades if disp(t) == "OB"]
    if rule_name == "FVG":
        return [t for t in trades if disp(t) == "FVG"]

    _tag = {
        "HARM_BAT": "HARM-BAT", "HARM_GART": "HARM-GART", "HARM_BAT_S": "HARM-GART",
        "MA_X_S": "MA-X", "MA_X_L": "MA-X",
        "SQZ_S": "SQZ", "SQZ_L": "SQZ",
        "ICT_SB_L": "ICT-SB", "ICT_SB_S": "ICT-SB",
        "ICT_MIT_S": "ICT-MIT", "ICT_MIT_L": "ICT-MIT",
        "ICT_BRK_L": "ICT-BRK", "ICT_BRK_S": "ICT-BRK",
        "ICT_OTE_L": "ICT-OTE", "ICT_OTE_S": "ICT-OTE",
        "ICT_AMD_L": "ICT-AMD", "ICT_AMD_S": "ICT-AMD",
        "ICT_MSS_L": "ICT-MSS", "ICT_MSS_S": "ICT-MSS",
        "ICT_JUD_L": "ICT-JUD", "ICT_JUD_S": "ICT-JUD",
        "ICT_NYO_L": "ICT-NYO", "ICT_NYO_S": "ICT-NYO",
        "SPIKE_SQS_L": "SPIKE-SQS", "SPIKE_SQS_S": "SPIKE-SQS",
        "SPIKE_BRK_L": "SPIKE-BRK", "SPIKE_BRK_S": "SPIKE-BRK",
        "SPIKE_VOL_L": "SPIKE-VOL", "SPIKE_VOL_S": "SPIKE-VOL",
        "FL": "FL", "QM": "QM", "BOS": "BOS", "WYCK": "WYCK",
    }
    if rule_name in _tag:
        want = _tag[rule_name]
        return [t for t in trades if disp(t) == want]
    if rule_name in EXT_DETECTORS or rule_name in STYLE_DETECTORS:
        return [t for t in trades if t.get("rule") == rule_name]
    dash = rule_name.replace("_", "-")
    return [t for t in trades if t.get("rule") == rule_name or disp(t) == dash]


def nested_setups_from(B, i, detector_fn, tag, htf_context, signal_time,
                       p=DEFAULT_PARAMS, htf_tfs=OPT_HTF_TFS,
                       max_ratio=OPT_NDS_MAX_RATIO, htf_lookback=OPT_NDS_HTF_LB,
                       require_fresh=False):
    """LTF detector setups whose zone is nested inside HTF demand/supply."""
    if htf_context is None or signal_time is None:
        return []
    out = []
    seen = set()
    for direction, prox, dist, _ in detector_fn(B, i, p):
        if not is_nds_setup(B, i, direction, prox, dist, htf_context, signal_time,
                            htf_tfs, max_ratio, htf_lookback, require_fresh):
            continue
        key = (tag, direction, round(prox, 2))
        if key not in seen:
            seen.add(key)
            out.append((direction, prox, dist, tag))
    return out


def detect_nds(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
               htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
               htf_lookback=OPT_NDS_HTF_LB):
    """Nested Demand/Supply: M15 D/S zone inside H4/H1 zone."""
    return nested_setups_from(B, i, detect_demand_supply, "NDS", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_ob(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                  htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                  htf_lookback=OPT_NDS_HTF_LB):
    return nested_setups_from(B, i, detect_order_block, "NDS-OB", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_fvg(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                   htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                   htf_lookback=OPT_NDS_HTF_LB):
    return nested_setups_from(B, i, detect_fvg, "NDS-FVG", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_bos(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                   htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                   htf_lookback=OPT_NDS_HTF_LB):
    return nested_setups_from(B, i, detect_bos, "NDS-BOS", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_wyck(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                    htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                    htf_lookback=OPT_NDS_HTF_LB):
    return nested_setups_from(B, i, detect_wyckoff, "NDS-WYCK", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_qm(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                  htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                  htf_lookback=OPT_NDS_HTF_LB):
    return nested_setups_from(B, i, detect_quasimodo, "NDS-QM", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback)


def detect_nds_fresh(B, i, p=DEFAULT_PARAMS, htf_context=None, signal_time=None,
                     htf_tfs=OPT_HTF_TFS, max_ratio=OPT_NDS_MAX_RATIO,
                     htf_lookback=OPT_NDS_HTF_LB):
    """Nested D/S on first retest of untouched HTF zone."""
    return nested_setups_from(B, i, detect_demand_supply, "NDS-FRESH", htf_context,
                              signal_time, p, htf_tfs, max_ratio, htf_lookback,
                              require_fresh=True)


NDS_FAMILY = {
    "NDS": detect_nds,
    "NDS_OB": detect_nds_ob,
    "NDS_FVG": detect_nds_fvg,
    "NDS_BOS": detect_nds_bos,
    "NDS_WYCK": detect_nds_wyck,
    "NDS_QM": detect_nds_qm,
    "NDS_FRESH": detect_nds_fresh,
}


def is_nds_rule(name):
    return name in NDS_FAMILY or name == "NDS"


def _risk_cfg(key, default):
    """Read RISK from floating_config once (lazy import keeps strategy standalone)."""
    if key not in _RISK_CACHE:
        val = default
        try:
            import floating_config as FC
            val = FC.RISK.get(key, default)
        except Exception:
            pass
        _RISK_CACHE[key] = val
    return _RISK_CACHE[key]


_RISK_CACHE = {}
SETUP_REJECTS = {"inverted": 0, "marketable": 0, "flat": 0}


def validate_setups(setups, B, i, rule=None):
    """Drop setups whose own geometry contradicts the trade they describe.

    Two failures were common enough to distort results:

    inverted    distal on the profit side of proximal, so the "invalidation"
                level sits where the trade is winning. min()/max() in the stop
                math then silently threw the rule's level away and used a
                generic swing instead, i.e. the rule was not trading itself.
                Butterfly/Crab did this on 100% of signals, CVD on 100%, and
                FVG on 36% once detect_tol_atr let the gap go slightly negative.

    marketable  a long's limit above the signal close (or a short's below), so
                the resting order is already through the market. The backtest
                fills it at that better-than-market price; live has to pay the
                market. ICT_BRK did this on ~78% of signals.

    A zero-width zone (distal == proximal) is kept: the rule simply declares no
    structural level of its own and the swing-plus-buffer stop takes over. It is
    counted so the behaviour stays visible.
    """
    if not setups or not _risk_cfg("validate_setups", True):
        return setups
    close = float(B.c[i])
    eps = max(abs(close), 1.0) * 1e-9
    out = []
    for st in setups:
        direction, prox, dist = st[0], float(st[1]), float(st[2])
        if direction == "long":
            if dist > prox + eps:
                SETUP_REJECTS["inverted"] += 1
                continue
            if prox > close + eps:
                SETUP_REJECTS["marketable"] += 1
                continue
        else:
            if dist < prox - eps:
                SETUP_REJECTS["inverted"] += 1
                continue
            if prox < close - eps:
                SETUP_REJECTS["marketable"] += 1
                continue
        if abs(dist - prox) <= eps:
            SETUP_REJECTS["flat"] += 1
        out.append(st)
    return out


def _raw_rule_setups(name, B, i, p, htf_context, signal_time, htf_tfs,
                     nds_max_ratio, spike_params):
    if name in SPIKE_DETECTORS:
        sp = spike_params if spike_params is not None else _spike.SPIKE_DEFAULT
        return SPIKE_DETECTORS[name](B, i, sp)
    if name in NDS_FAMILY:
        fn = NDS_FAMILY[name]
        return fn(B, i, p, htf_context, signal_time, htf_tfs, nds_max_ratio)
    if name in AB_DETECTORS:
        return AB_DETECTORS[name](B, i, _ab.AB_DEFAULT)
    if name in WYCK_DETECTORS:
        return WYCK_DETECTORS[name](B, i, _wy.WYCK_DEFAULT)
    if name in STYLE_DETECTORS:
        return STYLE_DETECTORS[name](B, i, _ws.STYLE_DEFAULT)
    if name in EXT_DETECTORS:
        return EXT_DETECTORS[name](B, i, _ext.EXT_DEFAULT)
    if name not in DETECTORS:
        return []
    return DETECTORS[name](B, i, p)


def iter_rule_setups(name, B, i, p=DEFAULT_PARAMS, htf_context=None,
                     signal_time=None, htf_tfs=OPT_HTF_TFS,
                     nds_max_ratio=OPT_NDS_MAX_RATIO, spike_params=None):
    """Yield validated setups for a rule; NDS family needs HTF context."""
    raw = _raw_rule_setups(name, B, i, p, htf_context, signal_time, htf_tfs,
                           nds_max_ratio, spike_params)
    return validate_setups(raw, B, i, rule=name)


def detect_flag_limit(B, i, p=DEFAULT_PARAMS):
    s = []
    atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)
    mult = p.get("fl_mult", 1.0)
    if (B.body[i - 1] < 0) and (B.body[i] > mult * atr_i) and (B.c[i] > B.h[i - 1]):
        s.append(("long", max(B.o[i - 1], B.c[i - 1]), B.l[i - 1], "FL"))
    if (B.body[i - 1] > 0) and (-B.body[i] > mult * atr_i) and (B.c[i] < B.l[i - 1]):
        s.append(("short", min(B.o[i - 1], B.c[i - 1]), B.h[i - 1], "FL"))
    return s


def detect_quasimodo(B, i, p=DEFAULT_PARAMS):
    lookback = p.get("qm_lookback", 20)
    s = []
    if i < lookback + 2:
        return s
    prior_low = B.l[i - lookback:i].min()
    prior_high = B.h[i - lookback:i].max()
    if np.any(B.l[i - 3:i + 1] < prior_low) and (B.c[i] > prior_high):
        s.append(("long", prior_low + 0.3 * (prior_high - prior_low), prior_low, "QM"))
    if np.any(B.h[i - 3:i + 1] > prior_high) and (B.c[i] < prior_low):
        s.append(("short", prior_high - 0.3 * (prior_high - prior_low), prior_high, "QM"))
    return s


def detect_order_block(B, i, p=DEFAULT_PARAMS):
    s = []
    atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)
    mult = p.get("impulse_mult", 1.2)
    dt = _atr_tol(atr_i, p, "detect_tol_atr")
    if (B.body[i - 1] < 0) and (B.body[i] > mult * atr_i - dt) and (B.c[i] > B.h[i - 1] - dt):
        s.append(("long", B.h[i - 1], B.l[i - 1], "OB"))
    if (B.body[i - 1] > 0) and (-B.body[i] > mult * atr_i - dt) and (B.c[i] < B.l[i - 1] + dt):
        s.append(("short", B.l[i - 1], B.h[i - 1], "OB"))
    return s


def detect_bos(B, i, p=DEFAULT_PARAMS):
    s = []
    lb = 20
    if i < lb + 1:
        return s
    prior_high = B.h[i - lb:i].max()
    prior_low = B.l[i - lb:i].min()
    recent_low = B.l[i - 10:i + 1].min()
    recent_high = B.h[i - 10:i + 1].max()
    if B.c[i] > prior_high:
        s.append(("long", prior_high, recent_low, "BOS"))
    if B.c[i] < prior_low:
        s.append(("short", prior_low, recent_high, "BOS"))
    return s


def detect_wyckoff(B, i, p=DEFAULT_PARAMS):
    """Legacy WYCK — delegates to wyckoff.py high-volume spring/UT."""
    wp = dict(_wy.WYCK_DEFAULT)
    wp["wyck_lb"] = p.get("wyck_lb", 40)
    wp["wyck_max_w"] = p.get("wyck_max_w", 8.0)
    wp["wyck_min_w"] = p.get("wyck_min_w", 1.5)
    wp["wyck_vol_hi"] = p.get("wyck_vol", 1.2)
    return _wy.detect_wyckoff_legacy(B, i, wp)


# The final, optimised rule set (LIQ removed: weakest; ACCDIST/SOS removed: redundant with BOS).
DETECTORS = {
    "WYCK":   detect_wyckoff,
    "FVG":    detect_fvg,
    "OB":     detect_order_block,
    "BOS":    detect_bos,
    "DEMAND": detect_demand_supply,
    "NDS":    detect_nds,   # use iter_rule_setups() — needs HTF context
    "FL":     detect_flag_limit,
    "QM":     detect_quasimodo,
}


def atr_regime_pass(B, bar_i, lookback=OPT_STABLE_ATR_LB, max_ratio=OPT_STABLE_ATR_RATIO):
    """Skip entries when volatility is abnormally high (e.g. gold crash/rally)."""
    if bar_i < lookback + 5:
        return True
    cur = B.atr[bar_i] if B.atr[bar_i] > 0 else B.rng[bar_i]
    med = float(np.median(B.atr[max(0, bar_i - lookback):bar_i]))
    if med <= 0:
        return True
    return cur <= max_ratio * med


def stable_settings(nds_mode=False, include_nds=True, nds_extended=False):
    """Default preset: OB+NDS+DEM, fib pullback, ATR regime, HTF trend, blend TP."""
    if nds_extended:
        rules = OPT_NDS_EXTENDED
    elif nds_mode:
        rules = ("OB",) + NDS_FAMILY_RULES
    elif include_nds:
        rules = OPT_STABLE_RULES
    else:
        rules = ("OB", "DEMAND")
    kw = _stable_kw(list(rules))
    kw["nds_mode"] = nds_mode
    kw["nds_extended"] = nds_extended
    return kw


def balanced_settings(plus=False):
    """
    More trades than STABLE with similar WR.
    Default (--balanced): STABLE + 2 concurrent positions.
    Plus (--balanced-plus): + NDS-FVG rule + session 8-22.
    """
    rules = list(OPT_BALANCED_RULES) if plus else list(OPT_STABLE_RULES)
    sess = OPT_BALANCED_SESSION if plus else (OPT_SESSION_START, OPT_SESSION_END)
    kw = _stable_kw(rules)
    kw["max_concurrent"] = OPT_BALANCED_CONCURRENT
    kw["session_start"] = sess[0]
    kw["session_end"] = sess[1]
    kw["balanced"] = True
    kw["balanced_plus"] = plus
    return kw


def fib_filter_pass(B, bar_i, direction, entry, rule_name, proximal, distal,
                    htf_context, signal_time, htf_tfs=OPT_HTF_TFS,
                    nds_max_ratio=OPT_NDS_MAX_RATIO, require_fib=False,
                    fib_nds_exempt=False, fib_ob_only=False, fib_spike_exempt=False,
                    fib_demand_exempt=False):
    """Fib retrace gate — exempt nested; fib_ob_only = fib on OB only (+ optional DEMAND)."""
    if not require_fib:
        return True
    if fib_spike_exempt and rule_name in SPIKE_RULE_KEYS:
        return True
    nested = is_nds_setup(
        B, bar_i, direction, proximal, distal, htf_context, signal_time,
        htf_tfs, nds_max_ratio) or is_nds_rule(rule_name)
    if fib_nds_exempt and nested:
        return True
    if fib_ob_only:
        if rule_name == "OB":
            return fib_retrace_pass(B, bar_i, direction, entry)
        if rule_name == "DEMAND" and not nested and not fib_demand_exempt:
            return fib_retrace_pass(B, bar_i, direction, entry)
        return True
    return fib_retrace_pass(B, bar_i, direction, entry)


def dense_settings():
    """
    More trades than STABLE with ~same win-rate.
    STABLE + 2 concurrent + skip fib filter for nested (NDS) setups + HARM_BAT.
    Nested zones already sit inside H4/H1 — fib retrace is redundant there.
    """
    kw = _stable_kw(list(OPT_DENSE_RULES))
    kw["max_concurrent"] = OPT_DENSE_CONCURRENT
    kw["fib_nds_exempt"] = OPT_DENSE_FIB_NDS_EXEMPT
    kw["dense"] = True
    return kw


def dense_plus_settings(wide=False):
    """
    DENSE with relaxed filters for more trades.
    Default (--dense-plus): skip ATR regime (~18 trades, ~78%% WR).
    Wide (--dense-wide): + session 8-22 (~24 trades, ~71%% WR).
    """
    kw = dense_settings()
    kw["require_atr_regime"] = OPT_DENSE_PLUS_ATR
    if wide:
        kw["session_start"] = 8
        kw["session_end"] = 22
    kw["dense"] = False
    kw["dense_plus"] = True
    kw["dense_wide"] = wide
    return kw


def dense_spike_settings(wide=True):
    """
    DENSE-WIDE + pre-spike rules (SPIKE-BRK, SPIKE-SQS).
    Spike rules bypass ATR high-vol block and fib filter — they target expansion.
    """
    kw = dense_plus_settings(wide=wide)
    kw["enabled"] = list(OPT_DENSE_SPIKE_RULES)
    kw["spike_mode"] = True
    kw["fib_spike_exempt"] = True
    kw["spike_params"] = dict(_spike.SPIKE_DEFAULT)
    return kw


def medium_settings():
    """
    Sweet spot: more trades than STABLE/DENSE, WR ~74%.
    DENSE logic (2 pos + fib nested exempt) + session 8-22.
    """
    kw = _stable_kw(list(OPT_STABLE_RULES))
    kw["max_concurrent"] = OPT_MEDIUM_CONCURRENT
    kw["fib_nds_exempt"] = True
    kw["session_start"] = 8
    kw["session_end"] = 22
    kw["medium"] = True
    return kw


def _stable_kw(rules):
    """Shared STABLE/BALANCED config body."""
    return {
        "enabled": list(rules),
        "exit_mode": OPT_EXIT_MODE,
        "fixed_tp_r": OPT_STABLE_TP_R,
        "be_trigger": OPT_STABLE_BE,
        "tp_mode": OPT_STABLE_TP_MODE,
        "htf_tfs": list(OPT_HTF_TFS),
        "min_tp_r": OPT_MIN_TP_R,
        "max_tp_r": OPT_MAX_TP_R,
        "htf_lookback": OPT_HTF_LOOKBACK,
        "htf_trend": OPT_HTF_TREND,
        "htf_ema": OPT_HTF_EMA,
        "max_concurrent": OPT_MAX_CONCURRENT,
        "min_risk_usd": OPT_STABLE_MIN_SL,
        "session_start": OPT_SESSION_START,
        "session_end": OPT_SESSION_END,
        "spread_usd": SPREAD_USD,
        "vp_mode": VP_MODE,
        "require_confluence": False,
        "require_bos": False,
        "require_fib": OPT_STABLE_FIB,
        "require_atr_regime": OPT_STABLE_ATR_REGIME,
        "atr_max_ratio": OPT_STABLE_ATR_RATIO,
        "atr_lookback": OPT_STABLE_ATR_LB,
        "nds_mode": False,
        "nds_extended": False,
        "nds_max_ratio": OPT_NDS_MAX_RATIO,
        "fib_nds_exempt": False,
        "fib_ob_only": False,
        "atr_nds_only": False,
        "htf_trend_tfs": None,
        "balanced": False,
        "balanced_plus": False,
        "dense": False,
        "medium": False,
        "dense_plus": False,
        "dense_wide": False,
    }


def optimized_settings(high_wr=False, more_trades=False, quality=False, fib=False,
                       aggressive=False, nds=False, nds_mode=False, no_nds=False,
                       nds_extended=False, balanced=False, balanced_plus=False,
                       dense=False, medium=False, dense_plus=False, dense_wide=False,
                       spike=False):
    """Preset router. Default (--optimized alone) = STABLE."""
    if spike and (dense_wide or dense_plus or dense):
        return dense_spike_settings(wide=dense_wide or not dense_plus)
    if dense_wide:
        return dense_plus_settings(wide=True)
    if dense_plus:
        return dense_plus_settings(wide=False)
    if medium:
        return medium_settings()
    if dense:
        return dense_settings()
    if balanced or balanced_plus:
        return balanced_settings(plus=balanced_plus)
    if not (aggressive or more_trades or high_wr or quality or fib):
        cfg = stable_settings(nds_mode=nds_mode, include_nds=not no_nds,
                              nds_extended=nds_extended)
        if nds and not nds_mode and not no_nds:
            cfg = dict(cfg)
            if "NDS" not in cfg["enabled"]:
                cfg["enabled"] = list(dict.fromkeys(["OB", "NDS"] + list(cfg["enabled"])))
        return cfg
    if more_trades or aggressive:
        sess = OPT_MORE_SESSION if more_trades else (OPT_SESSION_START, OPT_SESSION_END)
        return {
            "enabled": list(OPT_RULES),
            "exit_mode": OPT_EXIT_MODE,
            "fixed_tp_r": OPT_FIXED_TP_R,
            "be_trigger": OPT_BE_TRIGGER,
            "tp_mode": OPT_TP_MODE,
            "htf_tfs": list(OPT_MORE_HTF_TFS if more_trades else OPT_HTF_TFS),
            "min_tp_r": OPT_MIN_TP_R,
            "max_tp_r": OPT_MAX_TP_R,
            "htf_lookback": OPT_HTF_LOOKBACK,
            "htf_trend": OPT_HTF_TREND,
            "htf_ema": OPT_HTF_EMA,
            "max_concurrent": OPT_MAX_CONCURRENT,
            "min_risk_usd": OPT_MORE_MIN_SL if more_trades else OPT_MIN_RISK_USD,
            "session_start": sess[0] if more_trades else OPT_SESSION_START,
            "session_end": sess[1] if more_trades else OPT_SESSION_END,
            "spread_usd": SPREAD_USD,
            "vp_mode": VP_MODE,
            "require_confluence": quality,
            "require_bos": quality,
            "require_fib": fib,
            "require_atr_regime": False,
            "nds_max_ratio": OPT_NDS_MAX_RATIO,
        }
    return {
        "enabled": list(OPT_RULES_HIGH_WR if (high_wr and no_nds) else
                        ("OB", "NDS", "DEMAND") if high_wr else
                        (("OB", "DEMAND", "FVG") if no_nds else OPT_RULES)),
        "exit_mode": OPT_EXIT_MODE,
        "fixed_tp_r": OPT_FIXED_TP_R,
        "be_trigger": OPT_BE_TRIGGER,
        "tp_mode": OPT_TP_MODE,
        "htf_tfs": list(OPT_HTF_TFS),
        "min_tp_r": OPT_MIN_TP_R,
        "max_tp_r": OPT_MAX_TP_R,
        "htf_lookback": OPT_HTF_LOOKBACK,
        "htf_trend": OPT_HTF_TREND,
        "htf_ema": OPT_HTF_EMA,
        "max_concurrent": OPT_MAX_CONCURRENT,
        "min_risk_usd": OPT_MIN_RISK_USD,
        "session_start": OPT_SESSION_START,
        "session_end": OPT_SESSION_END,
        "spread_usd": SPREAD_USD,
        "vp_mode": VP_MODE,
        "require_confluence": quality,
        "require_bos": quality,
        "require_fib": fib,
        "require_atr_regime": False,
        "nds_max_ratio": OPT_NDS_MAX_RATIO,
    }


def setup_confluence_pass(B, bar_i, direction, tag, enabled_rules):
    """FVG only when OB or DEMAND also fires on the same bar (same direction)."""
    if tag != "FVG":
        return True
    for name in ("OB", "DEMAND"):
        if name not in enabled_rules:
            continue
        for d, *_ in DETECTORS[name](B, bar_i, DEFAULT_PARAMS):
            if d == direction:
                return True
    return False


def bos_confirm_pass(B, bar_i, direction, lookback=5):
    """Recent BOS in trade direction (structure confirmation)."""
    for j in range(max(3, bar_i - lookback), bar_i + 1):
        for d, *_ in DETECTORS["BOS"](B, j, DEFAULT_PARAMS):
            if d == direction:
                return True
    return False


def fib_retrace_pass(B, bar_i, direction, entry, lookback=OPT_FIB_LOOKBACK,
                     lo=OPT_FIB_LO, hi=OPT_FIB_HI):
    """Long in fib discount (38-78%% retrace of recent swing); short in premium."""
    lo_i = max(0, bar_i - lookback)
    sh = B.h[lo_i:bar_i + 1].max()
    sl = B.l[lo_i:bar_i + 1].min()
    rng = sh - sl
    if rng <= 0:
        return True
    if direction == "long":
        top = sh - lo * rng
        bot = sh - hi * rng
        return bot <= entry <= top
    bot = sl + lo * rng
    top = sl + hi * rng
    return bot <= entry <= top


def session_pass(ts, start_hour=OPT_SESSION_START, end_hour=OPT_SESSION_END):
    """Trade only during active gold hours (broker server time)."""
    h = pd.Timestamp(ts).hour
    return start_hour <= h < end_hour


def entry_filters_pass(B, bar_i, risk, min_risk_usd=0.0,
                       session_start=None, session_end=None,
                       direction=None, signal_time=None, htf_context=None,
                       htf_trend=False, htf_ema=20, htf_tfs=("H1",)):
    if min_risk_usd and risk < min_risk_usd:
        return False
    if session_start is not None and session_end is not None:
        if not session_pass(B.t[bar_i], session_start, session_end):
            return False
    if htf_trend and direction and signal_time is not None and htf_context:
        for tf in htf_tfs:
            if not htf_trend_pass(htf_context, signal_time, direction,
                                  tf=tf, ema_period=htf_ema):
                return False
    return True


def compute_adx_series(B, period=14):
    """Causal ADX / +DI / -DI arrays aligned to B (Wilder-style rolling mean).
    Same formula as the ADX rule detector, but computed once for the whole
    series so a per-bar gate is cheap. NaN until enough history."""
    h = pd.Series(B.h.astype(float))
    l = pd.Series(B.l.astype(float))
    c = pd.Series(B.c.astype(float))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.rolling(period).mean()
    pdi = 100 * pd.Series(pdm).rolling(period).mean() / atr
    mdi = 100 * pd.Series(mdm).rolling(period).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return (adx.to_numpy(), pdi.to_numpy(), mdi.to_numpy())


def adx_gate_pass(adx_arr, pdi_arr, mdi_arr, i, direction,
                  adx_min=22.0, require_dir=True):
    """Momentum/trend gate: only allow a trade when ADX shows real trend
    strength AND (optionally) the +DI/-DI direction agrees with the trade.
    No arrays -> always allow (feature off)."""
    if adx_arr is None or i >= len(adx_arr):
        return True
    a = adx_arr[i]
    if not np.isfinite(a) or a < adx_min:
        return False
    if require_dir:
        pdi, mdi = pdi_arr[i], mdi_arr[i]
        if not (np.isfinite(pdi) and np.isfinite(mdi)):
            return False
        if direction == "long" and not (pdi > mdi):
            return False
        if direction == "short" and not (mdi > pdi):
            return False
    return True


def build_regime_map(htf_context, tf="H1", detector="kaufman", win=40,
                     er_trend=0.35, slope_k=0.0006,
                     er_hi=0.40, er_lo=0.25, confirm=1,
                     h4_win=30, h4_lookback=60,
                     structure_break=False, brk_win=40,
                     shock_win=24, shock_baseline=720):
    """Build a RegimeMap from the same HTF context used everywhere else.

    detector='mtf' uses H4 structure + H1 ER (needs H1 and H4 in context).
    Returns None if the timeframe is unavailable, so callers can no-op safely.
    Identical inputs in backtest and live -> identical regime labels."""
    if not htf_context:
        return None
    if detector == "mtf":
        try:
            return _regime.build_mtf_map(
                htf_context, h4_win=h4_win, h1_win=win,
                er_hi=er_hi, er_lo=er_lo, confirm=max(confirm, 1),
                h4_lookback=h4_lookback,
                structure_break=structure_break, brk_win=brk_win,
                shock_win=shock_win, shock_baseline=shock_baseline)
        except Exception:
            return None
    src = htf_context.get(tf)
    if src is None:
        return None
    try:
        return _regime.RegimeMap(src, detector=detector, win=win,
                                 er_trend=er_trend, slope_k=slope_k,
                                 er_hi=er_hi, er_lo=er_lo, confirm=confirm)
    except Exception:
        return None


def regime_allows(rmap, signal_time, direction, rule=None,
                  shock_gate=False, shock_ratio=1.8,
                  flip_cooldown_h=0.0):
    """Directional gate: block longs in down-regimes and shorts in up-regimes.
    RANGE allows both sides unless MTF macro bias is set (then block counter-bias).

    Optional V-shape guards (both OFF by default; spike rules are exempt):
      shock_gate       block entries while realized vol is >= shock_ratio ×
                       its baseline (violent whipsaw — mean reversion bleeds).
      flip_cooldown_h  block entries for N hours after the macro bias flips
                       (fresh direction is unproven; the first pullback after
                       a V-turn is where the losing clusters happened).
    No map -> always allow (feature off)."""
    if rmap is None:
        return True
    reg = rmap.at(signal_time)
    d = str(direction).lower()
    if reg == "TREND_UP" and d == "short":
        return False
    if reg == "TREND_DOWN" and d == "long":
        return False
    if reg == "RANGE" and hasattr(rmap, "macro_at"):
        macro = rmap.macro_at(signal_time)
        if d == "long" and macro < 0:
            return False
        if d == "short" and macro > 0:
            return False
    spike_exempt = rule is not None and rule in SPIKE_RULE_KEYS
    if shock_gate and not spike_exempt and hasattr(rmap, "shock_at"):
        if rmap.shock_at(signal_time) >= shock_ratio:
            return False
    if flip_cooldown_h > 0 and not spike_exempt and hasattr(rmap, "macro_age_at"):
        if rmap.macro_age_at(signal_time) < flip_cooldown_h:
            return False
    return True


def htf_trend_pass(htf_context, signal_time, direction, tf="H1", ema_period=20):
    """Long only when H1 price >= EMA; short only when price <= EMA."""
    ctx = htf_context.get(tf) if htf_context else None
    if ctx is None:
        return True
    B = ctx if isinstance(ctx, Bars) else Bars(ctx)
    idx = htf_bar_index(B, signal_time)
    if idx < ema_period + 2:
        return True
    ema = pd.Series(B.c[:idx + 1]).ewm(span=ema_period, adjust=False).mean().iloc[-1]
    px = B.c[idx]
    if direction == "long":
        return px >= ema
    return px <= ema


def sl_buffer_atr(buffer_atr=None):
    """Room beyond the structural level, in ATR (floating_config.RISK)."""
    if buffer_atr is not None:
        return float(buffer_atr)
    return float(_risk_cfg("sl_buffer_atr", 0.50))


def sl_swing_lb(swing_lb=None):
    if swing_lb is not None:
        return int(swing_lb)
    return int(_risk_cfg("sl_swing_lb", 10))


def compute_entry_sl_risk(B, sig_i, direction, proximal, distal,
                          buffer_atr=None, swing_lb=None):
    """Structural stop + risk distance (shared by backtest and live)."""
    buffer_atr = sl_buffer_atr(buffer_atr)
    swing_lb = sl_swing_lb(swing_lb)
    atr_i = B.atr[sig_i] if B.atr[sig_i] > 0 else (B.rng[sig_i] + 1e-6)
    entry = proximal
    if direction == "long":
        swing = B.l[max(0, sig_i - swing_lb):sig_i + 1].min()
        sl = min(distal, swing) - buffer_atr * atr_i
        risk = entry - sl
    else:
        swing = B.h[max(0, sig_i - swing_lb):sig_i + 1].max()
        sl = max(distal, swing) + buffer_atr * atr_i
        risk = sl - entry
    return sl, entry, risk


def fixed_tp_price(direction, entry, risk, tp_r=OPT_FIXED_TP_R):
    if direction == "long":
        return entry + tp_r * risk
    return entry - tp_r * risk


# ─────────────────────── volume-profile momentum filter ───────────────────
def vp_levels(B, i, window=480, bins=50):
    lo_i = max(0, i - window)
    tp = B.tp[lo_i:i]; vol = B.vol[lo_i:i]
    if len(tp) < 20 or tp.max() <= tp.min():
        return None
    hist, edges = np.histogram(tp, bins=bins, weights=vol, range=(tp.min(), tp.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    return {"poc": centers[int(np.argmax(hist))]}


def vp_pass(B, i, direction, entry, mode=VP_MODE, window=480, bins=50,
            vp_tol_atr=0.0):
    """Momentum filter: long above POC, short below. vp_tol_atr softens the edge."""
    if not mode:
        return True
    vp = vp_levels(B, i, window, bins)
    if vp is None:
        return True
    poc = vp["poc"]
    tol = (vp_tol_atr * B.atr[i]) if vp_tol_atr and B.atr[i] > 0 else 0.0
    if mode == "poc_inv":
        if direction == "long":
            return entry >= poc - tol
        return entry <= poc + tol
    return True


def structural_tp(B, i, direction, tag, proximal, distal, p=DEFAULT_PARAMS):
    """
    Take-profit from the same rule logic that produced the entry.
    Returns a price level, or None if no valid structural target exists.
    """
    lb = 20
    atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)

    if tag == "WYCK" or (isinstance(tag, str) and tag.startswith("WYCK")):
        W = p.get("wyck_lb", 40)
        if i < W:
            return None
        lo_i = i - W
        rh = B.h[lo_i:i].max()
        rl = B.l[lo_i:i].min()
        return rh if direction == "long" else rl

    if tag == "FVG":
        if direction == "long":
            return max(B.h[i], B.h[max(0, i - lb):i + 1].max())
        return min(B.l[i], B.l[max(0, i - lb):i + 1].min())

    if tag in ("OB", "DEMAND", "FL", "SUPPLY"):
        return B.h[i] if direction == "long" else B.l[i]

    if tag == "BOS":
        lb_bos = 20
        if i < lb_bos + 1:
            return None
        prior_high = B.h[i - lb_bos:i].max()
        prior_low = B.l[i - lb_bos:i].min()
        recent_low = B.l[i - 10:i + 1].min()
        recent_high = B.h[i - 10:i + 1].max()
        if direction == "long":
            return prior_high + (prior_high - recent_low)
        return prior_low - (recent_high - prior_low)

    if tag == "QM":
        lookback = p.get("qm_lookback", 20)
        if i < lookback:
            return None
        prior_low = B.l[i - lookback:i].min()
        prior_high = B.h[i - lookback:i].max()
        return prior_high if direction == "long" else prior_low

    return None


def next_rule_tp(B, from_i, direction, tag, entry, search_bars=MAX_HOLD,
                 p=DEFAULT_PARAMS):
    """
    TP = entry (proximal) of the next same-rule setup in profit direction.
    Scans forward up to search_bars after the signal bar.
    Example: FL long -> proximal of the next FL long above entry.
    """
    det = DETECTORS.get(tag)
    if det is None:
        return None
    end = min(from_i + search_bars, B.n - 1)
    for j in range(from_i + 1, end + 1):
        for (d, prox, _dist, t) in det(B, j, p):
            if d != direction or t != tag:
                continue
            if direction == "long" and prox > entry:
                return prox
            if direction == "short" and prox < entry:
                return prox
    return None


def htf_bar_index(B, signal_time):
    """Last HTF bar that opened on or before signal_time."""
    idx = int(np.searchsorted(B.t, np.datetime64(signal_time), "right")) - 1
    return max(0, min(idx, B.n - 1))


def collect_htf_targets(B, bar_i, direction, entry, lookback=OPT_HTF_LOOKBACK,
                        rules=OPT_RULES):
    """Structural levels on a higher TF (swings + rule zones) for TP."""
    targets = []
    lo = max(3, bar_i - lookback)
    hi_i = min(bar_i, B.n - 2)
    for j in range(lo, hi_i + 1):
        if j >= 1 and j + 1 < B.n:
            if B.h[j] >= B.h[j - 1] and B.h[j] >= B.h[j + 1]:
                targets.append(float(B.h[j]))
            if B.l[j] <= B.l[j - 1] and B.l[j] <= B.l[j + 1]:
                targets.append(float(B.l[j]))
        for name in rules:
            if name not in DETECTORS:
                continue
            for d, prox, dist, _tag in DETECTORS[name](B, j, DEFAULT_PARAMS):
                targets.append(float(prox))
                targets.append(float(dist))
    if direction == "long":
        return sorted({t for t in targets if t > entry})
    return sorted({t for t in targets if t < entry}, reverse=True)


def htf_tp_price(signal_time, direction, entry, risk, htf_context,
                 min_r=OPT_MIN_TP_R, max_r=OPT_MAX_TP_R,
                 rules=OPT_RULES, priority=OPT_HTF_TFS, lookback=OPT_HTF_LOOKBACK):
    """
    TP from higher-TF zones (H4 first, then H1).
    Picks the nearest HTF level that is at least min_r away (not too close).
    """
    if not htf_context or risk <= 0:
        return None, None
    for tf_name in priority:
        ctx = htf_context.get(tf_name)
        if ctx is None:
            continue
        B = ctx if isinstance(ctx, Bars) else Bars(ctx)
        idx = htf_bar_index(B, signal_time)
        if idx < 5:
            continue
        targets = collect_htf_targets(B, idx, direction, entry, lookback, rules)
        if direction == "long":
            min_px = entry + min_r * risk
            max_px = entry + max_r * risk
            cands = [t for t in targets if min_px <= t <= max_px]
            if cands:
                return cands[0], f"htf_{tf_name.lower()}"
        else:
            max_px = entry - min_r * risk
            min_px = entry - max_r * risk
            cands = [t for t in targets if min_px <= t <= max_px]
            if cands:
                return cands[0], f"htf_{tf_name.lower()}"
    return None, None


def resolve_tp(B, sig_i, direction, tag, proximal, distal, entry, risk,
               tp_mode="fixed", tp_min_r=0.5, search_bars=MAX_HOLD,
               fixed_tp_r=None, htf_context=None, min_tp_r=OPT_MIN_TP_R,
               max_tp_r=OPT_MAX_TP_R, htf_lookback=OPT_HTF_LOOKBACK,
               htf_tfs=OPT_HTF_TFS, signal_time=None):
    """Unified TP: fixed R, HTF zones, or blend (max of floor + HTF)."""
    tp_mode = tp_mode.replace("-", "_")
    if signal_time is None:
        signal_time = B.t[sig_i]

    floor_px = None
    if fixed_tp_r is not None and fixed_tp_r > 0:
        floor_px = fixed_tp_price(direction, entry, risk, fixed_tp_r)

    if tp_mode == "fixed":
        if floor_px is None:
            return None, None
        return floor_px, "fixed_r"

    if tp_mode == "htf":
        raw, src = htf_tp_price(signal_time, direction, entry, risk, htf_context,
                                min_r=min_tp_r, max_r=max_tp_r,
                                lookback=htf_lookback, priority=htf_tfs)
        if raw is None:
            return floor_px, "fixed_fallback"
        return raw, src

    if tp_mode == "htf_blend":
        raw, src = htf_tp_price(signal_time, direction, entry, risk, htf_context,
                                min_r=min_tp_r, max_r=max_tp_r,
                                lookback=htf_lookback, priority=htf_tfs)
        if floor_px is None and raw is None:
            return None, None
        if raw is None:
            return floor_px, "fixed_r"
        if floor_px is None:
            return raw, src
        if direction == "long":
            pick = max(floor_px, raw)
            pick = min(pick, entry + max_tp_r * risk)
        else:
            pick = min(floor_px, raw)
            pick = max(pick, entry - max_tp_r * risk)
        if abs(pick - floor_px) < 1e-9:
            return pick, "fixed_r"
        if abs(pick - raw) < 1e-9:
            return pick, src
        return pick, "htf_blend"

    # legacy modes
    if tp_mode == "next_rule":
        raw = next_rule_tp(B, sig_i, direction, tag, entry, search_bars)
        source = "next_rule"
        if raw is None:
            raw = structural_tp(B, sig_i, direction, tag, proximal, distal)
            source = "structural_fallback"
    else:
        raw = structural_tp(B, sig_i, direction, tag, proximal, distal)
        source = "structural"
    tp = valid_structural_tp(direction, entry, raw, risk, min_r=tp_min_r)
    return tp, source


def compute_tp(B, sig_i, direction, tag, proximal, distal, entry, risk,
               tp_mode="structural", tp_min_r=0.5, search_bars=MAX_HOLD, **kw):
    """Resolve take-profit (delegates to resolve_tp)."""
    return resolve_tp(B, sig_i, direction, tag, proximal, distal, entry, risk,
                      tp_mode=tp_mode, tp_min_r=tp_min_r, search_bars=search_bars, **kw)


def valid_structural_tp(direction, entry, tp, risk, min_r=0.5):
    """TP must be on the profit side and at least min_r away."""
    if tp is None or risk <= 0:
        return None
    if direction == "long":
        if tp <= entry:
            return None
        if (tp - entry) / risk < min_r:
            return None
    else:
        if tp >= entry:
            return None
        if (entry - tp) / risk < min_r:
            return None
    return tp


# ─────────────────── 1-minute path-accurate exit engine ───────────────────
class M1Ctx:
    """Holds 1-minute data + time mapping for honest intrabar resolution."""
    def __init__(self, m1, signal_index):
        self.h = m1["high"].values.astype(float)
        self.l = m1["low"].values.astype(float)
        self.c = m1["close"].values.astype(float)
        self.t = m1.index.values
        self.n = len(m1)
        self.sig_t = signal_index.values

    def start(self, ts):
        return int(np.searchsorted(self.t, np.datetime64(ts), "left"))

    def end(self, ts):
        return int(np.searchsorted(self.t, np.datetime64(ts), "right"))

    def to_sig_idx(self, ts64):
        return int(np.searchsorted(self.sig_t, ts64, "right")) - 1


def _exit_long_m1(ctx, j0, jend, entry, sl, risk, tp=None,
                  trig=TRAIL_TRIGGER, dist=TRAIL_DIST, be=BREAKEVEN,
                  use_trail=True, breakeven_only=False, partial_frac=0.0):
    stop = sl; peak = entry; act = False
    partial_r = 0.0
    partial_taken = False
    runner_w = 1.0
    for j in range(j0, jend):
        if ctx.l[j] <= stop:
            r_out = (stop - entry) / risk
            if breakeven_only and abs(r_out) < 1e-9:
                r_out = 0.0
            if partial_taken:
                tag = "partial_be" if abs(r_out) < 1e-9 else "partial_sl"
                return partial_r + runner_w * r_out, j, tag
            if breakeven_only and abs(r_out) < 1e-9:
                return 0.0, j, "be"
            return r_out, j, "sl"
        if (tp is not None and ctx.h[j] >= tp and partial_frac > 0
                and not partial_taken):
            partial_taken = True
            tp_r = (tp - entry) / risk
            partial_r = partial_frac * tp_r
            runner_w = 1.0 - partial_frac
            tp = None
            use_trail = True
            breakeven_only = False
            peak = max(peak, ctx.h[j])
            act = True
            if be:
                stop = max(stop, entry)
            continue
        if tp is not None and ctx.h[j] >= tp:
            return (tp - entry) / risk, j, "tp"
        if ctx.h[j] > peak:
            peak = ctx.h[j]
        fav_r = (peak - entry) / risk
        if breakeven_only:
            if fav_r >= trig:
                stop = max(stop, entry)
        elif use_trail:
            if not act and fav_r >= trig:
                act = True
                if be:
                    stop = max(stop, entry)
            if act:
                stop = max(stop, peak - dist * risk)
    je = min(jend, ctx.n) - 1
    r_time = (ctx.c[je] - entry) / risk
    if partial_taken:
        return partial_r + runner_w * r_time, je, "partial_time"
    return r_time, je, "time"


def _exit_short_m1(ctx, j0, jend, entry, sl, risk, tp=None,
                   trig=TRAIL_TRIGGER, dist=TRAIL_DIST, be=BREAKEVEN,
                   use_trail=True, breakeven_only=False, partial_frac=0.0):
    stop = sl; trough = entry; act = False
    partial_r = 0.0
    partial_taken = False
    runner_w = 1.0
    for j in range(j0, jend):
        if ctx.h[j] >= stop:
            r_out = (entry - stop) / risk
            if breakeven_only and abs(r_out) < 1e-9:
                r_out = 0.0
            if partial_taken:
                tag = "partial_be" if abs(r_out) < 1e-9 else "partial_sl"
                return partial_r + runner_w * r_out, j, tag
            if breakeven_only and abs(r_out) < 1e-9:
                return 0.0, j, "be"
            return r_out, j, "sl"
        if (tp is not None and ctx.l[j] <= tp and partial_frac > 0
                and not partial_taken):
            partial_taken = True
            tp_r = (entry - tp) / risk
            partial_r = partial_frac * tp_r
            runner_w = 1.0 - partial_frac
            tp = None
            use_trail = True
            breakeven_only = False
            trough = min(trough, ctx.l[j])
            act = True
            if be:
                stop = min(stop, entry)
            continue
        if tp is not None and ctx.l[j] <= tp:
            return (entry - tp) / risk, j, "tp"
        if ctx.l[j] < trough:
            trough = ctx.l[j]
        fav_r = (entry - trough) / risk
        if breakeven_only:
            if fav_r >= trig:
                stop = min(stop, entry)
        elif use_trail:
            if not act and fav_r >= trig:
                act = True
                if be:
                    stop = min(stop, entry)
            if act:
                stop = min(stop, trough + dist * risk)
    je = min(jend, ctx.n) - 1
    r_time = (entry - ctx.c[je]) / risk
    if partial_taken:
        return partial_r + runner_w * r_time, je, "partial_time"
    return r_time, je, "time"


def exit_mode_params(exit_mode="trail", fixed_tp_r=None):
    """Map CLI exit mode to simulator flags."""
    mode = exit_mode.replace("-", "_")
    if mode == "trail":
        return {"use_structural_tp": False, "use_trail": True,
                "breakeven_only": False, "fixed_tp_r": None}
    if mode == "trail_tp":
        return {"use_structural_tp": True, "use_trail": True,
                "breakeven_only": False, "fixed_tp_r": None}
    if mode == "be1r_tp":
        return {"use_structural_tp": True, "use_trail": False,
                "breakeven_only": True, "fixed_tp_r": None}
    if mode == "be1r_fixed_tp":
        r = fixed_tp_r if fixed_tp_r is not None else OPT_FIXED_TP_R
        return {"use_structural_tp": False, "use_trail": False,
                "breakeven_only": True, "fixed_tp_r": r}
    raise ValueError(
        f"Unknown exit mode '{exit_mode}'. "
        "Use: trail, trail-tp, be1r-tp, be1r-fixed-tp"
    )


def prepare_htf_context(htf_dfs):
    """Wrap HTF DataFrames as Bars objects for TP lookup."""
    if not htf_dfs:
        return None
    return {tf: Bars(df) if not isinstance(df, Bars) else df
            for tf, df in htf_dfs.items()}


def _entry_fill_on_bar(B, j, direction, proximal, etol):
    """Classify how bar j triggered entry: exact limit touch vs tolerance band."""
    proximal = float(proximal)
    etol = float(etol)
    if direction == "long":
        wick = float(B.l[j])
        gap = wick - proximal
        if wick <= proximal:
            return "exact", gap
        if etol > 0 and wick <= proximal + etol:
            return "tol", gap
        return None, gap
    wick = float(B.h[j])
    gap = proximal - wick
    if wick >= proximal:
        return "exact", gap
    if etol > 0 and wick >= proximal - etol:
        return "tol", gap
    return None, gap


def _trade_fill_fields(fill_kind, entry_tol_atr, etol, fill_gap, entry_confirm_mode="none"):
    return {
        "fill_kind": fill_kind,
        "entry_tol_atr": float(entry_tol_atr),
        "entry_tol_usd": float(etol),
        "fill_gap": float(fill_gap),
        "entry_confirm_mode": entry_confirm_mode,
    }


def entry_zone_bounds(direction, proximal, etol):
    """Entry band [lo, hi] — same geometry as live zone mode."""
    proximal = float(proximal)
    etol = float(etol)
    if direction == "long":
        return proximal, proximal + etol
    return proximal - etol, proximal


LIMIT_AT = "band_edge"


def limit_entry_price(direction, proximal, etol, limit_at=LIMIT_AT):
    """Price where the resting LIMIT sits — and therefore the real entry.

    band_edge : near edge of the entry band, the first price the market
                reaches. Every touch of the band is a genuine fill.
    proximal  : far edge. Better price, but only fills when price trades all
                the way through the band.

    Booking the entry at `proximal` while triggering on a touch of the band
    hands the backtest up to entry_tol×ATR of price the broker never gives.
    """
    proximal = float(proximal)
    etol = float(etol or 0.0)
    if etol <= 0 or str(limit_at).lower() != "band_edge":
        return proximal
    return proximal + etol if direction == "long" else proximal - etol


def zone_touched_on_bar(B, bar_i, direction, proximal, etol):
    """True when bar wick enters the entry tolerance band."""
    lo, hi = entry_zone_bounds(direction, proximal, etol)
    if direction == "long":
        return float(B.l[bar_i]) <= hi
    return float(B.h[bar_i]) >= lo


def close_in_entry_zone(B, bar_i, direction, proximal, etol):
    lo, hi = entry_zone_bounds(direction, proximal, etol)
    c = float(B.c[bar_i])
    return lo <= c <= hi


def m15_close_entry_confirm(B, bar_i, direction, proximal, etol, touched_before=True):
    """
    Fill-time confirm: zone was touched + M15 close inside band + directional candle.
    touched_before: allow same-bar touch+confirm when True.
    """
    if not touched_before and not zone_touched_on_bar(B, bar_i, direction, proximal, etol):
        return False
    if not close_in_entry_zone(B, bar_i, direction, proximal, etol):
        return False
    if direction == "long":
        return float(B.c[bar_i]) > float(B.o[bar_i])
    return float(B.c[bar_i]) < float(B.o[bar_i])


def rejection_entry_confirm(B, bar_i, direction, proximal, etol):
    """
    SMC rejection: zone touched + close reclaims proximal in trade direction.
    Close may finish outside the tolerance band (strong rejection).
    """
    if not zone_touched_on_bar(B, bar_i, direction, proximal, etol):
        return False
    proximal = float(proximal)
    c = float(B.c[bar_i])
    o = float(B.o[bar_i])
    if direction == "long":
        return c > proximal and c > o
    return c < proximal and c < o


# Detectors that need fill-time confirm when entry_confirm_mode == "rule_confirm"
ENTRY_CONFIRM_RULES = frozenset({"CH_REV_L", "WYCK_SOW"})


def rule_needs_entry_confirm(rule_name, entry_confirm_rules=None):
    rule = str(rule_name or "").upper()
    rules = entry_confirm_rules or ENTRY_CONFIRM_RULES
    return rule in rules or any(rule.startswith(r) for r in rules)


def _effective_confirm_mode(entry_confirm_mode, rule_name=None, entry_confirm_rules=None):
    mode = (entry_confirm_mode or "none").lower()
    if mode == "rule_confirm":
        return "rejection" if rule_needs_entry_confirm(rule_name, entry_confirm_rules) else "none"
    return mode


def _bar_entry_confirm(B, bar_i, direction, proximal, etol, mode):
    if mode == "m15_close":
        return m15_close_entry_confirm(B, bar_i, direction, proximal, etol)
    if mode == "rejection":
        return rejection_entry_confirm(B, bar_i, direction, proximal, etol)
    return False


def _scan_entry_bar(B, sig_i, wait, direction, proximal, distal, etol, itol,
                    entry_confirm_mode="none", rule_name=None, entry_confirm_rules=None,
                    sl=None, tp=None, min_fill_rr=0.0, limit_at=LIMIT_AT):
    """Return (entry_idx, fill_kind, fill_gap) or (None, None, None).

    Confirm modes fill at the CLOSE of the confirming bar (live parity —
    the market order goes out when the bar closes, never at the zone edge).
    min_fill_rr skips confirmations whose remaining reward at that close is
    below min_fill_rr × actual risk; the zone stays armed for later bars.
    """
    touched = False
    mode = _effective_confirm_mode(entry_confirm_mode, rule_name, entry_confirm_rules)
    for j in range(sig_i + 1, min(sig_i + 1 + wait, B.n)):
        if direction == "long" and B.c[j] < distal - itol:
            return None, None, None
        if direction == "short" and B.c[j] > distal + itol:
            return None, None, None
        if zone_touched_on_bar(B, j, direction, proximal, etol):
            touched = True
        if mode in ("m15_close", "rejection"):
            if not touched:
                continue
            if not _bar_entry_confirm(B, j, direction, proximal, etol, mode):
                continue
            if sl is not None:
                entry_px = float(B.c[j])
                risk_j = (entry_px - sl) if direction == "long" else (sl - entry_px)
                if risk_j <= 0:
                    continue
                if min_fill_rr > 0 and tp is not None:
                    rew = (tp - entry_px) if direction == "long" else (entry_px - tp)
                    if rew <= 0 or rew / risk_j < min_fill_rr:
                        continue
            fk, gap = _entry_fill_on_bar(B, j, direction, proximal, etol)
            kind = fk or ("reject" if mode == "rejection" else "confirm")
            return j, kind, gap
        fk, gap = _entry_fill_on_bar(B, j, direction, proximal, etol)
        # A LIMIT resting at the far edge only fills on an exact touch; one at
        # the band edge fills on any touch of the band.
        if fk == "tol" and str(limit_at).lower() != "band_edge":
            continue
        if fk:
            return j, fk, gap
    return None, None, None


def simulate_trade_m1(B, ctx, direction, sig_i, proximal, distal, wait, max_hold,
                      buffer_atr=None, swing_lb=None, tf_min=15, tag=None,
                      use_structural_tp=False, tp_min_r=0.5, use_trail=True,
                      breakeven_only=False, exit_mode=None, tp_mode="structural",
                      max_hold_tp=MAX_HOLD, fixed_tp_r=None, htf_context=None,
                      min_tp_r=OPT_MIN_TP_R, max_tp_r=OPT_MAX_TP_R,
                      htf_lookback=OPT_HTF_LOOKBACK, htf_tfs=OPT_HTF_TFS,
                      be_trigger=TRAIL_TRIGGER, partial_tp=False,
                      partial_frac=0.5, partial_trail_dist=TRAIL_DIST,
                      entry_tol_atr=0.0, invalidate_tol_atr=0.0,
                      entry_confirm_mode="none", rule_name=None,
                      entry_confirm_rules=None, min_fill_rr=0.0,
                      limit_at=LIMIT_AT):
    """Decide entry on the signal timeframe; resolve the exit on 1-minute bars."""
    buffer_atr = sl_buffer_atr(buffer_atr)
    swing_lb = sl_swing_lb(swing_lb)
    if exit_mode is not None:
        p = exit_mode_params(exit_mode, fixed_tp_r=fixed_tp_r)
        use_structural_tp = p["use_structural_tp"]
        use_trail = p["use_trail"]
        breakeven_only = p["breakeven_only"]
        if p["fixed_tp_r"] is not None:
            fixed_tp_r = p["fixed_tp_r"]
    atr_i = B.atr[sig_i] if B.atr[sig_i] > 0 else (B.rng[sig_i] + 1e-6)
    etol = entry_tol_atr * atr_i
    itol = invalidate_tol_atr * atr_i
    eff_mode = _effective_confirm_mode(entry_confirm_mode, rule_name, entry_confirm_rules)
    confirm_fill = eff_mode in ("m15_close", "rejection")
    # Confirm rules market-in at the confirming close, so their entry is set
    # below. Touch rules rest a LIMIT — that order's price IS the entry.
    entry = (float(proximal) if confirm_fill
             else limit_entry_price(direction, proximal, etol, limit_at))
    if direction == "long":
        swing_low = B.l[max(0, sig_i - swing_lb): sig_i + 1].min()
        sl = min(distal, swing_low) - buffer_atr * atr_i
        risk = entry - sl
    else:
        swing_high = B.h[max(0, sig_i - swing_lb): sig_i + 1].max()
        sl = max(distal, swing_high) + buffer_atr * atr_i
        risk = sl - entry
    if risk <= 0:
        return None

    # TP first (depends only on the signal bar) so the fill scan can apply
    # the remaining-RR floor at each candidate confirm close (live parity).
    tp = None
    tp_source = None
    mode = tp_mode.replace("-", "_")
    if mode in ("htf", "htf_blend", "fixed") or fixed_tp_r or use_structural_tp:
        if mode == "structural" and not use_structural_tp and fixed_tp_r:
            mode = "fixed"
        tp, tp_source = resolve_tp(
            B, sig_i, direction, tag, proximal, distal, entry, risk,
            tp_mode=mode, tp_min_r=tp_min_r, search_bars=max_hold_tp,
            fixed_tp_r=fixed_tp_r, htf_context=htf_context,
            min_tp_r=min_tp_r, max_tp_r=max_tp_r,
            htf_lookback=htf_lookback, htf_tfs=htf_tfs,
            signal_time=B.t[sig_i],
        )

    entry_idx, fill_kind, fill_gap = _scan_entry_bar(
        B, sig_i, wait, direction, proximal, distal, etol, itol, entry_confirm_mode,
        rule_name=rule_name, entry_confirm_rules=entry_confirm_rules,
        sl=sl if confirm_fill else None, tp=tp, min_fill_rr=min_fill_rr,
        limit_at=limit_at)
    if entry_idx is None:
        return None

    entry_time = B.t[entry_idx]
    if confirm_fill:
        # Honest fill: the market order goes out when the confirming bar
        # CLOSES — entry price is that close, never the zone edge (which the
        # old model filled at intra-bar, i.e. before the confirmation existed).
        entry = float(B.c[entry_idx])
        risk = (entry - sl) if direction == "long" else (sl - entry)
        if risk <= 0:
            return None
        fill_time = np.datetime64(entry_time) + np.timedelta64(tf_min, "m")
        fill = ctx.start(fill_time)
        if fill >= ctx.n:
            return None
        end_time = fill_time + np.timedelta64(max_hold * tf_min, "m")
    else:
        start = ctx.start(entry_time)
        fill = None
        for j in range(start, min(start + tf_min, ctx.n)):
            if direction == "long" and ctx.l[j] <= entry:
                fill = j; break
            if direction == "short" and ctx.h[j] >= entry:
                fill = j; break
        if fill is None:
            return None
        end_time = np.datetime64(entry_time) + np.timedelta64(max_hold * tf_min, "m")
    jend = min(max(ctx.end(end_time), fill + 1), ctx.n)
    pf = partial_frac if partial_tp else 0.0
    trail_trig = be_trigger if breakeven_only else TRAIL_TRIGGER
    if direction == "long":
        R, jx, reason = _exit_long_m1(ctx, fill, jend, entry, sl, risk, tp=tp,
                                      use_trail=use_trail, breakeven_only=breakeven_only,
                                      trig=trail_trig, dist=partial_trail_dist,
                                      partial_frac=pf)
    else:
        R, jx, reason = _exit_short_m1(ctx, fill, jend, entry, sl, risk, tp=tp,
                                       use_trail=use_trail, breakeven_only=breakeven_only,
                                       trig=trail_trig, dist=partial_trail_dist,
                                       partial_frac=pf)
    exit_time = ctx.t[jx]
    exit_idx = max(ctx.to_sig_idx(exit_time), entry_idx)
    tp_r = ((tp - entry) / risk if direction == "long" else (entry - tp) / risk) if tp else None
    fill_meta = _trade_fill_fields(
        fill_kind or "exact", entry_tol_atr, etol, fill_gap, entry_confirm_mode)
    return {"dir": direction, "idx": entry_idx, "exit_idx": exit_idx, "R": R,
            "risk": risk, "entry": entry, "sl": sl, "tp": tp, "tp_r": tp_r,
            "tp_source": tp_source, "exit_reason": reason, "setup": None,
            "time": B.t[entry_idx], "exit_time": pd.Timestamp(exit_time),
            **fill_meta}


def simulate_trade_native(B, direction, sig_i, proximal, distal, wait, max_hold,
                          buffer_atr=None, swing_lb=None, tag=None,
                          use_structural_tp=False, tp_min_r=0.5, use_trail=True,
                          breakeven_only=False, exit_mode=None, tp_mode="structural",
                          max_hold_tp=MAX_HOLD, fixed_tp_r=None, htf_context=None,
                          min_tp_r=OPT_MIN_TP_R, max_tp_r=OPT_MAX_TP_R,
                          htf_lookback=OPT_HTF_LOOKBACK, htf_tfs=OPT_HTF_TFS,
                          be_trigger=TRAIL_TRIGGER, partial_tp=False,
                          partial_frac=0.5, partial_trail_dist=TRAIL_DIST,
                          entry_tol_atr=0.0, invalidate_tol_atr=0.0,
                          entry_confirm_mode="none", rule_name=None,
                          entry_confirm_rules=None, min_fill_rr=0.0,
                          limit_at=LIMIT_AT):
    """Entry + exit BOTH on the signal timeframe (no M1).

    Used when sub-signal data (M1/M5) is unavailable (e.g. multi-year history).
    Intrabar resolution is CONSERVATIVE: within one bar the stop is checked
    before the target, so a bar that touches both is scored as a loss. This is
    pessimistic on purpose — it never reports better than reality could be.
    """
    buffer_atr = sl_buffer_atr(buffer_atr)
    swing_lb = sl_swing_lb(swing_lb)
    if exit_mode is not None:
        p = exit_mode_params(exit_mode, fixed_tp_r=fixed_tp_r)
        use_structural_tp = p["use_structural_tp"]
        use_trail = p["use_trail"]
        breakeven_only = p["breakeven_only"]
        if p["fixed_tp_r"] is not None:
            fixed_tp_r = p["fixed_tp_r"]
    atr_i = B.atr[sig_i] if B.atr[sig_i] > 0 else (B.rng[sig_i] + 1e-6)
    etol = entry_tol_atr * atr_i
    itol = invalidate_tol_atr * atr_i
    eff_mode = _effective_confirm_mode(entry_confirm_mode, rule_name, entry_confirm_rules)
    confirm_fill = eff_mode in ("m15_close", "rejection")
    entry = (float(proximal) if confirm_fill
             else limit_entry_price(direction, proximal, etol, limit_at))
    if direction == "long":
        swing_low = B.l[max(0, sig_i - swing_lb): sig_i + 1].min()
        sl = min(distal, swing_low) - buffer_atr * atr_i
        risk = entry - sl
    else:
        swing_high = B.h[max(0, sig_i - swing_lb): sig_i + 1].max()
        sl = max(distal, swing_high) + buffer_atr * atr_i
        risk = sl - entry
    if risk <= 0:
        return None

    # TP first (depends only on the signal bar) so the fill scan can apply
    # the remaining-RR floor at each candidate confirm close (live parity).
    tp = None
    tp_source = None
    mode = tp_mode.replace("-", "_")
    if mode in ("htf", "htf_blend", "fixed") or fixed_tp_r or use_structural_tp:
        if mode == "structural" and not use_structural_tp and fixed_tp_r:
            mode = "fixed"
        tp, tp_source = resolve_tp(
            B, sig_i, direction, tag, proximal, distal, entry, risk,
            tp_mode=mode, tp_min_r=tp_min_r, search_bars=max_hold_tp,
            fixed_tp_r=fixed_tp_r, htf_context=htf_context,
            min_tp_r=min_tp_r, max_tp_r=max_tp_r,
            htf_lookback=htf_lookback, htf_tfs=htf_tfs,
            signal_time=B.t[sig_i],
        )

    entry_idx, fill_kind, fill_gap = _scan_entry_bar(
        B, sig_i, wait, direction, proximal, distal, etol, itol, entry_confirm_mode,
        rule_name=rule_name, entry_confirm_rules=entry_confirm_rules,
        sl=sl if confirm_fill else None, tp=tp, min_fill_rr=min_fill_rr,
        limit_at=limit_at)
    if entry_idx is None:
        return None

    if confirm_fill:
        # Honest fill: entry at the confirming bar's CLOSE (live parity);
        # the position exists only from the NEXT bar onward.
        entry = float(B.c[entry_idx])
        risk = (entry - sl) if direction == "long" else (sl - entry)
        if risk <= 0:
            return None
        exit_start = entry_idx + 1
        if exit_start >= B.n:
            return None
    else:
        # Limit-style fill: exit walks from the entry bar itself
        # (captures same-bar stop-outs = conservative).
        exit_start = entry_idx

    jend = min(exit_start + max_hold, B.n)
    pf = partial_frac if partial_tp else 0.0
    trail_trig = be_trigger if breakeven_only else TRAIL_TRIGGER
    if direction == "long":
        R, jx, reason = _exit_long_m1(B, exit_start, jend, entry, sl, risk, tp=tp,
                                      use_trail=use_trail, breakeven_only=breakeven_only,
                                      trig=trail_trig, dist=partial_trail_dist,
                                      partial_frac=pf)
    else:
        R, jx, reason = _exit_short_m1(B, exit_start, jend, entry, sl, risk, tp=tp,
                                       use_trail=use_trail, breakeven_only=breakeven_only,
                                       trig=trail_trig, dist=partial_trail_dist,
                                       partial_frac=pf)
    exit_idx = max(jx, entry_idx)
    tp_r = ((tp - entry) / risk if direction == "long" else (entry - tp) / risk) if tp else None
    fill_meta = _trade_fill_fields(
        fill_kind or "exact", entry_tol_atr, etol, fill_gap, entry_confirm_mode)
    return {"dir": direction, "idx": entry_idx, "exit_idx": exit_idx, "R": R,
            "risk": risk, "entry": entry, "sl": sl, "tp": tp, "tp_r": tp_r,
            "tp_source": tp_source, "exit_reason": reason, "setup": None,
            "time": B.t[entry_idx], "exit_time": pd.Timestamp(B.t[exit_idx]),
            **fill_meta}


def run_backtest(signal_df, m1, enabled=None, wait=WAIT_BARS, max_hold=MAX_HOLD,
                 max_concurrent=MAX_CONCURRENT, vp_mode=VP_MODE, vp_tol_atr=0.0,
                 vp_window=480, tf_min=15,
                 B=None, ctx=None, use_structural_tp=False, tp_min_r=0.5,
                 use_trail=True, exit_mode=None, tp_mode="structural",
                 fixed_tp_r=None, min_risk_usd=0.0,
                 session_start=None, session_end=None, htf_context=None,
                 min_tp_r=OPT_MIN_TP_R, max_tp_r=OPT_MAX_TP_R,
                 htf_lookback=OPT_HTF_LOOKBACK, htf_tfs=OPT_HTF_TFS,
                 be_trigger=TRAIL_TRIGGER, htf_trend=False, htf_ema=20,
                 require_confluence=False, require_bos=False, require_fib=False,
                 require_atr_regime=False, atr_max_ratio=OPT_STABLE_ATR_RATIO,
                 atr_lookback=OPT_STABLE_ATR_LB, nds_max_ratio=OPT_NDS_MAX_RATIO,
                 fib_nds_exempt=False, fib_ob_only=False, fib_spike_exempt=False,
                 fib_demand_exempt=False,
                 atr_nds_only=False, htf_trend_tfs=None,
                 spike_mode=False, spike_params=None, detector_params=None,
                 partial_tp=False, partial_frac=0.5, partial_trail_dist=TRAIL_DIST,
                 native_exit=False,
                 regime_gate=False, regime_tf="H1", regime_detector="kaufman",
                 regime_win=40, regime_er_trend=0.35, regime_slope_k=0.0006,
                 regime_er_hi=0.40, regime_er_lo=0.25, regime_confirm=1,
                 regime_h4_win=30, regime_h4_lookback=60,
                 regime_structure_break=False, regime_brk_win=40,
                 regime_shock_gate=False, regime_shock_ratio=1.8,
                 regime_shock_win=24, regime_shock_baseline=720,
                 regime_flip_cooldown_h=0.0,
                 meta_gate=None, meta_asset=None, meta_skip_if_no_rules=False,
                 adx_gate=False, adx_min=22.0, adx_require_dir=True,
                 adx_period=14, entry_confirm_mode="none",
                 entry_confirm_rules=None, min_fill_rr=0.0,
                 max_same_dir=0, limit_at=LIMIT_AT):
    """Run the full rule set on signal_df.

    Exits are 1-minute-accurate when `m1` is supplied. When `native_exit=True`
    (or m1 is None), exits resolve on the signal timeframe itself with a
    conservative same-bar (stop-before-target) assumption — used for multi-year
    history where sub-minute data is not available from the broker.
    """
    if enabled is None:
        enabled = list(DETECTORS.keys())
    det_p = dict(DEFAULT_PARAMS)
    if detector_params:
        det_p.update(detector_params)
    if B is None:
        B = Bars(signal_df)
    if m1 is None and ctx is None:
        native_exit = True
    if ctx is None and not native_exit:
        ctx = M1Ctx(m1, signal_df.index)
    trades = []
    # open book: (exit_idx, direction) — supports max_same_dir institutional cap
    open_book = []
    trend_tfs = htf_trend_tfs if htf_trend_tfs is not None else htf_tfs
    spike_set = SPIKE_RULE_KEYS if spike_mode else frozenset()
    same_dir_cap = int(max_same_dir or 0)
    rmap = None
    if regime_gate:
        rmap = build_regime_map(htf_context, tf=regime_tf,
                                detector=regime_detector, win=regime_win,
                                er_trend=regime_er_trend, slope_k=regime_slope_k,
                                er_hi=regime_er_hi, er_lo=regime_er_lo,
                                confirm=regime_confirm,
                                h4_win=regime_h4_win,
                                h4_lookback=regime_h4_lookback,
                                structure_break=regime_structure_break,
                                brk_win=regime_brk_win,
                                shock_win=regime_shock_win,
                                shock_baseline=regime_shock_baseline)
    adx_arr = pdi_arr = mdi_arr = None
    if adx_gate:
        adx_arr, pdi_arr, mdi_arr = compute_adx_series(B, period=adx_period)
    for i in range(3, B.n - 1):
        open_book = [t for t in open_book if t[0] >= i]
        if len(open_book) >= max_concurrent:
            continue
        atr_ok = (not require_atr_regime or atr_regime_pass(
            B, i, lookback=atr_lookback, max_ratio=atr_max_ratio))
        if not atr_ok and not atr_nds_only and not spike_set:
            continue
        cur_regime = None
        if rmap is not None:
            cur_regime = rmap.at(B.t[i])
            if meta_gate is not None and meta_asset and meta_skip_if_no_rules:
                if not meta_gate.has_any(meta_asset, cur_regime):
                    continue
        bar_rules = enabled
        if not atr_ok:
            if atr_nds_only:
                bar_rules = [n for n in enabled
                             if is_nds_rule(n) or n in ("NDS", "DEMAND")]
            elif spike_set:
                bar_rules = [n for n in enabled if n in spike_set]
            if not bar_rules:
                continue
        for name in bar_rules:
            if name not in DETECTORS and name not in NDS_FAMILY and name not in AB_DETECTORS \
                    and name not in WYCK_DETECTORS and name not in STYLE_DETECTORS \
                    and name not in EXT_DETECTORS and name not in SPIKE_DETECTORS:
                continue
            hit = False
            setups = iter_rule_setups(
                name, B, i, det_p, htf_context=htf_context,
                signal_time=B.t[i], htf_tfs=htf_tfs, nds_max_ratio=nds_max_ratio,
                spike_params=spike_params)
            for (direction, proximal, distal, tag) in setups:
                if same_dir_cap > 0:
                    n_dir = sum(1 for _, d in open_book if d == direction)
                    if n_dir >= same_dir_cap:
                        continue
                if not vp_pass(B, i, direction, proximal, vp_mode,
                               window=vp_window, vp_tol_atr=vp_tol_atr):
                    continue
                if require_confluence and not setup_confluence_pass(
                        B, i, direction, tag, enabled):
                    continue
                if require_bos and not bos_confirm_pass(B, i, direction):
                    continue
                _sl, _entry, risk = compute_entry_sl_risk(
                    B, i, direction, proximal, distal)
                if risk <= 0:
                    continue
                nested = is_nds_setup(
                    B, i, direction, proximal, distal, htf_context, B.t[i],
                    htf_tfs, nds_max_ratio) or is_nds_rule(name)
                if not fib_filter_pass(
                        B, i, direction, _entry, name, proximal, distal,
                        htf_context, B.t[i], htf_tfs, nds_max_ratio,
                        require_fib, fib_nds_exempt, fib_ob_only, fib_spike_exempt,
                        fib_demand_exempt):
                    continue
                if not regime_allows(rmap, B.t[i], direction, rule=name,
                                     shock_gate=regime_shock_gate,
                                     shock_ratio=regime_shock_ratio,
                                     flip_cooldown_h=regime_flip_cooldown_h):
                    continue
                if meta_gate is not None and meta_asset is not None:
                    reg = cur_regime if cur_regime is not None else (
                        rmap.at(B.t[i]) if rmap is not None else "RANGE")
                    if not meta_gate.allows(meta_asset, name, reg):
                        continue
                if adx_gate and not adx_gate_pass(
                        adx_arr, pdi_arr, mdi_arr, i, direction,
                        adx_min=adx_min, require_dir=adx_require_dir):
                    continue
                if not entry_filters_pass(B, i, risk, min_risk_usd=min_risk_usd,
                                          session_start=session_start,
                                          session_end=session_end,
                                          direction=direction, signal_time=B.t[i],
                                          htf_context=htf_context,
                                          htf_trend=htf_trend, htf_ema=htf_ema,
                                          htf_tfs=trend_tfs):
                    continue
                etol = det_p.get("entry_tol_atr", 0.0)
                itol = det_p.get("invalidate_tol_atr", 0.0)
                if native_exit:
                    tr = simulate_trade_native(B, direction, i, proximal, distal,
                                               wait, max_hold, tag=tag,
                                               use_structural_tp=use_structural_tp,
                                               tp_min_r=tp_min_r, use_trail=use_trail,
                                               exit_mode=exit_mode, tp_mode=tp_mode,
                                               max_hold_tp=max_hold, fixed_tp_r=fixed_tp_r,
                                               htf_context=htf_context,
                                               min_tp_r=min_tp_r, max_tp_r=max_tp_r,
                                               htf_lookback=htf_lookback, htf_tfs=htf_tfs,
                                               be_trigger=be_trigger,
                                               partial_tp=partial_tp, partial_frac=partial_frac,
                                               partial_trail_dist=partial_trail_dist,
                                               entry_tol_atr=etol, invalidate_tol_atr=itol,
                                               entry_confirm_mode=entry_confirm_mode,
                                               rule_name=name,
                                               entry_confirm_rules=entry_confirm_rules,
                                               min_fill_rr=min_fill_rr,
                                               limit_at=limit_at)
                else:
                    tr = simulate_trade_m1(B, ctx, direction, i, proximal, distal,
                                           wait, max_hold, tf_min=tf_min, tag=tag,
                                           use_structural_tp=use_structural_tp,
                                           tp_min_r=tp_min_r, use_trail=use_trail,
                                           exit_mode=exit_mode, tp_mode=tp_mode,
                                           max_hold_tp=max_hold, fixed_tp_r=fixed_tp_r,
                                           htf_context=htf_context,
                                           min_tp_r=min_tp_r, max_tp_r=max_tp_r,
                                           htf_lookback=htf_lookback, htf_tfs=htf_tfs,
                                           be_trigger=be_trigger,
                                           partial_tp=partial_tp, partial_frac=partial_frac,
                                           partial_trail_dist=partial_trail_dist,
                                           entry_tol_atr=etol, invalidate_tol_atr=itol,
                                           entry_confirm_mode=entry_confirm_mode,
                                           rule_name=name,
                                           entry_confirm_rules=entry_confirm_rules,
                                           min_fill_rr=min_fill_rr,
                                           limit_at=limit_at)
                if tr is not None:
                    tr["rule"] = name
                    tr["setup"] = resolve_trade_tag(
                        name, tag, B, i, direction, proximal, distal,
                        htf_context, B.t[i], htf_tfs, nds_max_ratio,
                        nds_enabled=any(is_nds_rule(n) for n in enabled))
                    trades.append(tr)
                    open_book.append((tr["exit_idx"], direction))
                    hit = True
                    break
            if hit:
                break
    trades.sort(key=lambda t: t["time"])
    return trades


def exit_stats(trades):
    """Summarise how trades closed (sl / tp / be / time)."""
    if not trades:
        return {}
    out = {"sl": 0, "tp": 0, "be": 0, "time": 0, "partial": 0}
    for t in trades:
        r = t.get("exit_reason", "sl")
        if r.startswith("partial"):
            out["partial"] = out.get("partial", 0) + 1
        else:
            out[r] = out.get(r, 0) + 1
    return out


# ─────────────────────────── account simulator ────────────────────────────
def simulate_account(trades, start_balance=START_BALANCE, risk_pct=RISK_PCT,
                     spread_price=SPREAD_USD, compound=True):
    """Event-driven $ account with compounding + monthly breakdown."""
    if not trades:
        return None
    ev = []
    for k, t in enumerate(trades):
        ev.append((t["time"], 0, k))
        ev.append((t["exit_time"], 1, k))
    ev.sort(key=lambda e: (e[0], e[1]))
    balance = start_balance; peak = start_balance
    min_balance = start_balance
    sizes = {}; riskd = {}
    n_taken = n_wins = 0; gp = gl = 0.0; max_dd = 0.0
    monthly = {}
    for (ts, kind, k) in ev:
        t = trades[k]
        if kind == 0:
            rd = risk_pct * (balance if compound else start_balance)
            rp = t.get("risk", 0)
            if rp <= 0:
                continue
            sizes[k] = rd / rp; riskd[k] = rd; n_taken += 1
        else:
            if k not in sizes:
                continue
            net = t["R"] * riskd[k] - sizes[k] * spread_price
            balance += net
            if net > 0:
                n_wins += 1; gp += net
            else:
                gl += -net
            monthly[f"{ts.year}-{ts.month:02d}"] = monthly.get(f"{ts.year}-{ts.month:02d}", 0.0) + net
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak)
            if balance < min_balance:
                min_balance = balance
    abs_dd = max(0.0, start_balance - min_balance)
    return {"final": balance, "ret_pct": (balance / start_balance - 1) * 100,
            "n": n_taken, "wr": (n_wins / n_taken * 100 if n_taken else 0),
            "pf": (gp / gl if gl > 0 else 999), "max_dd": max_dd * 100,
            "min_balance": min_balance,
            "abs_dd": abs_dd,
            "abs_dd_pct": (abs_dd / start_balance * 100) if start_balance > 0 else 0.0,
            "monthly": monthly}


def build_trade_ledger(trades, start_balance=START_BALANCE, risk_pct=RISK_PCT,
                       spread_price=SPREAD_USD, compound=True):
    """Per-trade P&L in $ (same math as simulate_account)."""
    if not trades:
        return []
    ev = []
    for k, t in enumerate(trades):
        ev.append((t["time"], 0, k))
        ev.append((t["exit_time"], 1, k))
    ev.sort(key=lambda e: (e[0], e[1]))
    balance = start_balance
    sizes = {}
    riskd = {}
    rows = []
    seq = 0
    for (ts, kind, k) in ev:
        t = trades[k]
        if kind == 0:
            rp = t.get("risk", 0)
            if rp <= 0:
                continue
            rd = risk_pct * (balance if compound else start_balance)
            sizes[k] = rd / rp
            riskd[k] = rd
        else:
            if k not in sizes:
                continue
            seq += 1
            net = t["R"] * riskd[k] - sizes[k] * spread_price
            balance += net
            direction = t.get("dir", "?")
            entry = t.get("entry", 0.0)
            risk = t.get("risk", 0.0)
            if direction == "long":
                exit_px = entry + t["R"] * risk
            else:
                exit_px = entry - t["R"] * risk
            tp_r = t.get("tp_r")
            rows.append({
                "n": seq,
                "rule": t.get("setup", "?"),
                "detector": t.get("rule", "?"),
                "dir": direction,
                "entry_time": t["time"],
                "exit_time": t["exit_time"],
                "entry": entry,
                "exit_px": exit_px,
                "sl": t.get("sl"),
                "tp": t.get("tp"),
                "R": t["R"],
                "tp_r": tp_r,
                "tp_src": t.get("tp_source"),
                "net_usd": net,
                "balance": balance,
                "exit": t.get("exit_reason", "?"),
                "fill_kind": t.get("fill_kind", "?"),
                "entry_tol_atr": t.get("entry_tol_atr"),
                "entry_tol_usd": t.get("entry_tol_usd"),
                "fill_gap": t.get("fill_gap"),
            })
    return rows
