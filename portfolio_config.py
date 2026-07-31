"""
Max-concurrent per asset (Gold: flat max 4).

  XAUUSD: max 4 open positions/orders at once.
  CH_REV_L: only bull context (TREND_UP or RANGE with macro>0).
"""
from __future__ import annotations

import meta_gate as MG

# ── portfolio universe ───────────────────────────────────────────────────────
DISABLED_ASSETS = ("US30", "US500", "BRENT", "BTCUSD")
PORTFOLIO_ASSETS = ("XAUUSD",)

RISK_MAP = {
    "XAUUSD": 0.035,
    "US500": 0.035,
    "US30": 0.015,
    "BRENT": 0.015,
    "BTCUSD": 0.015,
    "ETHUSD": 0.015,
}

MAX_CONCURRENT = {
    "XAUUSD": 4,
    "BTCUSD": 2,
    "US500": 2,
    "US30": 1,
    "BRENT": 2,
}
MAX_CONCURRENT_PER_ASSET = 2

# No tiering — flat cap only
TIERED_CONCURRENT: dict[str, dict] = {}

# Rules that may use premium slots (trending regime required)
GOLD_PREMIUM_RULES = frozenset({"DEMAND", "VWAP_L", "VWAP_S", "WYCK_SOW"})
TREND_REGIMES = frozenset({"TREND_UP", "TREND_DOWN"})

MAX_PORTFOLIO_RISK = 0.15
MAGIC_BASE = 779000

MAGIC = {
    "XAUUSD": 779001,
    "BTCUSD": 779002,
    "US30": 779003,
    "US500": 779004,
    "BRENT": 779005,
}

# Per-timeframe magic offset — lets an M5 bot and an M15 bot share one account
# with fully separated positions (max-concurrent, trailing, journal, history).
TF_MAGIC_OFFSET = {"M15": 0, "M5": 100}


def magic_for(asset: str, tf: str = "M15") -> int:
    base = MAGIC.get(str(asset).upper(), MAGIC_BASE + hash(str(asset).upper()) % 50)
    return base + TF_MAGIC_OFFSET.get(str(tf).upper(), 500)

SLICE_STATE_FILE = "data/portfolio_slices.json"


def tiered(asset: str) -> dict | None:
    return TIERED_CONCURRENT.get(str(asset).upper())


def max_concurrent(asset: str) -> int:
    t = tiered(asset)
    if t:
        return t["base"] + t["premium"]
    return MAX_CONCURRENT.get(str(asset).upper(), MAX_CONCURRENT_PER_ASSET)


# Sides a rule is allowed to trade, when it should not trade both. Empty means
# every enabled rule may trade both ways, which is the historical behaviour.
#
# Format: {ASSET: {RULE: ("short",) | ("long",) | ("long", "short")}}
#
# Pooling a falling 60-day window with a rising one separates rules that lose
# because the market went against them from rules that lose regardless. On gold
# that split is stark — ADX long returns PF 1.26 over 59 trades, while these
# four give back 20.1R between them and still earn their keep short:
#
#   "XAUUSD": {"NDS": ("short",), "FL": ("short",),
#              "NDS_BOS": ("short",), "NDS_FVG": ("short",)}
#
# Measure with rule_sides_test.py before pasting that in; it changes live.
RULE_SIDES: dict[str, dict[str, tuple[str, ...]]] = {}


# Sides allowed inside each regime label. Empty means every label trades both
# ways, which is the historical behaviour.
#
# Format: {ASSET: {REGIME: ("short",) | ("long",) | ("long", "short")}}
#
# This is a different question from RULE_SIDES above, and regime_audit.py is what
# answers it. That tool measures each label as a pure directional bias — mean
# favourable excursion over mean adverse excursion, in ATR units, on every H1 bar
# of the archive — with no rules involved. On four years of gold the RANGE label
# scores 1.056 for longs against 1.064 for an unfiltered long, i.e. it adds
# nothing directional, while TREND_UP adds +0.069 and TREND_DOWN +0.066.
#
# The book nonetheless takes 70% of its longs inside RANGE, and that is where
# 11.2R of its 13.3R long loss is booked. Shorts in RANGE are the opposite: they
# are the strongest cell measured, because a supply-zone short in a range is a
# mean-reversion trade that does not need the label to be directional.
#
#   "XAUUSD": {"RANGE": ("short",)}
#
# Measure with regime_sides_test.py before pasting that in; it changes live.
REGIME_SIDES: dict[str, dict[str, tuple[str, ...]]] = {}


def regime_sides(asset: str, regime: str) -> tuple[str, ...] | None:
    """Allowed sides inside a regime label, or None when both are permitted."""
    table = REGIME_SIDES.get(str(asset).upper())
    if not table:
        return None
    return table.get(str(regime).upper())


def rule_sides(asset: str, rule: str) -> tuple[str, ...] | None:
    """Allowed sides for a rule, or None when both sides are permitted.

    Rule tags reach here in several spellings (NDS-FVG, NDS_FVG, nds_fvg), so
    the lookup normalises separators rather than trusting the caller.
    """
    table = RULE_SIDES.get(str(asset).upper())
    if not table:
        return None
    key = str(rule).upper().replace("-", "_")
    return table.get(key)


def rule_allowed(asset: str, rule: str, regime: str,
                 direction: str | None = None, macro: int = 0) -> bool:
    """Regime + MTF macro bias filter, plus side restrictions.

    Two independent side filters apply, and a direction has to satisfy both:
    RULE_SIDES asks whether this rule may trade that way at all, REGIME_SIDES
    asks whether anything may trade that way in this market state.
    """
    asset = str(asset).upper()
    rule = str(rule).upper()
    regime = str(regime).upper()
    d = str(direction or "").lower()
    sides = rule_sides(asset, rule)
    if sides is not None and d and d not in sides:
        return False
    rsides = regime_sides(asset, regime)
    if rsides is not None and d and d not in rsides:
        return False
    if asset == "XAUUSD" and rule == "CH_REV_L":
        if regime == "TREND_UP":
            return True
        if regime == "RANGE" and macro > 0 and d in ("", "long"):
            return True
        return False
    return True


def is_golden_signal(
    asset: str,
    rule: str,
    regime: str,
    meta_gate: MG.MetaGate | None = None,
) -> bool:
    """
    Premium / golden slot: core gold rules in a trend regime (not RANGE).
    CH-REV never uses premium — only allowed in TREND_UP via rule_allowed().
    """
    asset = str(asset).upper()
    rule = str(rule).upper()
    regime = str(regime).upper()
    if tiered(asset) is None:
        return False
    if rule not in GOLD_PREMIUM_RULES:
        return False
    if regime not in TREND_REGIMES:
        return False
    if meta_gate is not None and not meta_gate.allows(asset, rule, regime):
        return False
    return True


def slot_available(asset: str, n_base: int, n_premium: int, golden: bool) -> bool:
    """Check tier caps before opening."""
    t = tiered(asset)
    if t is None:
        return (n_base + n_premium) < max_concurrent(asset)
    base_cap, prem_cap = t["base"], t["premium"]
    total = n_base + n_premium
    if total >= base_cap + prem_cap:
        return False
    if golden:
        return n_premium < prem_cap
    return n_base < base_cap


def tier_label(asset: str, golden: bool) -> str:
    if golden and tiered(asset):
        return "golden"
    return "base"


def describe_concurrent(asset: str) -> str:
    t = tiered(asset)
    if t:
        return f"{asset} {t['base']}base+{t['premium']}golden"
    mc = max_concurrent(asset)
    return f"{asset} {mc}"
