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


def rule_allowed(asset: str, rule: str, regime: str,
                 direction: str | None = None, macro: int = 0) -> bool:
    """Regime + MTF macro bias filter (CH-REV long only in bull context)."""
    asset = str(asset).upper()
    rule = str(rule).upper()
    regime = str(regime).upper()
    d = str(direction or "").lower()
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
