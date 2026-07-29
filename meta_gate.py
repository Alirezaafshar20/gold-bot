"""
Meta-Gate (Phase 2): allow a rule only in market regimes where it was
profitable on OOS TEST data.

Reads reports/oos_test.json (from oos_test.py) and builds:
    (asset, rule) -> allowed regimes

If no rule is allowed in the current regime, the asset skips trading
that bar — better than forcing a losing setup.
"""
from __future__ import annotations

import json
import os

DEFAULT_JSON = "reports/oos_test.json"
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE")


class MetaGate:
    """Per-asset, per-rule, per-regime allow table from frozen OOS results."""

    def __init__(self, table, min_regime_n=3, skip_empty=True):
        """
        table: dict[(asset, rule)] -> frozenset of regime names
        skip_empty: if True, block a rule with zero allowed regimes entirely
        """
        self.table = dict(table)
        self.min_regime_n = min_regime_n
        self.skip_empty = skip_empty

    def allows(self, asset, rule, regime):
        key = (str(asset).upper(), str(rule).upper())
        allowed = self.table.get(key)
        if allowed is None:
            return True
        if self.skip_empty and not allowed:
            return False
        return regime in allowed

    def active_rules(self, asset, regime):
        asset = str(asset).upper()
        out = []
        for (a, rule), regs in self.table.items():
            if a == asset and regime in regs:
                out.append(rule)
        return out

    def has_any(self, asset, regime):
        return bool(self.active_rules(asset, regime))

    def summary(self, assets=None):
        lines = []
        for (a, rule), regs in sorted(self.table.items()):
            if assets and a not in {x.upper() for x in assets}:
                continue
            lines.append(f"  {a} {rule:<12} -> {','.join(sorted(regs)) or 'BLOCKED'}")
        return "\n".join(lines)


def load_meta_gate(
    path=DEFAULT_JSON,
    assets=None,
    min_regime_n=3,
    require_positive_r=True,
    exclude_weak_rules=True,
    weak_pf=1.0,
):
    """
    Build MetaGate from oos_test.json per-rule TEST by_regime stats.

    A regime is allowed when:
      - n >= min_regime_n
      - tot_r > 0 (if require_positive_r)
    Rules with overall TEST PF < weak_pf are excluded entirely when
    exclude_weak_rules=True.
    """
    if not os.path.isfile(path):
        return None
    data = json.load(open(path, encoding="utf-8"))
    want = {a.upper() for a in assets} if assets else None
    table = {}

    for block in data:
        asset = block["asset"].upper()
        if want and asset not in want:
            continue
        for rule, rd in block.get("per_rule", {}).items():
            test = rd.get("test") or {}
            if exclude_weak_rules and test.get("n", 0) >= 5:
                if test.get("pf", 1) < weak_pf or test.get("ret", 0) < 0:
                    table[(asset, rule.upper())] = frozenset()
                    continue
            allowed = set()
            for reg, st in (test.get("by_regime") or {}).items():
                if not st:
                    continue
                n = st.get("n", 0)
                tot_r = st.get("tot_r", 0)
                if n < min_regime_n:
                    continue
                if require_positive_r and tot_r <= 0:
                    continue
                allowed.add(reg)
            table[(asset, rule.upper())] = frozenset(allowed)

    return MetaGate(table, min_regime_n=min_regime_n)


def regime_at(rmap, signal_time, default="RANGE"):
    if rmap is None:
        return default
    try:
        return rmap.at(signal_time)
    except Exception:
        return default


def meta_gate_from_table(table: dict, min_regime_n=3, skip_empty=True) -> MetaGate:
    """Build MetaGate from {(asset, rule): iterable_of_regimes}."""
    norm = {}
    for key, regs in table.items():
        if isinstance(key, tuple) and len(key) == 2:
            a, rule = key
        else:
            continue
        norm[(str(a).upper(), str(rule).upper())] = frozenset(regs or [])
    return MetaGate(norm, min_regime_n=min_regime_n, skip_empty=skip_empty)


def load_weekly_gate(path="reports/weekly_gate.json", assets=None, min_regime_n=3):
    """
    Load weekly adaptive gate built by weekly_adaptive.py.
    Returns MetaGate or None if missing/invalid.
    """
    if not os.path.isfile(path):
        return None
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    want = {a.upper() for a in assets} if assets else None
    table = {}
    for asset, rules in (data.get("table") or {}).items():
        asset = str(asset).upper()
        if want and asset not in want:
            continue
        for rule, regs in (rules or {}).items():
            table[(asset, str(rule).upper())] = frozenset(regs or [])
    if not table:
        return None
    return meta_gate_from_table(table, min_regime_n=min_regime_n)


def load_effective_gate(
    assets=None,
    oos_path=DEFAULT_JSON,
    weekly_path="reports/weekly_gate.json",
    weekly_enabled=True,
    min_regime_n=3,
    require_positive_r=True,
    exclude_weak_rules=True,
):
    """Prefer weekly_gate.json when enabled and present; else OOS meta-gate."""
    if weekly_enabled:
        wg = load_weekly_gate(weekly_path, assets=assets, min_regime_n=min_regime_n)
        if wg is not None:
            return wg, "weekly"
    mg = load_meta_gate(
        oos_path,
        assets=assets,
        min_regime_n=min_regime_n,
        require_positive_r=require_positive_r,
        exclude_weak_rules=exclude_weak_rules,
    )
    return mg, "oos"
