"""
Floating trading system — 4-phase configuration.

Phase 1  HONESTY     OOS test + fixed-risk + regime split (oos_test / oos_analyze)
Phase 2  META-GATE   Rule x regime filter from oos_test.json (meta_gate.py)
Phase 3  ADAPTIVE    Re-run OOS every N days on TRAIN only; never tune on TEST
Phase 4  PARITY      Daily backtest vs live comparison (parity_check.py)

All live + backtest paths read this file so behaviour stays aligned.
"""
import os
import sys
from datetime import datetime, timedelta

import meta_gate as MG

# ── active universe (only these trade) ─────────────────────────────────────
ACTIVE_ASSETS = ("XAUUSD",)

# ── OOS frozen rules ──────────────────────────────────────────────────────
# Gold is selected per TIMEFRAME. The M5 and M15 runs of 2026-07-29 (mtf
# regime, corrected engine) agree on the NDS core and disagree on everything
# else, so a single shared tuple would trade rules with no measured edge on
# the timeframe actually running. apply_tf_paths() swaps the gold entry.
#
# WYCK cleared M5 selection but is deliberately not here. It qualified on a
# 35-trade training sample, then produced 351 test trades at +0.20 R and a
# 28.7% win rate — losing in both trend regimes (-0.15 and -0.12 R) and
# profitable only in range. Dropping it moved the flat-risk drawdown from
# 11.4% to 6.4% and expectancy from +0.53 to +0.75 R over the same window.
OOS_RULES_TF = {
    "M15": ("NDS_FRESH", "NDS_FVG", "NDS", "ADX_L", "FL"),
    "M5": ("NDS_FRESH", "NDS_BOS", "NDS_FVG", "NDS"),
}

OOS_RULES = {
    "XAUUSD": OOS_RULES_TF["M15"],
    "BTCUSD": ("BOS", "FVG", "OB", "DEMAND", "ADX_L"),
    "ETHUSD": ("BOS", "FVG", "OB", "DEMAND", "ADX_L"),  # same set until ETH OOS run
    # EW_ABC pulled: it defines no invalidation of its own (proximal == distal on
    # all 521 audited signals) and its pivot scan read unconfirmed pivots, so 20%
    # of its signals could not be reproduced live. Re-add after re-running OOS.
    "BRENT": ("FL", "ADX_L"),  # BOS/PDC_RC_S blocked in OOS meta-gate
}

# ── Phase 2: regime detector (MTF v2 — H4 structure + H1 ER) ───────────────
REGIME = {
    "regime_gate": True,
    "regime_tf": "H1",
    "regime_detector": "mtf",
    "regime_win": 40,
    "regime_er_trend": 0.35,
    "regime_slope_k": 0.0006,
    "regime_er_hi": 0.32,
    "regime_er_lo": 0.20,
    "regime_confirm": 2,
    "regime_h4_win": 30,
    "regime_h4_lookback": 60,
    # V-shape / shock guards (validated via regime_guard_ab.py before enabling)
    "regime_structure_break": False,  # H1 donchian break overrides slow H4 macro
    "regime_brk_win": 40,
    "regime_shock_gate": False,       # block entries when realized vol explodes
    "regime_shock_ratio": 1.8,        # TR-sum(24 H1) >= ratio × 30d median
    "regime_shock_win": 24,
    "regime_shock_baseline": 720,
    "regime_flip_cooldown_h": 0.0,    # hours to distrust a fresh macro flip
}

# ── Phase 2: meta-gate ───────────────────────────────────────────────────
META = {
    "meta_gate": True,
    "meta_gate_json": "reports/oos_test.json",
    "meta_min_regime_n": 3,
    "meta_require_positive_r": True,
    "meta_exclude_weak_rules": True,
    "meta_skip_if_no_rules": True,   # no OOS edge in this regime -> skip bar
}

# ── Phase 3: weekly adaptive (rolling health on top of OOS) ───────────────
WEEKLY = {
    "enabled": True,
    "walkforward": True,   # replay trades week-by-week with causal rolling gate
    "json": "reports/weekly_gate.json",
    "rolling_weeks": 8,
    "regime_report_days": 7,
    "min_rolling_n": 3,
    "pause_pf": 0.70,
    "trades_csv": "reports/portfolio_trades.csv",
    # Live/backtest parity (2026-07-16):
    #   causal_week_cutoff    gate ignores trades closed inside the current ISO
    #                         week — identical basis to the walk-forward replay
    #   live_refresh_on_trades False = refresh only on ISO week change
    #                         (backtest never refreshes mid-week)
    #   live_refresh_hours    extra mid-week refresh cadence. 0 = off. Anything
    #                         else makes the gate depend on the wall-clock hour
    #                         the bot happened to be started, so two machines
    #                         (and a replay) can hold different gates mid-week.
    "causal_week_cutoff": True,
    "live_refresh_on_trades": False,
    "live_refresh_hours": 0,
}

# ── Phase 3b: full OOS recalibration schedule ────────────────────────────
RECALIBRATE = {
    "interval_days": 90,
    "train_cutoff": "2024-12-31",    # move forward only after full OOS re-run
    "oos_json": "reports/oos_test.json",
    "last_run_hint": "reports/oos_test.txt",
}

# ── Phase 4: parity ──────────────────────────────────────────────────────
PARITY = {
    "backtest_csv": "reports/portfolio_trades.csv",
    "compare_days": 1,
}

# ── Entry fill (zone retest) ──────────────────────────────────────────────
# Professional SMC split (why "always wait for close" was wrong):
#   Structure (BOS / zone create) already happened on the SIGNAL bar.
#   Retest entry = price RETURNS and TOUCHES the zone — a wick/shadow is
#   a valid fill (institutions leave limits at the proximal). Candle CLOSE
#   is the filter for INVALIDATION (close beyond distal), not for every entry.
#   Waiting for a reclaim close on EVERY rule was inherited from an old ML
#   probability model; it forces late market fills and collapses R:R.
#
# Modes:
#   none         : enter on first tick/wick in the entry band (classic retest)
#   m15_close    : touch + close inside band + directional candle
#   rejection    : touch + close reclaims proximal (candle-rejection pattern)
#   rule_confirm : none for most rules; rejection only for entry_confirm_rules
ENTRY = {
    "entry_confirm_mode": "rule_confirm",
    # Candle-pattern rules where the reclaim close IS the setup:
    "entry_confirm_rules": ("CH_REV_L", "WYCK_SOW"),
    # Skip fills whose remaining reward / actual risk < this (zone stays armed).
    # Applies to late market fills and honest confirm-close fills alike.
    "min_fill_rr": 1.0,
    # Touch/retest rules: arm a LIMIT at proximal (institutional). Rejection
    # rules still market on confirm close. If False, touch uses market-in-band.
    "touch_use_limit": True,
    # At most one open OR armed idea per direction (no NDS+VWAP double short).
    "max_same_dir": 1,
    # Windsor Brokers Prime (XAU): commission $0, cost is in spread.
    # limit_fill_mode for live_replay broker-sim (and docs for live intent):
    #   strict    — ask must reach the limit price (mid OHLC minus half-spread).
    #               This is what a real resting LIMIT does; default.
    #   proximal  — mid OHLC touches proximal (ignores the spread haircut)
    #   zone      — fill whenever M1 trades anywhere in the entry band, still
    #               booking the entry at proximal. Gifts up to entry_tol×ATR of
    #               entry price the broker never gives — replay-only fantasy.
    "broker": "windsor_prime",
    "commission_per_lot": 0.0,
    "limit_fill_mode": "strict",
    # Where the resting LIMIT actually sits inside the entry band:
    #   band_edge — near edge (first price the market reaches). Every touch of
    #               the band becomes a real fill at a real price.
    #   proximal  — far edge. Better price, far fewer fills.
    # Backtest, replay and live all read this, so they cannot drift apart.
    "limit_at": "band_edge",
}

# entry_tol: defines ENTRY ZONE height (live) and fill band (backtest)
#   LONG  zone [proximal … proximal+entry_tol×ATR]
#   SHORT zone [proximal-entry_tol×ATR … proximal]
# Live: M15 close registers zone → every 10s market if price inside
# detect/zone: softer pattern match on FVG, DEMAND, VWAP, CH-REV
# invalidate: extra room before canceling a pending zone (live + backtest)
TOLERANCE = {
    "entry_tol_atr": 0.38,
    "detect_tol_atr": 0.19,
    "zone_tol_atr": 0.23,
    "invalidate_tol_atr": 0.13,
}

# ── Clock: one canonical timeline for backtest, replay and live ────────────
# Dukascopy files are naive UTC. MT5 hands back the BROKER SERVER wall clock
# (measured UTC+3 on WM Markets), so mt5_data normalises it to UTC on the way
# in. Everything downstream therefore indexes in UTC and the two engines can
# no longer disagree about what hour or what day a bar belongs to.
#
#   session_tz       exchange clock the hour filters mean (handles DST itself)
#   day_anchor_hour  UTC hour where the trading DAY rolls over. 21:00 UTC is
#                    00:00 on a UTC+3 server ≈ the 17:00 New York close that
#                    defines "yesterday's high" for everyone who trades it.
CLOCK = {
    "server_utc_offset": None,   # None = auto-detect from the terminal
    "session_tz": "America/New_York",
    "day_anchor_hour": 21,
    "week_anchor_dow": 6,        # Sunday open
}

# ── Risk geometry ─────────────────────────────────────────────────────────
# sl_buffer_atr: room BEYOND the structural level before the stop sits.
# The old hard-coded 0.10 parked the stop a few cents under the most obvious
# swing on the chart. Sweeping the buffer over 1,889 XAUUSD M15 trades put the
# optimum at 0.50 ATR (avg R 0.310 -> 0.332, win rate 37.9% -> 41.1%).
RISK = {
    "sl_buffer_atr": 0.50,
    "sl_swing_lb": 10,
    # Reject setups whose own geometry is impossible instead of silently
    # falling back to a generic swing stop. See strategy.validate_setups().
    "validate_setups": True,
}


def _apply_gold_rules(t: str):
    """Point gold at the rule set its own OOS run selected for this timeframe.

    symbol_profiles copies the tuple into PROFILES at import time. When it is
    already loaded the live runner is holding the M15 default, so the profile
    has to be corrected in place as well as the source dict.
    """
    rules = OOS_RULES_TF.get(t, OOS_RULES_TF["M15"])
    OOS_RULES["XAUUSD"] = rules
    prof = sys.modules.get("symbol_profiles")
    if prof is not None:
        entry = getattr(prof, "PROFILES", {}).get("XAUUSD")
        if entry is not None:
            entry["rules_override"] = rules
    return rules


def apply_tf_paths(tf: str) -> None:
    """Isolate per-timeframe state files so M5 and M15 bots share one account.

    M15 keeps legacy names (oos_test.json, weekly_gate.json, …).
    M5 uses reports/*_M5.* — run: python oos_test.py --tf M5

    Gold's rule set is timeframe-specific too, so it is swapped here rather
    than baked into symbol_profiles: both bots import the same profile module.
    """
    t = str(tf).upper().replace(" ", "")
    if t in ("5M", "5"):
        t = "M5"
    if t in ("15M", "15", ""):
        t = "M15"
    _apply_gold_rules(t)

    if t == "M15":
        META["meta_gate_json"] = "reports/oos_test.json"
        RECALIBRATE["oos_json"] = "reports/oos_test.json"
        RECALIBRATE["last_run_hint"] = "reports/oos_test.txt"
        WEEKLY["json"] = "reports/weekly_gate.json"
        WEEKLY["trades_csv"] = "reports/portfolio_trades.csv"
        PARITY["backtest_csv"] = "reports/portfolio_trades.csv"
        return

    # M5 (or other non-M15): dedicated OOS + weekly + journal paths
    oos_json = f"reports/oos_test_{t}.json"
    META["meta_gate_json"] = oos_json
    # Until a TF-specific OOS exists, fall back to M15 gate so live still filters
    if not os.path.isfile(oos_json) and os.path.isfile("reports/oos_test.json"):
        META["meta_gate_json"] = "reports/oos_test.json"
    RECALIBRATE["oos_json"] = oos_json
    RECALIBRATE["last_run_hint"] = f"reports/oos_test_{t}.txt"
    WEEKLY["json"] = f"reports/weekly_gate_{t}.json"
    WEEKLY["trades_csv"] = f"reports/portfolio_trades_{t}.csv"
    PARITY["backtest_csv"] = f"reports/portfolio_trades_{t}.csv"


def tolerance_params():
    return dict(TOLERANCE)


def floating_opt_overrides():
    """Merge into symbol_profiles opt_overrides for active assets."""
    out = dict(REGIME)
    out.update(META)
    out.update(ENTRY)
    return out


def load_meta_gate(assets=None):
    if not META.get("meta_gate"):
        return None
    mg, _src = MG.load_effective_gate(
        assets=assets or ACTIVE_ASSETS,
        oos_path=META.get("meta_gate_json", MG.DEFAULT_JSON),
        weekly_path=WEEKLY.get("json", "reports/weekly_gate.json"),
        weekly_enabled=WEEKLY.get("enabled", False),
        min_regime_n=META.get("meta_min_regime_n", 3),
        require_positive_r=META.get("meta_require_positive_r", True),
        exclude_weak_rules=META.get("meta_exclude_weak_rules", True),
    )
    return mg


def meta_gate_source():
    """'weekly' | 'oos' | 'off' — which gate file is active."""
    if not META.get("meta_gate"):
        return "off"
    if WEEKLY.get("enabled") and os.path.isfile(WEEKLY.get("json", "")):
        return "weekly"
    return "oos"


def recalibration_due():
    """Phase 3: True if OOS json is older than interval_days."""
    path = RECALIBRATE.get("oos_json", "reports/oos_test.json")
    if not os.path.isfile(path):
        return True, "oos_test.json missing"
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    age = datetime.now() - mtime
    limit = timedelta(days=RECALIBRATE.get("interval_days", 90))
    if age > limit:
        return True, f"OOS file is {age.days}d old (limit {limit.days}d)"
    return False, f"OOS file is {age.days}d old — OK"


def print_status():
    """One-screen summary of floating system state."""
    print("=" * 72)
    print("  FLOATING SYSTEM STATUS")
    print("=" * 72)
    print(f"  Active assets : {', '.join(ACTIVE_ASSETS)}")
    print(f"  Gold rules    : {', '.join(OOS_RULES['XAUUSD'])}")
    print(f"  Regime        : {REGIME['regime_detector']} on {REGIME['regime_tf']}"
          f" confirm={REGIME['regime_confirm']}")
    src = meta_gate_source()
    gate_file = WEEKLY["json"] if src == "weekly" else META["meta_gate_json"]
    print(f"  Meta-gate     : {'ON' if META['meta_gate'] else 'OFF'}"
          f"  [{src}] ({gate_file})")
    due, msg = recalibration_due()
    print(f"  Recalibrate   : {'DUE' if due else 'OK'} — {msg}")
    print(f"  Tolerance     : entry={TOLERANCE['entry_tol_atr']} detect={TOLERANCE['detect_tol_atr']} "
          f"zone={TOLERANCE['zone_tol_atr']} invalidate={TOLERANCE['invalidate_tol_atr']} ATR")
    ecm = ENTRY.get("entry_confirm_mode", "none")
    if ecm == "rule_confirm":
        rules = ",".join(ENTRY.get("entry_confirm_rules") or ())
        lim = (f"LIMIT@{ENTRY.get('limit_at', 'band_edge')}"
               if ENTRY.get("touch_use_limit", True) else "market")
        print(f"  Entry mode    : rule_confirm — touch={lim}; "
              f"rejection close for [{rules}]; max_same_dir={ENTRY.get('max_same_dir', 1)}")
        print(f"  Broker fill   : {ENTRY.get('broker', '?')}  "
              f"limit_fill={ENTRY.get('limit_fill_mode', 'strict')}  "
              f"commission/lot=${float(ENTRY.get('commission_per_lot', 0) or 0):.2f}")
    else:
        print(f"  Entry mode    : {ecm}")
    print(f"  Stop geometry : {RISK['sl_buffer_atr']} ATR beyond structure"
          f"  (swing lookback {RISK['sl_swing_lb']} bars)"
          f"  validate={'ON' if RISK['validate_setups'] else 'OFF'}")
    print(f"  Clock         : bars in UTC; sessions on {CLOCK['session_tz']};"
          f" day rolls at {CLOCK['day_anchor_hour']:02d}:00 UTC")
    print(f"  Live journal  : {WEEKLY.get('trades_csv', 'reports/portfolio_trades.csv')}")
    _wr = ["ISO week change"]
    _wh = float(WEEKLY.get("live_refresh_hours", 0) or 0)
    if _wh > 0:
        _wr.append(f"every {_wh:g}h")
    if WEEKLY.get("live_refresh_on_trades"):
        _wr.append("after closed trades")
    print(f"  Weekly refresh: {' | '.join(_wr)}")
    mg = load_meta_gate()
    if mg:
        print("\n  Meta-gate table (active assets):")
        print(mg.summary(ACTIVE_ASSETS))
    print("=" * 72)


if __name__ == "__main__":
    print_status()
