"""
Per-asset calibration profiles for multi-symbol trading.

Each asset has its OWN:
  - rule subset (best performers from backtest research)
  - min SL (fixed or ATR-based — different pip/tick scale per market)
  - risk_pct (portfolio weight — lower on noisier alts)
  - TP mode / filters

Gold = dense live preset. Alts = tuned dense-wide.
P&L math is per-asset R × risk$; min SL must match that market's volatility.
"""
import strategy as S
import mt5_data as M
import floating_config as FC

# --- OOS FROZEN rules (mtf regime, oos_test 2026-07-29) ---
# Gold defaults to its M15 set here; FC.apply_tf_paths() rewrites
# PROFILES["XAUUSD"]["rules_override"] when an M5 bot starts.
_RULES_GOLD = FC.OOS_RULES["XAUUSD"]
_RULES_BTC = FC.OOS_RULES["BTCUSD"]
_RULES_ETH = FC.OOS_RULES["ETHUSD"]
_RULES_BRENT = FC.OOS_RULES["BRENT"]
_RULES_EUR = ("ICT_SB_L", "HARM_BAT")
_RULES_US30 = ("EW_W4", "FVG", "ADX_S", "ICT_BRK_S", "BK_HS_S")
_RULES_US500 = ("NDS_FVG", "BK_HS_S", "PDH_SW_S")

_FLOATING = FC.floating_opt_overrides()

PROFILES = {
    "XAUUSD": {
        "label": "Gold",
        "live_preset": "dense",
        "candidates": ["XAUUSD@", "XAUUSD", "XAUUSDm", "GOLD"],
        # Windsor Prime typical XAU mid-session (~0.25–0.40); live_spread()
        # overrides from MT5 tick when connected. Commission = $0 on Prime.
        "spread": 0.30,
        "min_sl": 3.0,
        "min_sl_mode": "fixed",
        "risk_pct": 0.01,
        "params": {"impulse_mult": 1.2, "base_max": 0.5, "fl_mult": 1.0, **FC.TOLERANCE},
        "session": None,  # 24/7 — backtest +857%→+2781% compound (90d)
        "search_contains": "XAU",
        "rules_override": _RULES_GOLD,
        "opt_overrides": {
            "session_start": None,
            "session_end": None,
            "htf_trend": True,
            "htf_trend_tfs": ("H4",),
            # Fib only on OB (gold has no OB) — DEMAND/VWAP/CH-REV/WYCK skip fib retrace
            "fib_ob_only": True,
            "fib_demand_exempt": True,
            "vp_mode": None,
            "max_concurrent": 4,
            **_FLOATING,
        },
    },
    "BTCUSD": {
        "label": "Bitcoin",
        "live_preset": "dense_wide",
        "candidates": ["BTCUSD@", "BTCUSD", "BTCUSDm", "BITCOIN"],
        "spread": 40.0,
        "spread_fixed": True,
        "min_sl": 200.0,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.35,
        "min_sl_floor": 100.0,
        "min_sl_cap": 500.0,
        "min_sl_pct_floor": 0.003,
        "min_sl_pct_cap": 0.008,
        "risk_pct": 0.005,
        "params": {"impulse_mult": 1.2, "base_max": 0.40, "fl_mult": 0.85, **FC.TOLERANCE},
        "session": (9, 21),
        "search_contains": "BTC",
        "rules_override": _RULES_BTC,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.6,
            "require_fib": True,
            "fib_ob_only": True,
            "fib_nds_exempt": True,
            "fib_demand_exempt": True,
            "vp_mode": None,
            "tp_mode": "fixed",
            "fixed_tp_r": 3.0,
            "be_trigger": 2.0,
            "htf_trend": True,
            "htf_trend_tfs": ("H4",),
            "max_concurrent": 2,
            **_FLOATING,
        },
    },
    "ETHUSD": {
        "label": "Ethereum",
        "live_preset": "dense_wide",
        "candidates": ["ETHUSD@", "ETHUSD", "ETHUSDm", "ETHEREUM"],
        "spread": 2.2,
        "spread_fixed": True,
        "min_sl": 15.0,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.35,
        "min_sl_floor": 8.0,
        "min_sl_cap": 80.0,
        "min_sl_pct_floor": 0.003,
        "min_sl_pct_cap": 0.008,
        "risk_pct": 0.015,
        "params": {"impulse_mult": 1.2, "base_max": 0.40, "fl_mult": 0.85, **FC.TOLERANCE},
        "session": (9, 21),
        "search_contains": "ETH",
        "rules_override": _RULES_ETH,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.6,
            "require_fib": True,
            "fib_ob_only": True,
            "fib_nds_exempt": True,
            "fib_demand_exempt": True,
            "vp_mode": None,
            "tp_mode": "fixed",
            "fixed_tp_r": 3.0,
            "be_trigger": 2.0,
            "htf_trend": True,
            "htf_trend_tfs": ("H4",),
            "max_concurrent": 2,
            "meta_gate": False,
            **_FLOATING,
        },
    },
    "BRENT": {
        "label": "Brent Oil",
        "live_preset": "dense_wide",
        "candidates": [
            "BRENTCASH", "UKBRENT.V26", "UKBRENT.Q26", "UKBRENT",
            "WTICASH", "WTI", "OIL", "BRENT",
        ],
        "spread": 0.05,
        "min_sl": 0.18,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.2,
        "min_sl_floor": 0.10,
        "min_sl_cap": 0.45,
        "risk_pct": 0.005,
        "params": {"impulse_mult": 1.05, "base_max": 0.42, "fl_mult": 0.95, **FC.TOLERANCE},
        "session": (9, 21),
        "search_contains": "BRENT",
        "rules_override": _RULES_BRENT,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.65,
            "fib_ob_only": True,
            "fib_demand_exempt": True,
            "fib_nds_exempt": True,
            "vp_mode": None,
            "tp_mode": "htf_blend",
            "fixed_tp_r": 2.0,
            "htf_trend": True,
            "htf_trend_tfs": ("H4",),
            "max_concurrent": 2,
            **_FLOATING,
        },
    },
    "EURUSD": {
        "label": "Euro / USD",
        "live_preset": "dense_wide",
        "candidates": ["EURUSD@", "EURUSD", "EURUSDm"],
        "spread": 0.00012,
        "min_sl": 0.0008,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.5,
        "min_sl_floor": 0.00055,
        "min_sl_cap": 0.0012,
        "risk_pct": 0.005,
        "params": {"impulse_mult": 0.85, "base_max": 0.30, "fl_mult": 0.75},
        "session": (8, 17),
        "search_contains": "EURUSD",
        "rules_override": _RULES_EUR,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.5,
            "require_fib": True,
            "fib_nds_exempt": True,
            "tp_mode": "fixed",
            "fixed_tp_r": 2.0,
            "min_tp_r": 1.5,
            "max_tp_r": 3.0,
            "htf_trend_tfs": ("H1",),
            "htf_tfs": ("H1",),
            "max_concurrent": 1,
        },
    },
    "US30": {
        "label": "Dow Jones",
        "live_preset": "dense_wide",
        "candidates": [
            "US30CASH", "US30@", "US30", "DJ30", "DJI", "DOW30", "WS30",
        ],
        "spread": 2.0,
        "min_sl": 50.0,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.2,
        "min_sl_floor": 35.0,
        "min_sl_cap": 120.0,
        "risk_pct": 0.005,
        "params": {"impulse_mult": 1.0, "base_max": 0.40, "fl_mult": 0.90},
        "session": (14, 22),
        "search_contains": "US30",
        "rules_override": _RULES_US30,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.65,
            "fib_nds_exempt": True,
            "tp_mode": "htf_blend",
            "fixed_tp_r": 2.0,
            "regime_gate": True,
            "regime_tf": "H1",
            "max_concurrent": 1,
        },
    },
    "US500": {
        "label": "S&P 500",
        "live_preset": "dense_wide",
        "candidates": [
            "US500CASH", "US500@", "US500", "SP500", "SPX500", "SPX",
        ],
        "spread": 2.0,
        "spread_fixed": True,
        "min_sl": 8.0,
        "min_sl_mode": "atr",
        "min_sl_atr_mult": 1.2,
        "min_sl_floor": 5.0,
        "min_sl_cap": 18.0,
        "risk_pct": 0.005,
        "params": {"impulse_mult": 1.0, "base_max": 0.40, "fl_mult": 0.90, **FC.TOLERANCE},
        "session": (13, 22),
        "search_contains": "US500",
        "rules_override": _RULES_US500,
        "opt_overrides": {
            "require_atr_regime": True,
            "atr_max_ratio": 1.65,
            "fib_ob_only": True,
            "fib_demand_exempt": True,
            "fib_nds_exempt": True,
            "vp_mode": None,
            "tp_mode": "htf_blend",
            "fixed_tp_r": 2.0,
            "htf_trend": True,
            "htf_trend_tfs": ("H4",),
            "max_concurrent": 2,
            **_FLOATING,
        },
    },
}

ASSET_KEYS = tuple(PROFILES.keys())


def preset_for_profile(prof, spike=False):
    name = prof.get("live_preset", "dense_wide")
    if spike:
        wide = name != "dense"
        return S.dense_spike_settings(wide=wide)
    if name == "dense":
        return S.dense_settings()
    if name == "dense_plus":
        return S.dense_plus_settings(wide=False)
    return S.dense_plus_settings(wide=True)


def resolve_live_opt(prof, spike=False, cli_dense=False, cli_dense_wide=False):
    if cli_dense_wide or spike:
        return preset_for_profile({**prof, "live_preset": "dense_wide"}, spike=spike)
    if cli_dense:
        return preset_for_profile({**prof, "live_preset": "dense"}, spike=False)
    return preset_for_profile(prof, spike=spike)


def get_profile(asset_or_symbol):
    key = str(asset_or_symbol).upper().replace("@", "").replace("/", "")
    if key in PROFILES:
        return key, PROFILES[key]
    if "XAU" in key or "GOLD" in key:
        return "XAUUSD", PROFILES["XAUUSD"]
    if "BTC" in key:
        return "BTCUSD", PROFILES["BTCUSD"]
    if "ETH" in key:
        return "ETHUSD", PROFILES["ETHUSD"]
    if "BRENT" in key or "WTI" in key or "OIL" in key or "UKBRENT" in key:
        return "BRENT", PROFILES["BRENT"]
    if "EURUSD" in key or key == "EUR":
        return "EURUSD", PROFILES["EURUSD"]
    if "US30" in key or "DJ30" in key or "DJI" in key or "DOW30" in key or key == "DOW":
        return "US30", PROFILES["US30"]
    if "US500" in key or "SP500" in key or "SPX" in key:
        return "US500", PROFILES["US500"]
    return "XAUUSD", PROFILES["XAUUSD"]


def _symbol_matches_profile(symbol, prof):
    """True if symbol belongs to this asset (not e.g. XAUUSD when asset is BRENT)."""
    if not symbol:
        return False
    su = str(symbol).upper().replace("@", "").replace("/", "")
    for c in prof.get("candidates", []):
        cu = str(c).upper().replace("@", "").replace("/", "")
        if su == cu or su.startswith(cu) or cu.startswith(su):
            return True
    needle = prof.get("search_contains", "")
    return bool(needle and needle in su)


def resolve_symbol_for_profile(asset_key, mt5=None, preferred=None):
    mt5 = mt5 or M.connect()
    _, prof = get_profile(asset_key)
    tried = []
    candidates = []
    if preferred and _symbol_matches_profile(preferred, prof):
        candidates.append(preferred)
    for s in prof["candidates"]:
        if s not in candidates:
            candidates.append(s)
    for sym in candidates:
        tried.append(sym)
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        if not info.visible and not mt5.symbol_select(sym, True):
            continue
        if mt5.symbol_info_tick(sym) is None:
            continue
        return sym
    needle = prof.get("search_contains", "")
    if needle:
        for info in sorted(mt5.symbols_get() or [], key=lambda x: x.name):
            if needle in info.name.upper():
                if mt5.symbol_select(info.name, True):
                    if mt5.symbol_info_tick(info.name) is not None:
                        return info.name
    raise RuntimeError(
        f"No symbol for {asset_key}. Tried: {tried}. "
        "Add symbol to Market Watch in MT5 or pass --symbol."
    )


def live_spread(symbol, profile, mt5=None):
    """Effective spread in price units for backtest P&L and live settings."""
    spread = profile["spread"]
    if profile.get("spread_fixed"):
        return spread
    if mt5 is None:
        return spread
    tick = mt5.symbol_info_tick(symbol)
    if tick and tick.ask and tick.bid:
        return max(spread, tick.ask - tick.bid)
    return spread


def merge_params(base_params, profile):
    p = dict(base_params or S.DEFAULT_PARAMS)
    p.update(profile.get("params") or {})
    return p


def apply_profile_to_settings(opt, profile, spread_override=None, min_sl_override=None):
    out = dict(opt)
    out["min_risk_usd"] = min_sl_override if min_sl_override is not None else profile["min_sl"]
    out["spread_usd"] = spread_override if spread_override is not None else profile["spread"]
    sess = profile.get("session")
    if sess:
        out["session_start"], out["session_end"] = sess
    for k, v in (profile.get("opt_overrides") or {}).items():
        out[k] = list(v) if k in ("htf_tfs", "htf_trend_tfs") and isinstance(v, tuple) else v
    if profile.get("rules_override"):
        out["enabled"] = list(profile["rules_override"])
    return out


def asset_backtest_kwargs(asset_key, opt=None, spread=None, spike=False, min_sl=None):
    _, prof = get_profile(asset_key)
    base = opt if opt is not None else preset_for_profile(prof, spike=spike)
    kw = apply_profile_to_settings(base, prof, spread, min_sl_override=min_sl)
    kw["detector_params"] = merge_params(S.DEFAULT_PARAMS, prof)
    if spike:
        import spike_strategies as sp
        kw["enabled"] = list(kw.get("enabled", opt.get("enabled", [])))
        for r in ("SPIKE_BRK_L", "SPIKE_BRK_S", "SPIKE_SQS_L", "SPIKE_SQS_S"):
            if r not in kw["enabled"]:
                kw["enabled"].append(r)
        kw["spike_mode"] = True
        kw["fib_spike_exempt"] = True
        kw["spike_params"] = dict(sp.SPIKE_DEFAULT)
        kw["spike_params"].update(prof.get("spike_params") or {})
    return kw, prof
