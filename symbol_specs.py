"""
MT5 symbol contract specs — correct lot sizing and spread cost per asset.

Backtest P&L uses R-multiples × account risk ($). That is symbol-agnostic IF
min SL (risk distance in price) is calibrated per asset. Live trading needs
tick_size / tick_value from the broker to convert SL distance → lot size.
"""
import numpy as np


def _broker_usd_per_unit(mt5, symbol, ref_price):
    """
    Exact $ loss for 1.0 lot per 1.0 price-unit move, straight from the broker
    via order_calc_profit. This is the ONLY value that matches realized P&L for
    every asset class (crypto/CFD tick_value is often misreported by MT5).
    Returns None if it can't be computed.
    """
    if mt5 is None or not ref_price or ref_price <= 0:
        return None
    try:
        # Loss for a 1.0-unit adverse move on a 1.0-lot BUY (price drops by 1.0).
        prof = mt5.order_calc_profit(
            mt5.ORDER_TYPE_BUY, symbol, 1.0, ref_price, ref_price - 1.0)
        if prof is not None and abs(prof) > 0:
            return abs(prof)
    except Exception:
        pass
    return None


# Offline contract assumptions used when MT5 is unavailable (Dukascopy OOS etc.).
# usd_per_unit = $ P&L per 1.0 price-unit move on 1.0 lot.
_OFFLINE_SPECS = {
    "XAUUSD": dict(point=0.01, tick_size=0.01, tick_value=1.0,
                   contract_size=100, usd_per_unit=100.0,
                   vol_min=0.01, vol_max=100.0, vol_step=0.01, digits=2),
    "BTCUSD": dict(point=0.01, tick_size=0.01, tick_value=0.01,
                   contract_size=1, usd_per_unit=1.0,
                   vol_min=0.01, vol_max=100.0, vol_step=0.01, digits=2),
    "US500":  dict(point=0.1, tick_size=0.1, tick_value=0.1,
                   contract_size=1, usd_per_unit=1.0,
                   vol_min=0.1, vol_max=100.0, vol_step=0.1, digits=1),
    "BRENT":  dict(point=0.01, tick_size=0.01, tick_value=1.0,
                   contract_size=100, usd_per_unit=100.0,
                   vol_min=0.01, vol_max=100.0, vol_step=0.01, digits=2),
}


def offline_spec(asset_key):
    """Synthetic broker spec for offline sims (fixed-lot / no MT5)."""
    base = _OFFLINE_SPECS.get(str(asset_key).upper())
    if base is None:
        return None
    out = dict(base)
    out["symbol"] = str(asset_key).upper()
    out["spread"] = 0.0
    return out


def get_spec(symbol, mt5=None):
    """Return dict with tick_size, tick_value, point, vol_min, vol_step, digits."""
    if mt5 is None:
        import mt5_data as M
        mt5 = M.connect()
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    tick = mt5.symbol_info_tick(symbol)
    spread = (tick.ask - tick.bid) if tick and tick.ask and tick.bid else 0.0
    point = info.point or 1e-5
    tick_size = info.trade_tick_size or point
    tick_value = info.trade_tick_value or 0.0
    ref_price = (tick.ask if tick and tick.ask else None) or getattr(info, "ask", None)
    return {
        "symbol": symbol,
        "point": point,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "contract_size": getattr(info, "trade_contract_size", 100000),
        "vol_min": info.volume_min or 0.01,
        "vol_max": info.volume_max or 100.0,
        "vol_step": info.volume_step or 0.01,
        "digits": info.digits or 5,
        "spread": spread,
        # Real broker $/lot per 1.0 price unit (preferred over tick_value).
        "usd_per_unit": _broker_usd_per_unit(mt5, symbol, ref_price),
    }


def loss_per_lot_usd(spec, sl_distance, mt5=None, symbol=None, ref_price=None):
    """
    USD loss if price moves `sl_distance` against a 1.0-lot position.

    Priority:
      1) broker order_calc_profit (live, exact — handles BTC/CFD correctly)
      2) usd_per_unit factor cached in spec (also from order_calc_profit)
      3) tick_value model (last-resort fallback; can be wrong for crypto)
    """
    if sl_distance <= 0 or spec is None:
        return 0.0
    sym = symbol or spec.get("symbol")
    # 1) Ask the broker directly for the exact loss over this SL distance.
    if mt5 is not None and sym:
        if ref_price is None:
            try:
                tick = mt5.symbol_info_tick(sym)
                ref_price = tick.ask if tick and tick.ask else None
            except Exception:
                ref_price = None
        if ref_price:
            try:
                prof = mt5.order_calc_profit(
                    mt5.ORDER_TYPE_BUY, sym, 1.0, ref_price, ref_price - sl_distance)
                if prof is not None and abs(prof) > 0:
                    return abs(prof)
            except Exception:
                pass
    # 2) Cached broker $/unit factor (linear — exact for CFDs).
    upu = spec.get("usd_per_unit")
    if upu and upu > 0:
        return upu * sl_distance
    # 3) Fallback: tick_value model (legacy).
    tick_size = spec["tick_size"]
    tick_value = spec["tick_value"]
    if tick_value > 0 and tick_size > 0:
        linear = (sl_distance / tick_size) * tick_value
        cs = spec.get("contract_size") or 1.0
        if cs == 1.0:
            return max(linear, sl_distance * cs)
        return linear
    cs = spec.get("contract_size") or 1.0
    return sl_distance * cs


def calc_lot_from_spec(spec, balance, risk_fraction, sl_distance, mt5=None):
    """
    Same lot math as calc_lot but on a pre-fetched spec dict.
    Uses the cached broker usd_per_unit factor so it matches live sizing exactly.
    """
    if spec is None or sl_distance <= 0 or balance <= 0 or risk_fraction <= 0:
        return spec["vol_min"] if spec else 0.01
    risk_money = balance * risk_fraction
    loss_per_lot = loss_per_lot_usd(spec, sl_distance, mt5=mt5)
    if loss_per_lot > 0:
        lot = risk_money / loss_per_lot
    else:
        lot = spec["vol_min"]
    step = spec["vol_step"]
    lot = max(spec["vol_min"], round(lot / step) * step)
    return min(lot, spec["vol_max"])


def calc_lot(symbol, balance, risk_fraction, sl_distance, mt5=None):
    """
    Lot size so that if SL is hit, loss ≈ balance × risk_fraction (account currency).
    Sizing uses the broker's exact order_calc_profit (via loss_per_lot_usd), which
    matches realized P&L for every asset including BTC/CFD.
    """
    if sl_distance <= 0 or balance <= 0 or risk_fraction <= 0:
        return 0.01
    spec = get_spec(symbol, mt5)
    if spec is None:
        return 0.01
    risk_money = balance * risk_fraction
    loss_per_lot = loss_per_lot_usd(spec, sl_distance, mt5=mt5, symbol=symbol)
    if loss_per_lot > 0:
        lot = risk_money / loss_per_lot
    else:
        lot = spec["vol_min"]
    step = spec["vol_step"]
    lot = max(spec["vol_min"], round(lot / step) * step)
    return min(lot, spec["vol_max"])


def spread_cost_r(spread_price, sl_distance):
    """Spread drag expressed in R (for comparing assets)."""
    if sl_distance <= 0:
        return 999.0
    return spread_price / sl_distance


def calibrate_min_sl(B, cutoff_time, profile):
    """
    Effective min SL for backtest/live from profile.
    mode=fixed → profile min_sl
    mode=atr  → median ATR in test window × mult, clamped to floor/cap
    """
    base = profile.get("min_sl", 0.0)
    if profile.get("min_sl_mode") != "atr":
        return base
    t_cut = np.datetime64(cutoff_time) if cutoff_time is not None else B.t[0]
    mask = B.t >= t_cut
    atrs = B.atr[mask]
    atrs = atrs[atrs > 0]
    if len(atrs) < 10:
        atrs = B.atr[B.atr > 0]
    med = float(np.median(atrs)) if len(atrs) else base
    mult = profile.get("min_sl_atr_mult", 1.25)
    sl = med * mult
    floor = profile.get("min_sl_floor", base * 0.5)
    cap = profile.get("min_sl_cap", base * 3.0)
    # Optional: clamp SL as % of price (important for BTC — scales with coin price)
    if profile.get("min_sl_pct_floor") or profile.get("min_sl_pct_cap"):
        idx = int(np.where(mask)[0][0]) if mask.any() else len(B.c) - 1
        price = float(B.c[idx])
        if profile.get("min_sl_pct_floor"):
            floor = max(floor, price * profile["min_sl_pct_floor"])
        if profile.get("min_sl_pct_cap"):
            cap = min(cap, price * profile["min_sl_pct_cap"])
    return float(max(floor, min(cap, sl)))


def profile_risk_pct(profile, default=0.01):
    return profile.get("risk_pct", default)
