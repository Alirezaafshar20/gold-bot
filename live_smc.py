"""
================================================================================
  LIVE TRADER  —  rule-based SMC/ICT/RTM/Wyckoff strategy on MetaTrader 5
================================================================================
All messages are in plain English (Windows PowerShell does not render Persian
correctly). When a trade opens, it tells you WHICH rule fired and WHY.

How it works (mirrors the back-test exactly):
  1. On every newly-closed candle it runs all rules on that closed candle.
  2. For each signal it places a pending LIMIT order at the entry price with a
     structural stop (the price must pull back to the zone to fill).
  3. If a candle closes beyond the zone before filling, the pending order is
     cancelled (the setup is invalidated) - same as the back-test.
  4. Once filled, it manages a TRAILING stop: at +1R the stop goes to break-even,
     then it trails 1R behind the best price. There is no fixed take-profit.

Usage:
  python live_smc.py --optimized                   # RECOMMENDED preset
  python live_smc.py --optimized --dry-run
  python live_smc.py --tf M5 --symbol XAUUSD@      # legacy (all rules + trail)
  python live_smc.py --update-csv             # also save fresh CSV for backtest.py

Data: live trading reads candles DIRECTLY from MetaTrader 5 every loop.
      No CSV download is required. Use --update-csv only if you want to
      refresh data/*.csv for offline back-testing.

Risk per trade = 1% of balance by default (override with --risk, same as backtest.py: 0.01 = 1%).
This model is RULE-BASED: there is no ML model, so it NEVER needs retraining.
================================================================================
"""
import argparse, sys, time, functools, os
import datetime as dt
sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.  Run:  pip install MetaTrader5")
    sys.exit(1)

TF_MIN = {"M5": 5, "M15": 15}
LIVE_BARS = 800          # enough for VP filter (480-bar window) + indicators


def get_bars(symbol, tf_str, n=LIVE_BARS, mt5_conn=None):
    try:
        return M.fetch_bars(symbol, tf_str, count=n, mt5=mt5_conn)
    except RuntimeError:
        return None


def save_csv_from_mt5(symbol, tf_str, bars=100000, mt5_conn=None):
    """Optional: save CSV archive (not required — live + backtest use MT5 directly)."""
    os.makedirs("data", exist_ok=True)
    safe = symbol.replace("@", "").replace("/", "_")
    for tf in ([tf_str, "M1"] if tf_str != "M1" else ["M1"]):
        df = M.fetch_bars(symbol, tf, count=bars, mt5=mt5_conn)
        path = f"data/{safe}_{tf}.csv"
        out = df.reset_index().rename(columns={"time": "datetime"})
        out.to_csv(path, index=False)
        print(f"[CSV] saved {len(out)} {tf} bars -> {path}")


def filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info and info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if info and info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def balance():
    info = mt5.account_info()
    return info.balance if info else 0.0


def my_positions(symbol, magic):
    pos = mt5.positions_get(symbol=symbol) or []
    return [p for p in pos if p.magic == magic]


def my_orders(symbol, magic):
    orders = mt5.orders_get(symbol=symbol) or []
    return [o for o in orders if o.magic == magic]


def place_market(symbol, direction, lot, sl, tag, magic, dry, tp=None,
                 deviation=20, via="ZONE"):
    """Market entry — used when price is within entry_tol of the limit level."""
    info = mt5.symbol_info(symbol)
    if not info:
        return False
    digits = info.digits
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    price = tick.ask if direction == "long" else tick.bid
    otype = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol, "volume": lot, "type": otype,
        "price": price, "sl": round(sl, digits),
        "deviation": deviation, "magic": magic, "comment": f"SMC-{tag}",
        "type_filling": filling_mode(symbol),
    }
    if tp is not None:
        req["tp"] = round(tp, digits)
    if dry:
        tp_s = f"  TP={tp:.{digits}f}" if tp else ""
        print(f"  [DryRun] place {direction.upper()} MARKET ({via}) {tag}  "
              f"lot={lot}  @ ~{price:.{digits}f}  SL={sl:.{digits}f}{tp_s}")
        return True
    res = mt5.order_send(req)
    ok = res and res.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        print(f"  [MT5] market order FAILED ({tag}): "
              f"{res.retcode if res else '?'} {res.comment if res else ''}")
    return ok


def close_position(p, dry, reason="TIME"):
    """Close an open position at market (backtest parity: max-hold time exit)."""
    tick = mt5.symbol_info_tick(p.symbol)
    if not tick:
        return False
    long_ = p.type == mt5.POSITION_TYPE_BUY
    price = tick.bid if long_ else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": p.symbol, "volume": p.volume,
        "type": mt5.ORDER_TYPE_SELL if long_ else mt5.ORDER_TYPE_BUY,
        "position": p.ticket, "price": price,
        "deviation": 20, "magic": p.magic,
        "comment": f"{(p.comment or 'SMC')[:20]}-{reason}",
        "type_filling": filling_mode(p.symbol),
    }
    if dry:
        print(f"  [DryRun] close #{p.ticket} @ ~{price} ({reason})")
        return True
    res = mt5.order_send(req)
    ok = res and res.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        print(f"  [MT5] close FAILED #{p.ticket}: "
              f"{res.retcode if res else '?'} {res.comment if res else ''}")
    return ok


def entry_zone_bounds(direction, proximal, etol):
    """Entry rectangle matching backtest entry_tol band."""
    if direction == "long":
        return float(proximal), float(proximal) + float(etol)
    return float(proximal) - float(etol), float(proximal)


def price_in_entry_zone(direction, zone_lo, zone_hi, bid, ask):
    """True when tick is inside the entry zone (same rule as backtest fill)."""
    if direction == "long":
        return ask <= zone_hi
    return bid >= zone_lo


def entry_tol_hit(direction, proximal, etol, bid, ask):
    """Alias — price entered the entry tolerance band."""
    lo, hi = entry_zone_bounds(direction, proximal, etol)
    return price_in_entry_zone(direction, lo, hi, bid, ask)


def place_limit(symbol, direction, lot, price, sl, tag, magic, expire_dt, dry, tp=None):
    """Place pending limit. Returns order ticket on success, None on failure.
    Dry-run returns a synthetic positive ticket so callers can track state."""
    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 2
    otype = mt5.ORDER_TYPE_BUY_LIMIT if direction == "long" else mt5.ORDER_TYPE_SELL_LIMIT
    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol, "volume": lot, "type": otype,
        "price": round(price, digits), "sl": round(sl, digits),
        "deviation": 10, "magic": magic, "comment": f"SMC-{tag}",
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": int(expire_dt.timestamp()),
        "type_filling": filling_mode(symbol),
    }
    if tp is not None:
        req["tp"] = round(tp, digits)
    if dry:
        tp_s = f"  TP={tp:.{digits}f}" if tp else ""
        print(f"  [DryRun] place {direction.upper()} LIMIT {tag}  lot={lot}  "
              f"@ {price:.{digits}f}  SL={sl:.{digits}f}{tp_s}")
        return 900000000 + (hash((symbol, tag, price)) % 1000000)
    res = mt5.order_send(req)
    ok = res and res.retcode == mt5.TRADE_RETCODE_DONE
    if not ok:
        print(f"  [MT5] pending order FAILED ({tag}): "
              f"{res.retcode if res else '?'} {res.comment if res else ''}")
        return None
    return int(res.order) if res.order else True


def cancel_order(ticket, dry):
    if dry:
        print(f"  [DryRun] cancel pending #{ticket} (setup invalidated)")
        return
    mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})


def modify_sl(pos, new_sl, dry):
    info = mt5.symbol_info(pos.symbol)
    digits = info.digits if info else 2
    new_sl = round(new_sl, digits)
    if abs(new_sl - pos.sl) < 10 ** (-digits):
        return False
    if dry:
        print(f"  [DryRun] trail #{pos.ticket}: SL {pos.sl:.{digits}f} -> {new_sl:.{digits}f}")
        return True
    res = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol,
                          "position": pos.ticket, "sl": new_sl, "tp": pos.tp,
                          "magic": pos.magic})
    return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)


# ────────────────────────────── main loop ─────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD@")
    ap.add_argument("--tf", default="M15", choices=["M5", "M15"])
    ap.add_argument("--magic", type=int, default=778899)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--optimized", action="store_true",
                    help="STABLE preset: OB+DEM, fib, ATR regime, HTF blend TP")
    ap.add_argument("--aggressive", action="store_true",
                    help="Legacy aggressive preset (old --optimized behavior)")
    ap.add_argument("--high-wr", action="store_true",
                    help="With --optimized: OB+DEMAND only (~54%% WR, fewer trades)")
    ap.add_argument("--more-trades", action="store_true",
                    help="With --optimized: more signals — FVG+OB+DEM, H1 trend, session 8-22")
    ap.add_argument("--quality", action="store_true",
                    help="With --optimized: FVG confluence + BOS confirm (~48-58%% WR)")
    ap.add_argument("--fib", action="store_true",
                    help="With --optimized: fib 38-78%% retrace zone filter")
    ap.add_argument("--nds", action="store_true",
                    help="With --optimized: add NDS rule")
    ap.add_argument("--balanced", action="store_true",
                    help="STABLE + 2 concurrent positions")
    ap.add_argument("--balanced-plus", action="store_true",
                    help="BALANCED + NDS-FVG + session 8-22")
    ap.add_argument("--dense-plus", action="store_true",
                    help="DENSE + no ATR regime (~18 trades, ~78%% WR)")
    ap.add_argument("--dense-wide", action="store_true",
                    help="DENSE + no ATR + session 8-22 (~24 trades, ~71%% WR)")
    ap.add_argument("--dense", action="store_true",
                    help="DENSE: OB+NDS+DEM+HARM_BAT+MA_X_S+SQZ_S+ICT_SB+ICT_MIT (~37 trades)")
    ap.add_argument("--spike", action="store_true",
                    help="Add pre-spike rules SPIKE-BRK + SPIKE-SQS")
    ap.add_argument("--asset", default=None, choices=P.ASSET_KEYS,
                    help="Calibrated profile for Gold/BTC/Brent/EURUSD")
    ap.add_argument("--medium", action="store_true",
                    help="DENSE + session 8-22 (~23 trades, ~74%% WR)")
    ap.add_argument("--nds-mode", action="store_true",
                    help="STABLE OB+NDS only (nested demand/supply inside H4/H1)")
    ap.add_argument("--update-csv", action="store_true",
                    help="also save fresh MT5 data to data/ (for backtest.py)")
    ap.add_argument("--risk", type=float, default=S.RISK_PCT,
                    help="risk per trade as balance fraction (default 0.01 = 1%%)")
    args = ap.parse_args()
    sym, tf, magic, dry = args.symbol, args.tf, args.magic, args.dry_run
    profile = None
    if args.asset:
        _, profile = P.get_profile(args.asset)
    risk_pct = args.risk
    if profile:
        risk_pct = profile.get("risk_pct", risk_pct)
    if risk_pct <= 0:
        print("[LIVE] --risk must be > 0 (e.g. 0.01 for 1%%)"); sys.exit(1)
    tf_min = TF_MIN[tf]
    if args.optimized:
        if args.asset:
            opt = P.resolve_live_opt(
                profile, spike=args.spike,
                cli_dense=args.dense, cli_dense_wide=args.dense_wide)
            opt = P.apply_profile_to_settings(opt, profile)
        else:
            opt = S.optimized_settings(
                high_wr=args.high_wr, more_trades=args.more_trades,
                quality=args.quality, fib=args.fib,
                nds_mode=args.nds_mode, no_nds=getattr(args, "no_nds", False),
                nds_extended=getattr(args, "nds_extended", False),
                balanced=args.balanced, balanced_plus=args.balanced_plus,
                dense=args.dense, medium=args.medium,
                dense_plus=args.dense_plus, dense_wide=args.dense_wide,
                spike=args.spike)
    else:
        opt = None
    det_params = P.merge_params(S.DEFAULT_PARAMS, profile) if profile else S.DEFAULT_PARAMS
    spike_params = opt.get("spike_params") if opt else None
    rule_names = opt["enabled"] if opt else list(S.DETECTORS.keys())
    max_pos = opt["max_concurrent"] if opt else S.MAX_CONCURRENT
    use_be_only = bool(opt)

    if not mt5.initialize():
        print(f"[MT5] connect failed: {mt5.last_error()}"); sys.exit(1)
    try:
        if args.asset:
            sym = P.resolve_symbol_for_profile(args.asset, mt5, preferred=sym)
        else:
            sym = M.resolve_symbol(sym)
    except RuntimeError as e:
        print(f"[MT5] {e}"); mt5.shutdown(); sys.exit(1)
    print(f"[MT5] connected to {mt5.terminal_info().company}")
    print(f"[LIVE] data source: MetaTrader 5 LIVE (no CSV needed)")
    if args.update_csv:
        print(f"[LIVE] optional CSV archive...")
        save_csv_from_mt5(sym, tf, mt5_conn=mt5)
    probe = get_bars(sym, tf, LIVE_BARS, mt5_conn=mt5)
    if probe is None or len(probe) < 60:
        print(f"[MT5] not enough {tf} bars for {sym}. Is the market open?")
        mt5.shutdown(); sys.exit(1)
    print(f"[LIVE] loaded {len(probe)} {tf} bars | last closed: {probe.index[-2]}")
    print(f"[LIVE] symbol={sym} tf={tf} risk={risk_pct*100:g}% dry_run={dry}")
    if opt:
        tag = "STABLE" if opt.get("require_atr_regime") else "OPTIMIZED"
        print(f"[LIVE] {tag} | rules: {', '.join(rule_names)}")
        print(f"[LIVE] exit: BE @ +{opt['be_trigger']}R + TP {opt['tp_mode']} ({opt['fixed_tp_r']}R floor)")
        print(f"[LIVE] filters: session {opt['session_start']}-{opt['session_end']} | min SL ${opt['min_risk_usd']}"
              + (f" | asset profile {args.asset}" if args.asset else "")
              + (" | HTF trend" if opt.get("htf_trend") else "")
              + (" | fib pullback" if opt.get("require_fib") and not opt.get("fib_nds_exempt") else "")
              + (" | fib (non-NDS only)" if opt.get("fib_nds_exempt") else "")
              + (" | fib (OB only)" if opt.get("fib_ob_only") else "")
              + (" | DENSE+ no ATR" if opt.get("dense_plus") and not opt.get("dense_wide") else "")
              + (" | DENSE-WIDE" if opt.get("dense_wide") else "")
              + (" | MEDIUM sess8-22" if opt.get("medium") else "")
              + (" | ATR regime (skip chaos)" if opt.get("require_atr_regime") else "")
              + (" | quality: FVG conf + BOS" if opt.get("require_confluence") else "")
              + (" | NDS nested zones" if opt.get("nds_mode") else ""))
        print(f"[LIVE] VP={S.VP_MODE} | max {max_pos} position(s)")
    else:
        print(f"[LIVE] active rules: {', '.join(rule_names)}")
        print(f"[LIVE] filters: VP={S.VP_MODE} | trail +{S.TRAIL_TRIGGER}R BE then {S.TRAIL_DIST}R")
        print(f"[LIVE] max {max_pos} positions")
    print(f"[LIVE] Ctrl+C to stop\n")

    last_bar = None
    pos_state = {}            # ticket -> {entry, risk, dir, tag, peak, activated}
    known_tickets = set()

    while True:
        try:
            df = get_bars(sym, tf, 600)
            if df is None or len(df) < 60:
                time.sleep(20); continue
            closed_time = df.index[-2]                  # last FULLY closed candle

            # ---- manage trailing on open positions (every cycle) ----
            positions = my_positions(sym, magic)
            for p in positions:
                stt = pos_state.get(p.ticket)
                if stt is None:                          # discovered after restart
                    risk = abs(p.price_open - p.sl) if p.sl else df["close"].iloc[-2] * 0.001
                    stt = {"entry": p.price_open, "risk": max(risk, 1e-6),
                           "dir": "long" if p.type == mt5.POSITION_TYPE_BUY else "short",
                           "tag": (p.comment or "SMC-?").replace("SMC-", ""),
                           "peak": p.price_open, "activated": False}
                    pos_state[p.ticket] = stt
                hi = float(df["high"].iloc[-2]); lo = float(df["low"].iloc[-2])
                if stt["dir"] == "long":
                    stt["peak"] = max(stt["peak"], hi)
                    fav = (stt["peak"] - stt["entry"]) / stt["risk"]
                    if not stt["activated"] and fav >= (opt["be_trigger"] if opt else S.TRAIL_TRIGGER):
                        stt["activated"] = True
                        modify_sl(p, max(p.sl, stt["entry"]), dry)
                    if stt["activated"] and not use_be_only:
                        modify_sl(p, max(p.sl, stt["peak"] - S.TRAIL_DIST * stt["risk"]), dry)
                else:
                    stt["peak"] = min(stt["peak"], lo)
                    fav = (stt["entry"] - stt["peak"]) / stt["risk"]
                    if not stt["activated"] and fav >= (opt["be_trigger"] if opt else S.TRAIL_TRIGGER):
                        stt["activated"] = True
                        modify_sl(p, min(p.sl, stt["entry"]), dry)
                    if stt["activated"] and not use_be_only:
                        modify_sl(p, min(p.sl, stt["peak"] + S.TRAIL_DIST * stt["risk"]), dry)

            # ---- report newly-filled positions (rule + explanation) ----
            for p in positions:
                if p.ticket not in known_tickets:
                    known_tickets.add(p.ticket)
                    tag = (p.comment or "SMC-?").replace("SMC-", "")
                    side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
                    print("\n" + "=" * 64)
                    print(f"  TRADE OPENED  |  {side}  {sym}  lot={p.volume}")
                    print(f"  Rule: {tag}   entry={p.price_open}  stop={p.sl}  tp={p.tp}")
                    print(f"  Why : {S.RULE_EXPLANATIONS.get(tag, 'price-action setup')}")
                    print("=" * 64 + "\n")

            # ---- only act once per newly-closed candle ----
            if last_bar is not None and closed_time <= last_bar:
                time.sleep(15); continue
            last_bar = closed_time

            # ---- invalidate pending orders whose zone the candle closed beyond ----
            for o in my_orders(sym, magic):
                tag = (o.comment or "").replace("SMC-", "")
                close = float(df["close"].iloc[-2])
                is_buy = o.type == mt5.ORDER_TYPE_BUY_LIMIT
                # distal stored implicitly via sl side; invalidate if price ran away past entry
                if is_buy and close < o.sl:
                    cancel_order(o.ticket, dry)
                elif (not is_buy) and close > o.sl:
                    cancel_order(o.ticket, dry)

            # ---- look for new setups on the closed candle ----
            B = S.Bars(df)
            i = B.n - 2                                  # the closed candle index
            if opt and opt.get("require_atr_regime") and not opt.get("spike_mode") and not S.atr_regime_pass(
                    B, i, lookback=opt.get("atr_lookback", S.OPT_STABLE_ATR_LB),
                    max_ratio=opt.get("atr_max_ratio", S.OPT_STABLE_ATR_RATIO)):
                time.sleep(10); continue
            n_open = len(my_positions(sym, magic)) + len(my_orders(sym, magic))
            if n_open >= max_pos:
                time.sleep(10); continue

            if opt and not S.session_pass(closed_time, opt["session_start"], opt["session_end"]):
                time.sleep(10); continue

            bal = balance()
            tick = mt5.symbol_info_tick(sym)
            cur_price = tick.ask if tick else float(df["close"].iloc[-1])
            htf_ctx = None
            if opt:
                htf_dfs = {}
                for htf in opt["htf_tfs"]:
                    hdf = get_bars(sym, htf, 400, mt5_conn=mt5)
                    if hdf is not None:
                        htf_dfs[htf] = hdf
                htf_ctx = S.prepare_htf_context(htf_dfs)
                for name in rule_names:
                    placed = False
                    if (name not in S.DETECTORS and name not in S.NDS_FAMILY
                            and name not in S.AB_DETECTORS and name not in S.WYCK_DETECTORS
                            and name not in S.STYLE_DETECTORS and name not in S.EXT_DETECTORS
                            and name not in S.SPIKE_DETECTORS):
                        continue
                    for (direction, proximal, distal, tag) in S.iter_rule_setups(
                            name, B, i, det_params, htf_context=htf_ctx,
                            signal_time=B.t[i],
                            htf_tfs=tuple(opt.get("htf_tfs", ("H4", "H1"))) if opt else ("H4", "H1"),
                            nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO) if opt else S.OPT_NDS_MAX_RATIO,
                            spike_params=spike_params):
                        if not S.vp_pass(B, i, direction, proximal,
                                         opt.get("vp_mode", S.VP_MODE) if opt else S.VP_MODE,
                                         window=opt.get("vp_window", 480) if opt else 480,
                                         vp_tol_atr=opt.get("vp_tol_atr", 0.0) if opt else 0.0):
                            continue
                        if opt and opt.get("require_confluence") and not S.setup_confluence_pass(
                                B, i, direction, tag, rule_names):
                            continue
                        if opt and opt.get("require_bos") and not S.bos_confirm_pass(B, i, direction):
                            continue
                        if direction == "long" and proximal >= cur_price:
                            continue
                        if direction == "short" and proximal <= cur_price:
                            continue
                        sl, _entry, risk = S.compute_entry_sl_risk(
                            B, i, direction, proximal, distal)
                        if risk <= 0:
                            continue
                        if opt and not S.fib_filter_pass(
                                B, i, direction, _entry, name, proximal, distal,
                                htf_ctx, B.t[i],
                                tuple(opt.get("htf_tfs", ("H4", "H1"))),
                                opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO),
                                opt.get("require_fib", False),
                                opt.get("fib_nds_exempt", False),
                                opt.get("fib_ob_only", False),
                                opt.get("fib_spike_exempt", False),
                                opt.get("fib_demand_exempt", False)):
                            continue
                        if opt and not S.entry_filters_pass(
                                B, i, risk, min_risk_usd=opt["min_risk_usd"],
                                session_start=opt["session_start"],
                                session_end=opt["session_end"],
                                direction=direction, signal_time=B.t[i],
                                htf_context=htf_ctx if opt.get("htf_trend") else None,
                                htf_trend=opt.get("htf_trend", False),
                                htf_ema=opt.get("htf_ema", 20),
                                htf_tfs=tuple(opt.get("htf_tfs", ("H1",)))):
                            continue
                        trade_tag = S.resolve_trade_tag(
                            name, tag, B, i, direction, proximal, distal,
                            htf_ctx, B.t[i],
                            htf_tfs=tuple(opt.get("htf_tfs", ("H4", "H1"))) if opt else ("H4", "H1"),
                            nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO) if opt else S.OPT_NDS_MAX_RATIO,
                            nds_enabled=any(S.is_nds_rule(n) for n in rule_names))
                        tp = None
                        tp_src = ""
                        if opt:
                            tp, tp_src = S.resolve_tp(
                                B, i, direction, trade_tag, proximal, distal, proximal, risk,
                                tp_mode=opt["tp_mode"], fixed_tp_r=opt["fixed_tp_r"],
                                htf_context=htf_ctx, min_tp_r=opt["min_tp_r"],
                                max_tp_r=opt["max_tp_r"], htf_lookback=opt["htf_lookback"],
                                htf_tfs=tuple(opt["htf_tfs"]), signal_time=B.t[i],
                            )
                        lot = X.calc_lot(sym, bal, risk_pct, risk, mt5)
                        expire = dt.datetime.now() + dt.timedelta(minutes=S.WAIT_BARS * tf_min)
                        tp_s = f" TP {tp:.2f}" if tp else ""
                        if tp and tp_src:
                            tp_s += f" [{tp_src}]"
                        print(f"[{closed_time}] signal {trade_tag} {direction.upper()} @ {proximal:.2f} SL {sl:.2f}{tp_s}")
                        if place_limit(sym, direction, lot, proximal, sl, trade_tag, magic, expire, dry, tp=tp):
                            placed = True
                            break
                    if placed:
                        break

            time.sleep(10)

        except KeyboardInterrupt:
            print("\n[LIVE] stopped by user."); break
        except Exception as e:
            print(f"[LIVE] error: {e}")
            time.sleep(20)

    mt5.shutdown()
    print("[MT5] disconnected.")


if __name__ == "__main__":
    main()
