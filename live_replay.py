"""
Live-engine replay on historical bars — same `process_once` as live_portfolio.

  bar/M1 clock → LP.process_once (invalidate/register/arm/confirm/check fills)
  ReplayBrokerIO implements place/cancel/positions; LIMIT fills via broker-sim
  hook then sync_limit_fills (identical live path).

Usage:
  python live_replay.py --days 14 --tf M15
  python live_replay.py --days 7 --tf M5 --fill live
  python live_replay.py --days 7 --tf M5 --fill market
  python live_replay.py --days 3 --dual
  python live_replay.py --dual --days 60 --balance 1000 --risk 0.02 --flat-risk
  python live_replay.py --dual --days 60 --balance 1000 --flat 0.02
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import sys

sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd

import MetaTrader5 as mt5
import strategy as S
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C
import floating_config as FC
import mt5_data as M
import live_portfolio as LP
import live_flight_recorder as FR
from live_smc import TF_MIN


# Don't start trading until a full live-sized signal window exists behind the
# cursor, otherwise the first replayed bars run on a shorter history than live.
WARMUP_BARS = LP.SIG_BARS

_TF_ALIASES = {
    "M5": "M5", "5M": "M5", "5": "M5",
    "M15": "M15", "15M": "M15", "15": "M15",
}

_FILL_ALIASES = {
    "live": "live",
    "limit": "live",
    "proximal": "live",
    "market": "market",
}


def normalize_tf(raw: str) -> str:
    key = str(raw).strip().upper().replace(" ", "")
    tf = _TF_ALIASES.get(key)
    if tf is None:
        raise SystemExit(
            f"Unsupported --tf {raw!r}. Use M5/5M or M15/15M.")
    return tf


def normalize_fill(raw: str) -> str:
    key = str(raw).strip().lower()
    mode = _FILL_ALIASES.get(key)
    if mode is None:
        raise SystemExit(
            f"Unsupported --fill {raw!r}. Use live|market (proximal=live).")
    return mode


def _df_asof(full: pd.DataFrame, end_i: int,
             count: int | None = None) -> pd.DataFrame:
    """Bars up to end_i as 'closed', plus a synthetic forming bar so live
    code's iloc[-2] is the just-closed candle (same as MT5 feed).

    `count` is the caller's requested window size and must be honoured: MT5
    returns exactly `count` rows with the last one still forming, so replay
    has to serve count-1 closed bars. Handing over an expanding window instead
    feeds full-array indicators (volume profile, prior-day levels, swing
    pivots, regime map) a different history than live ever sees.
    """
    lo = 0
    if count and int(count) > 1:
        lo = max(0, end_i + 2 - int(count))
    closed = full.iloc[lo: end_i + 1]
    last = closed.iloc[-1]
    forming = closed.iloc[[-1]].copy()
    step = (closed.index[-1] - closed.index[-2]) if len(closed) > 1 else pd.Timedelta(minutes=15)
    forming.index = [closed.index[-1] + step]
    for col in ("open", "high", "low", "close"):
        if col in forming.columns:
            forming.iloc[0, forming.columns.get_loc(col)] = float(last["close"])
    return pd.concat([closed, forming])


def _spread_half(spread: float) -> float:
    return max(float(spread), 0.0) * 0.5


def _tick_from_px(px: float, spread: float):
    h = _spread_half(spread)
    return type("T", (), {"bid": px - h, "ask": px + h})()


def _profit_factor(trades: list[dict]) -> float:
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = sum(-t["net"] for t in trades if t["net"] < 0)
    if gl <= 0:
        return 999.0 if gp > 0 else 0.0
    return gp / gl


class LiveReplay:
    """Bar+M1 driver that calls LP.process_once — one engine with live."""

    def __init__(self, asset: str, tf: str, days: int, balance: float,
                 fill_mode: str = "live", mirror_vps: bool = False,
                 opt_overrides: dict | None = None, record: bool = False,
                 risk_pct: float | None = None, flat_risk: bool = False,
                 fixed_lot: float | None = None, no_cap: bool = False):
        self.asset = asset
        self.tf = tf
        self.tf_min = TF_MIN[tf]
        self.days = days
        self.start_balance = balance
        self.balance = balance
        self.fillspread = 0.0
        self.mirror_vps = bool(mirror_vps)
        self.record = bool(record)
        # risk_pct=None → profile / RISK_MAP default. flat_risk sizes every
        # trade from start_balance so the equity curve is not compounded.
        # fixed_lot ( --flat LOT ) wins over both: every fill uses that volume.
        self.risk_pct_override = (float(risk_pct) if risk_pct is not None
                                  else None)
        self.flat_risk = bool(flat_risk)
        self.fixed_lot = (float(fixed_lot) if fixed_lot is not None else None)
        self.no_cap = bool(no_cap)
        self._orig_calc_lot = None
        if self.mirror_vps:
            fill_mode = "market"
        self.fill_mode = normalize_fill(fill_mode)
        self.opt_overrides = dict(opt_overrides or {})
        self.trades: list[dict] = []
        self.ideas: list[dict] = []
        self.open_pos: list[dict] = []
        self._asof_i = WARMUP_BARS
        self._broker_orders: dict[int, dict] = {}
        self._next_ticket = 900000001
        self._next_pos_ticket = 800000001
        self._tick = None
        self._now = None
        self._sig_cache = None
        self._sig_cache_key = None
        self._m1_lo = None
        self._m1_hi = None
        self._orig_symbol_info_tick = None
        self._orig_lp: dict | None = None
        self._patched = False
        self._prev_pos_tickets: dict | None = None
        self.peak_equity = balance
        self.max_dd = 0.0
        self.max_dd_usd = 0.0
        self.max_dd_time = None
        self.min_equity = balance
        self.min_equity_time = None
        self.peak_balance = balance
        self.max_bal_dd = 0.0
        self.max_bal_dd_usd = 0.0
        self.max_open = 0
        self.max_open_risk_pct = 0.0

    def load(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 connect failed: {mt5.last_error()}")
        FC.apply_tf_paths(self.tf)
        FR.set_tf(self.tf)
        FR.set_enabled(self.record)
        if self.record:
            print(f"  Flight record → {FR.recorder_path(self.tf)}")
        _, prof = P.get_profile(self.asset)
        self.prof = prof
        self.sym = P.resolve_symbol_for_profile(self.asset, mt5)
        self.spread = P.live_spread(self.sym, prof, mt5)
        self.fillspread = self.spread
        opt = P.resolve_live_opt(prof)
        opt = P.apply_profile_to_settings(opt, prof, self.spread)
        opt = dict(opt)
        if self.fill_mode == "market":
            opt["touch_use_limit"] = False
        opt["max_concurrent"] = C.max_concurrent(self.asset)
        if self.mirror_vps:
            opt["touch_use_limit"] = False
            opt["max_concurrent"] = max(int(opt.get("max_concurrent") or 4), 8)
            opt["max_same_dir"] = int(opt["max_concurrent"])
            opt["min_fill_rr"] = 0.0
            opt["meta_gate"] = False
            opt["entry_confirm_mode"] = "none"
            opt["require_confluence"] = False
            opt["require_bos"] = False
            opt["htf_trend"] = False
            opt["vp_mode"] = None
        if self.opt_overrides:
            opt.update(self.opt_overrides)
        risk = C.RISK_MAP.get(self.asset, prof.get("risk_pct", S.RISK_PCT))
        if self.risk_pct_override is not None:
            risk = self.risk_pct_override
        magic = C.magic_for(self.asset, self.tf)
        self.st = LP.AssetRunner(
            self.asset, self.sym, prof, opt, magic, risk, tf_tag=self.tf)
        if self.mirror_vps or not opt.get("meta_gate"):
            self.st.meta_gate = None
        if self.mirror_vps:
            self.st.max_pos = int(opt.get("max_concurrent") or 8)
            opt["regime_gate"] = False

        # Preload the test window plus a full live-sized signal window, and let
        # M1 reach just as far back so fetch_pair's overlap clip cannot eat it.
        pad = LP.SIG_BARS + 60
        span_bars = int(self.days * 1440 / self.tf_min) + 1
        _, self.sig, self.m1, self.cutoff = M.fetch_pair(
            self.sym, self.tf, mt5=mt5, days=self.days,
            signal_count=min(99999, span_bars + pad),
            m1_count=min(99999, int(self.days * 1440) + pad * self.tf_min))
        htf_tfs = list(opt.get("htf_tfs", ("H4", "H1")))
        if opt.get("regime_gate"):
            rtf = opt.get("regime_tf", "H1")
            if rtf not in htf_tfs:
                htf_tfs.append(rtf)
        # +LP.HTF_BARS so even the first replayed bar can serve the same 400
        # HTF candles live gets; otherwise the regime map runs on a short series.
        _, htf_dfs = M.fetch_htf_bars(
            self.sym, days=self.days, mt5=mt5, tfs=tuple(htf_tfs),
            warmup=LP.HTF_BARS + 60)
        self.htf_dfs = htf_dfs
        # MT5 position 0 is the candle still being built. Live only ever acts
        # on index[-2], so replay must not treat that unfinished bar as closed
        # — its high/low keep moving and two runs minutes apart disagree.
        if len(self.sig) > 1:
            self.sig = self.sig.iloc[:-1]
        self.ctx = S.M1Ctx(self.m1, self.sig.index)
        self._cache_arrays()

        n = int(LP.CALIB_DAYS * 1440 / M.tf_minutes(self.tf)) + 60
        try:
            dc = M.fetch_bars(self.sym, self.tf, count=n, mt5=mt5)
            self.st.min_sl = X.calibrate_min_sl(S.Bars(dc), S.Bars(dc).t[0], prof)
        except Exception:
            self.st.min_sl = X.calibrate_min_sl(
                S.Bars(self.sig), self.sig.index[0], prof)
        self.st.min_sl_date = dt.date.today()
        if opt.get("meta_gate") and self.st.meta_gate is None:
            self.st.meta_gate = FC.load_meta_gate(assets=[self.asset])
        if self.mirror_vps:
            self.st.meta_gate = None

        FR.log_session(
            asset=self.st.key, tf=self.tf, mode="replay",
            fp=LP.engine_fingerprint(self.st), dry=True,
            sig_bars=LP.SIG_BARS, htf_bars=LP.HTF_BARS,
            risk=float(self.st.risk_pct), rules=",".join(self.st.rule_names))

        self._install_hooks()

    def _install_hooks(self):
        if self._patched:
            return
        self._orig_lp = {
            "get_bars": LP.get_bars,
            "my_positions": LP.my_positions,
            "my_orders": LP.my_orders,
            "place_limit": LP.place_limit,
            "place_market": LP.place_market,
            "cancel_order": LP.cancel_order,
            "close_position": LP.close_position,
            "modify_sl": LP.modify_sl,
        }
        LP.get_bars = self._get_bars
        LP.my_positions = self._fake_positions
        LP.my_orders = self._fake_orders
        LP.place_limit = self._place_limit
        LP.place_market = self._place_market
        LP.cancel_order = self._cancel_order
        LP.close_position = self._close_position
        LP.modify_sl = self._modify_sl
        # LP.arm_touch_limit / _execute_zone_fill call X.calc_lot — patch it
        # so --flat LOT reaches every fill path without forking the live engine.
        self._orig_calc_lot = X.calc_lot
        X.calc_lot = self._calc_lot
        LP.ALLOW_DRY_LIMIT_FILL = True
        LP.set_broker_sim_step(self._broker_sim_step)
        self._orig_symbol_info_tick = mt5.symbol_info_tick
        mt5.symbol_info_tick = self._symbol_info_tick
        self._patched = True

    def _uninstall_hooks(self):
        if not self._patched:
            return
        if self._orig_symbol_info_tick is not None:
            mt5.symbol_info_tick = self._orig_symbol_info_tick
        if self._orig_lp:
            for name, fn in self._orig_lp.items():
                setattr(LP, name, fn)
        if self._orig_calc_lot is not None:
            X.calc_lot = self._orig_calc_lot
            self._orig_calc_lot = None
        LP.set_broker_sim_step(None)
        LP.ALLOW_DRY_LIMIT_FILL = False
        LP.set_clock(None)
        self._patched = False

    def _clamp_lot(self, symbol: str, lot: float) -> float:
        """Round a requested lot to the broker's step / min / max."""
        try:
            spec = X.get_spec(symbol, mt5)
            if not spec:
                return float(lot)
            step = float(spec["vol_step"] or 0.01)
            lo = float(spec["vol_min"] or 0.01)
            hi = float(spec["vol_max"] or 100.0)
            lot = max(lo, round(float(lot) / step) * step)
            return min(lot, hi)
        except Exception:
            return float(lot)

    def _calc_lot(self, symbol, balance, risk_fraction, sl_distance, mt5_mod=None):
        if self.fixed_lot is not None:
            return self._clamp_lot(symbol, self.fixed_lot)
        fn = self._orig_calc_lot or X.calc_lot
        return fn(symbol, balance, risk_fraction, sl_distance, mt5_mod)

    def _symbol_info_tick(self, symbol=None):
        if self._tick is not None:
            return self._tick
        if self._orig_symbol_info_tick is not None:
            return self._orig_symbol_info_tick(symbol or self.sym)
        return None

    def _set_tick_from_mid(self, mid: float):
        self._tick = _tick_from_px(mid, self.spread)

    def _fake_positions(self, sym, magic):
        out = []
        for p in self.open_pos:
            out.append(type("P", (), {
                "ticket": int(p["ticket"]),
                "type": (mt5.POSITION_TYPE_BUY if p["dir"] == "long"
                         else mt5.POSITION_TYPE_SELL),
                "symbol": sym,
                "magic": magic,
                "volume": float(p.get("lot") or 0.01),
                "price_open": p["entry"],
                "sl": p["sl"],
                "tp": p.get("tp") or 0.0,
                "comment": f"SMC-{p.get('tag', '')}",
                "time": int(pd.Timestamp(p["entry_time"]).timestamp()),
            })())
        return out

    def _fake_orders(self, sym, magic):
        out = []
        for ticket, o in self._broker_orders.items():
            out.append(type("O", (), {
                "ticket": ticket,
                "symbol": sym,
                "magic": magic,
                "price_open": o["price"],
                "type": (mt5.ORDER_TYPE_BUY_LIMIT if o["direction"] == "long"
                         else mt5.ORDER_TYPE_SELL_LIMIT),
                "comment": f"SMC-{o.get('tag', '')}",
            })())
        return out

    def _place_limit(self, symbol, direction, lot, price, sl, tag, magic,
                     expire_dt, dry, tp=None):
        ticket = self._next_ticket
        self._next_ticket += 1
        self._broker_orders[ticket] = {
            "direction": direction,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp) if tp is not None else None,
            "tag": tag,
            "lot": lot,
        }
        return ticket

    def _cancel_order(self, ticket, dry):
        self._broker_orders.pop(int(ticket), None)

    def _place_market(self, symbol, direction, lot, sl, tag, magic, dry,
                      tp=None, deviation=20, via="ZONE"):
        tick = self._symbol_info_tick(symbol)
        if not tick:
            return False
        entry = float(tick.ask if direction == "long" else tick.bid)
        long_ = direction == "long"
        risk_act = (entry - float(sl)) if long_ else (float(sl) - entry)
        if risk_act <= 0:
            return False
        clean_tag = str(tag).split("@", 1)[0]
        ticket = self._next_pos_ticket
        self._next_pos_ticket += 1
        fill_ts = LP.clock_now()
        self.open_pos.append({
            "ticket": ticket,
            "tag": clean_tag,
            "rule": clean_tag,
            "dir": direction,
            "entry": entry,
            "sl": float(sl),
            "tp": float(tp) if tp is not None else None,
            "risk": risk_act,
            "risk_usd": self._risk_usd(float(lot or 0.01), risk_act),
            "lot": float(lot or 0.01),
            "via": str(via or "MARKET"),
            "entry_time": fill_ts,
            "entry_i": self._asof_i,
        })
        self._update_equity(fill_ts, tick.bid, tick.ask)
        return True

    def _modify_sl(self, pos, new_sl, dry):
        for p in self.open_pos:
            if int(p["ticket"]) == int(pos.ticket):
                if abs(float(new_sl) - float(p["sl"])) < 1e-9:
                    return False
                p["sl"] = float(new_sl)
                return True
        return False

    def _close_position(self, pos, dry, reason="TIME"):
        tick = self._symbol_info_tick(pos.symbol)
        if not tick:
            return False
        long_ = pos.type == mt5.POSITION_TYPE_BUY
        px = float(tick.bid if long_ else tick.ask)
        kept = []
        closed = None
        for p in self.open_pos:
            if int(p["ticket"]) != int(pos.ticket):
                kept.append(p)
                continue
            closed = p
        self.open_pos = kept
        if closed is None:
            return False
        risk = max(closed["risk"], 1e-9)
        if closed["dir"] == "long":
            R = (px - closed["entry"]) / risk
        else:
            R = (closed["entry"] - px) / risk
        net = R * closed["risk_usd"] - (closed["risk_usd"] / risk) * self.fillspread
        exit_t = LP.clock_now()
        row = {
            "asset": self.asset,
            "entry_time": closed["entry_time"],
            "exit_time": exit_t,
            "rule": closed.get("tag") or closed.get("rule"),
            "dir": closed["dir"],
            "entry": closed["entry"],
            "sl": closed["sl"],
            "tp": closed.get("tp"),
            "R": R,
            "risk_usd": closed["risk_usd"],
            "net": net,
            "exit": reason,
            "via": closed.get("via", "?"),
            "balance": self.balance + net,
            "tf": self.tf,
        }
        self.trades.append(row)
        self.balance += net
        self._update_balance_dd()
        if self.record:
            FR.log_close(
                self.st, rule=row["rule"], side=row["dir"],
                entry=row["entry"], exit_px=px, R=R, pnl=net,
                reason=reason, ticket=closed["ticket"], ts=exit_t)
        return True

    def _cache_arrays(self):
        """Plain numpy views of the bar data the hot loop reads every step.

        Boolean masks and .iloc lookups against the full frames cost more than
        the engine step itself once you are polling once per M1 candle.
        """
        self._m1_idx = self.m1.index
        self._m1_open = self.m1["open"].to_numpy(dtype=float)
        self._m1_high = self.m1["high"].to_numpy(dtype=float)
        self._m1_low = self.m1["low"].to_numpy(dtype=float)
        self._m1_close = self.m1["close"].to_numpy(dtype=float)
        self._m1_vol = (self.m1["volume"].to_numpy(dtype=float)
                        if "volume" in self.m1.columns
                        else np.zeros(len(self.m1)))
        self._sig_high = self.sig["high"].to_numpy(dtype=float)
        self._sig_low = self.sig["low"].to_numpy(dtype=float)
        self._sig_close = self.sig["close"].to_numpy(dtype=float)

    def _partial_htf_bar(self, open_ts, now, template):
        """The HTF candle as far as it has printed at `now`.

        MT5 hands live the bar that is still being built, so replay must not
        hand the engine the same candle already finished — that is future data.
        """
        a = int(self._m1_idx.searchsorted(open_ts, side="left"))
        b = int(self._m1_idx.searchsorted(pd.Timestamp(now), side="left"))
        bar = template.iloc[[0]].copy()
        bar.index = [open_ts]
        if b <= a:
            px = float(self._sig_close[self._asof_i])
            vals = {"open": px, "high": px, "low": px, "close": px, "volume": 0.0}
        else:
            vals = {
                "open": float(self._m1_open[a]),
                "high": float(self._m1_high[a:b].max()),
                "low": float(self._m1_low[a:b].min()),
                "close": float(self._m1_close[b - 1]),
                "volume": float(self._m1_vol[a:b].sum()),
            }
        for col, v in vals.items():
            if col in bar.columns:
                bar.iloc[0, bar.columns.get_loc(col)] = v
        return bar

    def _get_bars(self, sym, tf, count=600):
        # The engine polls once per M1 tick but the bar window only moves on a
        # TF close, so the same frame gets rebuilt ~15x per candle. Nothing in
        # live_portfolio writes to the frame, so handing back the cached object
        # is the same data, not a copy of it.
        if tf == self.tf:
            key = (self._asof_i, int(count or 0))
            if self._sig_cache_key != key:
                self._sig_cache = _df_asof(self.sig, self._asof_i, count)
                self._sig_cache_key = key
            return self._sig_cache
        hdf = self.htf_dfs.get(tf)
        if hdf is None:
            return None
        # Cut at the decision instant, not the signal bar's OPEN stamp: live is
        # polling after the candle closed and already sees the HTF bar that
        # opened at that moment. Cutting at the open time left replay a whole
        # H4/H1 candle behind live on every boundary.
        now = pd.Timestamp(self._now) if self._now is not None else (
            self.sig.index[self._asof_i] + pd.Timedelta(minutes=self.tf_min))
        cut = hdf[hdf.index <= now]
        if len(cut) < 30:
            return None
        step = pd.Timedelta(minutes=M.tf_minutes(tf))
        if cut.index[-1] + step > now:
            open_ts = cut.index[-1]
            cut = pd.concat([cut.iloc[:-1],
                             self._partial_htf_bar(open_ts, now, hdf)])
        return cut.iloc[-min(len(cut), count):]

    def _size_balance(self) -> float:
        """Balance used for lot sizing (and intended risk_usd fallback)."""
        return self.start_balance if self.flat_risk else self.balance

    def _risk_usd(self, lot: float, sl_distance: float) -> float:
        """Money actually at risk for `lot`, not the intended risk_pct.

        Live rounds every lot to vol_step and floors it at vol_min, so a small
        account routinely risks more (or less) than risk_pct. Pricing the trade
        off the intended risk hides that gap from the replay P&L.
        """
        fallback = self.st.risk_pct * self._size_balance()
        try:
            spec = X.get_spec(self.sym, mt5)
            lpl = X.loss_per_lot_usd(spec, sl_distance, mt5=mt5, symbol=self.sym)
        except Exception:
            return fallback
        if not lpl or lpl <= 0 or not lot:
            return fallback
        return float(lot) * float(lpl)

    def _limit_fill_mode(self) -> str:
        return str(self.st.opt.get("limit_fill_mode", "strict") or "strict").lower()

    def _open_limit_fill(self, z, entry: float, via: str = "LIMIT") -> bool:
        """Open synthetic position when replay book fills a resting LIMIT."""
        if len(self.open_pos) >= self.st.max_pos:
            return False
        if LP._dir_open_count(self.st, z["direction"]) >= LP._max_same_dir(self.st):
            return False
        long_ = z["direction"] == "long"
        risk_act = (entry - z["sl"]) if long_ else (z["sl"] - entry)
        if risk_act <= 0:
            return False
        min_rr = float(self.st.opt.get("min_fill_rr", 0.0) or 0.0)
        tp = z.get("tp")
        if min_rr > 0 and tp is not None:
            rew = (tp - entry) if long_ else (entry - tp)
            if rew <= 0 or rew / risk_act < min_rr:
                return False
        ticket = z.get("limit_ticket")
        lot = 0.0
        if ticket:
            order = self._broker_orders.pop(int(ticket), None)
            if order:
                lot = float(order.get("lot") or 0.0)
        if lot <= 0:
            # Goes through the hooked X.calc_lot so --flat LOT applies here too.
            lot = X.calc_lot(self.sym, self._size_balance(), self.st.risk_pct,
                             risk_act, mt5)
        pos_ticket = self._next_pos_ticket
        self._next_pos_ticket += 1
        fill_ts = LP.clock_now()
        self.open_pos.append({
            "ticket": pos_ticket,
            "tag": z["tag"],
            "rule": z.get("rule"),
            "dir": z["direction"],
            "entry": float(entry),
            "sl": float(z["sl"]),
            "tp": float(tp) if tp else None,
            "risk": risk_act,
            "risk_usd": self._risk_usd(lot, risk_act),
            "lot": lot,
            "via": via,
            "entry_time": fill_ts,
            "entry_i": self._asof_i,
        })
        tick = _tick_from_px(entry, self.spread)
        self._update_equity(fill_ts, tick.bid, tick.ask)
        return True

    def _broker_sim_step(self, st):
        """Remove filled LIMITs from the book (sync_limit_fills completes the zone).

        Also market-in-band chase when fill_mode=market (touch_use_limit=False).
        """
        if st is not self.st:
            return
        tick = self._tick
        if tick is None:
            return

        # Instant marketable LIMITs at current ask/bid
        if self.fill_mode == "live" and LP._touch_use_limit(self.st):
            for z in list(self.st.pending_zones):
                ticket = z.get("limit_ticket")
                if not ticket or int(ticket) not in self._broker_orders:
                    continue
                if LP._zone_bar_confirm_mode(self.st, z) is not None:
                    continue
                limit_px = LP.zone_limit_px(z)
                if LP.limit_marketable_now(
                        z["direction"], limit_px, tick.bid, tick.ask):
                    if self._open_limit_fill(z, limit_px, via="LIMIT"):
                        break

            # M1 OHLC band / proximal / strict
            lo, hi = self._m1_lo, self._m1_hi
            if lo is not None and hi is not None:
                mode = self._limit_fill_mode()
                h = _spread_half(self.spread)
                for z in list(self.st.pending_zones):
                    ticket = z.get("limit_ticket")
                    if not ticket or int(ticket) not in self._broker_orders:
                        continue
                    if LP._zone_bar_confirm_mode(self.st, z) is not None:
                        continue
                    limit_px = LP.zone_limit_px(z)
                    if not LP.limit_would_fill(
                            z["direction"], limit_px, lo, hi,
                            zone_lo=z.get("zone_lo"), zone_hi=z.get("zone_hi"),
                            mode=mode, spread_half=h):
                        continue
                    if self._open_limit_fill(z, limit_px, via="LIMIT"):
                        break

        # Market chase is handled inside LP.check_zone_fills when limits off.

    def _check_sl_tp(self):
        """Broker SL/TP hits on current M1 range (live uses real broker)."""
        if self._tick is None:
            return
        lo = self._m1_lo if self._m1_lo is not None else self._tick.bid
        hi = self._m1_hi if self._m1_hi is not None else self._tick.ask
        h = _spread_half(self.spread)
        still = []
        for p in self.open_pos:
            entry, sl, tp = p["entry"], p["sl"], p.get("tp")
            risk = max(p["risk"], 1e-9)
            hit = None
            exit_px = None
            if p["dir"] == "long":
                # adverse = bid low; favorable = bid high ≈ mid-hi - half
                bid_lo, bid_hi = lo - h, hi - h
                if bid_lo <= sl:
                    hit, exit_px = "SL", float(sl)
                elif tp is not None and bid_hi >= tp:
                    hit, exit_px = "TP", float(tp)
            else:
                ask_lo, ask_hi = lo + h, hi + h
                if ask_hi >= sl:
                    hit, exit_px = "SL", float(sl)
                elif tp is not None and ask_lo <= tp:
                    hit, exit_px = "TP", float(tp)
            if hit is None:
                still.append(p)
                continue
            if p["dir"] == "long":
                R = (exit_px - entry) / risk
            else:
                R = (entry - exit_px) / risk
            net = R * p["risk_usd"] - (p["risk_usd"] / risk) * self.fillspread
            exit_t = LP.clock_now()
            row = {
                "asset": self.asset,
                "entry_time": p["entry_time"],
                "exit_time": exit_t,
                "rule": p.get("tag") or p.get("rule"),
                "dir": p["dir"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "R": R,
                "risk_usd": p["risk_usd"],
                "net": net,
                "exit": hit,
                "via": p.get("via", "?"),
                "balance": self.balance + net,
                "tf": self.tf,
            }
            self.trades.append(row)
            self.balance += net
            self._update_balance_dd()
            # Drop from book; cleanup_closed on next process_once clears pos_state
            if self.record:
                FR.log_close(
                    self.st, rule=row["rule"], side=row["dir"],
                    entry=entry, exit_px=exit_px, R=R, pnl=net,
                    reason=hit, ticket=p.get("ticket"), ts=exit_t)
        self.open_pos = still

    def _open_count(self):
        return len(self.open_pos)

    def _floating_pnl(self, bid: float, ask: float) -> float:
        total = 0.0
        for p in self.open_pos:
            risk = max(p["risk"], 1e-9)
            if p["dir"] == "long":
                R = (bid - p["entry"]) / risk
            else:
                R = (p["entry"] - ask) / risk
            total += R * p["risk_usd"]
        return total

    def _floating_pnl_adverse(self, lo: float, hi: float) -> float:
        h = _spread_half(self.spread)
        total = 0.0
        for p in self.open_pos:
            risk = max(p["risk"], 1e-9)
            if p["dir"] == "long":
                R = ((lo - h) - p["entry"]) / risk
            else:
                R = (p["entry"] - (hi + h)) / risk
            total += R * p["risk_usd"]
        return total

    def _open_risk_usd(self) -> float:
        return sum(p["risk_usd"] for p in self.open_pos)

    def _record_equity(self, ts, equity: float):
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd_usd = self.peak_equity - equity
        dd = dd_usd / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd > self.max_dd:
            self.max_dd = dd
            self.max_dd_usd = dd_usd
            self.max_dd_time = pd.Timestamp(ts)
        if equity < self.min_equity:
            self.min_equity = equity
            self.min_equity_time = pd.Timestamp(ts)
        n_open = self._open_count()
        if n_open > self.max_open:
            self.max_open = n_open
        if self.balance > 0:
            risk_pct = self._open_risk_usd() / self.balance
            if risk_pct > self.max_open_risk_pct:
                self.max_open_risk_pct = risk_pct

    def _update_equity(self, ts, bid: float, ask: float):
        self._record_equity(ts, self.balance + self._floating_pnl(bid, ask))

    def _update_equity_bar(self, ts, lo: float, hi: float, mid: float):
        self._record_equity(ts, self.balance + self._floating_pnl_adverse(lo, hi))
        tick = _tick_from_px(mid, self.spread)
        self._record_equity(ts, self.balance + self._floating_pnl(tick.bid, tick.ask))

    def _update_balance_dd(self):
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd_usd = self.peak_balance - self.balance
        dd = dd_usd / self.peak_balance if self.peak_balance > 0 else 0.0
        if dd > self.max_bal_dd:
            self.max_bal_dd = dd
            self.max_bal_dd_usd = dd_usd

    def stats(self) -> dict:
        return {
            "n": len(self.trades),
            "final": self.balance,
            "profit": self.balance - self.start_balance,
            "ret_pct": (self.balance / self.start_balance - 1.0) * 100
            if self.start_balance else 0.0,
            "wr": (sum(1 for t in self.trades if t["net"] > 0) / len(self.trades) * 100
                   if self.trades else 0.0),
            "pf": _profit_factor(self.trades),
            "max_dd": self.max_dd * 100,
            "max_dd_usd": self.max_dd_usd,
            "max_dd_time": self.max_dd_time,
            "min_equity": self.min_equity,
            "min_equity_time": self.min_equity_time,
            "max_bal_dd": self.max_bal_dd * 100,
            "max_bal_dd_usd": self.max_bal_dd_usd,
            "max_open": self.max_open,
            "max_open_risk_pct": self.max_open_risk_pct * 100,
            "peak_equity": self.peak_equity,
        }

    def _collect_ideas(self, closed_time):
        sig_key = str(closed_time)[:16]
        for z in self.st.pending_zones:
            if z.get("signal_bar") != sig_key:
                continue
            self.ideas.append({
                "tf": self.tf,
                "rule": z.get("tag") or z.get("rule"),
                "dir": z["direction"],
                "side": z["direction"],
                "entry_time": closed_time,
                "entry": LP.zone_limit_px(z),
                "sl": float(z["sl"]),
                "tp": float(z["tp"]) if z.get("tp") is not None else None,
                "kind": "armed",
                "zone_id": z.get("id"),
            })

    def _engine_step(self, now, *, m1_lo=None, m1_hi=None, mid=None):
        """One shared-engine poll at historical `now`."""
        if mid is not None:
            self._set_tick_from_mid(float(mid))
        self._m1_lo = m1_lo
        self._m1_hi = m1_hi
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        captured = now
        self._now = now
        prev_clock = LP._CLOCK
        LP.set_clock(lambda: captured)
        try:
            last_bar_before = self.st.last_bar
            # Same gate as the live loop: without it replay keeps filling zones
            # that live refuses once the account is already loaded up.
            bal = self._size_balance()
            port_risk = LP.portfolio_open_risk_pct([self.st], bal)
            risk_ok = self.no_cap or port_risk < C.MAX_PORTFOLIO_RISK - 0.001
            self._prev_pos_tickets, _ = LP.process_once(
                [self.st],
                bal=bal,
                risk_ok=risk_ok,
                dry=True,
                tf=self.tf,
                tf_min=self.tf_min,
                prev_pos_tickets=self._prev_pos_tickets,
                now=None,  # clock already set for this whole step
                sync_chart=False,
            )
            if self.st.last_bar is not None and self.st.last_bar != last_bar_before:
                self._collect_ideas(self.st.last_bar)
            self._check_sl_tp()
            if self._tick is not None:
                self._update_equity(now, self._tick.bid, self._tick.ask)
        finally:
            LP.set_clock(prev_clock)

    def _force_close_remaining(self):
        if not self.open_pos:
            return
        last = float(self.sig["close"].iloc[-1])
        self._set_tick_from_mid(last)
        until = self.sig.index[-1]
        if hasattr(until, "to_pydatetime"):
            until = until.to_pydatetime()
        captured = until
        LP.set_clock(lambda: captured)
        try:
            for p in list(self.open_pos):
                fake = type("P", (), {
                    "ticket": int(p["ticket"]),
                    "type": (mt5.POSITION_TYPE_BUY if p["dir"] == "long"
                             else mt5.POSITION_TYPE_SELL),
                    "symbol": self.sym,
                    "magic": self.st.magic,
                    "volume": float(p.get("lot") or 0.01),
                    "price_open": p["entry"],
                    "sl": p["sl"],
                    "tp": p.get("tp") or 0.0,
                    "comment": f"SMC-{p.get('tag', '')}",
                    "time": int(pd.Timestamp(p["entry_time"]).timestamp()),
                })()
                self._close_position(fake, True, reason="eod")
        finally:
            LP.set_clock(None)

    def run(self, shutdown_mt5: bool = True):
        self.load()
        start_i = max(WARMUP_BARS, int(self.sig.index.searchsorted(self.cutoff)))
        n = len(self.sig)
        use_lim = bool(self.st.opt.get("touch_use_limit", True))
        mode_tag = "mirror-vps" if self.mirror_vps else f"fill={self.fill_mode}"
        if self.fixed_lot is not None:
            size_mode = f"FIXED LOT {self.fixed_lot:g}"
        elif self.flat_risk:
            size_mode = "flat from start_balance"
        else:
            size_mode = "compound on running balance"
        print(f"\n  LIVE REPLAY  |  {self.asset}  {self.tf}  |  {mode_tag}")
        print(f"  bars {start_i}→{n - 1}  start_bal=${self.start_balance:,.0f}  "
              f"entry={self.st.opt.get('entry_confirm_mode')}")
        if self.fixed_lot is not None:
            print(f"  sizing={size_mode}  (risk% ignored for volume)")
        else:
            print(f"  risk={self.st.risk_pct * 100:.2f}%  sizing={size_mode}")
        print(f"  min_fill_rr={self.st.opt.get('min_fill_rr')}  "
              f"max_pos={self.st.max_pos}  max_same_dir={self.st.opt.get('max_same_dir')}  "
              f"touch_limit={use_lim}  spread={self.spread}")
        print(f"  engine: LP.process_once (shared with live)  "
              f"limit_fill_mode={self._limit_fill_mode()}")
        if self.mirror_vps:
            print("  profile: mirror-vps (legacy loose VPS) — not identity mode")

        try:
            step = pd.Timedelta(minutes=self.tf_min)
            for i in range(start_i, n):
                self._asof_i = i
                mid = float(self._sig_close[i])
                # MT5 stamps a bar with its OPEN time: bar i is only closed at
                # t_i + tf. Live arms zones at that instant with price sitting
                # at the close — never with the bar's own high/low.
                closed = self.sig.index[i] + step
                self._engine_step(closed, m1_lo=None, m1_hi=None, mid=mid)
                self._update_equity_bar(
                    closed, float(self._sig_low[i]), float(self._sig_high[i]),
                    mid)

                # M1 polls between this bar's close and the next bar's close
                # (same role as the 10s live polls). searchsorted gives the
                # same rows as an index mask without rescanning the frame.
                t0 = closed
                t1 = (self.sig.index[i + 1] + step) if i + 1 < n else None
                a = int(self._m1_idx.searchsorted(t0, side="left"))
                b = (len(self._m1_idx) if t1 is None
                     else int(self._m1_idx.searchsorted(t1, side="left")))
                for k in range(a, b):
                    self._engine_step(
                        self._m1_idx[k],
                        m1_lo=float(self._m1_low[k]),
                        m1_hi=float(self._m1_high[k]),
                        mid=float(self._m1_close[k]),
                    )

            self._force_close_remaining()
            last = self.sig.iloc[-1]
            self._update_equity_bar(
                self.sig.index[-1], float(last["low"]), float(last["high"]),
                float(last["close"]))
        finally:
            self._uninstall_hooks()
            if shutdown_mt5:
                mt5.shutdown()

        for t in self.trades:
            t["tf"] = self.tf
            tag = str(t.get("rule") or "")
            if "@" not in tag:
                t["rule_tf"] = f"{tag}@{self.tf}"
            else:
                t["rule_tf"] = tag
        return self.trades


def run_dual(asset: str, days: int, balance: float, mirror_vps: bool = False,
             fill_mode: str = "live", record: bool = False,
             risk_pct: float | None = None, flat_risk: bool = False,
             fixed_lot: float | None = None, no_cap: bool = False
             ) -> tuple[list[dict], list[LiveReplay], list[dict]]:
    """Run M5 then M15 (shared start balance per bot, like two VPS processes)."""
    engines: list[LiveReplay] = []
    all_trades: list[dict] = []
    all_ideas: list[dict] = []
    for i, tf in enumerate(("M5", "M15")):
        eng = LiveReplay(
            asset, tf, days, balance, fill_mode=fill_mode,
            mirror_vps=mirror_vps, record=record,
            risk_pct=risk_pct, flat_risk=flat_risk, fixed_lot=fixed_lot,
            no_cap=no_cap)
        trades = eng.run(shutdown_mt5=(i == 1))
        engines.append(eng)
        all_trades.extend(trades)
        all_ideas.extend(eng.ideas)
    all_trades.sort(key=lambda t: pd.Timestamp(t["entry_time"]))
    all_ideas.sort(key=lambda t: pd.Timestamp(t["entry_time"]))
    return all_trades, engines, all_ideas


def _print_stats(st: dict):
    print(f"\n  ACCOUNT (equity DD = balance + floating, M1 adverse extremes)")
    print(f"  Final balance : ${st['final']:,.2f}  |  Profit ${st['profit']:+,.2f}  "
          f"({st['ret_pct']:+.1f}%)")
    print(f"  Trades: {st['n']}  |  WR: {st['wr']:.1f}%  |  PF: {st['pf']:.2f}")
    print(f"  Max equity DD : {st['max_dd']:.1f}%  (${st['max_dd_usd']:,.2f})"
          + (f"  @ {st['max_dd_time']}" if st.get("max_dd_time") is not None else ""))
    print(f"  Min equity    : ${st['min_equity']:,.2f}"
          + (f"  @ {st['min_equity_time']}" if st.get("min_equity_time") is not None else ""))
    print(f"  Max balance DD: {st['max_bal_dd']:.1f}%  (${st['max_bal_dd_usd']:,.2f})  "
          f"(closed trades only - comparable to portfolio_backtest)")
    print(f"  Peak equity   : ${st['peak_equity']:,.2f}")
    print(f"  Max open seen : {st['max_open']} concurrent  "
          f"(cap max_pos / max_same_dir still apply separately)")
    print(f"  Max open risk : {st['max_open_risk_pct']:.1f}% of balance")


def _print_trades(trades, title):
    print(f"\n  --- {title} ({len(trades)} trades) ---")
    if not trades:
        print("  (none)")
        return
    tot = 0.0
    wins = 0
    for i, t in enumerate(trades, 1):
        tot += t["net"]
        if t["net"] > 0:
            wins += 1
        side = "LONG" if t["dir"] == "long" else "SHORT"
        et = pd.Timestamp(t["entry_time"]).strftime("%Y-%m-%d %H:%M")
        print(f"  {i:>3}  {et}  {t.get('rule', '?'):<10}  {side:<5}  "
              f"entry={t['entry']:.2f}  R={t['R']:+.2f}  "
              f"${t['net']:+.2f}  {t.get('via', '?'):<10}  {t['exit']}")
    pf = _profit_factor(trades)
    print(f"  TOTAL ${tot:+.2f}  |  WR {wins / len(trades) * 100:.1f}%  |  PF {pf:.2f}")


def _compare_backtest(asset, tf, days, balance):
    print("\n  Running portfolio_backtest path for comparison…")
    FC.apply_tf_paths(tf)
    oos_mg = FC.load_meta_gate(assets=[asset])
    trades, wf, _ = __import__(
        "portfolio_backtest", fromlist=["fetch_portfolio_trades"]
    ).fetch_portfolio_trades(days, mt5, oos_mg, tf=tf)
    trades = [t for t in trades if t.get("_asset") == asset]
    sim = __import__(
        "portfolio_backtest", fromlist=["simulate_portfolio"]
    ).simulate_portfolio(trades, balance)
    print(f"  Backtest path: {sim['n']} trades  WR={sim['wr']:.1f}%  "
          f"PF={sim['pf']:.2f}  PnL=${sim['profit']:+.2f}  "
          f"MaxDD={sim['max_dd']:.1f}%")
    return sim


def main():
    ap = argparse.ArgumentParser(description="Replay live zone engine on history")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--tf", default="M15",
                    help="Signal timeframe: M5/5M or M15/15M (default M15)")
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument(
        "--risk", type=float, default=None, metavar="FRAC",
        help="Risk fraction per trade (e.g. 0.02 = 2%%). "
             "Default: profile / RISK_MAP (gold is usually 1%%)")
    ap.add_argument(
        "--flat-risk", action="store_true",
        help="Size every trade from --balance (no compounding). "
             "PnL still accumulates; only the lot formula stays fixed")
    ap.add_argument(
        "--flat", type=float, default=None, metavar="LOT",
        help="Fixed lot size every trade (e.g. --flat 0.02). "
             "Ignores --risk / --flat-risk for volume")
    ap.add_argument(
        "--fill", default="live",
        help="live=LIMIT@proximal broker-sim (default, shared LP path); "
             "market=chase touch price; proximal is an alias for live")
    ap.add_argument(
        "--mirror-vps", action="store_true",
        help="Legacy loose VPS profile (not identity mode)")
    ap.add_argument(
        "--dual", action="store_true",
        help="Run M5 + M15 (like two bots on one account) and merge trades")
    ap.add_argument(
        "--no-cap", action="store_true",
        help="Disable the portfolio open-risk cap (live enforces it by default)")
    ap.add_argument(
        "--record", action="store_true",
        help="Write flight_recorder JSONL — decision journal for parity_diff")
    ap.add_argument(
        "--journal-path", default=None, metavar="FILE",
        help="Send the flight recorder to FILE instead of reports/"
             "flight_recorder_<TF>.jsonl (keeps the live tape untouched)")
    ap.add_argument("--compare", action="store_true",
                    help="Also run portfolio_backtest on the same window")
    args = ap.parse_args()
    if args.mirror_vps:
        args.fill = "market"
    else:
        args.fill = normalize_fill(args.fill)
    if args.risk is not None and args.risk <= 0:
        ap.error("--risk must be a positive fraction (e.g. 0.02)")
    if args.flat is not None and args.flat <= 0:
        ap.error("--flat LOT must be > 0 (e.g. 0.02)")
    if args.flat is not None and (args.risk is not None or args.flat_risk):
        print("  NOTE: --flat LOT wins; --risk / --flat-risk ignored for volume")
    if args.journal_path:
        FR.set_path(args.journal_path)
        args.record = True

    if args.dual:
        trades, engines, _ideas = run_dual(
            args.asset, args.days, args.balance,
            mirror_vps=args.mirror_vps, fill_mode=args.fill,
            record=args.record, risk_pct=args.risk,
            flat_risk=args.flat_risk, fixed_lot=args.flat,
            no_cap=args.no_cap)
        tag = "mirror-vps" if args.mirror_vps else f"fill={args.fill}"
        if args.flat is not None:
            risk_tag = f"FIXED LOT {args.flat:g}"
        else:
            risk_tag = (f"risk={engines[0].st.risk_pct * 100:.2f}%"
                        + (" flat" if args.flat_risk else " compound"))
        _print_trades(trades, f"LIVE-REPLAY DUAL M5+M15 {tag} {risk_tag}")
        bal = args.balance + sum(t["net"] for t in trades)
        print(f"\n  DUAL merged: {len(trades)} trades  "
              f"PnL=${bal - args.balance:+.2f}  end≈${bal:,.2f}  "
              f"(per-TF books independent; equity path not cross-merged)")
        for eng in engines:
            print(f"  · {eng.tf}: {len(eng.trades)} trades  "
                  f"max_same_dir={eng.st.opt.get('max_same_dir')}  "
                  f"touch_limit={eng.st.opt.get('touch_use_limit')}")
    else:
        args.tf = normalize_tf(args.tf)
        eng = LiveReplay(
            args.asset, args.tf, args.days, args.balance, args.fill,
            mirror_vps=args.mirror_vps, record=args.record,
            risk_pct=args.risk, flat_risk=args.flat_risk,
            fixed_lot=args.flat, no_cap=args.no_cap)
        trades = eng.run()
        tag = "mirror-vps" if args.mirror_vps else f"fill={args.fill}"
        _print_trades(trades, f"LIVE-REPLAY {args.tf} {tag}")
        _print_stats(eng.stats())
        if args.compare:
            _compare_backtest(args.asset, args.tf, args.days, args.balance)

    if args.mirror_vps:
        print("  • --mirror-vps is a legacy profile, not live≡replay identity.")
    else:
        print("  • Identity mode: LP.process_once + same ENTRY config as live.")
    if args.flat is not None:
        print(f"  • --flat {args.flat:g}: every trade uses that lot "
              f"(rounded to broker vol_step).")
    elif args.flat_risk:
        print("  • --flat-risk: lot sized from start balance every trade "
              "(not live default — live compounds).")


if __name__ == "__main__":
    main()
