"""
Live multi-asset portfolio — zone-based execution (shared with live_replay).

  • Bar close → invalidate → register zones → arm LIMIT @ proximal
  • Rejection rules → market on confirm close
  • Every poll → sync LIMIT fills / optional market chase
  • `process_once` is the shared engine step (live 10s poll + replay M1 clock)

Usage:
  python live_portfolio.py --dry-run
  python live_portfolio.py
"""
import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import strategy as S
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C
import floating_config as FC
import weekly_adaptive as WA
import live_trade_journal as LTJ
import live_flight_recorder as FR
import mt5_zone_draw as ZD

from live_smc import (
    TF_MIN, get_bars, balance, my_positions, my_orders,
    place_market, place_limit, cancel_order, modify_sl, close_position,
    entry_zone_bounds, price_in_entry_zone,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: pip install MetaTrader5")
    sys.exit(1)


CALIB_DAYS = 90
LOOP_SEC = 10
FILLED_ZONE_KEEP_H = 24  # filled zones stay on the chart this long
ZONE_STATE_FILE = "data/pending_zones.json"  # per-TF via set_zone_state_file()

# Bar windows handed to the engine. Replay must serve exactly these sizes or
# every full-array indicator (VP, prior-day levels, pivots, regime map) is
# computed over a different history than live and the two forks apart.
SIG_BARS = 600   # signal TF (M5/M15) per poll — last row is the forming bar
HTF_BARS = 400   # H1/H4 context per zone registration

# Injectable clock — live uses wall time; replay sets historical bar/M1 time.
_CLOCK = None  # None → datetime.now(); else zero-arg callable → datetime

# When True, dry-run synthetic LIMIT tickets (ticket>=900000000) may fill via
# sync_limit_fills once they disappear from my_orders (replay broker book).
# Live --dry-run keeps False so synthetic tickets are not instantly "filled".
ALLOW_DRY_LIMIT_FILL = False

# Optional hook: replay removes filled LIMITs / opens synthetic positions
# immediately before sync_limit_fills (same predicates as live broker).
_BROKER_SIM_STEP = None  # callable(st) | None


def clock_now() -> dt.datetime:
    if _CLOCK is not None:
        return _CLOCK()
    return dt.datetime.now()


# Undated flight-recorder events follow the same clock as the engine, so a
# replayed fill is stamped with the bar it happened on, not with real time.
FR.set_clock(clock_now)


def set_clock(fn=None) -> None:
    """Set clock provider (callable → datetime) or None to restore wall clock."""
    global _CLOCK
    _CLOCK = fn


def set_broker_sim_step(fn=None) -> None:
    global _BROKER_SIM_STEP
    _BROKER_SIM_STEP = fn


def set_zone_state_file(tf: str) -> None:
    """M15 keeps the legacy file; other TFs get their own state file."""
    global ZONE_STATE_FILE
    t = str(tf).upper()
    if t not in ("", "M15"):
        ZONE_STATE_FILE = f"data/pending_zones_{t}.json"


class AssetRunner:
    __slots__ = (
        "key", "label", "sym", "profile", "opt", "magic", "risk_pct",
        "rule_names", "det_params", "spike_params", "max_pos", "use_be_only",
        "last_bar", "pos_state", "known_tickets", "min_sl", "min_sl_date",
        "pending_zones", "filled_zones", "dead_zones", "meta_gate", "tf_tag",
    )

    def __init__(self, key, sym, profile, opt, magic, risk_pct, tf_tag="M15"):
        self.key = key
        self.label = profile.get("label", key)
        self.sym = sym
        self.profile = profile
        self.opt = opt
        self.magic = magic
        self.risk_pct = risk_pct
        self.tf_tag = str(tf_tag).upper()
        self.rule_names = list(opt["enabled"])
        self.det_params = P.merge_params(S.DEFAULT_PARAMS, profile)
        self.spike_params = opt.get("spike_params")
        self.max_pos = C.max_concurrent(key)
        self.use_be_only = True
        self.last_bar = None
        self.pos_state = {}
        self.known_tickets = set()
        self.min_sl = profile.get("min_sl", 0.0)
        self.min_sl_date = None
        self.pending_zones = []
        self.filled_zones = []  # kept on chart after entry (kind=filled)
        self.dead_zones = []    # kept on chart after invalidation (kind=dead)
        self.meta_gate = FC.load_meta_gate(assets=[key]) if opt.get("meta_gate") else None


def open_position_count(st):
    return len(my_positions(st.sym, st.magic))


def pending_limit_count(st):
    return len(my_orders(st.sym, st.magic))


def exposure_count(st):
    """Open positions + working limits (slot usage for max_concurrent)."""
    return open_position_count(st) + pending_limit_count(st)


def _max_same_dir(st) -> int:
    return max(1, int(st.opt.get("max_same_dir", 1) or 1))


def _touch_use_limit(st) -> bool:
    return bool(st.opt.get("touch_use_limit", True))


def _dir_open_count(st, direction: str) -> int:
    long_ = direction == "long"
    n = 0
    for p in my_positions(st.sym, st.magic):
        is_long = p.type == mt5.POSITION_TYPE_BUY
        if is_long == long_:
            n += 1
    return n


def _dir_armed_count(st, direction: str) -> int:
    """Pending zones (touch limit or rejection wait) in this direction."""
    return sum(1 for z in st.pending_zones if z.get("direction") == direction)


def direction_slot_free(st, direction: str) -> bool:
    """Institutional: one idea per direction — open OR armed, not both stacked."""
    cap = _max_same_dir(st)
    return (_dir_open_count(st, direction) + _dir_armed_count(st, direction)) < cap


# Broker/session conditions, not trading rules: these are re-measured on every
# start (live spread, ATR-calibrated min stop) and would otherwise make the
# fingerprint drift with the market instead of with the config.
_FP_SKIP = ("spread_usd", "min_risk_usd", "min_sl")


def _fp_value(v):
    """Config values only — drop live objects whose repr changes every run."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_fp_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _fp_value(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return None


def engine_fingerprint(st) -> str:
    """Hash of the rules+settings a zone was born under.

    Zones persisted by an older config are not comparable to what the current
    engine would produce, so a restart after a config change must not resurrect
    them — that is exactly how live drifts away from a clean replay.
    """
    opt = {k: v for k, v in st.opt.items() if k not in _FP_SKIP}
    payload = {
        "tf": st.tf_tag,
        "rules": sorted(str(r) for r in st.rule_names),
        "risk": round(float(st.risk_pct), 6),
        "opt": _fp_value(opt),
        "det": _fp_value(dict(getattr(st, "det_params", {}) or {})),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def save_zone_state(states):
    """Persist pending zones so a restart (code update) doesn't drop them.

    Backtest never 'restarts' — without this, every live restart silently
    deletes armed zones and their touched flags => missed fills vs backtest."""
    try:
        os.makedirs(os.path.dirname(ZONE_STATE_FILE), exist_ok=True)
        doc = {}
        for st in states:
            rows = []
            for z in st.pending_zones:
                r = dict(z)
                r["expires"] = z["expires"].isoformat()
                rows.append(r)
            doc[st.key] = rows
            frows = []
            for z in st.filled_zones:
                r = dict(z)
                r["expires"] = z["expires"].isoformat()
                r["filled_at"] = z["filled_at"].isoformat()
                frows.append(r)
            doc[f"{st.key}::filled"] = frows
            drows = []
            for z in st.dead_zones:
                r = dict(z)
                r["expires"] = z["expires"].isoformat()
                r["died_at"] = z["died_at"].isoformat()
                drows.append(r)
            doc[f"{st.key}::dead"] = drows
            doc[f"{st.key}::fp"] = engine_fingerprint(st)
        tmp = f"{ZONE_STATE_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, ZONE_STATE_FILE)
    except OSError as e:
        print(f"[ZONE] state save failed: {e}")


def load_zone_state(states):
    if not os.path.isfile(ZONE_STATE_FILE):
        return
    try:
        with open(ZONE_STATE_FILE, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[ZONE] state load failed: {e}")
        return
    now = clock_now()
    for st in states:
        saved_fp = doc.get(f"{st.key}::fp")
        cur_fp = engine_fingerprint(st)
        if saved_fp and saved_fp != cur_fp:
            n_drop = len(doc.get(st.key) or [])
            print(f"[ZONE] {st.key}: config changed since last run "
                  f"({saved_fp} → {cur_fp}) — dropped {n_drop} stale zone(s)")
            st.pending_zones = []
            continue
        rows = doc.get(st.key) or []
        restored = []
        for r in rows:
            try:
                r["expires"] = dt.datetime.fromisoformat(r["expires"])
            except (KeyError, ValueError):
                continue
            if r["expires"] > now:
                restored.append(r)
        st.pending_zones = restored
        filled = []
        keep_until = now - dt.timedelta(hours=FILLED_ZONE_KEEP_H)
        for r in doc.get(f"{st.key}::filled") or []:
            try:
                r["expires"] = dt.datetime.fromisoformat(r["expires"])
                r["filled_at"] = dt.datetime.fromisoformat(r["filled_at"])
            except (KeyError, ValueError):
                continue
            if r["filled_at"] > keep_until:
                filled.append(r)
        st.filled_zones = filled
        dead = []
        for r in doc.get(f"{st.key}::dead") or []:
            try:
                r["expires"] = dt.datetime.fromisoformat(r["expires"])
                r["died_at"] = dt.datetime.fromisoformat(r["died_at"])
            except (KeyError, ValueError):
                continue
            if r["died_at"] > keep_until:
                dead.append(r)
        st.dead_zones = dead
        if restored or filled or dead:
            print(f"[ZONE] restored {len(restored)} pending / {len(filled)} "
                  f"filled / {len(dead)} dead zone(s) for {st.key} after restart")


def reload_meta_gates(states):
    mg = FC.load_meta_gate(assets=[st.key for st in states])
    for st in states:
        st.meta_gate = mg
    return mg


def refresh_weekly_system(states, mt5_conn, reason: str = "", verbose: bool = True):
    if not FC.WEEKLY.get("enabled", True):
        return None
    doc = WA.auto_refresh(mt5=mt5_conn, silent=True, write=True)
    mg = reload_meta_gates(states)
    tag = f" [{reason}]" if reason else ""
    src = FC.meta_gate_source()
    if not verbose:
        n = (doc or {}).get("meta", {}).get("trades_in_window", "?")
        print(f"[WEEKLY] gate updated{tag} — source={src}  trades_in_window={n}")
        return doc
    print(f"\n[WEEKLY] gate refreshed{tag} — source={src}")
    if doc:
        meta = doc.get("meta", {})
        print(f"  week={meta.get('week_id')}  trades_in_window={meta.get('trades_in_window')}")
        for asset, rep in (doc.get("regime_report") or {}).items():
            if rep.get("error"):
                continue
            pct = rep.get("pct", {})
            print(f"  {asset} regime 7d: UP {pct.get('TREND_UP', 0):.0f}%  "
                  f"DOWN {pct.get('TREND_DOWN', 0):.0f}%  "
                  f"RANGE {pct.get('RANGE', 0):.0f}%  "
                  f"now={rep.get('current')} bias={rep.get('bias')}")
    if mg:
        print(mg.summary([st.key for st in states]))
    print()
    return doc


def should_refresh_weekly(last_week_id: str | None, last_refresh: dt.datetime | None) -> bool:
    now = clock_now()
    week_id = now.strftime("%G-W%V")
    if last_week_id is None or week_id != last_week_id:
        return True
    if last_refresh is None:
        return True
    # Mid-week refreshes are off by default: their timing depends on when the
    # process was started, so the same week can run on different gates.
    hours = float(FC.WEEKLY.get("live_refresh_hours", 0) or 0)
    if hours <= 0:
        return False
    return (now - last_refresh) >= dt.timedelta(hours=hours)


def open_exposure_usd(st, account_bal):
    return open_position_count(st) * st.risk_pct * account_bal


def portfolio_open_risk_pct(states, account_bal):
    if account_bal <= 0:
        return 0.0
    return sum(open_exposure_usd(st, account_bal) for st in states) / account_bal


def refresh_min_sl(st, tf, tf_min):
    today = dt.date.today()
    if st.min_sl_date == today:
        return
    if st.profile.get("min_sl_mode") != "atr":
        st.min_sl = st.profile.get("min_sl", 0.0)
        st.min_sl_date = today
        return
    n = int(CALIB_DAYS * 1440 / tf_min) + 60
    df = get_bars(st.sym, tf, n)
    if df is None or len(df) < 60:
        return
    B = S.Bars(df)
    st.min_sl = X.calibrate_min_sl(B, B.t[0], st.profile)
    st.min_sl_date = today


def build_asset_runner(key, mt5_conn, tf="M15"):
    _, profile = P.get_profile(key)
    sym = P.resolve_symbol_for_profile(key, mt5_conn)
    opt = P.resolve_live_opt(profile)
    spread = P.live_spread(sym, profile, mt5_conn)
    opt = P.apply_profile_to_settings(opt, profile, spread)
    opt["max_concurrent"] = C.max_concurrent(key)
    risk = C.RISK_MAP.get(key, profile.get("risk_pct", S.RISK_PCT))
    magic = C.magic_for(key, tf)
    return AssetRunner(key, sym, profile, opt, magic, risk, tf_tag=tf)


def _sync_chart_zones(states):
    tf_tag = states[0].tf_tag if states else "M15"
    try:
        if ZD.sync_zones(states, tf_tag=tf_tag):
            return
        print("[ZONE] chart sync skipped (MT5 Files path unavailable)")
    except OSError as exc:
        print(f"[ZONE] chart sync failed ({exc}); trading continues")


def cancel_legacy_limits(st, dry):
    """On startup: clear orphan limits, then re-arm from restored touch zones."""
    tracked = {z.get("limit_ticket") for z in st.pending_zones if z.get("limit_ticket")}
    for o in my_orders(st.sym, st.magic):
        if o.ticket not in tracked:
            cancel_order(o.ticket, dry)
            print(f"  [ZONE] cancelled orphan limit #{o.ticket}")


def _cancel_zone_limit(st, z, dry, reason=""):
    ticket = z.pop("limit_ticket", None)
    if not ticket:
        return
    # Dry-run / replay synthetic tickets live only in the local order book
    if dry and int(ticket) >= 900000000:
        cancel_order(ticket, dry)
        print(f"[LIMIT] cancelled #{ticket} {z.get('tag')} {reason}")
        FR.log_limit(st, z, "limit_cancel", reason=reason, ticket=ticket)
        return
    for o in my_orders(st.sym, st.magic):
        if o.ticket == ticket:
            cancel_order(ticket, dry)
            print(f"[LIMIT] cancelled #{ticket} {z.get('tag')} {reason}")
            FR.log_limit(st, z, "limit_cancel", reason=reason, ticket=ticket)
            return


def zone_limit_px(z) -> float:
    """Price the zone's LIMIT rests at (zones saved before limit_px existed
    fall back to proximal)."""
    px = z.get("limit_px")
    return float(px) if px is not None else float(z["proximal"])


def arm_touch_limit(st, z, account_bal, dry) -> bool:
    """Place the LIMIT at the band edge for touch/retest zones."""
    if not _touch_use_limit(st):
        return False
    if _zone_bar_confirm_mode(st, z) is not None:
        return False
    if z.get("limit_ticket"):
        return True
    bal = max(account_bal, 1.0)
    lot = X.calc_lot(st.sym, bal, st.risk_pct, z["risk"], mt5)
    if lot <= 0:
        return False
    limit_px = zone_limit_px(z)
    # min_fill_rr at the order price (planned R:R)
    min_rr = float(st.opt.get("min_fill_rr", 0.0) or 0.0)
    tp = z.get("tp")
    if min_rr > 0 and tp is not None and z["risk"] > 0:
        rew = (tp - limit_px) if z["direction"] == "long" else (limit_px - tp)
        if rew <= 0 or rew / z["risk"] < min_rr:
            print(f"[LIMIT] skip arm {z['tag']} — planned RR "
                  f"{rew / z['risk'] if z['risk'] else 0:.2f} < {min_rr:.2f}")
            return False
    tick = mt5.symbol_info_tick(st.sym)
    if tick and limit_marketable_now(z["direction"], limit_px, tick.bid, tick.ask):
        # Band edge already through the market — the broker would reject a
        # pending order here, so take it at market (the backtest's equivalent
        # is filling on the very first bar of the entry scan).
        return _execute_zone_fill(st, z, account_bal, dry, via="TOUCH")
    expire = z["expires"]
    if isinstance(expire, str):
        expire = dt.datetime.fromisoformat(expire)
    ticket = place_limit(
        st.sym, z["direction"], lot, limit_px, z["sl"],
        f"{z['tag']}@{st.tf_tag}", st.magic, expire, dry, tp=tp)
    if not ticket:
        return False
    z["limit_ticket"] = int(ticket)
    print(f"[LIMIT] armed {z['tag']} {z['direction'].upper()} "
          f"@ {limit_px:.5g}  SL {z['sl']:.5g}  #{z['limit_ticket']}")
    FR.log_limit(st, z, "limit_armed", lot=lot)
    return True


def arm_pending_touch_limits(st, account_bal, dry):
    # Copy: an immediately-marketable zone fills at market and leaves the list.
    for z in list(st.pending_zones):
        if z.get("limit_ticket"):
            continue
        if _zone_bar_confirm_mode(st, z) is not None:
            continue
        if exposure_count(st) >= st.max_pos:
            break
        arm_touch_limit(st, z, account_bal, dry)


def sync_limit_fills(st, account_bal, dry):
    """If a zone's limit disappeared from the order book → zone filled.

    Cancel paths pop `limit_ticket` first, so cancelled zones never reach here.
    Live dry-run keeps synthetic tickets unfilled unless ALLOW_DRY_LIMIT_FILL
    (replay) is set and the replay book removed the order.
    """
    live_tickets = {o.ticket for o in my_orders(st.sym, st.magic)}
    kept = []
    for z in st.pending_zones:
        ticket = z.get("limit_ticket")
        if not ticket or ticket in live_tickets:
            kept.append(z)
            continue
        if (dry and int(ticket) >= 900000000 and not ALLOW_DRY_LIMIT_FILL):
            kept.append(z)
            continue
        # Limit gone — broker filled (live) or replay book removed it
        z["filled_at"] = clock_now()
        st.filled_zones.append(z)
        limit_px = zone_limit_px(z)
        print(f"[LIMIT-FILL] {st.key} {z['tag']} {z['direction'].upper()} "
              f"@ {limit_px:.5g} (broker filled #{ticket})")
        FR.log_fill(st, z, entry=limit_px, via="LIMIT", ticket=ticket)
    st.pending_zones = kept


def process_trailing(st, df, opt, dry):
    positions = my_positions(st.sym, st.magic)
    for p in positions:
        sst = st.pos_state.get(p.ticket)
        if sst is None:
            risk = abs(p.price_open - p.sl) if p.sl else df["close"].iloc[-2] * 0.001
            sst = {
                "entry": p.price_open, "risk": max(risk, 1e-6),
                "dir": "long" if p.type == mt5.POSITION_TYPE_BUY else "short",
                "tag": (p.comment or "SMC-?").replace("SMC-", ""),
                "peak": p.price_open, "activated": False,
                "entry_time": dt.datetime.fromtimestamp(int(p.time)),
                "open_ticket": p.ticket,
            }
            st.pos_state[p.ticket] = sst
        hi = float(df["high"].iloc[-2])
        lo = float(df["low"].iloc[-2])
        trig = opt["be_trigger"] if opt else S.TRAIL_TRIGGER
        if sst["dir"] == "long":
            sst["peak"] = max(sst["peak"], hi)
            fav = (sst["peak"] - sst["entry"]) / sst["risk"]
            if not sst["activated"] and fav >= trig:
                sst["activated"] = True
                modify_sl(p, max(p.sl, sst["entry"]), dry)
            if sst["activated"] and not st.use_be_only:
                modify_sl(p, max(p.sl, sst["peak"] - S.TRAIL_DIST * sst["risk"]), dry)
        else:
            sst["peak"] = min(sst["peak"], lo)
            fav = (sst["entry"] - sst["peak"]) / sst["risk"]
            if not sst["activated"] and fav >= trig:
                sst["activated"] = True
                modify_sl(p, min(p.sl, sst["entry"]), dry)
            if sst["activated"] and not st.use_be_only:
                modify_sl(p, min(p.sl, sst["peak"] + S.TRAIL_DIST * sst["risk"]), dry)
    return positions


def enforce_max_hold(st, positions, tf_min, dry):
    """Backtest parity: run_backtest force-closes after max_hold bars.
    Live had NO time exit — stale trades drifted for days and diverged."""
    max_hold = int(st.opt.get("max_hold", S.MAX_HOLD))
    if max_hold <= 0:
        return
    now = clock_now()
    for p in positions:
        sst = st.pos_state.get(p.ticket)
        entry_time = (sst or {}).get("entry_time") \
            or dt.datetime.fromtimestamp(int(p.time))
        held_min = (now - entry_time).total_seconds() / 60.0
        if held_min >= max_hold * tf_min:
            print(f"[TIME-EXIT] {st.key} #{p.ticket} "
                  f"{(p.comment or '').replace('SMC-', '')} held "
                  f"{held_min/60:.1f}h ≥ max_hold {max_hold}×{tf_min}m → close")
            close_position(p, dry, reason="TIME")


def cleanup_closed(st, prev_tickets, account_bal, dry):
    current = {p.ticket for p in my_positions(st.sym, st.magic)}
    closed = prev_tickets - current
    for ticket in closed:
        sst = st.pos_state.pop(ticket, None)
        if sst and not dry:
            row = LTJ.log_closed_position(st, ticket, sst, account_bal,
                                          since=sst.get("entry_time"))
            if row:
                print(f"[JOURNAL] {st.key} closed {row['rule']} {row['side']} "
                      f"R={row['R']:+.2f} ${row['pnl_usd']:+.2f} -> {LTJ.journal_path()}")
                FR.log_close(
                    st, rule=row["rule"], side=row["side"],
                    entry=row["entry"], exit_px=row["exit"],
                    R=row["R"], pnl=row["pnl_usd"],
                    reason=row.get("exit_reason", "unknown"), ticket=ticket)
    return closed


def _zone_fingerprint(tag, proximal, direction, signal_bar):
    return f"{tag}|{direction}|{round(float(proximal), 2)}|{signal_bar}"


def _prune_zones(st, now=None, dry=False):
    now = now or clock_now()
    kept = []
    for z in st.pending_zones:
        exp = z["expires"]
        if isinstance(exp, str):
            exp = dt.datetime.fromisoformat(exp)
            z["expires"] = exp
        if exp > now:
            kept.append(z)
        else:
            _cancel_zone_limit(st, z, dry, reason="expired")
    st.pending_zones = kept
    keep_until = now - dt.timedelta(hours=FILLED_ZONE_KEEP_H)
    st.filled_zones = [z for z in st.filled_zones
                       if z.get("filled_at") and z["filled_at"] > keep_until]
    st.dead_zones = [z for z in st.dead_zones
                     if z.get("died_at") and z["died_at"] > keep_until]


def register_zones(st, df, closed_time, tf_min, account_bal):
    """On M15 close: detect setups and register entry zones (no orders)."""
    opt = dict(st.opt)
    B = S.Bars(df)
    i = B.n - 2
    opt["min_risk_usd"] = st.min_sl

    def _bar_log(gate, **kw):
        FR.log_bar(st, df=df, upto=i, ts=closed_time, gate=gate,
                   close=float(B.c[i]), **kw)

    def _rej(rule, direction, reason, **kw):
        FR.log_cand(st, rule=rule, side=direction, reason=reason,
                    ts=closed_time, **kw)

    if opt.get("require_atr_regime") and not opt.get("spike_mode") and not S.atr_regime_pass(
            B, i, lookback=opt.get("atr_lookback", S.OPT_STABLE_ATR_LB),
            max_ratio=opt.get("atr_max_ratio", S.OPT_STABLE_ATR_RATIO)):
        _bar_log("atr_regime")
        return

    sess_start, sess_end = opt.get("session_start"), opt.get("session_end")
    if sess_start is not None and sess_end is not None:
        if not S.session_pass(closed_time, sess_start, sess_end):
            _bar_log("session")
            return

    htf_dfs = {}
    htf_needed = list(opt["htf_tfs"])
    if opt.get("regime_gate"):
        rtf = opt.get("regime_tf", "H1")
        if rtf not in htf_needed:
            htf_needed.append(rtf)
    for htf in htf_needed:
        hdf = get_bars(st.sym, htf, HTF_BARS)
        if hdf is not None:
            htf_dfs[htf] = hdf
    htf_ctx = S.prepare_htf_context(htf_dfs)
    rmap = None
    if opt.get("regime_gate"):
        rmap = S.build_regime_map(
            htf_ctx, tf=opt.get("regime_tf", "H1"),
            detector=opt.get("regime_detector", "hybrid"),
            win=opt.get("regime_win", 40),
            er_trend=opt.get("regime_er_trend", 0.35),
            slope_k=opt.get("regime_slope_k", 0.0006),
            er_hi=opt.get("regime_er_hi", 0.40),
            er_lo=opt.get("regime_er_lo", 0.25),
            confirm=opt.get("regime_confirm", 2),
            h4_win=opt.get("regime_h4_win", 30),
            h4_lookback=opt.get("regime_h4_lookback", 60),
            structure_break=opt.get("regime_structure_break", False),
            brk_win=opt.get("regime_brk_win", 40),
            shock_win=opt.get("regime_shock_win", 24),
            shock_baseline=opt.get("regime_shock_baseline", 720))

    htf_fp = {tf_name: FR.frame_fingerprint(hdf)
              for tf_name, hdf in htf_dfs.items()} if FR.journal_on() else {}

    cur_regime = None
    if rmap is not None:
        cur_regime = rmap.at(B.t[i])
        if (opt.get("meta_gate") and st.meta_gate
                and opt.get("meta_skip_if_no_rules")
                and not st.meta_gate.has_any(st.key, cur_regime)):
            _bar_log("meta_no_rules", htf=htf_fp, regime=cur_regime)
            return

    existing = {_zone_fingerprint(z["tag"], z["proximal"], z["direction"], z["signal_bar"])
                for z in st.pending_zones}
    n_cand = n_armed = 0
    wait_min = S.WAIT_BARS * tf_min
    expires = clock_now() + dt.timedelta(minutes=wait_min)
    signal_key = str(closed_time)[:16]

    for name in st.rule_names:
        if (name not in S.DETECTORS and name not in S.NDS_FAMILY
                and name not in S.AB_DETECTORS and name not in S.WYCK_DETECTORS
                and name not in S.STYLE_DETECTORS and name not in S.EXT_DETECTORS
                and name not in S.SPIKE_DETECTORS):
            continue
        for (direction, proximal, distal, tag) in S.iter_rule_setups(
                name, B, i, st.det_params, htf_context=htf_ctx,
                signal_time=B.t[i],
                htf_tfs=tuple(opt.get("htf_tfs", ("H4", "H1"))),
                nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO),
                spike_params=st.spike_params):
            n_cand += 1
            if not S.vp_pass(B, i, direction, proximal,
                             opt.get("vp_mode", S.VP_MODE),
                             window=opt.get("vp_window", 480),
                             vp_tol_atr=opt.get("vp_tol_atr", 0.0)):
                _rej(tag, direction, "vp", proximal=float(proximal),
                     distal=float(distal))
                continue
            if opt.get("require_confluence") and not S.setup_confluence_pass(
                    B, i, direction, tag, st.rule_names):
                _rej(tag, direction, "confluence", proximal=float(proximal))
                continue
            if opt.get("require_bos") and not S.bos_confirm_pass(B, i, direction):
                _rej(tag, direction, "bos", proximal=float(proximal))
                continue
            atr_i = B.atr[i] if B.atr[i] > 0 else (B.rng[i] + 1e-6)
            etol = st.det_params.get("entry_tol_atr", 0.0) * atr_i
            itol = st.det_params.get("invalidate_tol_atr", 0.0) * atr_i
            # The LIMIT rests at the band edge, so THAT price is the entry —
            # risk, lot and RR all have to be measured from it.
            limit_px = S.limit_entry_price(
                direction, proximal, etol, opt.get("limit_at", S.LIMIT_AT))
            sl, _entry, _risk = S.compute_entry_sl_risk(B, i, direction, proximal, distal)
            risk = (limit_px - sl) if direction == "long" else (sl - limit_px)
            _entry = limit_px
            if risk <= 0:
                _rej(tag, direction, "risk<=0", proximal=float(proximal),
                     distal=float(distal), sl=float(sl), limit=float(limit_px))
                continue
            if not S.regime_allows(
                    rmap, B.t[i], direction, rule=name,
                    shock_gate=opt.get("regime_shock_gate", False),
                    shock_ratio=opt.get("regime_shock_ratio", 1.8),
                    flip_cooldown_h=opt.get("regime_flip_cooldown_h", 0.0)):
                _rej(tag, direction, "regime", proximal=float(proximal),
                     regime=cur_regime)
                continue
            if st.meta_gate is not None:
                reg = cur_regime if cur_regime is not None else (
                    rmap.at(B.t[i]) if rmap is not None else "RANGE")
                if not st.meta_gate.allows(st.key, name, reg):
                    _rej(tag, direction, "meta_gate",
                         proximal=float(proximal), regime=reg)
                    continue
            reg = cur_regime if cur_regime is not None else (
                rmap.at(B.t[i]) if rmap is not None else "RANGE")
            if not C.rule_allowed(st.key, name, reg,
                                  direction=direction,
                                  macro=rmap.macro_at(B.t[i]) if rmap and hasattr(rmap, "macro_at") else 0):
                _rej(tag, direction, "rule_gate", proximal=float(proximal),
                     regime=reg)
                continue
            if not S.fib_filter_pass(
                    B, i, direction, _entry, name, proximal, distal,
                    htf_ctx, B.t[i], tuple(opt.get("htf_tfs", ("H4", "H1"))),
                    opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO),
                    opt.get("require_fib", False),
                    opt.get("fib_nds_exempt", False),
                    opt.get("fib_ob_only", False),
                    opt.get("fib_spike_exempt", False),
                    opt.get("fib_demand_exempt", False)):
                _rej(tag, direction, "fib", proximal=float(proximal),
                     entry=float(_entry))
                continue
            if not S.entry_filters_pass(
                    B, i, risk, min_risk_usd=opt["min_risk_usd"],
                    session_start=opt["session_start"], session_end=opt["session_end"],
                    direction=direction, signal_time=B.t[i],
                    htf_context=htf_ctx if opt.get("htf_trend") else None,
                    htf_trend=opt.get("htf_trend", False),
                    htf_ema=opt.get("htf_ema", 20),
                    htf_tfs=tuple(opt.get("htf_trend_tfs") or opt.get("htf_tfs", ("H1",)))):
                _rej(tag, direction, "entry_filters", proximal=float(proximal),
                     risk=float(risk), min_risk_usd=float(opt["min_risk_usd"]))
                continue
            trade_tag = S.resolve_trade_tag(
                name, tag, B, i, direction, proximal, distal,
                htf_ctx, B.t[i], tuple(opt.get("htf_tfs", ("H4", "H1"))),
                nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO),
                nds_enabled=any(S.is_nds_rule(n) for n in st.rule_names))
            tp, tp_src = S.resolve_tp(
                B, i, direction, trade_tag, proximal, distal, limit_px, risk,
                tp_mode=opt["tp_mode"], fixed_tp_r=opt["fixed_tp_r"],
                htf_context=htf_ctx, min_tp_r=opt["min_tp_r"],
                max_tp_r=opt["max_tp_r"], htf_lookback=opt["htf_lookback"],
                htf_tfs=tuple(opt["htf_tfs"]), signal_time=B.t[i],
            )
            zone_lo, zone_hi = entry_zone_bounds(direction, proximal, etol)
            fp = _zone_fingerprint(trade_tag, proximal, direction, signal_key)
            if fp in existing:
                _rej(trade_tag, direction, "dup", proximal=float(proximal))
                continue
            # One idea per direction: skip if already open or armed that way.
            if not direction_slot_free(st, direction):
                FR.log_skip_register(
                    st, rule=trade_tag, side=direction, reason="same_dir",
                    closed_time=closed_time, proximal=float(proximal), sl=float(sl))
                continue
            zone = {
                "id": uuid.uuid4().hex[:8],
                "tag": trade_tag,
                "rule": name,
                "direction": direction,
                "proximal": float(proximal),
                "limit_px": float(limit_px),
                "distal": float(distal),
                "zone_lo": zone_lo,
                "zone_hi": zone_hi,
                "etol": etol,
                "itol": itol,
                "sl": float(sl),
                "tp": float(tp) if tp is not None else None,
                "tp_src": tp_src,
                "risk": float(risk),
                "signal_bar": signal_key,
                "expires": expires,
                "touched": False,
                "limit_ticket": None,
            }
            st.pending_zones.append(zone)
            existing.add(fp)
            n_armed += 1
            FR.log_armed(st, zone, closed_time, regime=cur_regime)
            tp_s = f" TP {tp:.2f}" if tp else ""
            if tp and tp_src:
                tp_s += f" [{tp_src}]"
            print(f"[ZONE] {closed_time} {st.key} {trade_tag} {direction.upper()} "
                  f"entry [{zone_lo:.5g} – {zone_hi:.5g}] limit {limit_px:.5g}  "
                  f"SL {sl:.5g}{tp_s}  wait={wait_min}m")

    _bar_log("ok", htf=htf_fp, regime=cur_regime,
             cands=n_cand, armed=n_armed, pending=len(st.pending_zones))


def invalidate_zones(st, df, dry=False):
    """Drop zones when close breaks distal; cancel any armed LIMIT."""
    close = float(df["close"].iloc[-2])
    bar_time = df.index[-2]
    kept = []
    for z in st.pending_zones:
        itol = z.get("itol", 0.0)
        distal = z["distal"]
        direction = z["direction"]
        dead = False
        if direction == "long" and close < distal - itol:
            dead = True
        elif direction == "short" and close > distal + itol:
            dead = True
        if dead:
            print(f"[ZONE] {bar_time} invalidated {z['tag']} {direction.upper()} "
                  f"close={close:.5g} distal={distal:.5g}")
            FR.log_invalidate(st, z, close=close, bar_time=bar_time)
            _cancel_zone_limit(st, z, dry, reason="invalidated")
            z["died_at"] = clock_now()
            st.dead_zones.append(z)
        else:
            kept.append(z)
    st.pending_zones = kept


def _entry_confirm_mode(st):
    return str(st.opt.get("entry_confirm_mode", "none")).lower()


def _zone_bar_confirm_mode(st, z):
    """Per-zone confirm style (None = immediate tick fill)."""
    mode = _entry_confirm_mode(st)
    if mode == "rule_confirm":
        rules = st.opt.get("entry_confirm_rules")
        if S.rule_needs_entry_confirm(z.get("rule"), rules):
            return "rejection"
        return None
    if mode in ("m15_close", "rejection"):
        return mode
    return None


def _uses_touch_poll(st):
    return _entry_confirm_mode(st) in ("m15_close", "rejection", "rule_confirm")


def touch_zones(st, bid, ask):
    """Mark zones whose entry band has been touched by the current tick."""
    for z in st.pending_zones:
        if price_in_entry_zone(z["direction"], z["zone_lo"], z["zone_hi"], bid, ask):
            z["touched"] = True


def _execute_zone_fill(st, z, account_bal, dry, via="ZONE"):
    """Market fill — used for rejection confirm (and touch if limits disabled)."""
    if exposure_count(st) >= st.max_pos:
        FR.log_skip_fill(st, z, "max_pos", via=via)
        return False
    # Same-dir: count open only (this zone is already in pending_zones)
    if _dir_open_count(st, z["direction"]) >= _max_same_dir(st):
        print(f"[ZONE] skip fill {z['tag']} — same-dir slot full")
        FR.log_skip_fill(st, z, "same_dir", via=via)
        return False
    bal = max(account_bal, 1.0)
    tick = mt5.symbol_info_tick(st.sym)
    if not tick:
        return False
    px = tick.ask if z["direction"] == "long" else tick.bid
    long_ = z["direction"] == "long"
    risk_act = (px - z["sl"]) if long_ else (z["sl"] - px)
    if risk_act <= 0:
        print(f"[ZONE] skip fill {z['tag']} — price beyond SL")
        FR.log_skip_fill(st, z, "beyond_sl", via=via, entry=px)
        return False
    min_rr = float(st.opt.get("min_fill_rr", 0.0) or 0.0)
    tp = z.get("tp")
    if min_rr > 0 and tp is not None:
        rew = (tp - px) if long_ else (px - tp)
        if rew <= 0 or rew / risk_act < min_rr:
            rr = rew / risk_act if risk_act > 0 else 0.0
            print(f"[ZONE] skip fill {z['tag']} {z['direction'].upper()} "
                  f"— RR {rr:.2f} < {min_rr:.2f} (zone stays armed)")
            FR.log_skip_fill(st, z, "min_fill_rr", via=via, entry=px, rr=rr)
            return False
    risk_used = risk_act
    lot = X.calc_lot(st.sym, bal, st.risk_pct, risk_used, mt5)
    print(f"[ZONE-FILL] {st.key} {z['tag']} {z['direction'].upper()} "
          f"zone=[{z['zone_lo']:.5g}–{z['zone_hi']:.5g}] tick={px:.5g} → MARKET ({via})")
    ok = place_market(
        st.sym, z["direction"], lot, z["sl"], f"{z['tag']}@{st.tf_tag}",
        st.magic, dry, tp=z.get("tp"), via=via)
    if ok:
        _cancel_zone_limit(st, z, dry, reason="market-fill")
        st.pending_zones = [x for x in st.pending_zones if x["id"] != z["id"]]
        z["filled_at"] = clock_now()
        st.filled_zones.append(z)
        FR.log_fill(st, z, entry=px, via=via)
    return ok


def _bar_confirm_pass(B, bar_i, z, confirm_mode):
    if confirm_mode == "m15_close":
        return S.m15_close_entry_confirm(
            B, bar_i, z["direction"], z["proximal"], z["etol"])
    if confirm_mode == "rejection":
        return S.rejection_entry_confirm(
            B, bar_i, z["direction"], z["proximal"], z["etol"])
    return False


def limit_price_reached(direction: str, limit_px: float, lo: float, hi: float,
                        spread_half: float = 0.0) -> bool:
    """True if an M1 bar range would fill a resting LIMIT at limit_px (strict).

    Buy limit fills when the ask traded down to the limit; sell limit when
    bid traded up to it. With mid OHLC, approximate ask≈mid+half, bid≈mid-half.
    """
    h = max(float(spread_half), 0.0)
    if direction == "long":
        return (float(lo) + h) <= float(limit_px)
    return (float(hi) - h) >= float(limit_px)


def limit_would_fill(direction: str, limit_px: float, lo: float, hi: float,
                     *, zone_lo: float | None = None, zone_hi: float | None = None,
                     mode: str = "zone", spread_half: float = 0.0) -> bool:
    """Broker-sim LIMIT fill check. Default `zone` = Windsor Prime friendly.

    zone     : M1 trades anywhere in the entry rectangle → fill at proximal
    proximal : mid OHLC touches the limit price (no spread haircut)
    strict   : proximal with half-spread haircut (harsh on M1 vs real ticks)
    """
    m = str(mode or "zone").lower()
    if m == "zone" and zone_lo is not None and zone_hi is not None:
        return float(lo) <= float(zone_hi) and float(hi) >= float(zone_lo)
    if m == "strict":
        return limit_price_reached(direction, limit_px, lo, hi, spread_half)
    # proximal
    if direction == "long":
        return float(lo) <= float(limit_px)
    return float(hi) >= float(limit_px)


def limit_marketable_now(direction: str, limit_px: float, bid: float, ask: float) -> bool:
    """True if a freshly placed LIMIT is already through the market (instant fill)."""
    if direction == "long":
        return float(ask) <= float(limit_px)
    return float(bid) >= float(limit_px)


def confirm_zone_fills(st, df, account_bal, dry, portfolio_risk_ok, now=None):
    """On bar close: rejection zones market-fill; touch zones use LIMIT fills.

    Touch/retest: LIMIT already sits at proximal — only mark touched / sync.
    If touch_use_limit is False, wick still triggers a market fill (legacy).
    Rejection (CH_REV / WYCK): reclaim close → market.
    `now` lets live_replay prune on historical clock (not wall time).
    """
    if not portfolio_risk_ok:
        return
    if exposure_count(st) >= st.max_pos:
        return
    if df is None or len(df) < 5:
        return
    _prune_zones(st, now=now, dry=dry)
    bar_i = len(df) - 2
    B = S.Bars(df)
    cur_bar_key = str(df.index[bar_i])[:16]
    use_limit = _touch_use_limit(st)
    for z in list(st.pending_zones):
        if exposure_count(st) >= st.max_pos:
            break
        # Backtest parity: _scan_entry_bar starts at sig_i + 1.
        if z.get("signal_bar") == cur_bar_key:
            continue
        confirm_mode = _zone_bar_confirm_mode(st, z)
        wick_hit = S.zone_touched_on_bar(
            B, bar_i, z["direction"], z["proximal"], z["etol"])
        if wick_hit:
            z["touched"] = True

        if confirm_mode is None:
            # Touch/retest — LIMIT handles the fill when armed.
            if use_limit:
                continue
            if not wick_hit:
                continue
            if _execute_zone_fill(st, z, account_bal, dry, via="TOUCH"):
                break
            continue

        if not z.get("touched"):
            continue
        if not _bar_confirm_pass(B, bar_i, z, confirm_mode):
            continue
        label = "REJECTION" if confirm_mode == "rejection" else "M15-CONFIRM"
        if _execute_zone_fill(st, z, account_bal, dry, via=label):
            break


def check_zone_fills(st, account_bal, dry, portfolio_risk_ok, now=None):
    """Every poll: sync limit fills; market-fill touch only if limits disabled."""
    if _BROKER_SIM_STEP is not None:
        _BROKER_SIM_STEP(st)
    sync_limit_fills(st, account_bal, dry)
    mode = _entry_confirm_mode(st)
    if mode in ("rejection", "m15_close"):
        tick = mt5.symbol_info_tick(st.sym)
        if tick:
            touch_zones(st, tick.bid, tick.ask)
        return
    if _touch_use_limit(st):
        # Institutional path: working LIMITs only — no chase-market in band.
        return
    if not portfolio_risk_ok:
        return
    if exposure_count(st) >= st.max_pos:
        return
    tick = mt5.symbol_info_tick(st.sym)
    if not tick:
        return
    bid, ask = tick.bid, tick.ask
    _prune_zones(st, now=now, dry=dry)

    for z in list(st.pending_zones):
        if exposure_count(st) >= st.max_pos:
            break
        if _zone_bar_confirm_mode(st, z) is not None:
            continue
        if not price_in_entry_zone(z["direction"], z["zone_lo"], z["zone_hi"], bid, ask):
            continue
        if _execute_zone_fill(st, z, account_bal, dry, via="ZONE"):
            break


def process_once(states, *, bal, risk_ok, dry, tf, tf_min,
                 prev_pos_tickets=None, now=None, sync_chart=True):
    """One shared engine step used by live (10s poll) and replay (M1 clock).

    Order matches the historical live main loop:
      trail → max_hold → cleanup → (on new bar) invalidate/register/arm/confirm
      → check_zone_fills

    If `now` is set, the injectable clock is temporarily pointed at it.
    Returns (prev_pos_tickets, n_closed).
    """
    clock_tok = None
    if now is not None:
        if hasattr(now, "to_pydatetime"):
            now = now.to_pydatetime()
        elif not isinstance(now, dt.datetime):
            now = dt.datetime.fromisoformat(str(now)[:19])
        captured = now
        clock_tok = _CLOCK
        set_clock(lambda: captured)

    if prev_pos_tickets is None:
        prev_pos_tickets = {st.key: set() for st in states}

    n_closed = 0
    try:
        for st in states:
            df = get_bars(st.sym, tf, SIG_BARS)
            if df is None or len(df) < 60:
                continue

            refresh_min_sl(st, tf, tf_min)
            positions = process_trailing(st, df, st.opt, dry)
            enforce_max_hold(st, positions, tf_min, dry)
            cur_tix = {p.ticket for p in positions}
            closed = cleanup_closed(
                st, prev_pos_tickets.get(st.key, set()), bal, dry)
            if closed:
                n_closed += len(closed)
            prev_pos_tickets[st.key] = cur_tix

            for p in positions:
                if p.ticket not in st.known_tickets:
                    st.known_tickets.add(p.ticket)
                    side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
                    print(f"\n{'='*64}\n  TRADE OPEN  |  {st.label}  {side}  {st.sym}  "
                          f"lot={p.volume}\n  Rule: {(p.comment or '').replace('SMC-','')}  "
                          f"entry={p.price_open}  sl={p.sl}  tp={p.tp}\n{'='*64}\n")
                    FR.log_event(
                        "open_seen", asset=st.key, tf=st.tf_tag, symbol=st.sym,
                        rule=(p.comment or "").replace("SMC-", ""),
                        side="long" if p.type == mt5.POSITION_TYPE_BUY else "short",
                        entry=float(p.price_open), sl=float(p.sl or 0),
                        tp=float(p.tp or 0), ticket=int(p.ticket),
                        lot=float(p.volume))

            closed_time = df.index[-2]
            new_bar = st.last_bar is None or closed_time > st.last_bar
            if new_bar:
                st.last_bar = closed_time
                invalidate_zones(st, df, dry=dry)
                register_zones(st, df, closed_time, tf_min, bal)
                arm_pending_touch_limits(st, bal, dry)
                confirm_zone_fills(
                    st, df, bal, dry, risk_ok, now=clock_now())
                if sync_chart:
                    _sync_chart_zones(states)

            check_zone_fills(st, bal, dry, risk_ok, now=clock_now())

        if sync_chart:
            _sync_chart_zones(states)
    finally:
        if now is not None:
            set_clock(clock_tok)

    return prev_pos_tickets, n_closed


def main():
    ap = argparse.ArgumentParser(description="Live multi-asset portfolio (zone execution)")
    ap.add_argument("--tf", default="M15", choices=["M5", "M15"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assets", default=",".join(C.PORTFOLIO_ASSETS))
    ap.add_argument("--no-cap", action="store_true",
                    help="Disable portfolio-wide open risk cap")
    ap.add_argument("--risk-scale", type=float, default=1.0,
                    help="Multiply per-trade risk (e.g. 0.5 when running two "
                         "TF bots on one account)")
    ap.add_argument("--no-journal", action="store_true",
                    help="Stop writing the per-bar decision journal "
                         "(bar/cand events used by parity_diff.py)")
    args = ap.parse_args()
    tf = args.tf
    tf_min = TF_MIN[tf]
    dry = args.dry_run
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    # Per-TF isolation: magic numbers, weekly gate, journal, zone files —
    # lets an M5 and an M15 instance trade the same account side by side.
    FC.apply_tf_paths(tf)
    set_zone_state_file(tf)
    FR.set_tf(tf)
    FR.set_journal(not args.no_journal)

    if not mt5.initialize():
        print(f"[MT5] connect failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"[MT5] connected to {mt5.terminal_info().company}")
    print(f"  Flight recorder: {FR.recorder_path(tf)}")
    states = []
    for key in assets:
        if key not in P.PROFILES:
            print(f"[LIVE] unknown asset {key}")
            continue
        try:
            st = build_asset_runner(key, mt5, tf=tf)
            if args.risk_scale != 1.0:
                st.risk_pct *= args.risk_scale
            states.append(st)
        except RuntimeError as e:
            print(f"[LIVE] skip {key}: {e}")

    if not states:
        print("[LIVE] no symbols loaded")
        mt5.shutdown()
        sys.exit(1)

    bal = balance()
    sep = "=" * 72
    print(sep)
    print(f"  LIVE PORTFOLIO  |  ZONE mode  |  tf={tf}  |  poll={LOOP_SEC}s  |  dry={dry}")
    ecm = FC.ENTRY.get("entry_confirm_mode", "none")
    use_lim = FC.ENTRY.get("touch_use_limit", True)
    same_dir = FC.ENTRY.get("max_same_dir", 1)
    if ecm == "rule_confirm" and use_lim:
        print(f"  {tf} close → arm zones  |  TOUCH → LIMIT @ proximal  |  "
              f"REJECTION close → market  |  max_same_dir={same_dir}")
    elif ecm == "rule_confirm":
        print(f"  {tf} close → arm zones  |  TOUCH/wick → market  |  "
              f"REJECTION close → market  |  max_same_dir={same_dir}")
    elif ecm == "rejection":
        print(f"  {tf} close → arm zones  |  poll → touch  |  {tf} close → REJECTION → market")
    elif ecm == "m15_close":
        print(f"  {tf} close → arm zones  |  poll → touch  |  {tf} close → confirm → market")
    else:
        print(f"  {tf} close → arm zones  |  "
              f"{'LIMIT @ proximal' if use_lim else 'tick/wick → market'}")
    print(f"  balance=${bal:,.2f}")
    mc_desc = ", ".join(C.describe_concurrent(st.key) for st in states)
    print(f"  Max open/symbol: {mc_desc}", end="")
    if args.no_cap:
        print()
    else:
        print(f"  |  portfolio cap {C.MAX_PORTFOLIO_RISK*100:.0f}%")
    print(sep)
    FC.print_status()
    print(sep)
    for st in states:
        refresh_min_sl(st, tf, tf_min)
        print(f"  {st.label:<12} {st.sym:<14} risk={st.risk_pct*100:.0f}%  "
              f"min_sl={st.min_sl:.5g}  magic={st.magic}  rules={len(st.rule_names)}")
    print(sep + "\n")

    for st in states:
        FR.log_session(asset=st.key, tf=st.tf_tag, mode="live",
                       fp=engine_fingerprint(st), dry=bool(dry),
                       sig_bars=SIG_BARS, htf_bars=HTF_BARS,
                       risk=float(st.risk_pct), rules=",".join(st.rule_names))

    refresh_weekly_system(states, mt5, reason="startup")
    load_zone_state(states)
    for st in states:
        cancel_legacy_limits(st, dry)
        arm_pending_touch_limits(st, bal, dry)
    _sync_chart_zones(states)
    zpath = ZD.zones_csv_path(tf)
    if zpath:
        print(f"  Zone chart file: {zpath}")
        print(f"  Attach EA: mql5/SMCZoneDrawer.mq5 → compile → XAUUSD {tf} chart"
              f" (input ZoneFile={ZD.zone_file_name(tf)})\n")

    prev_pos_tickets = {st.key: set() for st in states}
    last_week_id = clock_now().strftime("%G-W%V")
    last_weekly_refresh = clock_now()
    trades_since_refresh = 0

    while True:
        try:
            bal = balance()
            port_risk = portfolio_open_risk_pct(states, bal)
            risk_ok = args.no_cap or port_risk < C.MAX_PORTFOLIO_RISK - 0.001

            prev_pos_tickets, n_closed = process_once(
                states, bal=bal, risk_ok=risk_ok, dry=dry,
                tf=tf, tf_min=tf_min, prev_pos_tickets=prev_pos_tickets,
                sync_chart=True)
            if n_closed:
                trades_since_refresh += n_closed
            save_zone_state(states)

            # Parity: the backtest walk-forward rebuilds the gate once per ISO
            # week. Refreshing after every closed trade makes the live gate
            # drift from the replayed one mid-week (missed/extra trades).
            refresh_on_trades = FC.WEEKLY.get("live_refresh_on_trades", False)
            due_trades = refresh_on_trades and trades_since_refresh > 0
            if due_trades or should_refresh_weekly(last_week_id, last_weekly_refresh):
                reason = "new_trades" if due_trades else "schedule"
                verbose = reason != "new_trades"
                refresh_weekly_system(states, mt5, reason=reason, verbose=verbose)
                last_week_id = clock_now().strftime("%G-W%V")
                last_weekly_refresh = clock_now()
                trades_since_refresh = 0

            time.sleep(LOOP_SEC)

        except KeyboardInterrupt:
            print("\n[LIVE] stopped.")
            break
        except Exception as e:
            print(f"[LIVE] error: {e}")
            time.sleep(20)

    mt5.shutdown()


if __name__ == "__main__":
    main()
