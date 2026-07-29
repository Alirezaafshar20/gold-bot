"""
Long-horizon WALK-FORWARD rule calibration (2022-04-01 -> now).

Goal: find rules that work across MANY market regimes, not just the last 90 days,
so the live model stops being overfit to one recent window.

Data: Dukascopy 1-minute (download with data: see dukascopy_loader / README).
  - M1 resampled to M15 or M5  -> SIGNALS (--tf)
  - raw M1                     -> EXIT resolution (true intrabar)
  - M1 resampled to H1/H4      -> HTF context (same as live)

  python calibrate_long.py --tf M15                 # default
  python calibrate_long.py --tf M5 --asset XAUUSD
  python calibrate_long.py --source mt5 --days 120 --tf M5
"""
import argparse
import inspect
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import strategy as S
import symbol_profiles as P
import portfolio_config as C
import dukascopy_loader as DK
import regime as RG

META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)

BAL = 1000.0
# Minimum trades before a rule may be judged at all. Over a multi-year feed a
# real rule produces hundreds; anything in single digits is noise, and picking
# on it is how a gate ends up fitted to three lucky trades.
MIN_N = 30
DEFAULT_FROM = "2022-04-01"

_TF_ALIASES = {
    "M5": "M5", "5M": "M5", "5": "M5",
    "M15": "M15", "15M": "M15", "15": "M15",
}


TF_MINUTES = {"M5": 5, "M15": 15}


def tf_minutes(tf: str) -> int:
    return TF_MINUTES.get(str(tf).upper(), 15)


def normalize_tf(raw: str) -> str:
    key = str(raw).strip().upper().replace(" ", "")
    tf = _TF_ALIASES.get(key)
    if tf is None:
        raise SystemExit(f"Unsupported --tf {raw!r}. Use M5 or M15.")
    return tf


def default_calibrate_out(tf: str) -> str:
    return ("reports/calibrate_long.txt" if tf == "M15"
            else f"reports/calibrate_long_{tf}.txt")


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def _resample(m1, rule):
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = m1.resample(rule, label="left", closed="left", origin="epoch").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def load_dukascopy(key, start, tf="M15", end=None):
    sig, m1 = DK.load_pair(key, start=start, end=end, tf=tf)
    htf = {"H1": _resample(m1, "1h"), "H4": _resample(m1, "4h")}
    return sig, m1, htf


def load_mt5(key, days, tf="M15"):
    import mt5_data as M
    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile(key, mt5)
        _, sig, m1, _ = M.fetch_pair(sym, tf, mt5=mt5, days=days)
        _, htf = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=("H4", "H1"))
    finally:
        M.shutdown(mt5)
    return sig, m1, htf


# --------------------------------------------------------------------------- #
#  min-SL (trailing, per fold — identical to live)
# --------------------------------------------------------------------------- #
def trailing_min_sl(B, fold_start, prof, lookback_days=90):
    base = prof.get("min_sl", 0.0)
    if prof.get("min_sl_mode") != "atr":
        return base
    t0 = np.datetime64(pd.Timestamp(fold_start) - pd.Timedelta(days=lookback_days))
    t1 = np.datetime64(pd.Timestamp(fold_start))
    mask = (B.t >= t0) & (B.t < t1)
    atrs = B.atr[np.asarray(mask)]
    atrs = atrs[atrs > 0]
    if len(atrs) < 10:
        return base
    med = float(np.median(atrs))
    sl = med * prof.get("min_sl_atr_mult", 1.25)
    floor = prof.get("min_sl_floor", base * 0.5)
    cap = prof.get("min_sl_cap", base * 3.0)
    if prof.get("min_sl_pct_floor") or prof.get("min_sl_pct_cap"):
        idx = int(np.searchsorted(B.t, t1, "left"))
        idx = min(max(idx, 0), len(B.c) - 1)
        price = float(B.c[idx])
        if prof.get("min_sl_pct_floor"):
            floor = max(floor, price * prof["min_sl_pct_floor"])
        if prof.get("min_sl_pct_cap"):
            cap = min(cap, price * prof["min_sl_pct_cap"])
    return float(max(floor, min(cap, sl)))


# --------------------------------------------------------------------------- #
#  rule universe (same source as asset_rule_research)
# --------------------------------------------------------------------------- #
# Rules held out of rule SELECTION until their own logic is repaired. They are
# not merely unprofitable — their R is not measured against a risk their logic
# defines, so ranking on expectancy promotes them for the wrong reason.
#
#   no invalidation level of their own (proximal == distal on ~every signal, so
#   the stop is whatever the generic swing happens to be):
#     EW_ABC, DIV_RSI_*, DIV_MACD_*, ICT_MIT_*, WYCK_LPS, WYCK_LPSY, ICT_SB_*
#   invalidation on the profit side of the entry on every signal:
#     CVD_L, CVD_S
#   limit sits through the market on most signals, so the backtest books a
#   price the broker never offers:
#     ICT_BRK_L, ICT_BRK_S, WYCK_BC, WYCK_SC
#
# Measured on 35,420 XAUUSD M15 bars (2025-01 .. 2026-07). Re-run the audit and
# delete a name from here once its detector defines a real stop.
QUARANTINED_RULES = frozenset({
    "EW_ABC",
    "DIV_RSI_L", "DIV_RSI_S", "DIV_MACD_L", "DIV_MACD_S",
    "ICT_MIT_L", "ICT_MIT_S",
    "ICT_SB_L", "ICT_SB_S",
    "WYCK_LPS", "WYCK_LPSY",
    "CVD_L", "CVD_S",
    "ICT_BRK_L", "ICT_BRK_S",
    "WYCK_BC", "WYCK_SC",
})


def collect_rules(scope, include_quarantined=False):
    import extended_strategies as EX
    import wave_styles as WS
    import wyckoff as WY
    import al_brooks as AB
    import spike_strategies as SP
    rules = {}

    def add(name, fam):
        if name and name not in rules:
            rules[name] = fam
    for k in S.DETECTORS:
        add(k, "SMC Core")
    for k in S.NDS_FAMILY:
        add(k, "NDS Family")
    for k in S.OPT_DENSE_RULES:
        add(k, "DENSE Live")
    for k in EX.EXT_ALL_RULES:
        add(k, EX.EXT_RULE_META.get(k, {}).get("label", "Extended"))
    for k in WS.STYLE_ALL_RULES:
        add(k, WS.STYLE_RULE_META.get(k, {}).get("label", "Wave Styles"))
    for k in WY.WYCK_ALL_RULES:
        add(k, WY.WYCK_RULE_META.get(k, {}).get("label", "Wyckoff"))
    for k in AB.AB_ALL_RULES:
        add(k, "Al Brooks")
    for k in SP.SPIKE_DETECTORS:
        add(k, "Spike")
    if not include_quarantined:
        rules = {k: v for k, v in rules.items() if k not in QUARANTINED_RULES}
    if scope == "core":
        keep = {"SMC Core", "NDS Family", "DENSE Live"}
        return {k: v for k, v in rules.items() if v in keep}
    if scope == "dense":
        drop = {"Al Brooks", "Spike", "Wave Styles"}
        return {k: v for k, v in rules.items()
                if v not in drop and "Wyckoff" not in v}
    return rules


# --------------------------------------------------------------------------- #
#  backtest one rule over the full period
# --------------------------------------------------------------------------- #
def run_rule(rule, m15, m1, B, ctx, htf, prof, spread, tf_min=15):
    o = P.apply_profile_to_settings(P.preset_for_profile(prof), prof, spread,
                                    min_sl_override=prof.get("min_sl_floor", 0.0))
    o["enabled"] = [rule]
    # Broker/execution keys (limit_fill_mode, broker, commission…) describe how
    # orders get filled live — run_backtest has no parameter for them.
    engine_args = set(inspect.signature(S.run_backtest).parameters)
    k = {x: o[x] for x in o if x not in META and x in engine_args}
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    # run with a permissive floor; per-fold min_sl applied later as a filter
    k["min_risk_usd"] = 0.0
    return S.run_backtest(m15, m1, B=B, ctx=ctx, tf_min=tf_min, **k)


def fold_bounds(t_index, start, fold_days):
    t0 = pd.Timestamp(start)
    tend = pd.Timestamp(t_index[-1])
    folds = []
    cur = t0
    while cur < tend:
        nxt = cur + pd.Timedelta(days=fold_days)
        folds.append((cur, min(nxt, tend)))
        cur = nxt
    return folds


def stat_block(trades, risk, spread):
    if not trades:
        return None
    acc = S.simulate_account(trades, BAL, risk, spread)
    Rs = np.array([t["R"] - spread / t["risk"] for t in trades if t["risk"] > 0])
    w, l = Rs[Rs > 0], Rs[Rs <= 0]
    pf = float(w.sum() / (-l.sum())) if l.sum() < 0 else 999.0
    # ret/dd are compounded at `risk`, so they depend on leverage and on the
    # ORDER of the trades. Rule ranking uses avg_r/tot_r instead — expectancy
    # per trade is what actually transfers out of sample.
    return {"n": len(trades), "wr": acc["wr"], "ret": acc["ret_pct"],
            "pf": pf, "dd": acc["max_dd"], "tot_r": float(Rs.sum()),
            "avg_r": float(Rs.mean()) if len(Rs) else 0.0}


def tag_regimes(trades, rmap):
    """Attach the market regime at entry to every trade (no lookahead)."""
    if rmap is None:
        return trades
    for t in trades:
        t["_regime"] = rmap.at(t["time"])
    return trades


def evaluate(trades, folds, B, prof, risk, spread):
    """Per-rule walk-forward: filter each fold by its trailing min-SL."""
    fold_rets, fold_rs, fold_pf_ok, fold_n = [], [], 0, 0
    kept_all = []
    for (f0, f1) in folds:
        msl = trailing_min_sl(B, f0, prof)
        sub = [t for t in trades
               if f0 <= pd.Timestamp(t["time"]) < f1 and t["risk"] >= msl]
        if not sub:
            continue
        kept_all.extend(sub)
        st = stat_block(sub, risk, spread)
        fold_n += 1
        fold_rets.append(st["ret"])
        fold_rs.append(st["avg_r"])
        if st["pf"] > 1.0:
            fold_pf_ok += 1
    overall = stat_block(sorted(kept_all, key=lambda t: t["time"]), risk, spread)
    if overall is None or fold_n == 0:
        return None
    overall["folds"] = fold_n
    overall["folds_profitable"] = fold_pf_ok
    overall["robust_pct"] = 100.0 * fold_pf_ok / fold_n
    overall["median_fold_ret"] = float(np.median(fold_rets)) if fold_rets else 0.0
    overall["median_fold_r"] = float(np.median(fold_rs)) if fold_rs else 0.0

    # per-regime breakdown over the kept (live-like) trades
    by_reg = {}
    for reg in RG.REGIMES:
        sub = [t for t in kept_all if t.get("_regime") == reg]
        by_reg[reg] = stat_block(sub, risk, spread)
    overall["by_regime"] = by_reg
    # regime score: how many regimes are net-positive, and the worst-regime return
    rets, r_exp, pos = [], [], 0
    for reg in RG.REGIMES:
        st = by_reg[reg]
        if st and st["n"] >= 3:
            rets.append(st["ret"])
            r_exp.append(st["avg_r"])
            if st["tot_r"] > 0:
                pos += 1
    overall["regimes_pos"] = pos
    overall["worst_regime_ret"] = float(min(rets)) if rets else 0.0
    overall["worst_regime_r"] = float(min(r_exp)) if r_exp else 0.0
    return overall


# --------------------------------------------------------------------------- #
#  parallel workers (one heavy build per process, then many rules)
# --------------------------------------------------------------------------- #
_W = {}


def build_regime_map(htf_dfs, reg_cfg):
    """Regime labeller for calibration/OOS.

    Must stay label-for-label identical to what live_portfolio builds, or the
    meta-gate's per-regime allow table is keyed on regimes live never sees.
    """
    if not reg_cfg:
        return None
    detector = reg_cfg.get("detector", "kaufman")

    if detector == "mtf":
        # MTF needs both H1 and H4; it is built from the whole HTF context.
        return RG.build_mtf_map(
            htf_dfs,
            h4_win=reg_cfg.get("h4_win", 30),
            h1_win=reg_cfg.get("win", 40),
            er_hi=reg_cfg.get("er_hi", 0.32),
            er_lo=reg_cfg.get("er_lo", 0.20),
            confirm=reg_cfg.get("confirm", 2),
            h4_lookback=reg_cfg.get("h4_lookback", 60),
            structure_break=reg_cfg.get("structure_break", False),
            brk_win=reg_cfg.get("brk_win", 40),
            shock_win=reg_cfg.get("shock_win", 24),
            shock_baseline=reg_cfg.get("shock_baseline", 720))

    tf = reg_cfg.get("tf", "H1")
    df = htf_dfs.get(tf)
    if df is None or len(df) == 0:
        return None
    return RG.RegimeMap(df, detector=detector,
                        win=reg_cfg.get("win", 40),
                        er_trend=reg_cfg.get("er_trend", 0.35),
                        slope_k=reg_cfg.get("slope_k", 0.0006),
                        er_hi=reg_cfg.get("er_hi", 0.40),
                        er_lo=reg_cfg.get("er_lo", 0.25),
                        confirm=reg_cfg.get("confirm", 1))


def _init_worker(m15, m1, htf_dfs, prof, spread, risk, folds, reg_cfg, tf_min=15):
    _W["tf_min"] = tf_min
    _W["m15"] = m15
    _W["m1"] = m1
    _W["B"] = S.Bars(m15)
    _W["ctx"] = S.M1Ctx(m1, m15.index)
    _W["htf"] = S.prepare_htf_context(htf_dfs)
    _W["prof"] = prof
    _W["spread"] = spread
    _W["risk"] = risk
    _W["folds"] = folds
    _W["rmap"] = build_regime_map(htf_dfs, reg_cfg)


def _eval_rule_worker(item):
    rule, fam = item
    try:
        trades = run_rule(rule, _W["m15"], _W["m1"], _W["B"], _W["ctx"],
                          _W["htf"], _W["prof"], _W["spread"],
                          tf_min=_W.get("tf_min", 15))
    except Exception as exc:
        return rule, {"error": str(exc)[:60], "family": fam}
    tag_regimes(trades, _W["rmap"])
    ev = evaluate(trades, _W["folds"], _W["B"], _W["prof"], _W["risk"], _W["spread"])
    if ev:
        ev["family"] = fam
    return rule, ev


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def calibrate_asset(key, source, days, start, scope, fold_days, log, workers=1,
                    reg_cfg=None, tf="M15"):
    _, prof = P.get_profile(key)
    label = prof.get("label", key)
    log("\n" + "=" * 92)
    log(f"  {label.upper()}  ({key})   scope={scope}   source={source}   tf={tf}")
    log("=" * 92)

    if source == "mt5":
        m15, m1, htf_dfs = load_mt5(key, days, tf=tf)
    else:
        m15, m1, htf_dfs = load_dukascopy(key, start, tf=tf)

    B = S.Bars(m15)
    ctx = S.M1Ctx(m1, m15.index)
    htf = S.prepare_htf_context(htf_dfs)
    tf_min = tf_minutes(tf)
    risk = prof.get("risk_pct", 0.01)
    spread = prof.get("spread", 0.0)
    folds = fold_bounds(B.t, m15.index[0], fold_days)
    rmap = build_regime_map(htf_dfs, reg_cfg)
    log(f"  {tf} bars: {len(m15):,}  |  {m15.index[0]:%Y-%m-%d} -> {m15.index[-1]:%Y-%m-%d}"
        f"  |  folds: {len(folds)} x {fold_days}d")
    log(f"  spread={spread}  risk={risk*100:.2f}%  (per-fold trailing min-SL, live-like)")
    if rmap is not None:
        cov = rmap.coverage()
        log(f"  regime: detector={reg_cfg['detector']} tf={reg_cfg['tf']} "
            f"win={reg_cfg['win']}  | coverage  "
            f"UP {cov['TREND_UP']:.0f}%  DOWN {cov['TREND_DOWN']:.0f}%  "
            f"RANGE {cov['RANGE']:.0f}%")

    rules = collect_rules(scope)
    results = {}
    items = sorted(rules.items())
    n = len(items)
    if workers > 1:
        log(f"  parallel: {workers} worker processes")
        with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(m15, m1, htf_dfs, prof, spread, risk, folds, reg_cfg,
                          tf_min)) as ex:
            done = 0
            for rule, ev in ex.map(_eval_rule_worker, items):
                done += 1
                print(f"  [{key}] {done}/{n} done        ", end="\r", flush=True)
                if ev and "error" not in ev:
                    results[rule] = ev
                elif ev:
                    results[rule] = ev
    else:
        for i, (rule, fam) in enumerate(items, 1):
            print(f"  [{key}] {i}/{n} {rule}        ", end="\r", flush=True)
            try:
                trades = run_rule(rule, m15, m1, B, ctx, htf, prof, spread,
                                  tf_min=tf_min)
            except Exception as exc:
                results[rule] = {"error": str(exc)[:60], "family": fam}
                continue
            tag_regimes(trades, rmap)
            ev = evaluate(trades, folds, B, prof, risk, spread)
            if ev:
                ev["family"] = fam
                results[rule] = ev
    print(" " * 70, end="\r")

    usable = {r: v for r, v in results.items()
              if v and "error" not in v and v["n"] >= MIN_N}

    # Ranked on expectancy per trade, not compounded return (see stat_block).
    robust = sorted(usable.items(),
                    key=lambda kv: (-kv[1]["robust_pct"],
                                    -kv[1].get("median_fold_r", 0.0),
                                    -kv[1]["pf"]))
    profit = sorted(usable.items(),
                    key=lambda kv: (-kv[1]["tot_r"], -kv[1].get("avg_r", 0.0)))
    # regime-robust: most regimes net-positive, then best worst-regime expectancy
    reg_robust = sorted(usable.items(),
                        key=lambda kv: (-kv[1].get("regimes_pos", 0),
                                        -kv[1].get("worst_regime_r", -1e9),
                                        -kv[1]["robust_pct"]))

    def table(title, ordered):
        log(f"\n  {title}")
        log(f"  {'RULE':<16}{'fam':<14}{'n':>5}{'WR%':>7}{'PF':>8}"
            f"{'avgR':>7}{'totR':>8}{'ret%':>8}{'DD%':>6}{'robust':>8}")
        log("  " + "-" * 90)
        for rule, v in ordered[:20]:
            pf_disp = min(v["pf"], 99.99)
            log(f"  {rule:<16}{str(v['family'])[:13]:<14}{v['n']:>5}{v['wr']:>7.1f}"
                f"{pf_disp:>8.2f}{v.get('avg_r', 0.0):>+7.2f}{v['tot_r']:>+8.1f}"
                f"{v['ret']:>+8.1f}{v['dd']:>6.1f}{v['robust_pct']:>7.0f}%")

    def regime_table(title, ordered):
        log(f"\n  {title}")
        log(f"  {'RULE':<16}{'+regs':>6}  "
            f"{'UP  n/ret%/pf':>20}{'DOWN  n/ret%/pf':>22}{'RANGE  n/ret%/pf':>22}")
        log("  " + "-" * 86)
        for rule, v in ordered[:20]:
            br = v.get("by_regime", {})
            cells = ""
            for reg in RG.REGIMES:
                st = br.get(reg)
                if st:
                    cells += f"{st['n']:>4}/{st['ret']:>+6.1f}/{min(st['pf'],99.9):>4.1f}  "
                else:
                    cells += f"{'—':>16}  "
            log(f"  {rule:<16}{v.get('regimes_pos',0):>6}  {cells}")

    table("MOST ROBUST (profitable in most folds — anti-overfit)", robust)
    table(f"MOST PROFITABLE (raw total edge over {start}->now)", profit)
    if rmap is not None:
        regime_table("REGIME-ROBUST (works in UP *and* DOWN *and* RANGE)", reg_robust)

    rec_robust = tuple(r for r, _ in robust[:5])
    rec_profit = tuple(r for r, _ in profit[:5])
    rec_regime = tuple(r for r, _ in reg_robust[:5])
    log(f"\n  Current live rules_override: {prof.get('rules_override')}")
    log(f"  >> ROBUST pick           : {rec_robust}")
    log(f"  >> PROFIT pick           : {rec_profit}")
    if rmap is not None:
        log(f"  >> REGIME-ROBUST (recommended): {rec_regime}")
    return {"asset": key, "tf": tf, "robust": rec_robust, "profit": rec_profit,
            "regime_robust": rec_regime,
            "results": {r: {kk: vv for kk, vv in v.items()}
                        for r, v in usable.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None,
                    help="single asset key (default: all portfolio assets)")
    ap.add_argument("--tf", default="M15",
                    help="Signal timeframe: M5 or M15 (default M15)")
    ap.add_argument("--source", choices=("dukascopy", "mt5"), default="dukascopy")
    ap.add_argument("--from", dest="start", default=DEFAULT_FROM)
    ap.add_argument("--days", type=int, default=120, help="only for --source mt5")
    ap.add_argument("--scope", choices=("core", "dense", "full"), default="dense")
    ap.add_argument("--fold-days", type=int, default=90)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel processes (default: CPU cores - 1)")
    ap.add_argument("--regime-detector", dest="reg_det", default="kaufman",
                    choices=("kaufman", "slope", "donchian", "ma", "hybrid",
                             "mtf", "none"),
                    help="market-regime detector (mtf = the one live uses)")
    ap.add_argument("--regime-tf", dest="reg_tf", default="H1",
                    choices=("H1", "H4"))
    ap.add_argument("--regime-win", dest="reg_win", type=int, default=40,
                    help="lookback bars on the regime TF")
    ap.add_argument("--er-trend", dest="er_trend", type=float, default=0.35,
                    help="kaufman: ER above this = trending, below = range")
    ap.add_argument("--slope-k", dest="slope_k", type=float, default=0.0006,
                    help="slope: |per-bar %% slope| above this = trending")
    ap.add_argument("--er-hi", dest="er_hi", type=float, default=0.40,
                    help="hybrid: enter a trend when ER >= er_hi")
    ap.add_argument("--er-lo", dest="er_lo", type=float, default=0.25,
                    help="hybrid: leave a trend only when ER < er_lo (hysteresis)")
    ap.add_argument("--confirm", type=int, default=1,
                    help="bars a new regime must persist before committing")
    ap.add_argument("--out", default=None,
                    help="report path (default: calibrate_long.txt or _M5.txt)")
    args = ap.parse_args()
    args.tf = normalize_tf(args.tf)
    if args.out is None:
        args.out = default_calibrate_out(args.tf)

    reg_cfg = None
    if args.reg_det != "none":
        reg_cfg = {"detector": args.reg_det, "tf": args.reg_tf, "win": args.reg_win,
                   "er_trend": args.er_trend, "slope_k": args.slope_k,
                   "er_hi": args.er_hi, "er_lo": args.er_lo, "confirm": args.confirm}

    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 92)
    log("  LONG WALK-FORWARD CALIBRATION  (live logic & gold defaults untouched)")
    log(f"  tf={args.tf}  from={args.start}  scope={args.scope}  fold={args.fold_days}d  "
        f"source={args.source}  workers={args.workers}")
    if reg_cfg:
        log(f"  regime detector={args.reg_det}  tf={args.reg_tf}  win={args.reg_win}")
    log("=" * 92)

    summary = []
    for key in assets:
        try:
            summary.append(calibrate_asset(
                key, args.source, args.days, args.start,
                args.scope, args.fold_days, log,
                workers=args.workers, reg_cfg=reg_cfg, tf=args.tf))
        except FileNotFoundError as e:
            log(f"\n  {key}: SKIP — {e}")
        except Exception as e:
            log(f"\n  {key}: ERROR — {e}")

    log("\n" + "=" * 92)
    log("  SUMMARY — recommended rules_override per asset")
    log("=" * 92)
    for s in summary:
        log(f"  {s['asset']:<8}  tf={s.get('tf', '?')}  ROBUST {s['robust']}")
        log(f"  {'':<8}  PROFIT {s['profit']}")
        if s.get("regime_robust"):
            log(f"  {'':<8}  REGIME {s['regime_robust']}")
    log(f"\n  Generated: {datetime.now().isoformat(timespec='seconds')}")
    log("  Apply by editing ONLY rules_override per asset in symbol_profiles.py")
    log("=" * 92)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
