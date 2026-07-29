"""
TRUE out-of-sample (OOS) test — the honest verdict on whether the model has edge.

  1. SELECT rules using ONLY data up to --cutoff (default 2024-12-31)  [TRAIN]
  2. FREEZE those rules and TEST them on the UNSEEN period cutoff->now [TEST]
  3. Simulate ONE compounding account over the unseen period

  python oos_test.py --tf M15                         # default → reports/oos_test.*
  python oos_test.py --tf M5 --asset XAUUSD           # → reports/oos_test_M5.*
  python oos_test.py --cutoff 2024-12-31 --pick regime --top 5
  python oos_test.py --asset BTCUSD --workers 6
  python oos_test.py --source mt5 --days 400 --cutoff 2026-04-01 --tf M5
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import strategy as S
import symbol_profiles as P
import portfolio_config as C
import regime as RG
import calibrate_long as CL
import floating_config as FC

BAL = CL.BAL
MIN_N = CL.MIN_N


def default_oos_paths(tf: str) -> dict:
    """Canonical report paths per signal TF (M15 keeps legacy names)."""
    t = CL.normalize_tf(tf)
    if t == "M15":
        return {
            "txt": "reports/oos_test.txt",
            "json": "reports/oos_test.json",
            "chart": "reports/oos_equity.png",
            "csv": "reports/oos_trades.csv",
        }
    return {
        "txt": f"reports/oos_test_{t}.txt",
        "json": f"reports/oos_test_{t}.json",
        "chart": f"reports/oos_equity_{t}.png",
        "csv": f"reports/oos_trades_{t}.csv",
    }


# --------------------------------------------------------------------------- #
#  parallel: run every candidate rule ONCE over the full feed, return trades
# --------------------------------------------------------------------------- #
_W = {}


def _init_worker(m15, m1, htf_dfs, prof, spread, tf_min=15):
    _W["tf_min"] = tf_min
    _W["m15"] = m15
    _W["m1"] = m1
    _W["B"] = S.Bars(m15)
    _W["ctx"] = S.M1Ctx(m1, m15.index)
    _W["htf"] = S.prepare_htf_context(htf_dfs)
    _W["prof"] = prof
    _W["spread"] = spread


def _run_rule_worker(item):
    rule, fam = item
    try:
        trades = CL.run_rule(rule, _W["m15"], _W["m1"], _W["B"], _W["ctx"],
                             _W["htf"], _W["prof"], _W["spread"],
                             tf_min=_W.get("tf_min", 15))
    except Exception as exc:
        return rule, {"error": str(exc)[:80], "family": fam}
    return rule, trades


def run_all_rules(rules, m15, m1, B, ctx, htf, prof, spread, htf_dfs,
                  workers, key, log, tf_min=15):
    items = sorted(rules.items())
    n = len(items)
    out = {}
    if workers > 1:
        log(f"  parallel: {workers} worker processes over {n} rules")
        with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(m15, m1, htf_dfs, prof, spread, tf_min)) as ex:
            done = 0
            for rule, res in ex.map(_run_rule_worker, items):
                done += 1
                print(f"  [{key}] run {done}/{n}         ", end="\r", flush=True)
                out[rule] = res
    else:
        for i, (rule, fam) in enumerate(items, 1):
            print(f"  [{key}] run {i}/{n} {rule}        ", end="\r", flush=True)
            try:
                out[rule] = CL.run_rule(rule, m15, m1, B, ctx, htf, prof,
                                        spread, tf_min=tf_min)
            except Exception as exc:
                out[rule] = {"error": str(exc)[:80]}
    print(" " * 70, end="\r")
    return out


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def folds_between(start, end, fold_days):
    out = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur < end:
        nxt = cur + pd.Timedelta(days=fold_days)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def eval_window(trades, lo, hi, folds, B, prof, risk, spread, rmap):
    """Filter a rule's trades to [lo,hi), tag regimes, walk-forward evaluate."""
    sub = []
    for t in trades:
        ts = pd.Timestamp(t["time"])
        if (lo is None or ts >= lo) and (hi is None or ts < hi):
            sub.append(dict(t))
    if not sub:
        return None
    CL.tag_regimes(sub, rmap)
    return CL.evaluate(sub, folds, B, prof, risk, spread)


def collect_kept(trades, lo, hi, folds, B, prof):
    """The exact trades that survive each fold's trailing min-SL (live-like)."""
    kept = []
    for (f0, f1) in folds:
        msl = CL.trailing_min_sl(B, f0, prof)
        for t in trades:
            ts = pd.Timestamp(t["time"])
            if (lo is None or ts >= lo) and (hi is None or ts < hi) \
                    and f0 <= ts < f1 and t["risk"] >= msl:
                kept.append(dict(t))
    return kept


def pick_rules(train_eval, mode, top):
    items = [(r, v) for r, v in train_eval.items() if v and v["n"] >= MIN_N]
    # Rank on expectancy per trade, never on compounded return: the latter is
    # amplified by risk_pct and by the order the trades happened to arrive.
    if mode == "profit":
        items.sort(key=lambda kv: (-kv[1]["tot_r"], -kv[1].get("avg_r", 0.0)))
    elif mode == "robust":
        items.sort(key=lambda kv: (-kv[1]["robust_pct"],
                                   -kv[1].get("median_fold_r", 0.0), -kv[1]["pf"]))
    else:  # regime (default)
        items.sort(key=lambda kv: (-kv[1].get("regimes_pos", 0),
                                   -kv[1].get("worst_regime_r", -1e9),
                                   -kv[1]["robust_pct"]))
    return [r for r, _ in items[:top]]


def fmt(ev):
    if not ev:
        return "no trades"
    return (f"n={ev['n']:>4}  WR={ev['wr']:>5.1f}%  PF={min(ev['pf'],99.9):>5.2f}  "
            f"ret={ev['ret']:>+7.1f}%  DD={ev['dd']:>4.1f}%  robust={ev['robust_pct']:>3.0f}%")


# --------------------------------------------------------------------------- #
#  per asset: select on TRAIN, freeze, test on TEST
# --------------------------------------------------------------------------- #
def process_asset(key, source, days, start, cutoff, scope, fold_days, pick, top,
                  reg_cfg, workers, log, only_rules=None, tf="M15",
                  risk_override=None, end=None):
    import symbol_specs as X

    _, prof = P.get_profile(key)
    label = prof.get("label", key)
    log("\n" + "=" * 92)
    log(f"  {label.upper()}  ({key})    tf={tf}  cutoff={cutoff}   "
        f"pick={pick}   top={top}")
    log("=" * 92)

    if source == "mt5":
        m15, m1, htf_dfs = CL.load_mt5(key, days, tf=tf)
    else:
        m15, m1, htf_dfs = CL.load_dukascopy(key, start, tf=tf, end=end)

    B = S.Bars(m15)
    ctx = S.M1Ctx(m1, m15.index)
    htf = S.prepare_htf_context(htf_dfs)
    risk = (float(risk_override) if risk_override is not None
            else C.RISK_MAP.get(key, prof.get("risk_pct", 0.01)))
    spread = prof.get("spread", 0.0)
    rmap = CL.build_regime_map(htf_dfs, reg_cfg)
    spec = X.offline_spec(key)

    cut = pd.Timestamp(cutoff)
    t0, tend = m15.index[0], m15.index[-1]
    if not (t0 < cut < tend):
        log(f"  SKIP — cutoff {cutoff} not inside data range "
            f"{t0:%Y-%m-%d}..{tend:%Y-%m-%d}")
        return None
    train_folds = folds_between(t0, cut, fold_days)
    test_folds = folds_between(cut, tend, fold_days)
    log(f"  {tf} bars: {len(m15):,}   {t0:%Y-%m-%d} -> {tend:%Y-%m-%d}")
    log(f"  TRAIN {t0:%Y-%m-%d}..{cut:%Y-%m-%d} ({len(train_folds)} folds)   "
        f"TEST {cut:%Y-%m-%d}..{tend:%Y-%m-%d} ({len(test_folds)} folds)")
    log(f"  risk={risk*100:.2f}%  spread={spread}")

    # An explicit --rules request may name a quarantined rule on purpose (that
    # is how you re-check one after repairing it); automatic selection may not.
    rules = CL.collect_rules(scope, include_quarantined=bool(only_rules))
    if only_rules:
        want = {r.strip().upper() for r in only_rules}
        rules = {r: f for r, f in rules.items() if r.upper() in want}
        missing = want - {r.upper() for r in rules}
        if missing:
            log(f"  NOTE: rules not found in universe and ignored: {sorted(missing)}")
        if not rules:
            log("  SKIP — none of the requested --rules exist for this asset")
            return None
        held = sorted(r for r in rules if r in CL.QUARANTINED_RULES)
        if held:
            log(f"  WARNING: quarantined rule(s) included by request: {tuple(held)}")
        log(f"  restricted to {len(rules)} rule(s): {tuple(sorted(rules))}")
    else:
        log(f"  universe: {len(rules)} rules"
            f"  ({len(CL.QUARANTINED_RULES)} quarantined, see calibrate_long)")
    rule_trades = run_all_rules(rules, m15, m1, B, ctx, htf, prof, spread,
                                htf_dfs, workers, key, log,
                                tf_min=CL.tf_minutes(tf))
    rule_trades = {r: v for r, v in rule_trades.items()
                   if isinstance(v, list)}

    # ---- SELECT on TRAIN only ------------------------------------------------
    train_eval = {}
    for rule, trades in rule_trades.items():
        ev = eval_window(trades, t0, cut, train_folds, B, prof, risk, spread, rmap)
        if ev and ev["n"] >= MIN_N:
            train_eval[rule] = ev
    if not train_eval:
        log("  SKIP — no rules with enough TRAIN trades")
        return None
    picks = pick_rules(train_eval, pick, top)
    log(f"\n  FROZEN rules (chosen on TRAIN only): {tuple(picks)}")
    log(f"  {'rule':<14}{'TRAIN':<66}{'TEST'}")
    log("  " + "-" * 100)

    # ---- TEST the frozen rules (per rule, no re-selection) -------------------
    per_rule = {}
    for rule in picks:
        tr_ev = train_eval.get(rule)
        te_ev = eval_window(rule_trades[rule], cut, tend, test_folds,
                            B, prof, risk, spread, rmap)
        per_rule[rule] = {"train": tr_ev, "test": te_ev}
        log(f"  {rule:<14}{fmt(tr_ev):<66}{fmt(te_ev)}")

    # ---- SET-level TRAIN vs TEST (all frozen rules merged) -------------------
    def merged(lo, hi, folds):
        m = []
        for r in picks:
            for t in rule_trades[r]:
                ts = pd.Timestamp(t["time"])
                if lo <= ts < hi:
                    m.append(dict(t))
        CL.tag_regimes(m, rmap)
        return CL.evaluate(m, folds, B, prof, risk, spread)

    set_train = merged(t0, cut, train_folds)
    set_test = merged(cut, tend, test_folds)
    log("  " + "-" * 100)
    log(f"  {'SET (all)':<14}{fmt(set_train):<66}{fmt(set_test)}")

    # ---- kept TEST trades for the combined portfolio sim ---------------------
    kept = []
    for rule in picks:
        for t in collect_kept(rule_trades[rule], cut, tend, test_folds, B, prof):
            tc = dict(t)
            tc["_asset"] = key
            tc["_risk_pct"] = risk
            tc["_spread"] = spread
            tc["_max_concurrent"] = C.MAX_CONCURRENT_PER_ASSET
            tc["_spec"] = spec
            kept.append(tc)

    return {"asset": key, "tf": tf, "picks": list(picks),
            "set_train": set_train, "set_test": set_test,
            "per_rule": per_rule, "kept_test": kept}


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--tf", default="M15",
                    help="Signal timeframe: M5 or M15 (default M15)")
    ap.add_argument("--source", choices=("dukascopy", "mt5"), default="dukascopy")
    ap.add_argument("--from", dest="start", default=CL.DEFAULT_FROM)
    ap.add_argument("--to", dest="end", default=None,
                    help="clip data END (YYYY-MM-DD). Use with --from/--cutoff to "
                         "score a specific historical era, e.g. the 2012-2015 bear")
    ap.add_argument("--cutoff", default="2024-12-31",
                    help="TRAIN uses data < cutoff; TEST uses data >= cutoff")
    ap.add_argument("--days", type=int, default=400, help="only for --source mt5")
    ap.add_argument("--scope", choices=("core", "dense", "full"), default="dense")
    ap.add_argument("--rules", default=None,
                    help="comma-separated rule keys to test ONLY these "
                         "(e.g. CH_REV_L,CH_REV_S,CH_BO_L,CH_BO_S)")
    ap.add_argument("--fold-days", type=int, default=90)
    ap.add_argument("--pick", choices=("regime", "robust", "profit"),
                    default="regime", help="how to choose rules on TRAIN")
    ap.add_argument("--top", type=int, default=5, help="rules kept per asset")
    ap.add_argument("--min-n", dest="min_n", type=int, default=CL.MIN_N,
                    help=f"min TRAIN trades before a rule is eligible "
                         f"(default {CL.MIN_N})")
    ap.add_argument("--balance", type=float, default=BAL)
    ap.add_argument("--risk", type=float, default=None,
                    help="override risk fraction per trade (e.g. 0.005 = 0.5%%). "
                         "Default: RISK_MAP / profile")
    ap.add_argument("--flat", type=float, default=None, metavar="LOT",
                    help="fixed lot size, no compounding (e.g. --flat 0.10). "
                         "P&L uses offline contract $/unit × lot")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    # Regime defaults come from floating_config.REGIME, which is what the LIVE
    # bot uses. If these drift apart the meta-gate is keyed on regime labels
    # live never produces, and the whole gate silently misfires.
    _R = FC.REGIME
    ap.add_argument("--regime-detector", dest="reg_det",
                    default=_R.get("regime_detector", "mtf"),
                    choices=("kaufman", "slope", "donchian", "ma", "hybrid",
                             "mtf", "none"))
    ap.add_argument("--regime-tf", dest="reg_tf",
                    default=_R.get("regime_tf", "H1"), choices=("H1", "H4"))
    ap.add_argument("--regime-win", dest="reg_win", type=int,
                    default=_R.get("regime_win", 40))
    ap.add_argument("--er-trend", dest="er_trend", type=float,
                    default=_R.get("regime_er_trend", 0.35))
    ap.add_argument("--slope-k", dest="slope_k", type=float,
                    default=_R.get("regime_slope_k", 0.0006))
    ap.add_argument("--er-hi", dest="er_hi", type=float,
                    default=_R.get("regime_er_hi", 0.32))
    ap.add_argument("--er-lo", dest="er_lo", type=float,
                    default=_R.get("regime_er_lo", 0.20))
    ap.add_argument("--confirm", type=int, default=_R.get("regime_confirm", 2))
    ap.add_argument("--h4-win", dest="h4_win", type=int,
                    default=_R.get("regime_h4_win", 30),
                    help="mtf only: H4 swing window")
    ap.add_argument("--h4-lookback", dest="h4_lookback", type=int,
                    default=_R.get("regime_h4_lookback", 60),
                    help="mtf only: H4 structure lookback")
    ap.add_argument("--out", default=None, help="txt report (default per --tf)")
    ap.add_argument("--chart", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    args.tf = CL.normalize_tf(args.tf)
    global MIN_N
    MIN_N = int(args.min_n)
    if args.risk is not None and args.risk <= 0:
        ap.error("--risk must be > 0 (e.g. 0.005 for 0.5%)")
    if args.flat is not None and args.flat <= 0:
        ap.error("--flat LOT must be > 0 (e.g. 0.10)")
    paths = default_oos_paths(args.tf)
    if args.out is None:
        args.out = paths["txt"]
    if args.chart is None:
        args.chart = paths["chart"]
    if args.csv is None:
        args.csv = paths["csv"]

    reg_cfg = None
    if args.reg_det != "none":
        _R = FC.REGIME
        reg_cfg = {"detector": args.reg_det, "tf": args.reg_tf, "win": args.reg_win,
                   "er_trend": args.er_trend, "slope_k": args.slope_k,
                   "er_hi": args.er_hi, "er_lo": args.er_lo, "confirm": args.confirm,
                   "h4_win": args.h4_win, "h4_lookback": args.h4_lookback,
                   "structure_break": _R.get("regime_structure_break", False),
                   "brk_win": _R.get("regime_brk_win", 40),
                   "shock_win": _R.get("regime_shock_win", 24),
                   "shock_baseline": _R.get("regime_shock_baseline", 720)}

    only_rules = ([r for r in args.rules.split(",") if r.strip()]
                  if args.rules else None)
    assets = [args.asset.upper()] if args.asset else list(C.PORTFOLIO_ASSETS)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 92)
    log("  OUT-OF-SAMPLE TEST — rules chosen on TRAIN only, judged on UNSEEN data")
    log(f"  tf={args.tf}  from={args.start}"
        + (f"  to={args.end}" if args.end else "")
        + f"  cutoff={args.cutoff}  scope={args.scope}  "
          f"pick={args.pick}  top={args.top}  min_n={MIN_N}  source={args.source}")
    if args.risk is not None:
        log(f"  risk override = {args.risk*100:.3f}% per trade")
    if args.flat is not None:
        log(f"  sizing = FIXED LOT {args.flat:g} (no compounding)")
    elif args.risk is not None:
        log(f"  sizing = compound at risk={args.risk*100:.3f}%")
    else:
        log("  sizing = compound at RISK_MAP (default)")
    if reg_cfg:
        log(f"  regime detector={args.reg_det} tf={args.reg_tf} win={args.reg_win} "
            f"er_hi={args.er_hi} er_lo={args.er_lo} confirm={args.confirm}"
            + (f" h4_win={args.h4_win} h4_lookback={args.h4_lookback}"
               if args.reg_det == "mtf" else ""))
        if args.reg_det != FC.REGIME.get("regime_detector"):
            log(f"  WARNING: live uses '{FC.REGIME.get('regime_detector')}' — this "
                f"gate will be keyed on regimes live does not produce.")
    if only_rules:
        log(f"  RULES FILTER: testing only {only_rules}")
    log("=" * 92)

    results = []
    for key in assets:
        try:
            r = process_asset(
                key, args.source, args.days, args.start, args.cutoff,
                args.scope, args.fold_days, args.pick, args.top,
                reg_cfg, args.workers, log, only_rules=only_rules, tf=args.tf,
                risk_override=args.risk, end=args.end)
            if r:
                results.append(r)
        except FileNotFoundError as e:
            log(f"\n  {key}: SKIP — {e}")
        except Exception as e:
            log(f"\n  {key}: ERROR — {e}")

    # ---- combined portfolio on the UNSEEN period -----------------------------
    import portfolio_backtest as PB
    all_kept = []
    for r in results:
        all_kept.extend(r["kept_test"])
    all_kept.sort(key=lambda t: (t["time"], t.get("exit_time")))

    log("\n" + "=" * 92)
    log("  TRAIN vs TEST per asset (frozen rule SET)")
    log("=" * 92)
    log(f"  {'asset':<10}{'TRAIN':<66}{'TEST'}")
    log("  " + "-" * 100)
    for r in results:
        log(f"  {r['asset']:<10}{fmt(r['set_train']):<66}{fmt(r['set_test'])}")

    if all_kept:
        if args.flat is not None:
            port = PB.simulate_portfolio(
                all_kept, args.balance,
                max_portfolio_risk_pct=None,
                fixed_lot=args.flat)
            size_note = f"FIXED LOT {args.flat:g}, no compounding"
        else:
            port = PB.simulate_portfolio(
                all_kept, args.balance,
                max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK)
            if args.risk is not None:
                size_note = (f"compound, risk={args.risk*100:.3f}%, "
                             f"cap {C.MAX_PORTFOLIO_RISK*100:.0f}%")
            else:
                size_note = (f"compound, RISK_MAP, "
                             f"cap {C.MAX_PORTFOLIO_RISK*100:.0f}%")
        log("\n" + "=" * 92)
        log(f"  COMBINED ACCOUNT on UNSEEN data ({size_note})")
        log("=" * 92)
        log(f"  Trades: {port['n']}  |  WR: {port['wr']:.1f}%  |  "
            f"PF: {port['pf']:.2f}  |  Max DD: {port['max_dd']:.1f}%")
        log(f"  ${args.balance:,.2f}  ->  ${port['final']:,.2f}  |  "
            f"Profit ${port['profit']:+,.2f}  ({port['ret_pct']:+.1f}%)")
        mb_t = port.get("min_balance_time")
        mb_t_s = (str(mb_t)[:16] if mb_t is not None else "—")
        log(f"  Lowest balance: ${port.get('min_balance', args.balance):,.2f}  "
            f"at {mb_t_s}")
        log(f"  Absolute DD:    ${port.get('abs_dd', 0.0):,.2f}  "
            f"({port.get('abs_dd_pct', 0.0):.1f}% of deposit)  |  "
            f"Relative max DD: {port['max_dd']:.1f}%")
        log(f"  {'asset':<10}{'trades':>8}{'net $':>14}")
        log("  " + "-" * 34)
        for r in results:
            ba = port["by_asset"].get(r["asset"], {})
            log(f"  {r['asset']:<10}{ba.get('n', 0):>8}{ba.get('net', 0):>+14.2f}")
        if port.get("monthly"):
            log("\n  Monthly P&L ($) on unseen data:")
            for m in sorted(port["monthly"]):
                log(f"    {m}:  {port['monthly'][m]:>+10.2f}")

        # ---- expectancy: the leverage-free view ------------------------------
        Rs = np.array([float(t["R"]) for t in all_kept if t.get("R") is not None])
        avg_r = float(Rs.mean()) if len(Rs) else 0.0
        t_stat = (avg_r / (Rs.std(ddof=1) / np.sqrt(len(Rs)))
                  if len(Rs) > 2 and Rs.std(ddof=1) > 0 else 0.0)
        flat_r = np.cumsum(Rs * 0.005)
        flat_dd = (float(np.max(np.maximum.accumulate(flat_r) - flat_r) * 100)
                   if len(Rs) else 0.0)
        log("\n" + "=" * 92)
        log("  EXPECTANCY on UNSEEN data (independent of risk_pct and of trade order)")
        log("=" * 92)
        log(f"  avg R = {avg_r:+.3f}   t-stat = {t_stat:.2f}   n = {len(Rs)}")
        log(f"  at a flat 0.5% risk, no compounding: "
            f"{flat_r[-1] * 100 if len(Rs) else 0:+.1f}%  |  max DD {flat_dd:.1f}%")

        # ---- verdict ---------------------------------------------------------
        log("\n" + "=" * 92)
        pf = port["pf"]
        # Judge on expectancy + significance, never on the compounded curve:
        # at 4% risk a thin edge still prints an absurd final balance.
        if avg_r >= 0.10 and t_stat >= 3.0 and pf >= 1.15:
            verdict = ("EDGE HOLDS OUT-OF-SAMPLE. Expectancy is positive and "
                       "statistically solid on data the rules never saw. Worth a "
                       "fixed-config forward test on demo before any real money.")
        elif avg_r > 0 and t_stat >= 2.0:
            verdict = ("MARGINAL. Positive expectancy but thin — a bad regime or "
                       "a wider spread could erase it. Do not scale up risk.")
        elif avg_r > 0:
            verdict = ("NOT PROVEN. Expectancy is positive but within noise "
                       f"(t={t_stat:.1f}); this is not yet evidence of an edge.")
        else:
            verdict = ("NO REAL EDGE OUT-OF-SAMPLE. The backtest profit did NOT "
                       "survive on unseen data => it was overfit. Do NOT risk real "
                       "money on this configuration.")
        log(f"  VERDICT: {verdict}")
        log("=" * 92)

        try:
            PB.export_equity_chart(port["equity_curve"], args.balance, port,
                                   args.chart)
            log(f"  Saved chart: {args.chart}")
        except Exception as e:
            log(f"  (chart skipped: {e})")
        try:
            PB.export_ledger_csv(port["ledger"], args.csv)
            log(f"  Saved CSV:   {args.csv}")
        except Exception as e:
            log(f"  (csv skipped: {e})")
    else:
        log("\n  No test-period trades produced — nothing to simulate.")

    log(f"\n  Generated: {datetime.now().isoformat(timespec='seconds')}")
    log(f"  Meta-gate file for live --tf {args.tf}: {paths['json']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "kept_test"}
                   for r in results], f, indent=2, default=str)
    print(f"\n  Saved: {args.out}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
