"""
Honest back-test — reads data DIRECTLY from MetaTrader 5 (no CSV needed).

Usage:
  python backtest.py --optimized --days 180          # STABLE: OB + NDS + DEM
  python backtest.py --nds-only --days 180           # NDS nested zones (OB+NDS)
  python backtest.py --optimized --compare-nds --days 180   # STABLE vs NDS side-by-side

NDS = Nested Demand/Supply: M15 zone inside H4/H1 zone (same direction).

Requirements: MetaTrader 5 must be open and logged in.
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X

SIGNAL_TFS = ["M5", "M15", "M30", "H1", "H4"]
EXIT_LABELS = {
    "trail": "Trail only (SL + trail after +1R)",
    "trail-tp": "Trail + structural TP",
    "be1r-tp": "BE at +1R + structural TP (no trail)",
    "be1r-fixed-tp": "BE at +1R + fixed TP (OPTIMIZED)",
}


def filter_trades(trades, cutoff, date_to=None):
    if cutoff is not None:
        trades = [t for t in trades if t["time"] >= cutoff]
    if date_to is not None:
        hi = np.datetime64(date_to)
        trades = [t for t in trades if t["time"] <= hi]
    return trades


def print_trade_ledger(trades, balance, risk, spread, title="TRADE LIST"):
    rows = S.build_trade_ledger(trades, start_balance=balance, risk_pct=risk,
                                spread_price=spread)
    if not rows:
        return
    print(f"\n  --- {title} ({len(rows)} trades) ---")
    print(f"  {'#':>3}  {'entry':<17}  {'rule':<10}  {'side':<5}  "
          f"{'entry$':>10}  {'exit$':>10}  {'R':>6}  {'tpR':>5}  {'P&L$':>9}  {'bal$':>9}  exit")
    print("  " + "-" * 100)
    tot = 0.0
    for r in rows:
        tot += r["net_usd"]
        et = pd_ts(r["entry_time"])
        side = "LONG" if r["dir"] == "long" else "SHORT"
        tp_r_s = f"{r['tp_r']:.1f}" if r.get("tp_r") is not None else "  -"
        print(f"  {r['n']:>3}  {et}  {r['rule']:<10}  {side:<5}  "
              f"{r['entry']:>10.2f}  {r['exit_px']:>10.2f}  {r['R']:>+6.2f}  "
              f"{tp_r_s:>5}  {r['net_usd']:>+9.2f}  {r['balance']:>9.2f}  {r['exit']}")
    print("  " + "-" * 100)
    print(f"  {'TOTAL':>3}  {'':17}  {'':6}  {'':5}  {'':8}  {'':8}  {'':6}  "
          f"{tot:>+8.2f}")


def pd_ts(ts):
    return str(ts)[:16]


def print_summary(label, trades, balance, risk, spread, show_trades=False):
    r = S.simulate_account(trades, start_balance=balance, risk_pct=risk,
                           spread_price=spread)
    print(f"\n  --- {label} ---")
    if r is None or r["n"] == 0:
        print("  No trades in this period.")
        return r
    print(f"  Trades: {r['n']}  |  Win-rate: {r['wr']:.1f}%  |  PF: {r['pf']:.2f}")
    print(f"  Final balance: ${r['final']:,.2f}   (return {r['ret_pct']:+.1f}%)")
    print(f"  Max drawdown:  {r['max_dd']:.1f}%")
    stats = S.exit_stats(trades)
    if stats:
        partial_n = stats.get("partial", 0)
        extra = f"  partial={partial_n}" if partial_n else ""
        print(f"  Exits: SL={stats.get('sl', 0)}  TP={stats.get('tp', 0)}  "
              f"BE={stats.get('be', 0)}  time={stats.get('time', 0)}{extra}")
    tp_rs = [t["tp_r"] for t in trades if t.get("tp_r") is not None]
    if tp_rs:
        print(f"  Avg TP target: {np.mean(tp_rs):.2f}R")
    srcs = {}
    for t in trades:
        s = t.get("tp_source") or "?"
        srcs[s] = srcs.get(s, 0) + 1
    if srcs:
        print(f"  TP sources: " + "  ".join(f"{k}={v}" for k, v in sorted(srcs.items())))
    rows = S.build_trade_ledger(trades, start_balance=balance, risk_pct=risk,
                                spread_price=spread)
    if rows:
        nets = [x["net_usd"] for x in rows]
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x < 0]
        print(f"  P&L per trade: max win ${max(wins):+.2f}  |  max loss ${min(losses):+.2f}"
              if wins and losses else "")
        if wins:
            print(f"  Avg WIN ${np.mean(wins):+.2f} ({len(wins)} wins)  |  "
                  f"Avg LOSS ${np.mean(losses):+.2f} ({len(losses)} losses)" if losses else
                  f"  Avg WIN ${np.mean(wins):+.2f} ({len(wins)} wins)")
        be_n = sum(1 for x in rows if abs(x["net_usd"]) < 0.50)
        if be_n:
            print(f"  Near-zero trades (BE/spread): {be_n}  — these drag avg $/trade down")
    avg_r = np.mean([t["R"] for t in trades])
    print(f"  Avg R per trade: {avg_r:+.2f}")
    if show_trades:
        print_trade_ledger(trades, balance, risk, spread, title=f"TRADES — {label}")
    return r


def run_bt(d, m1, B, ctx, tf_min, cutoff, date_to, spread, balance, risk, **kw):
    all_trades = S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=tf_min, **kw)
    return filter_trades(all_trades, cutoff, date_to)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M15", choices=SIGNAL_TFS)
    ap.add_argument("--symbol", default=M.DEFAULT_SYMBOL)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--bars", type=int, default=None)
    ap.add_argument("--m1-bars", type=int, default=None)
    ap.add_argument("--risk", type=float, default=S.RISK_PCT)
    ap.add_argument("--spread", type=float, default=None)
    ap.add_argument("--balance", type=float, default=S.START_BALANCE)
    ap.add_argument("--optimized", action="store_true",
                    help="STABLE preset: OB+DEM, fib, ATR regime, HTF blend TP (default)")
    ap.add_argument("--aggressive", action="store_true",
                    help="Legacy: FVG+OB+DEM, fixed TP 2.5R, BE@2R (old --optimized)")
    ap.add_argument("--high-wr", action="store_true",
                    help="With --optimized: OB+DEMAND only (~54%% WR, fewer trades)")
    ap.add_argument("--more-trades", action="store_true",
                    help="With --optimized: FVG+OB+DEM, H1 trend, session 8-22, min SL $2")
    ap.add_argument("--quality", action="store_true",
                    help="With --optimized: FVG confluence + BOS confirm (~48-58%% WR)")
    ap.add_argument("--fib", action="store_true",
                    help="With --optimized: fib 38-78%% retrace filter (better WR + PF)")
    ap.add_argument("--nds", action="store_true",
                    help="With --optimized: add NDS rule alongside OB/DEM")
    ap.add_argument("--nds-mode", action="store_true",
                    help="STABLE with OB+NDS only (nested zones inside H4/H1)")
    ap.add_argument("--no-nds", action="store_true",
                    help="With --optimized: disable NDS (OB+DEM only)")
    ap.add_argument("--nds-only", action="store_true",
                    help="Shortcut: same as --optimized --nds-mode (OB+NDS only)")
    ap.add_argument("--compare-nds", action="store_true",
                    help="With --optimized: compare STABLE (OB+DEM) vs NDS mode (OB+NDS)")
    ap.add_argument("--partial-tp", action="store_true",
                    help="50%% close at TP, trail remaining 50%% (1R trail after TP)")
    ap.add_argument("--compare-partial-tp", action="store_true",
                    help="Compare full TP vs partial TP (50%% + trail)")
    ap.add_argument("--nds-extended", action="store_true",
                    help="STABLE + full NDS family (OB/FVG/BOS/WYCK/QM/FRESH nested in HTF)")
    ap.add_argument("--compare-nds-extended", action="store_true",
                    help="Compare STABLE vs NDS extended family")
    ap.add_argument("--balanced", action="store_true",
                    help="STABLE + 2 concurrent (~14 trades/180d, ~86%% WR)")
    ap.add_argument("--balanced-plus", action="store_true",
                    help="BALANCED + NDS-FVG + session 8-22 (~18 trades, ~78%% WR)")
    ap.add_argument("--compare-balanced", action="store_true",
                    help="Compare STABLE vs --balanced vs --balanced-plus")
    ap.add_argument("--dense", action="store_true",
                    help="STABLE + 2 concurrent + fib exempt for nested (~17 trades, ~82%% WR)")
    ap.add_argument("--dense-plus", action="store_true",
                    help="DENSE + no ATR regime (~18 trades, ~78%% WR)")
    ap.add_argument("--dense-wide", action="store_true",
                    help="DENSE + no ATR + session 8-22 (~24 trades, ~71%% WR)")
    ap.add_argument("--spike", action="store_true",
                    help="Add pre-spike rules (SPIKE-BRK + SPIKE-SQS) — compression→expansion")
    ap.add_argument("--asset", default=None,
                    choices=list(P.ASSET_KEYS),
                    help="Apply calibrated profile for this asset (min SL, spread, params)")
    ap.add_argument("--multi", action="store_true",
                    help="Run dense-wide backtest on all calibrated assets (BTC, Brent, EUR, Gold)")
    ap.add_argument("--assets", default=None,
                    help="With --multi: comma list e.g. XAUUSD,BRENT (default: all)")
    ap.add_argument("--compare-dense", action="store_true",
                    help="Compare STABLE vs BALANCED vs DENSE")
    ap.add_argument("--compare-dense-tiers", action="store_true",
                    help="Compare DENSE vs DENSE+ vs DENSE-WIDE vs MEDIUM")
    ap.add_argument("--medium", action="store_true",
                    help="DENSE + session 8-22 (~23 trades/180d, ~74%% WR)")
    ap.add_argument("--compare-medium", action="store_true",
                    help="Compare STABLE vs DENSE vs MEDIUM")
    ap.add_argument("--partial-frac", type=float, default=0.5,
                    help="Fraction to close at TP when using --partial-tp (default 0.5)")
    ap.add_argument("--compare-old", action="store_true",
                    help="With --optimized: also run old all-rules trail exit")
    ap.add_argument("--exit", choices=["trail", "trail-tp", "be1r-tp", "be1r-fixed-tp"],
                    default="trail")
    ap.add_argument("--compare-exits", action="store_true")
    ap.add_argument("--tp-min-r", type=float, default=0.5)
    ap.add_argument("--tp-mode",
                    choices=["fixed", "htf", "htf_blend", "structural", "next-rule"],
                    default=None,
                    help="TP: htf_blend = max(2R, nearest H4/H1 zone)")
    ap.add_argument("--compare-tp-mode", action="store_true")
    ap.add_argument("--compare-tp", action="store_true",
                    help="Compare fixed 2R vs HTF blend (--optimized)")
    ap.add_argument("--fixed-tp-r", type=float, default=None,
                    help="Fixed TP in R (default 2.0 with --optimized)")
    ap.add_argument("--min-risk", type=float, default=None,
                    help="Min SL distance in price units (default 2.0 with --optimized)")
    ap.add_argument("--session-start", type=int, default=None)
    ap.add_argument("--session-end", type=int, default=None)
    ap.add_argument("--no-trades", action="store_true",
                    help="Hide the per-trade P&L list")
    args = ap.parse_args()
    show_trades = not args.no_trades

    if args.nds_only:
        args.optimized = True
        args.nds_mode = True
    if args.compare_nds:
        args.optimized = True
    if args.compare_partial_tp:
        args.optimized = True
    if args.compare_nds_extended or args.nds_extended:
        args.optimized = True
    if args.balanced or args.balanced_plus or args.compare_balanced:
        args.optimized = True
    if args.dense or args.compare_dense or args.dense_plus or args.dense_wide or args.compare_dense_tiers or args.spike or args.multi:
        args.optimized = True
    if args.medium or args.compare_medium:
        args.optimized = True
    if args.multi and not args.dense and not args.dense_wide:
        pass  # --multi uses per-asset live_preset from symbol_profiles
    elif args.multi and not args.dense_wide:
        args.dense_wide = True

    opt = S.optimized_settings(high_wr=args.high_wr, more_trades=args.more_trades,
                               quality=args.quality, fib=args.fib,
                               aggressive=args.aggressive, nds=args.nds,
                               nds_mode=args.nds_mode, no_nds=args.no_nds,
                               nds_extended=args.nds_extended or args.compare_nds_extended,
                               balanced=args.balanced, balanced_plus=args.balanced_plus,
                               dense=args.dense and not args.compare_dense and not args.compare_medium
                               and not args.compare_dense_tiers and not args.dense_plus
                               and not args.dense_wide and not args.spike and not args.multi,
                               medium=args.medium and not args.compare_medium,
                               dense_plus=args.dense_plus and not args.compare_dense_tiers,
                               dense_wide=(args.dense_wide or args.multi) and not args.compare_dense_tiers,
                               spike=args.spike) if args.optimized else None

    if args.multi:
        import types
        from multi_symbol_calibrate import run_multi
        run_multi(types.SimpleNamespace(
            days=args.days or 90, risk=args.risk, spike=args.spike, calibrate=False,
            no_trades=args.no_trades, balance=args.balance, assets=args.assets,
            risk_override=("--risk" in sys.argv)))
        return

    profile = None
    if args.asset and args.optimized:
        _, profile = P.get_profile(args.asset)
        opt = P.resolve_live_opt(
            profile, spike=args.spike,
            cli_dense=args.dense, cli_dense_wide=args.dense_wide)
        opt = P.apply_profile_to_settings(opt, profile)
        if args.min_risk is None:
            args.min_risk = profile["min_sl"]
        if args.session_start is None and profile.get("session"):
            args.session_start, args.session_end = profile["session"]
        if "--risk" not in sys.argv:
            args.risk = X.profile_risk_pct(profile, args.risk)
    elif args.asset and opt:
        _, profile = P.get_profile(args.asset)
        opt = P.apply_profile_to_settings(opt, profile)
        if "--risk" not in sys.argv:
            args.risk = X.profile_risk_pct(profile, args.risk)
    spread = args.spread if args.spread is not None else (
        profile["spread"] if profile else (opt["spread_usd"] if opt else S.SPREAD_USD))

    if args.optimized and args.tf == "M15" and "--tf" not in sys.argv:
        pass  # default M15 is fine
    if args.optimized:
        args.exit = opt["exit_mode"]
        if args.fixed_tp_r is None:
            args.fixed_tp_r = opt["fixed_tp_r"]
        if args.min_risk is None:
            args.min_risk = opt["min_risk_usd"]
        if args.session_start is None:
            args.session_start = opt["session_start"]
        if args.session_end is None:
            args.session_end = opt["session_end"]
        if args.tp_mode is None:
            args.tp_mode = opt["tp_mode"]
    if args.tp_mode is None:
        args.tp_mode = "structural"

    if args.days is None and args.date_from is None:
        period = "all downloaded history"
    elif args.days and not args.date_from:
        period = f"last {args.days} days"
    else:
        period = f"{args.date_from or 'start'} -> {args.date_to or 'latest'}"

    tf_min = M.tf_minutes(args.tf)

    print("[MT5] connecting...")
    mt5 = M.connect()
    try:
        if args.asset:
            pref = args.symbol if P._symbol_matches_profile(args.symbol, profile or P.get_profile(args.asset)[1]) else None
            sym = P.resolve_symbol_for_profile(args.asset, mt5, preferred=pref)
        else:
            sym = M.resolve_symbol(args.symbol, mt5)
        print(f"[MT5] symbol: {sym}")
        print(f"[MT5] downloading {args.tf} + M1 + HTF | test window: {period} ...")
        _, d, m1, cutoff = M.fetch_pair(
            sym, args.tf, m1_count=args.m1_bars, signal_count=args.bars, mt5=mt5,
            days=args.days if not args.date_from else None,
            date_from=args.date_from, date_to=args.date_to,
        )
        htf_tfs = tuple(opt["htf_tfs"]) if opt else ("H4", "H1")
        need_htf = args.tp_mode in ("htf", "htf_blend") or args.compare_tp or bool(opt)
        htf_dfs = {}
        if need_htf:
            _, htf_dfs = M.fetch_htf_bars(
                sym, days=args.days if not args.date_from else None,
                mt5=mt5, tfs=htf_tfs,
            )
        if profile:
            spread = P.live_spread(sym, profile, mt5)
    finally:
        M.shutdown(mt5)

    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf_context = S.prepare_htf_context(htf_dfs) if htf_dfs else None

    eff_min_sl = args.min_risk or 0.0
    if profile and profile.get("min_sl_mode") == "atr":
        eff_min_sl = X.calibrate_min_sl(B, cutoff, profile)
        args.min_risk = eff_min_sl

    base_kw = dict(
        tp_min_r=args.tp_min_r, exit_mode=args.exit, tp_mode=args.tp_mode,
        fixed_tp_r=args.fixed_tp_r, min_risk_usd=eff_min_sl,
        session_start=args.session_start, session_end=args.session_end,
        htf_context=htf_context,
        min_tp_r=opt["min_tp_r"] if opt else S.OPT_MIN_TP_R,
        max_tp_r=opt["max_tp_r"] if opt else S.OPT_MAX_TP_R,
        htf_lookback=opt["htf_lookback"] if opt else S.OPT_HTF_LOOKBACK,
        htf_tfs=tuple(opt["htf_tfs"]) if opt else S.OPT_HTF_TFS,
        be_trigger=opt["be_trigger"] if opt else S.TRAIL_TRIGGER,
        max_hold=opt.get("max_hold", S.MAX_HOLD) if opt else S.MAX_HOLD,
        htf_trend=opt["htf_trend"] if opt else False,
        htf_ema=opt["htf_ema"] if opt else 20,
        require_confluence=opt.get("require_confluence", False) if opt else False,
        require_bos=opt.get("require_bos", False) if opt else False,
        require_fib=opt.get("require_fib", False) if opt else False,
        require_atr_regime=opt.get("require_atr_regime", False) if opt else False,
        atr_max_ratio=opt.get("atr_max_ratio", S.OPT_STABLE_ATR_RATIO) if opt else S.OPT_STABLE_ATR_RATIO,
        atr_lookback=opt.get("atr_lookback", S.OPT_STABLE_ATR_LB) if opt else S.OPT_STABLE_ATR_LB,
        nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO) if opt else S.OPT_NDS_MAX_RATIO,
        fib_nds_exempt=opt.get("fib_nds_exempt", False) if opt else False,
        fib_ob_only=opt.get("fib_ob_only", False) if opt else False,
        fib_spike_exempt=opt.get("fib_spike_exempt", False) if opt else False,
        spike_mode=opt.get("spike_mode", False) if opt else False,
        spike_params=opt.get("spike_params") if opt else None,
        detector_params=P.merge_params(S.DEFAULT_PARAMS, profile) if profile else None,
        atr_nds_only=opt.get("atr_nds_only", False) if opt else False,
        htf_trend_tfs=opt.get("htf_trend_tfs") if opt else None,
        partial_tp=args.partial_tp,
        partial_frac=args.partial_frac,
    )
    if opt:
        base_kw.update(enabled=opt["enabled"], max_concurrent=opt["max_concurrent"],
                       vp_mode=opt["vp_mode"])

    common = dict(d=d, m1=m1, B=B, ctx=ctx, tf_min=tf_min, cutoff=cutoff,
                  date_to=args.date_to, spread=spread, balance=args.balance,
                  risk=args.risk)

    if args.compare_old and args.optimized:
        trades_old = run_bt(exit_mode="trail", enabled=list(S.DETECTORS.keys()),
                            max_concurrent=S.MAX_CONCURRENT, min_risk_usd=0.0,
                            session_start=None, session_end=None, **common)
        trades_new = run_bt(**base_kw, **common)
        trades = trades_new
        mode_label = "OPTIMIZED vs OLD (all rules + trail)"
    elif args.compare_exits:
        modes = ["trail", "trail-tp", "be1r-tp", "be1r-fixed-tp"]
        results = {m: run_bt(exit_mode=m, **base_kw, **common) for m in modes}
        trades = results[args.exit]
        mode_label = "COMPARE exits"
    elif args.compare_tp:
        kw = {k: v for k, v in base_kw.items() if k != "tp_mode"}
        trades_fixed = run_bt(tp_mode="fixed", **kw, **common)
        trades_htf = run_bt(tp_mode="htf_blend", **kw, **common)
        trades = trades_htf
        mode_label = "COMPARE TP: fixed 2R vs HTF blend (H4/H1)"
    elif args.compare_tp_mode:
        if args.exit == "trail":
            args.exit = "be1r-tp"
        trades_struct = run_bt(exit_mode=args.exit, tp_mode="structural", **base_kw, **common)
        trades_next = run_bt(exit_mode=args.exit, tp_mode="next-rule", **base_kw, **common)
        trades = trades_next
        mode_label = "COMPARE TP modes"
    elif args.compare_nds and opt:
        kw_stable = {**base_kw, "enabled": ["OB", "DEMAND"]}
        kw_full = {**base_kw, "enabled": list(S.OPT_STABLE_RULES)}
        trades_stable = run_bt(**kw_stable, **common)
        trades_full = run_bt(**kw_full, **common)
        trades = trades_full
        mode_label = "COMPARE: OB+DEM vs OB+NDS+DEM (full STABLE)"
    elif args.compare_partial_tp and opt:
        kw_full = {**base_kw, "partial_tp": False}
        kw_part = {**base_kw, "partial_tp": True, "partial_frac": args.partial_frac}
        trades_full = run_bt(**kw_full, **common)
        trades_partial = run_bt(**kw_part, **common)
        trades = trades_partial
        mode_label = (f"COMPARE: full TP vs partial {args.partial_frac*100:.0f}% @ TP + trail")
    elif args.compare_nds_extended and opt:
        kw_stable = {**base_kw, "enabled": list(S.OPT_STABLE_RULES)}
        kw_ext = {**base_kw, "enabled": list(S.OPT_NDS_EXTENDED)}
        trades_stable = run_bt(**kw_stable, **common)
        trades_ext = run_bt(**kw_ext, **common)
        trades = trades_ext
        mode_label = "COMPARE: STABLE vs NDS EXTENDED (all nested triggers)"
    elif args.compare_balanced and opt:
        skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
                "dense", "medium", "dense_plus", "dense_wide")
        kw_stable = {**base_kw, **{k: v for k, v in S.stable_settings().items() if k not in skip}}
        kw_bal = {**base_kw, **{k: v for k, v in S.balanced_settings(False).items() if k not in skip}}
        kw_plus = {**base_kw, **{k: v for k, v in S.balanced_settings(True).items() if k not in skip}}
        trades_stable = run_bt(**kw_stable, **common)
        trades_bal = run_bt(**kw_bal, **common)
        trades_plus = run_bt(**kw_plus, **common)
        trades = trades_bal
        mode_label = "COMPARE: STABLE vs BALANCED vs BALANCED+"
    elif args.compare_dense and opt:
        meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
                "dense", "medium", "dense_plus", "dense_wide")
        kw_stable = {**base_kw, **{k: v for k, v in S.stable_settings().items() if k not in meta}}
        kw_bal = {**base_kw, **{k: v for k, v in S.balanced_settings(False).items() if k not in meta}}
        kw_dense = {**base_kw, **{k: v for k, v in S.dense_settings().items() if k not in meta}}
        trades_stable = run_bt(**kw_stable, **common)
        trades_bal = run_bt(**kw_bal, **common)
        trades_dense = run_bt(**kw_dense, **common)
        trades = trades_dense
        mode_label = "COMPARE: STABLE vs BALANCED vs DENSE"
    elif args.compare_medium and opt:
        meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
                "dense", "medium")
        kw_stable = {**base_kw, **{k: v for k, v in S.stable_settings().items() if k not in meta}}
        kw_dense = {**base_kw, **{k: v for k, v in S.dense_settings().items() if k not in meta}}
        kw_med = {**base_kw, **{k: v for k, v in S.medium_settings().items() if k not in meta}}
        trades_stable = run_bt(**kw_stable, **common)
        trades_dense = run_bt(**kw_dense, **common)
        trades_med = run_bt(**kw_med, **common)
        trades = trades_med
        mode_label = "COMPARE: STABLE vs DENSE vs MEDIUM"
    elif args.compare_dense_tiers and opt:
        meta = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
                "dense", "medium", "dense_plus", "dense_wide")
        kw_d = {**base_kw, **{k: v for k, v in S.dense_settings().items() if k not in meta}}
        kw_dp = {**base_kw, **{k: v for k, v in S.dense_plus_settings(False).items() if k not in meta}}
        kw_m = {**base_kw, **{k: v for k, v in S.medium_settings().items() if k not in meta}}
        kw_dw = {**base_kw, **{k: v for k, v in S.dense_plus_settings(True).items() if k not in meta}}
        trades_d = run_bt(**kw_d, **common)
        trades_dp = run_bt(**kw_dp, **common)
        trades_m = run_bt(**kw_m, **common)
        trades_dw = run_bt(**kw_dw, **common)
        trades = trades_d
        mode_label = "COMPARE: DENSE tiers"
    else:
        trades = run_bt(**base_kw, **common)
        if opt:
            is_stable = opt.get("require_atr_regime") and opt.get("require_fib")
            mode_label = (
                (f"STABLE | rules {','.join(opt['enabled'])} | "
                 f"BE@{opt['be_trigger']}R + TP {args.tp_mode} {args.fixed_tp_r}R floor | "
                 f"fib + ATR regime | session {args.session_start}-{args.session_end} | minSL ${args.min_risk}"
                 if is_stable else
                 f"OPTIMIZED | rules {','.join(opt['enabled'])} | "
                 f"BE@{opt['be_trigger']}R + TP {args.fixed_tp_r}R ({args.tp_mode}) | "
                 f"session {args.session_start}-{args.session_end} | minSL ${args.min_risk}")
                + (" | HTF trend H4+H1" if opt.get("htf_trend") and tuple(opt.get("htf_tfs", ())) == ("H4", "H1")
                   else " | HTF trend H1" if opt.get("htf_trend") else "")
                + (" | quality: FVG conf + BOS" if opt.get("require_confluence") else "")
                + (" | NDS mode" if opt.get("nds_mode") else "")
                + (" | +NDS rule" if "NDS" in opt.get("enabled", []) and not opt.get("nds_mode") else "")
                + (" | NDS extended" if opt.get("nds_extended") else "")
                + (" | BALANCED 2pos" if opt.get("balanced") and not opt.get("balanced_plus") else "")
                + (" | BALANCED+ FVG/sess8-22" if opt.get("balanced_plus") else "")
                + (" | DENSE 2pos+fibNDS exempt" if opt.get("dense") else "")
                + (" | DENSE+ no ATR" if opt.get("dense_plus") and not opt.get("dense_wide") else "")
                + (" | DENSE-WIDE noATR+sess8-22" if opt.get("dense_wide") else "")
                + (" | MEDIUM 2pos+sess8-22" if opt.get("medium") else "")
                + (f" | partial {args.partial_frac*100:.0f}%@TP+trail" if args.partial_tp else "")
            )
        else:
            mode_label = EXIT_LABELS.get(args.exit, args.exit)

    if trades:
        lo = min(t["time"] for t in trades)
        hi = max(t["exit_time"] for t in trades)
    elif cutoff is not None:
        lo, hi = cutoff, d.index.max()
    else:
        lo, hi = d.index[0], d.index[-1]

    print("=" * 64)
    print(f"  RULE-BASED STRATEGY  |  {args.tf} signals  |  1-min accurate exits")
    print(f"  Data: MetaTrader 5  |  symbol {sym}")
    print(f"  Mode: {mode_label}")
    print(f"  Test window: {lo} -> {hi}  ({len(trades)} trades in window)")
    print(f"  Bars loaded: {len(d)} {args.tf} + {len(m1)} M1 (incl. warm-up)")
    print(f"  Account ${args.balance:,.0f} | risk {args.risk*100:.2g}% | spread ${spread}")
    print("=" * 64)

    if args.compare_old and args.optimized:
        print_summary("OLD (all rules + trail, no filters)", trades_old,
                      args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("OPTIMIZED (FVG+OB+DEMAND, BE+TP2R, filters)", trades_new,
                          args.balance, args.risk, spread, show_trades=show_trades)
    elif args.compare_exits:
        r = None
        for m in ["trail", "trail-tp", "be1r-tp", "be1r-fixed-tp"]:
            r = print_summary(EXIT_LABELS[m].upper(), results[m],
                              args.balance, args.risk, spread,
                              show_trades=show_trades)
    elif args.compare_tp:
        print_summary("TP FIXED 2R", trades_fixed, args.balance, args.risk, spread,
                      show_trades=show_trades)
        r = print_summary("TP HTF BLEND (H4/H1)", trades_htf, args.balance, args.risk, spread,
                          show_trades=show_trades)
    elif args.compare_tp_mode:
        print_summary("TP structural", trades_struct, args.balance, args.risk, spread,
                      show_trades=show_trades)
        r = print_summary("TP next-rule", trades_next, args.balance, args.risk, spread,
                          show_trades=show_trades)
    elif args.compare_nds and opt:
        print_summary("WITHOUT NDS — OB + DEMAND", trades_stable, args.balance, args.risk, spread,
                      show_trades=show_trades)
        r = print_summary("WITH NDS — OB + NDS + DEMAND", trades_full,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Monthly P&L — OB+DEM (no NDS):")
        rs = S.simulate_account(trades_stable, start_balance=args.balance,
                                risk_pct=args.risk, spread_price=spread)
        if rs:
            for mk in sorted(rs["monthly"]):
                print(f"    {mk}:  {rs['monthly'][mk]:>+10.2f}")
        print("\n  Monthly P&L — OB+NDS+DEM (full):")
        if r:
            for mk in sorted(r["monthly"]):
                print(f"    {mk}:  {r['monthly'][mk]:>+10.2f}")
    elif args.compare_partial_tp and opt:
        r0 = print_summary("FULL TP (100% at HTF target)", trades_full,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary(
            f"PARTIAL TP ({args.partial_frac*100:.0f}% at TP + {100-args.partial_frac*100:.0f}% trail 1R)",
            trades_partial, args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r:
            print(f"    Return:  {r0['ret_pct']:+.1f}%  ->  {r['ret_pct']:+.1f}%  "
                  f"({r['ret_pct']-r0['ret_pct']:+.1f}%)")
            print(f"    Win-rate: {r0['wr']:.1f}%  ->  {r['wr']:.1f}%")
            print(f"    Max DD:   {r0['max_dd']:.1f}%  ->  {r['max_dd']:.1f}%")
            avg0 = np.mean([t["R"] for t in trades_full])
            avg1 = np.mean([t["R"] for t in trades_partial])
            print(f"    Avg R:    {avg0:+.2f}  ->  {avg1:+.2f}")
        print("\n  Monthly P&L — FULL TP:")
        if r0:
            for mk in sorted(r0["monthly"]):
                print(f"    {mk}:  {r0['monthly'][mk]:>+10.2f}")
        print(f"\n  Monthly P&L — PARTIAL {args.partial_frac*100:.0f}%:")
        if r:
            for mk in sorted(r["monthly"]):
                print(f"    {mk}:  {r['monthly'][mk]:>+10.2f}")
    elif args.compare_nds_extended and opt:
        r0 = print_summary("STABLE — OB + NDS + DEMAND", trades_stable,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("NDS EXTENDED — all nested triggers", trades_ext,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r:
            print(f"    Return:  {r0['ret_pct']:+.1f}%  ->  {r['ret_pct']:+.1f}%  "
                  f"({r['ret_pct']-r0['ret_pct']:+.1f}%)")
            print(f"    Trades:  {r0['n']}  ->  {r['n']}")
            print(f"    WR:      {r0['wr']:.1f}%  ->  {r['wr']:.1f}%")
            avg0 = np.mean([t["R"] for t in trades_stable])
            avg1 = np.mean([t["R"] for t in trades_ext])
            print(f"    Avg R:   {avg0:+.2f}  ->  {avg1:+.2f}")
    elif args.compare_balanced and opt:
        r0 = print_summary("STABLE — 1 position, session 10-20", trades_stable,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r1 = print_summary("BALANCED — 2 concurrent, same filters", trades_bal,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("BALANCED+ — +NDS-FVG + session 8-22", trades_plus,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r1 and r:
            print(f"    {'':12}  STABLE   BALANCED  BALANCED+")
            print(f"    Trades:     {r0['n']:>5}    {r1['n']:>5}      {r['n']:>5}")
            print(f"    Win-rate:   {r0['wr']:>5.1f}%   {r1['wr']:>5.1f}%     {r['wr']:>5.1f}%")
            print(f"    Return:     {r0['ret_pct']:>+5.1f}%   {r1['ret_pct']:>+5.1f}%     {r['ret_pct']:>+5.1f}%")
            print(f"    Max DD:     {r0['max_dd']:>5.1f}%   {r1['max_dd']:>5.1f}%     {r['max_dd']:>5.1f}%")
            print(f"    PF:         {r0['pf']:>5.2f}   {r1['pf']:>5.2f}     {r['pf']:>5.2f}")
    elif args.compare_dense_tiers and opt:
        r0 = print_summary("DENSE — full filters", trades_d,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r1 = print_summary("DENSE+ — no ATR regime", trades_dp,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r2 = print_summary("MEDIUM — session 8-22", trades_m,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("DENSE-WIDE — no ATR + sess 8-22", trades_dw,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r1 and r2 and r:
            print(f"    {'':14}  DENSE   DENSE+  MEDIUM  D-WIDE")
            print(f"    Trades:       {r0['n']:>5}   {r1['n']:>5}   {r2['n']:>5}   {r['n']:>5}")
            print(f"    Win-rate:     {r0['wr']:>5.1f}%  {r1['wr']:>5.1f}%  {r2['wr']:>5.1f}%  {r['wr']:>5.1f}%")
            print(f"    Return:       {r0['ret_pct']:>+5.1f}%  {r1['ret_pct']:>+5.1f}%  {r2['ret_pct']:>+5.1f}%  {r['ret_pct']:>+5.1f}%")
            print(f"    Max DD:       {r0['max_dd']:>5.1f}%  {r1['max_dd']:>5.1f}%  {r2['max_dd']:>5.1f}%  {r['max_dd']:>5.1f}%")
    elif args.compare_medium and opt:
        r0 = print_summary("STABLE — full fib, 1 pos", trades_stable,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r1 = print_summary("DENSE — fib exempt NDS, 2 pos", trades_dense,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("MEDIUM — 2 pos + sess 8-22", trades_med,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r1 and r:
            print(f"    {'':12}  STABLE     DENSE     MEDIUM")
            print(f"    Trades:     {r0['n']:>5}    {r1['n']:>5}      {r['n']:>5}")
            print(f"    Win-rate:   {r0['wr']:>5.1f}%   {r1['wr']:>5.1f}%     {r['wr']:>5.1f}%")
            print(f"    Return:     {r0['ret_pct']:>+5.1f}%   {r1['ret_pct']:>+5.1f}%     {r['ret_pct']:>+5.1f}%")
            print(f"    Max DD:     {r0['max_dd']:>5.1f}%   {r1['max_dd']:>5.1f}%     {r['max_dd']:>5.1f}%")
            print(f"    PF:         {r0['pf']:>5.2f}   {r1['pf']:>5.2f}     {r['pf']:>5.2f}")
    elif args.compare_dense and opt:
        r0 = print_summary("STABLE — 1 pos, full fib", trades_stable,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r1 = print_summary("BALANCED — 2 concurrent", trades_bal,
                           args.balance, args.risk, spread, show_trades=show_trades)
        r = print_summary("DENSE — 2 pos + fib exempt for NDS", trades_dense,
                          args.balance, args.risk, spread, show_trades=show_trades)
        print("\n  Side-by-side:")
        if r0 and r1 and r:
            print(f"    {'':12}  STABLE   BALANCED    DENSE")
            print(f"    Trades:     {r0['n']:>5}    {r1['n']:>5}      {r['n']:>5}")
            print(f"    Win-rate:   {r0['wr']:>5.1f}%   {r1['wr']:>5.1f}%     {r['wr']:>5.1f}%")
            print(f"    Return:     {r0['ret_pct']:>+5.1f}%   {r1['ret_pct']:>+5.1f}%     {r['ret_pct']:>+5.1f}%")
            print(f"    Max DD:     {r0['max_dd']:>5.1f}%   {r1['max_dd']:>5.1f}%     {r['max_dd']:>5.1f}%")
            print(f"    PF:         {r0['pf']:>5.2f}   {r1['pf']:>5.2f}     {r['pf']:>5.2f}")
    else:
        r = print_summary(mode_label.upper(), trades, args.balance, args.risk, spread,
                          show_trades=show_trades)

    if r is None or r["n"] == 0:
        print("=" * 64)
        return
    if not (args.compare_old or args.compare_exits or args.compare_tp_mode
            or args.compare_tp or args.compare_nds or args.compare_partial_tp
            or args.compare_nds_extended or args.compare_balanced or args.compare_dense
            or args.compare_medium or args.compare_dense_tiers):
        print("\n  Monthly net P&L ($):")
        for mk in sorted(r["monthly"]):
            print(f"    {mk}:  {r['monthly'][mk]:>+10.2f}")

    rules = opt["enabled"] if opt else list(S.DETECTORS.keys())
    if args.compare_nds_extended:
        rules = sorted(set(S.OPT_NDS_EXTENDED) | {t.get("setup", "?") for t in trades})
    elif args.compare_nds:
        rules = ["OB", "NDS", "DEMAND", "FVG"]
    per_kw = {k: v for k, v in base_kw.items() if k not in ("enabled", "max_concurrent")}
    print("\n  Per-rule performance (display tag — matches trade list; NDS includes nested demand):")
    print(f"  {'RULE':<12}{'trades':>8}{'WR%':>8}{'PF':>7}{'totR':>9}")
    if trades:
        rule_keys = sorted({t.get("setup") or t.get("rule", "?") for t in trades}) \
            if args.compare_nds_extended else rules
        for name in rule_keys:
            t = S.trades_for_enabled_rule(name, trades) if not args.compare_nds_extended \
                else [x for x in trades if (x.get("setup") or x.get("rule")) == name]
            if not t:
                if not args.compare_nds_extended:
                    print(f"  {name:<12}{'-':>8}")
                continue
            R = np.array([x["R"] - spread / x["risk"] for x in t])
            w = R[R > 0]
            l = R[R <= 0]
            pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999
            print(f"  {name:<12}{len(R):>8}{(R>0).mean()*100:>7.1f}%{pf:>7.2f}{R.sum():>+9.1f}")
    else:
        for name in rules:
            print(f"  {name:<12}{'-':>8}")
    print("=" * 64)


if __name__ == "__main__":
    main()
