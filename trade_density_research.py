"""
Deep research: increase trade count while preserving STABLE win-rate.
Instruments filter funnel + sweeps parameter combos + new feature flags.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

DAYS = 180
SPREAD = S.SPREAD_USD
BAL = 1000.0
RISK = 0.01


def sim(trades, cutoff):
    tr = [t for t in trades if t["time"] >= cutoff]
    r = S.simulate_account(tr, start_balance=BAL, risk_pct=RISK, spread_price=SPREAD)
    if not r or r["n"] == 0:
        return dict(n=0, wr=0.0, ret=0.0, pf=0.0, dd=0.0, avg_r=0.0)
    return dict(n=r["n"], wr=r["wr"], ret=r["ret_pct"], pf=r["pf"],
                dd=r["max_dd"], avg_r=float(np.mean([t["R"] for t in tr])))


def base_kw(opt, htf_context):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus")
    kw = {k: v for k, v in opt.items() if k not in skip}
    kw["htf_context"] = htf_context
    return kw


def run_funnel(B, ctx, htf_context, opt, tf_min=15):
    """Count how many raw setups fail at each filter stage (STABLE filters)."""
    enabled = list(opt["enabled"])
    counts = dict(
        bars=0, raw_setups=0, vp_fail=0, risk_fail=0, fib_fail=0,
        atr_fail=0, session_fail=0, minsl_fail=0, htf_trend_fail=0,
        fill_fail=0, concurrent_skip=0, would_trade=0,
        rule_hits={n: 0 for n in enabled},
    )
    open_until = []
    for i in range(3, B.n - 1):
        counts["bars"] += 1
        open_until = [x for x in open_until if x >= i]
        if len(open_until) >= opt["max_concurrent"]:
            counts["concurrent_skip"] += 1
            continue
        if opt.get("require_atr_regime") and not S.atr_regime_pass(
                B, i, lookback=opt.get("atr_lookback", S.OPT_STABLE_ATR_LB),
                max_ratio=opt.get("atr_max_ratio", S.OPT_STABLE_ATR_RATIO)):
            counts["atr_fail"] += 1
            continue
        for name in enabled:
            setups = S.iter_rule_setups(
                name, B, i, S.DEFAULT_PARAMS, htf_context=htf_context,
                signal_time=B.t[i], htf_tfs=tuple(opt["htf_tfs"]),
                nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO))
            for (direction, proximal, distal, tag) in setups:
                counts["raw_setups"] += 1
                if not S.vp_pass(B, i, direction, proximal, opt.get("vp_mode", S.VP_MODE)):
                    counts["vp_fail"] += 1
                    continue
                _sl, _entry, risk = S.compute_entry_sl_risk(
                    B, i, direction, proximal, distal)
                if risk <= 0:
                    counts["risk_fail"] += 1
                    continue
                if opt.get("require_fib") and not S.fib_retrace_pass(B, i, direction, _entry):
                    counts["fib_fail"] += 1
                    continue
                if opt.get("min_risk_usd") and risk < opt["min_risk_usd"]:
                    counts["minsl_fail"] += 1
                    continue
                if opt.get("session_start") is not None:
                    if not S.session_pass(B.t[i], opt["session_start"], opt["session_end"]):
                        counts["session_fail"] += 1
                        continue
                if opt.get("htf_trend") and htf_context:
                    fail = False
                    for tf in opt["htf_tfs"]:
                        if not S.htf_trend_pass(htf_context, B.t[i], direction,
                                                tf=tf, ema_period=opt.get("htf_ema", 20)):
                            fail = True
                            break
                    if fail:
                        counts["htf_trend_fail"] += 1
                        continue
                # simulate fill
                tr = S.simulate_trade_m1(
                    B, ctx, direction, i, proximal, distal,
                    S.WAIT_BARS, S.MAX_HOLD, tf_min=tf_min, tag=tag,
                    exit_mode=opt["exit_mode"], tp_mode=opt["tp_mode"],
                    fixed_tp_r=opt["fixed_tp_r"], htf_context=htf_context,
                    min_tp_r=opt["min_tp_r"], max_tp_r=opt["max_tp_r"],
                    htf_lookback=opt["htf_lookback"], htf_tfs=tuple(opt["htf_tfs"]),
                    be_trigger=opt["be_trigger"])
                if tr is None:
                    counts["fill_fail"] += 1
                    continue
                counts["would_trade"] += 1
                counts["rule_hits"][name] = counts["rule_hits"].get(name, 0) + 1
                open_until.append(tr["exit_idx"])
                break
            else:
                continue
            break
    return counts


def run_backtest_multi(B, ctx, htf_context, opt, tf_min=15,
                       scan_all_rules=False, fib_nds_exempt=False,
                       htf_trend_tfs=None):
    """
    Extended backtest with optional features:
    - scan_all_rules: try every enabled rule on each bar (pick first that fills)
    - fib_nds_exempt: skip fib filter for nested (NDS) setups
    - htf_trend_tfs: override which TFs used for trend filter
    """
    enabled = list(opt["enabled"])
    trend_tfs = htf_trend_tfs or tuple(opt["htf_tfs"])
    trades = []
    open_until = []
    for i in range(3, B.n - 1):
        open_until = [x for x in open_until if x >= i]
        if len(open_until) >= opt["max_concurrent"]:
            continue
        if opt.get("require_atr_regime") and not S.atr_regime_pass(
                B, i, lookback=opt.get("atr_lookback", S.OPT_STABLE_ATR_LB),
                max_ratio=opt.get("atr_max_ratio", S.OPT_STABLE_ATR_RATIO)):
            continue
        candidates = []
        rules = enabled if scan_all_rules else enabled
        for name in rules:
            setups = S.iter_rule_setups(
                name, B, i, S.DEFAULT_PARAMS, htf_context=htf_context,
                signal_time=B.t[i], htf_tfs=tuple(opt["htf_tfs"]),
                nds_max_ratio=opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO))
            for (direction, proximal, distal, tag) in setups:
                if not S.vp_pass(B, i, direction, proximal, opt.get("vp_mode", S.VP_MODE)):
                    continue
                _sl, _entry, risk = S.compute_entry_sl_risk(
                    B, i, direction, proximal, distal)
                if risk <= 0:
                    continue
                is_nds = S.is_nds_setup(
                    B, i, direction, proximal, distal, htf_context, B.t[i],
                    tuple(opt["htf_tfs"]), opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO))
                if opt.get("require_fib") and not fib_nds_exempt:
                    if not S.fib_retrace_pass(B, i, direction, _entry):
                        continue
                elif opt.get("require_fib") and fib_nds_exempt and not is_nds:
                    if not S.fib_retrace_pass(B, i, direction, _entry):
                        continue
                if not S.entry_filters_pass(
                        B, i, risk, min_risk_usd=opt.get("min_risk_usd", 0),
                        session_start=opt.get("session_start"),
                        session_end=opt.get("session_end"),
                        direction=direction, signal_time=B.t[i],
                        htf_context=htf_context,
                        htf_trend=opt.get("htf_trend", False),
                        htf_ema=opt.get("htf_ema", 20),
                        htf_tfs=trend_tfs):
                    continue
                candidates.append((name, direction, proximal, distal, tag, risk))
        if not candidates:
            continue
        # priority: NDS family > OB > DEMAND
        prio = {"NDS": 0, "NDS_OB": 1, "NDS_FVG": 2, "NDS_FRESH": 3,
                "OB": 4, "DEMAND": 5}
        candidates.sort(key=lambda c: prio.get(c[0], 9))
        for name, direction, proximal, distal, tag, risk in candidates:
            tr = S.simulate_trade_m1(
                B, ctx, direction, i, proximal, distal,
                S.WAIT_BARS, S.MAX_HOLD, tf_min=tf_min, tag=tag,
                exit_mode=opt["exit_mode"], tp_mode=opt["tp_mode"],
                fixed_tp_r=opt["fixed_tp_r"], htf_context=htf_context,
                min_tp_r=opt["min_tp_r"], max_tp_r=opt["max_tp_r"],
                htf_lookback=opt["htf_lookback"], htf_tfs=tuple(opt["htf_tfs"]),
                be_trigger=opt["be_trigger"])
            if tr is not None:
                tr["rule"] = name
                tr["setup"] = S.resolve_trade_tag(
                    name, tag, B, i, direction, proximal, distal,
                    htf_context, B.t[i], tuple(opt["htf_tfs"]),
                    opt.get("nds_max_ratio", S.OPT_NDS_MAX_RATIO),
                    nds_enabled=any(S.is_nds_rule(n) for n in enabled))
                trades.append(tr)
                open_until.append(tr["exit_idx"])
                break
    trades.sort(key=lambda t: t["time"])
    return trades


def load_data(tf="M15"):
    mt5 = M.connect()
    try:
        sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
        _, d, m1, cutoff = M.fetch_pair(
            sym, tf, mt5=mt5, days=DAYS)
        opt = S.stable_settings()
        _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=tuple(opt["htf_tfs"]))
    finally:
        M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    tf_min = M.tf_minutes(tf)
    return B, ctx, htf, cutoff, tf_min


def main():
    print("=" * 72)
    print("  TRADE DENSITY RESEARCH — preserve STABLE win-rate")
    print("=" * 72)

    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    kw = base_kw(S.stable_settings(), htf)

    tr0 = S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **kw)
    s0 = sim(tr0, cutoff)
    print(f"\n  BASELINE STABLE: n={s0['n']} WR={s0['wr']:.1f}% ret={s0['ret']:+.1f}% "
          f"PF={s0['pf']:.2f} DD={s0['dd']:.1f}% avgR={s0['avg_r']:+.2f}")

    print("\n  --- FILTER FUNNEL (per-bar rejections) ---")
    funnel = run_funnel(B, ctx, htf, S.stable_settings(), tf_min=15)
    print(f"  Bars scanned:           {funnel['bars']}")
    print(f"  Raw setups found:       {funnel['raw_setups']}")
    print(f"  Would trade (no conc):  {funnel['would_trade']}")
    print(f"  Skipped (concurrent):   {funnel['concurrent_skip']} bars")
    print(f"  Rejected — ATR regime:  {funnel['atr_fail']} bars")
    print(f"  Rejected — VP filter:   {funnel['vp_fail']}")
    print(f"  Rejected — fib:         {funnel['fib_fail']}")
    print(f"  Rejected — min SL:      {funnel['minsl_fail']}")
    print(f"  Rejected — session:     {funnel['session_fail']}")
    print(f"  Rejected — HTF trend:   {funnel['htf_trend_fail']}")
    print(f"  Rejected — no fill:     {funnel['fill_fail']}")
    print(f"  Rule hits: {funnel['rule_hits']}")

    # parameter + feature sweep
    print("\n  --- SWEEP (target WR >= 75%) ---")
    print(f"  {'label':<42} {'n':>4} {'WR':>6} {'ret':>7} {'PF':>6} {'DD':>5}")
    print("  " + "-" * 72)

    sweeps = []

    def add(label, **overrides):
        scan_all = overrides.pop("scan_all_rules", False)
        fib_exempt = overrides.pop("fib_nds_exempt", False)
        trend_tfs = overrides.pop("htf_trend_tfs", None)
        tf = overrides.pop("signal_tf", "M15")
        o = dict(S.stable_settings())
        o.update(overrides)
        k = base_kw(o, htf)
        if tf != "M15":
            mt5 = M.connect()
            sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
            _, d2, m1_2, cut2 = M.fetch_pair(sym, tf, mt5=mt5, days=DAYS)
            M.shutdown(mt5)
            B2 = S.Bars(d2)
            ctx2 = S.M1Ctx(m1_2, d2.index)
            tmin = M.tf_minutes(tf)
            if scan_all or fib_exempt or trend_tfs:
                tr = run_backtest_multi(B2, ctx2, htf, o, tf_min=tmin,
                                        scan_all_rules=scan_all,
                                        fib_nds_exempt=fib_exempt,
                                        htf_trend_tfs=trend_tfs)
            else:
                tr = S.run_backtest(d2, m1_2, B=B2, ctx=ctx2, tf_min=tmin, **k)
            s = sim(tr, cut2)
        elif scan_all or fib_exempt or trend_tfs:
            tr = run_backtest_multi(B, ctx, htf, o, tf_min=15,
                                    scan_all_rules=scan_all,
                                    fib_nds_exempt=fib_exempt,
                                    htf_trend_tfs=trend_tfs)
            s = sim(tr, cutoff)
        else:
            tr = S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
            s = sim(tr, cutoff)
        sweeps.append((label, s, tr))
        flag = " ***" if s["wr"] >= 75 and s["n"] > s0["n"] else ""
        print(f"  {label:<42} {s['n']:>4} {s['wr']:>5.1f}% {s['ret']:>+6.1f}% "
              f"{s['pf']:>6.2f} {s['dd']:>4.1f}%{flag}")

    add("STABLE baseline")
    add("max_concurrent=2", max_concurrent=2)
    add("max_concurrent=3", max_concurrent=3)
    add("+ NDS_FVG rule", enabled=["OB", "NDS", "NDS_FVG", "DEMAND"])
    add("mc2 + NDS_FVG", max_concurrent=2, enabled=["OB", "NDS", "NDS_FVG", "DEMAND"])
    add("+ NDS_OB rule", enabled=["OB", "NDS", "NDS_OB", "DEMAND"])
    add("+ NDS_FRESH rule", enabled=["OB", "NDS", "NDS_FRESH", "DEMAND"])
    add("mc2 + NDS_OB", max_concurrent=2, enabled=["OB", "NDS", "NDS_OB", "DEMAND"])
    add("mc2 + NDS_FRESH", max_concurrent=2, enabled=["OB", "NDS", "NDS_FRESH", "DEMAND"])
    add("HTF trend H1 only", htf_trend_tfs=("H1",))
    add("mc2 + H1 trend only", max_concurrent=2, htf_trend_tfs=("H1",))
    add("fib exempt for NDS", fib_nds_exempt=True)
    add("mc2 + fib NDS exempt", max_concurrent=2, fib_nds_exempt=True)
    add("session 9-21", session_start=9, session_end=21)
    add("session 8-22", session_start=8, session_end=22)
    add("mc2 + session 9-21", max_concurrent=2, session_start=9, session_end=21)
    add("atr ratio 1.7", atr_max_ratio=1.7)
    add("mc2 + atr 1.7", max_concurrent=2, atr_max_ratio=1.7)
    add("min SL $2.8", min_risk_usd=2.8)
    add("mc2 + minSL 2.8", max_concurrent=2, min_risk_usd=2.8)
    add("nds ratio 0.65", nds_max_ratio=0.65)
    add("mc2 + nds 0.65", max_concurrent=2, nds_max_ratio=0.65)
    add("scan all rules/bar", scan_all_rules=True)
    add("mc2 + scan all rules", max_concurrent=2, scan_all_rules=True)
    add("M5 signals", signal_tf="M5")
    add("M30 signals", signal_tf="M30")
    add("mc2 + M5", max_concurrent=2, signal_tf="M5")
    add("dual: M15+mc2 + M5 add", max_concurrent=2)  # placeholder

    # combos that looked promising
    add("mc2+NDS_FVG+H1trend", max_concurrent=2,
        enabled=["OB", "NDS", "NDS_FVG", "DEMAND"], htf_trend_tfs=("H1",))
    add("mc2+NDS_OB+fib exempt", max_concurrent=2,
        enabled=["OB", "NDS", "NDS_OB", "DEMAND"], fib_nds_exempt=True)
    add("mc2+NDS_FVG+sess9-21", max_concurrent=2,
        enabled=["OB", "NDS", "NDS_FVG", "DEMAND"], session_start=9, session_end=21)
    add("mc2+atr1.7+NDS_FVG", max_concurrent=2, atr_max_ratio=1.7,
        enabled=["OB", "NDS", "NDS_FVG", "DEMAND"])

    # best candidates
    good = [x for x in sweeps if x[1]["wr"] >= 75 and x[1]["n"] > s0["n"]]
    good.sort(key=lambda x: (-x[1]["n"], -x[1]["wr"], -x[1]["ret"]))
    print("\n  --- TOP CANDIDATES (WR>=75%, more trades than baseline) ---")
    if not good:
        print("  None hit WR>=75%. Showing WR>=70%:")
        good = [x for x in sweeps if x[1]["wr"] >= 70 and x[1]["n"] > s0["n"]]
        good.sort(key=lambda x: (-x[1]["n"], -x[1]["wr"], -x[1]["ret"]))
    for label, s, tr in good[:8]:
        wins = sum(1 for t in tr if t["R"] > 0)
        print(f"  {label}: n={s['n']} WR={s['wr']:.1f}% ret={s['ret']:+.1f}% "
              f"(wins={wins})")

    print("=" * 72)


if __name__ == "__main__":
    main()
