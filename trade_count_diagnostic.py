"""
Diagnostic: why so few trades? Raw signals vs filter funnel vs fill rate.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M

DAYS = 180


def count_raw(B, enabled, htf=None):
    """Count all setups detectors fire (no filters)."""
    n = 0
    by_rule = {}
    for i in range(3, B.n - 1):
        for name in enabled:
            setups = S.iter_rule_setups(
                name, B, i, S.DEFAULT_PARAMS, htf_context=htf,
                signal_time=B.t[i], htf_tfs=("H4", "H1"))
            c = len(setups)
            if c:
                by_rule[name] = by_rule.get(name, 0) + c
                n += c
    return n, by_rule


def funnel(B, ctx, htf, opt, tf_min=15):
    """Per-stage rejection counts mimicking run_backtest."""
    enabled = list(opt["enabled"])
    trend_tfs = opt.get("htf_trend_tfs") or tuple(opt["htf_tfs"])
    counts = dict(
        bars=0, raw=0, atr_bar_skip=0, concurrent_skip=0,
        vp=0, risk=0, fib=0, minsl=0, session=0, htf=0, no_fill=0,
        trades=0, by_rule={},
    )
    open_until = []
    for i in range(3, B.n - 1):
        counts["bars"] += 1
        open_until = [x for x in open_until if x >= i]
        if len(open_until) >= opt["max_concurrent"]:
            counts["concurrent_skip"] += 1
            continue
        atr_ok = (not opt.get("require_atr_regime") or S.atr_regime_pass(
            B, i, lookback=opt.get("atr_lookback", 100),
            max_ratio=opt.get("atr_max_ratio", 1.5)))
        if not atr_ok:
            counts["atr_bar_skip"] += 1
            continue
        for name in enabled:
            setups = S.iter_rule_setups(
                name, B, i, S.DEFAULT_PARAMS, htf_context=htf,
                signal_time=B.t[i], htf_tfs=tuple(opt["htf_tfs"]),
                nds_max_ratio=opt.get("nds_max_ratio", 0.6))
            for (direction, proximal, distal, tag) in setups:
                counts["raw"] += 1
                if not S.vp_pass(B, i, direction, proximal, opt.get("vp_mode")):
                    counts["vp"] += 1
                    continue
                _sl, _entry, risk = S.compute_entry_sl_risk(
                    B, i, direction, proximal, distal)
                if risk <= 0:
                    counts["risk"] += 1
                    continue
                nested = S.is_nds_setup(
                    B, i, direction, proximal, distal, htf, B.t[i],
                    tuple(opt["htf_tfs"]), opt.get("nds_max_ratio", 0.6))
                if opt.get("require_fib") and not (opt.get("fib_nds_exempt") and nested):
                    if not S.fib_retrace_pass(B, i, direction, _entry):
                        counts["fib"] += 1
                        continue
                if opt.get("min_risk_usd") and risk < opt["min_risk_usd"]:
                    counts["minsl"] += 1
                    continue
                if opt.get("session_start") is not None:
                    if not S.session_pass(B.t[i], opt["session_start"], opt["session_end"]):
                        counts["session"] += 1
                        continue
                if opt.get("htf_trend") and htf:
                    fail = False
                    for tf in trend_tfs:
                        if not S.htf_trend_pass(htf, B.t[i], direction, tf=tf,
                                                ema_period=opt.get("htf_ema", 20)):
                            fail = True
                            break
                    if fail:
                        counts["htf"] += 1
                        continue
                tr = S.simulate_trade_m1(
                    B, ctx, direction, i, proximal, distal,
                    S.WAIT_BARS, S.MAX_HOLD, tf_min=tf_min, tag=tag,
                    exit_mode=opt["exit_mode"], tp_mode=opt["tp_mode"],
                    fixed_tp_r=opt["fixed_tp_r"], htf_context=htf,
                    min_tp_r=opt["min_tp_r"], max_tp_r=opt["max_tp_r"],
                    htf_lookback=opt["htf_lookback"], htf_tfs=tuple(opt["htf_tfs"]),
                    be_trigger=opt["be_trigger"])
                if tr is None:
                    counts["no_fill"] += 1
                    continue
                counts["trades"] += 1
                counts["by_rule"][name] = counts["by_rule"].get(name, 0) + 1
                open_until.append(tr["exit_idx"])
                break
            else:
                continue
            break
    return counts


def run_preset(label, opt, d, m1, B, ctx, htf, cutoff):
    skip = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus", "dense")
    k = {x: opt[x] for x in opt if x not in skip}
    k["htf_context"] = htf
    tr_all = S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
    tr = [t for t in tr_all if t["time"] >= cutoff]
    r = S.simulate_account(tr, 1000, 0.01, 0.28) if tr else None
    raw_n, raw_by = count_raw(B, opt["enabled"], htf)
    fn = funnel(B, ctx, htf, opt)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Rules: {', '.join(opt['enabled'])}")
    print(f"  max_concurrent={opt['max_concurrent']} | fib={opt.get('require_fib')} "
          f"| fib_nds_exempt={opt.get('fib_nds_exempt')} | ATR={opt.get('require_atr_regime')} "
          f"| HTF trend={opt.get('htf_trend')} | session {opt.get('session_start')}-{opt.get('session_end')} "
          f"| minSL ${opt.get('min_risk_usd')}")
    print(f"  RAW setups (no filters):     {raw_n:>5}  {raw_by}")
    print(f"  --- funnel on test window ---")
    print(f"  Bars scanned:                {fn['bars']:>5}")
    print(f"  ATR regime (whole bar skip): {fn['atr_bar_skip']:>5}")
    print(f"  Setups entering filter chain:{fn['raw']:>5}")
    print(f"  Rejected VP:                 {fn['vp']:>5}")
    print(f"  Rejected fib:                {fn['fib']:>5}")
    print(f"  Rejected HTF trend:          {fn['htf']:>5}")
    print(f"  Rejected session:            {fn['session']:>5}")
    print(f"  Rejected min SL:             {fn['minsl']:>5}")
    print(f"  No limit fill (wait={S.WAIT_BARS}): {fn['no_fill']:>5}")
    print(f"  Concurrent skip (bars):      {fn['concurrent_skip']:>5}")
    print(f"  => TRADES (funnel):          {fn['trades']:>5}  {fn['by_rule']}")
    if r:
        print(f"  => TRADES (backtest in window): {len(tr):>3}  WR={r['wr']:.1f}% ret={r['ret_pct']:+.1f}%")
    else:
        print(f"  => TRADES (backtest in window): 0")


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=180, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    print("=" * 70)
    print("  TRADE COUNT DIAGNOSTIC — are rules applied correctly?")
    print(f"  Symbol: {sym} | TF: M15 | Window: {DAYS}d | Bars: {B.n}")
    print("=" * 70)

    # 1) No filters at all — sanity check detectors work
    all_rules = list(S.DETECTORS.keys())
    raw_all, raw_by = count_raw(B, all_rules, htf)
    print(f"\n  SANITY: ALL detectors raw setups = {raw_all}")
    print(f"    per rule: {raw_by}")

    # 2) STABLE vs DENSE vs no filters vs one-filter-off sweeps
    run_preset("STABLE (--optimized)", S.stable_settings(), d, m1, B, ctx, htf, cutoff)
    run_preset("DENSE (--dense)", S.dense_settings(), d, m1, B, ctx, htf, cutoff)

    # No filters except exit
    loose = S.stable_settings()
    loose.update(require_fib=False, require_atr_regime=False, htf_trend=False,
                 min_risk_usd=0, session_start=None, session_end=None, max_concurrent=3)
    run_preset("NO FILTERS (all rules OB+NDS+DEM)", loose, d, m1, B, ctx, htf, cutoff)

    loose2 = dict(loose)
    loose2["enabled"] = list(S.DETECTORS.keys())
    run_preset("NO FILTERS + ALL 8 SMC rules", loose2, d, m1, B, ctx, htf, cutoff)

    # One filter off at a time from STABLE
    base = S.stable_settings()
    for label, kw in [
        ("STABLE minus fib", dict(require_fib=False)),
        ("STABLE minus ATR", dict(require_atr_regime=False)),
        ("STABLE minus HTF trend", dict(htf_trend=False)),
        ("STABLE minus VP", dict(vp_mode=None)),
        ("STABLE session 8-22", dict(session_start=8, session_end=22)),
        ("STABLE minSL $2", dict(min_risk_usd=2.0)),
    ]:
        o = dict(base)
        o.update(kw)
        run_preset(label, o, d, m1, B, ctx, htf, cutoff)

    # Rule order: only one rule at a time with STABLE
    print(f"\n{'='*70}")
    print("  EACH RULE ALONE with STABLE filters:")
    for name in ["OB", "NDS", "DEMAND", "FVG", "WYCK", "BOS"]:
        o = S.stable_settings()
        o["enabled"] = [name]
        fn = funnel(B, ctx, htf, o)
        raw, _ = count_raw(B, [name], htf)
        tr = [t for t in S.run_backtest(
            d, m1, B=B, ctx=ctx, tf_min=15,
            **{x: o[x] for x in o if x not in ("spread_usd", "nds_mode", "nds_extended",
                                                 "balanced", "balanced_plus", "dense")},
            htf_context=htf) if t["time"] >= cutoff]
        print(f"    {name:<8} raw={raw:>4} -> funnel={fn['trades']:>2} -> backtest={len(tr):>2}")

    # NDS tagging: DEMAND setups that become NDS
    print(f"\n  NDS note: NDS rule key often fires 0 — nested trades come via DEMAND tag resolution")

    print("\n" + "=" * 70)
    print("  CONCLUSION keys printed above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
