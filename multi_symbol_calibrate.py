"""
Multi-asset backtest with calibrated profiles (Gold, BTC, Brent, EURUSD).

Usage:
  python multi_symbol_calibrate.py
  python backtest.py --multi --days 90
  python backtest.py --multi --days 90 --no-trades   # hide trade ledgers
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import inspect
import numpy as np
import pandas as pd
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X

DAYS = 90
BAL, RISK = 1000.0, 0.01
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
        "fib_spike_exempt", "spike_params",
        # floating-system config — not run_backtest() kwargs
        "meta_gate", "meta_gate_json", "meta_min_regime_n",
        "meta_require_positive_r", "meta_exclude_weak_rules",
        # live execution style — backtest uses max_same_dir only
        "touch_use_limit")
SEP = "=" * 96
THIN = "-" * 96


def _opt(prof, spike=False):
    return P.preset_for_profile(prof, spike=spike)


CALIB_DAYS = 90  # min-SL ATR window — MUST equal live_portfolio.CALIB_DAYS


def _calibrate_min_sl_90d(sym, prof, B_test, cutoff, mt5, tf="M15"):
    """
    Min-SL from a fixed 90-day trailing window (same method as the live bot),
    so the threshold does not drift with the --days you pass to the backtest.
    Falls back to the test-window calibration if the 90-day fetch is unavailable.
    """
    if prof.get("min_sl_mode") != "atr":
        return X.calibrate_min_sl(B_test, cutoff, prof)
    try:
        n = int(CALIB_DAYS * 1440 / M.tf_minutes(tf)) + 60
        dc = M.fetch_bars(sym, tf, count=n, mt5=mt5)
        Bc = S.Bars(dc)
        return X.calibrate_min_sl(Bc, Bc.t[0], prof)
    except Exception:
        return X.calibrate_min_sl(B_test, cutoff, prof)


def _run(sym, prof, mt5, days=DAYS, spike=False, asset_key="?", risk_pct=None,
         max_concurrent=None, tol_override=None, meta_gate_override=None,
         date_from=None, date_to=None, opt_extra=None, tf="M15"):
    spread = P.live_spread(sym, prof, mt5)
    opt = _opt(prof, spike=spike)
    _, d, m1, cutoff = M.fetch_pair(sym, tf, mt5=mt5, days=days,
                                     date_from=date_from, date_to=date_to)
    htf_tfs = list(opt.get("htf_tfs", ("H4", "H1")))
    if opt.get("regime_gate"):
        rtf = opt.get("regime_tf", "H1")
        if rtf not in htf_tfs:
            htf_tfs.append(rtf)
    htf_tfs = tuple(htf_tfs)
    _, htf_dfs = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=htf_tfs)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    # Calibrate min-SL on a FIXED 90-day trailing window — identical to live
    # (live_portfolio.refresh_min_sl). Independent of --days so a 10-day and a
    # 90-day backtest use the SAME threshold the live bot uses → same trades.
    eff_sl = _calibrate_min_sl_90d(sym, prof, B, cutoff, mt5, tf=tf)
    rp = risk_pct if risk_pct is not None else X.profile_risk_pct(prof, RISK)
    o = P.apply_profile_to_settings(opt, prof, spread, min_sl_override=eff_sl)
    if opt_extra:
        o.update(opt_extra)
    # Broker/execution keys (limit_fill_mode, broker, commission…) describe how
    # orders get filled live — run_backtest has no parameter for them.
    _engine_args = set(inspect.signature(S.run_backtest).parameters)
    k = {x: o[x] for x in o if x not in META and x in _engine_args}
    if max_concurrent is not None:
        k["max_concurrent"] = max_concurrent
    k["htf_context"] = htf
    det = P.merge_params(S.DEFAULT_PARAMS, prof)
    if tol_override:
        det.update(tol_override)
    k["detector_params"] = det
    if o.get("meta_gate"):
        import floating_config as FC
        k["meta_gate"] = meta_gate_override if meta_gate_override is not None else (
            FC.load_meta_gate(assets=[asset_key]))
        k["meta_asset"] = asset_key
        k["meta_skip_if_no_rules"] = o.get("meta_skip_if_no_rules", False)
    if spike:
        k["spike_mode"] = True
        k["fib_spike_exempt"] = True
        import spike_strategies as sp
        k["spike_params"] = dict(sp.SPIKE_DEFAULT)
        k["spike_params"].update(prof.get("spike_params") or {})
        k["enabled"] = list(S.OPT_DENSE_SPIKE_RULES)
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=M.tf_minutes(tf), **k)
          if t["time"] >= cutoff]
    if date_to is not None:
        hi = pd.Timestamp(date_to)
        tr = [t for t in tr if t["time"] < hi]
    empty = dict(n=0, wr=0.0, ret=0.0, pf=0.0, dd=0.0, rules={}, spike_n=0,
                 sym=sym, min_sl=prof["min_sl"], eff_sl=prof["min_sl"], trades=[], spread=spread,
                 spread_r=0.0, risk_pct=rp, enabled=list(o.get("enabled", [])),
                 asset=prof.get("label", asset_key))
    if not tr:
        return empty
    r = S.simulate_account(tr, BAL, rp, spread)
    spike_n = sum(1 for t in tr if str(t.get("setup", "")).startswith("SPIKE"))
    rules = {}
    for t in tr:
        rk = t.get("setup") or t.get("rule", "?")
        rules[rk] = rules.get(rk, 0) + 1
    spr_r = X.spread_cost_r(spread, eff_sl)
    return dict(n=r["n"], wr=r["wr"], ret=r["ret_pct"], pf=r["pf"], dd=r["max_dd"],
                rules=rules, spike_n=spike_n, sym=sym, min_sl=eff_sl, eff_sl=eff_sl,
                spread=spread, spread_r=spr_r, risk_pct=rp,
                trades=tr, enabled=list(o.get("enabled", [])),
                asset=prof.get("label", asset_key), monthly=r.get("monthly", {}))


def _fmt_min_sl(asset, val):
    if asset == "EURUSD":
        return f"{val:.5f}"
    if asset == "BRENT":
        return f"{val:.3f}"
    if asset == "BTCUSD":
        return f"{val:.0f}"
    if asset == "US30":
        return f"{val:.1f}"
    if asset == "US500":
        return f"{val:.2f}"
    return f"{val:.2f}"


def print_overview_table(rows):
    """Portfolio summary — one row per asset."""
    print(f"\n  PORTFOLIO OVERVIEW")
    print(f"  {'#':>2}  {'asset':<8} {'preset':<11} {'symbol':<14} {'minSL':>8}  "
          f"{'risk%':>5}  {'spr/R':>5}  {'trades':>6}  {'WR%':>6}  {'PF':>6}  {'return':>8}  {'DD%':>5}")
    print("  " + THIN)
    for i, row in enumerate(rows, 1):
        if row.get("error"):
            print(f"  {i:>2}  {row['asset_key']:<8} {row.get('preset', '?'):<11} "
                  f"{'NOT FOUND':<14}  {'—':>8}  {'—':>5}  {'—':>5}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>8}  {'—':>5}")
            continue
        s = row["stats"]
        rp = s.get("risk_pct", RISK) * 100
        sr = s.get("spread_r", 0)
        if s["n"] == 0:
            print(f"  {i:>2}  {row['asset_key']:<8} {row['preset']:<11} {s['sym']:<14} "
                  f"{_fmt_min_sl(row['asset_key'], s['min_sl']):>8}  {rp:>4.1f}%  "
                  f"{sr:>5.2f}  {0:>6}  {'—':>6}  {'—':>6}  {'—':>8}  {'—':>5}")
            continue
        print(f"  {i:>2}  {row['asset_key']:<8} {row['preset']:<11} {s['sym']:<14} "
              f"{_fmt_min_sl(row['asset_key'], s['min_sl']):>8}  {rp:>4.1f}%  "
              f"{sr:>5.2f}  {s['n']:>6}  {s['wr']:>5.1f}%  {s['pf']:>6.2f}  "
              f"{s['ret']:>+7.1f}%  {s['dd']:>4.1f}%")


def print_per_rule(trades, enabled, spread):
    """Per-rule stats table (same layout as backtest.py)."""
    print("\n  Per-rule performance:")
    print(f"  {'RULE':<12}{'trades':>8}{'WR%':>8}{'PF':>7}{'totR':>9}")
    print("  " + "-" * 44)
    if not trades:
        for name in enabled:
            print(f"  {name:<12}{'-':>8}")
        return
    for name in enabled:
        t = S.trades_for_enabled_rule(name, trades)
        if not t:
            print(f"  {name:<12}{'-':>8}")
            continue
        R = np.array([x["R"] - spread / x["risk"] for x in t])
        w, l = R[R > 0], R[R <= 0]
        pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
        print(f"  {name:<12}{len(R):>8}{(R > 0).mean() * 100:>7.1f}%{pf:>7.2f}{R.sum():>+9.1f}")


def print_trade_ledger(trades, balance, risk, spread, title="TRADES"):
    rows = S.build_trade_ledger(trades, start_balance=balance, risk_pct=risk,
                                spread_price=spread)
    if not rows:
        print("\n  (no trades)")
        return
    print(f"\n  --- {title} ({len(rows)} trades) ---")
    print(f"  {'#':>3}  {'entry':<17}  {'rule':<10}  {'side':<5}  "
          f"{'entry$':>10}  {'exit$':>10}  {'R':>6}  {'tpR':>5}  {'P&L$':>9}  "
          f"{'bal$':>9}  {'fill':>5}  {'etol$':>5}  exit")
    print("  " + THIN)
    tot = 0.0
    for r in rows:
        tot += r["net_usd"]
        et = str(r["entry_time"])[:16]
        side = "LONG" if r["dir"] == "long" else "SHORT"
        tp_r_s = f"{r['tp_r']:.1f}" if r.get("tp_r") is not None else "  -"
        fill = "TOL" if str(r.get("fill_kind", "")).lower() == "tol" else "EXACT"
        etol = r.get("entry_tol_usd")
        etol_s = f"{float(etol):>5.2f}" if etol is not None else "   -"
        print(f"  {r['n']:>3}  {et}  {r['rule']:<10}  {side:<5}  "
              f"{r['entry']:>10.2f}  {r['exit_px']:>10.2f}  {r['R']:>+6.2f}  "
              f"{tp_r_s:>5}  {r['net_usd']:>+9.2f}  {r['balance']:>9.2f}  "
              f"{fill:>5}  {etol_s:>5}  {r['exit']}")
    print("  " + THIN)
    print(f"  {'TOTAL':>3}  {'':17}  {'':10}  {'':5}  {'':10}  {'':10}  {'':6}  "
          f"{'':5}  {tot:>+9.2f}")


def print_asset_summary(label, trades, spread, balance, risk, show_trades=True,
                        enabled=None):
    """Full per-asset block: stats + per-rule + ledger + monthly."""
    from backtest import print_summary
    r = print_summary(label, trades, balance, risk, spread, show_trades=False)
    print(f"  Risk per trade: {risk * 100:.2f}%  |  Spread/SL: {X.spread_cost_r(spread, trades[0]['risk'] if trades else 1):.2f}R")
    print_per_rule(trades, enabled or list(S.OPT_DENSE_RULES), spread)
    if r and r.get("monthly"):
        print("\n  Monthly net P&L ($):")
        for mk in sorted(r["monthly"]):
            print(f"    {mk}:  {r['monthly'][mk]:>+10.2f}")
    if show_trades and trades:
        print_trade_ledger(trades, balance, risk, spread, title=f"TRADES — {label}")
    return r


def print_asset_section(idx, total, asset_key, prof, stats, balance, risk, show_trades):
    preset = prof.get("live_preset", "dense_wide")
    label = f"{asset_key} ({prof.get('label', asset_key)}) | {preset} | {stats['sym']}"
    print(f"\n{SEP}")
    print(f"  [{idx}/{total}]  {label}")
    print(f"  minSL={_fmt_min_sl(asset_key, stats['min_sl'])}  |  "
          f"risk={stats.get('risk_pct', RISK)*100:.1f}%  |  "
          f"spread/SL={stats.get('spread_r', 0):.2f}R  |  "
          f"rules: {', '.join(stats.get('enabled') or [])}")
    print(SEP)
    if stats["n"] == 0:
        print("  No trades in this period.")
        return None
    return print_asset_summary(label, stats["trades"], stats["spread"],
                               BAL, stats.get("risk_pct", RISK), show_trades=show_trades,
                               enabled=stats.get("enabled"))


def calibrate_asset(asset_key, mt5, spike=False):
    """Sweep min_sl to find best trade count × quality for this broker."""
    _, prof = P.get_profile(asset_key)
    try:
        sym = P.resolve_symbol_for_profile(asset_key, mt5)
    except RuntimeError as e:
        return None, str(e)
    base_sl = prof["min_sl"]
    if asset_key == "XAUUSD":
        grid = [base_sl * x for x in (0.5, 0.75, 1.0, 1.25, 1.5)]
    elif asset_key == "BTCUSD":
        grid = [base_sl * x for x in (0.5, 0.75, 1.0, 1.5, 2.0)]
    elif asset_key == "BRENT":
        grid = [base_sl * x for x in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]
    else:
        grid = [base_sl * x for x in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    best = None
    rows = []
    for sl in grid:
        p = dict(prof)
        p["min_sl"] = round(sl, 8)
        p["min_sl_mode"] = "fixed"
        s = _run(sym, p, mt5, spike=spike, asset_key=asset_key)
        if not s or s["n"] == 0:
            rows.append((sl, 0, 0, 0, 0))
            continue
        score = s["n"] * max(s["pf"], 0.1) * (s["wr"] / 100.0)
        rows.append((sl, s["n"], s["wr"], s["pf"], s["ret"]))
        if best is None or score > best["score"]:
            best = {**s, "score": score, "cal_sl": sl}
    return {"sym": sym, "rows": rows, "best": best}, None


def run_multi(cli_args=None):
    global DAYS, RISK, BAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--risk", type=float, default=RISK)
    ap.add_argument("--balance", type=float, default=BAL)
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-trades", action="store_true", help="Hide per-asset trade ledgers")
    ap.add_argument("--assets", default=None,
                    help="Comma-separated asset keys (default: all in symbol_profiles)")
    if cli_args is None:
        ns = ap.parse_args()
    elif hasattr(cli_args, "days"):
        ns = cli_args
        for attr, default in (("calibrate", False), ("no_trades", False),
                              ("balance", BAL), ("spike", False), ("assets", None)):
            if not hasattr(ns, attr):
                setattr(ns, attr, default)
    else:
        ns = ap.parse_args(cli_args)
    DAYS = ns.days
    RISK = ns.risk
    BAL = ns.balance
    show_trades = not ns.no_trades
    # --risk on CLI overrides per-asset profile risk_pct; default uses each profile
    risk_override = getattr(ns, "risk_override", None)
    if risk_override is None:
        risk_override = "--risk" in sys.argv
    cli_risk = RISK if risk_override else None

    mt5 = M.connect()
    mode = "per-asset tuned profiles"
    if ns.spike:
        mode += " + SPIKE"
    print(SEP)
    risk_hdr = (f"risk {RISK * 100:.1f}% all assets"
                if cli_risk is not None else "risk per asset profile")
    print(f"  MULTI-ASSET PORTFOLIO  |  {mode}  |  M15  |  {DAYS}d  |  "
          f"{risk_hdr}  |  balance ${BAL:,.0f}")
    print(SEP)

    if ns.calibrate:
        print(f"\n  {'asset':<8} {'symbol':<16} {'best minSL':>10} {'n':>4} {'WR':>6} {'PF':>6} {'ret':>8}")
        print("  " + "-" * 70)
        for asset in P.ASSET_KEYS:
            res, err = calibrate_asset(asset, mt5, spike=ns.spike)
            if err:
                print(f"  {asset:<8} {'NOT FOUND':<16}  —  {err}")
                continue
            b = res["best"]
            if not b:
                print(f"  {asset:<8} {res['sym']:<16}  no trades")
                continue
            print(f"  {asset:<8} {res['sym']:<16} {b['cal_sl']:>10.5g} {b['n']:>4} "
                  f"{b['wr']:>5.1f}% {b['pf']:>6.2f} {b['ret']:>+7.1f}%")
        M.shutdown(mt5)
        print(SEP)
        return

    if getattr(ns, "assets", None):
        assets = [a.strip().upper() for a in str(ns.assets).split(",") if a.strip()]
        bad = [a for a in assets if a not in P.ASSET_KEYS]
        if bad:
            print(f"  Unknown assets: {bad}. Valid: {', '.join(P.ASSET_KEYS)}")
            M.shutdown(mt5)
            return
    else:
        assets = list(P.ASSET_KEYS)
    overview = []
    for asset in assets:
        _, prof = P.get_profile(asset)
        preset = prof.get("live_preset", "dense_wide")
        try:
            sym = P.resolve_symbol_for_profile(asset, mt5)
            stats = _run(sym, prof, mt5, days=DAYS, spike=ns.spike, asset_key=asset,
                         risk_pct=cli_risk)
            overview.append(dict(asset_key=asset, preset=preset, stats=stats))
        except RuntimeError:
            overview.append(dict(asset_key=asset, preset=preset, error=True))

    print_overview_table(overview)

    results = []
    for i, row in enumerate(overview, 1):
        if row.get("error"):
            print(f"\n{SEP}")
            print(f"  [{i}/{len(assets)}]  {row['asset_key']}  —  symbol not found in MT5")
            print(SEP)
            continue
        _, prof = P.get_profile(row["asset_key"])
        r = print_asset_section(i, len(assets), row["asset_key"], prof, row["stats"],
                                BAL, RISK, show_trades)
        if r:
            results.append(r)

    if results:
        comb_n = sum(x["n"] for x in results)
        comb_ret = sum(x["ret_pct"] for x in results) / len(results)
        avg_wr = np.mean([x["wr"] for x in results])
        print(f"\n{SEP}")
        print(f"  PORTFOLIO TOTAL  |  {len(results)} assets  |  {comb_n} trades")
        print(f"  avg WR {avg_wr:.1f}%  |  avg return {comb_ret:+.1f}% per asset "
              f"(not compounded across symbols)")
        print(SEP)

    M.shutdown(mt5)


if __name__ == "__main__":
    run_multi()
