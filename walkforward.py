"""
Walk-forward / out-of-sample test — does the strategy have a REAL edge, or is it
just curve-fit to history?

Method (per asset):
  1. Split the available history into TRAIN (first part) and TEST (held-out tail).
  2. Calibrate min-SL on TRAIN only (no look-ahead into TEST).
  3. Test A  — run the CURRENT live rules (profile rules_override) on TRAIN and TEST
               separately. If TEST collapses vs TRAIN → the live rules are fragile.
  4. Test B  (optional --reselect) — re-select the "best" rule subset on TRAIN only,
               then apply it to TEST. The gap between in-sample TRAIN and
               out-of-sample TEST = how much the SELECTION itself overfits.

Nothing here modifies the strategy or any profile. Read-only research.

Usage:
  python walkforward.py
  python walkforward.py --train-frac 0.6
  python walkforward.py --reselect
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import numpy as np
import pandas as pd

import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
import portfolio_config as C
from multi_symbol_calibrate import _opt, META
from portfolio_backtest import simulate_portfolio

BAL = 1000.0


def acc_stats(trades, risk, spread):
    if not trades:
        return dict(n=0, wr=0.0, pf=0.0, ret=0.0, dd=0.0)
    acc = S.simulate_account(trades, BAL, risk, spread)
    return dict(n=acc["n"], wr=acc["wr"], pf=acc["pf"],
                ret=acc["ret_pct"], dd=acc["max_dd"])


def run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, rules, max_concurrent):
    o = P.apply_profile_to_settings(_opt(prof), prof, spread, min_sl_override=min_sl)
    o["enabled"] = list(rules)
    k = {x: o[x] for x in o if x not in META}
    k["max_concurrent"] = max_concurrent
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    return S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)


def split_trades(trades, split_ts):
    tr = [t for t in trades if t.get("exit_idx") is not None or True]
    train = [t for t in tr if pd.Timestamp(t["time"]) < split_ts]
    test = [t for t in tr if pd.Timestamp(t["time"]) >= split_ts]
    return train, test


def train_min_sl(d, split_ts, prof):
    d_train = d[d.index < split_ts]
    if len(d_train) < 60:
        d_train = d
    B_train = S.Bars(d_train)
    return X.calibrate_min_sl(B_train, B_train.t[0], prof)


def greedy_select(d, m1, B, ctx, htf, prof, spread, min_sl, split_ts, pool, mc, risk):
    """Rank each candidate rule on TRAIN, greedily build the best subset on TRAIN."""
    scored = []
    for rule in pool:
        try:
            tr = run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, [rule], mc)
        except Exception:
            continue
        train, _ = split_trades(tr, split_ts)
        if len(train) < 3:
            continue
        s = acc_stats(train, risk, spread)
        score = s["n"] * max(s["pf"], 0.05) * (s["wr"] / 100.0)
        scored.append((rule, score, s))
    scored.sort(key=lambda x: -x[1])
    cands = [r for r, _, _ in scored[:10]]

    kept, best = [], -1.0
    while True:
        best_add = None
        for rule in cands:
            if rule in kept:
                continue
            tr = run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, kept + [rule], mc)
            train, _ = split_trades(tr, split_ts)
            if not train:
                continue
            s = acc_stats(train, risk, spread)
            score = s["n"] * max(s["pf"], 0.05) * (s["wr"] / 100.0)
            if score > best:
                best_add = (rule, score)
        if best_add is None or best_add[1] <= best:
            break
        kept.append(best_add[0])
        best = best_add[1]
    return kept


def dense_pool():
    pool = []
    for k in S.OPT_DENSE_RULES:
        if k not in pool:
            pool.append(k)
    for k in S.DETECTORS:
        if k not in pool:
            pool.append(k)
    for k in S.NDS_FAMILY:
        if k not in pool:
            pool.append(k)
    return pool


def main():
    ap = argparse.ArgumentParser(description="Walk-forward out-of-sample test")
    ap.add_argument("--days", type=int, default=68,
                    help="History to fetch (M1 limited ~69d). Default 68.")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="Fraction of history used for TRAIN (rest = TEST).")
    ap.add_argument("--reselect", action="store_true",
                    help="Also re-select rules on TRAIN and test out-of-sample (slow).")
    args = ap.parse_args()

    assets = list(C.PORTFOLIO_ASSETS)
    mt5 = M.connect()
    sep = "=" * 92
    print(sep)
    print(f"  WALK-FORWARD / OUT-OF-SAMPLE TEST   |  history ~{args.days}d  "
          f"|  train {args.train_frac*100:.0f}% / test {(1-args.train_frac)*100:.0f}%")
    print("  Strategy & profiles: READ-ONLY (nothing modified)")
    print(sep)

    A_train_all, A_test_all = [], []
    B_test_all = []
    pool = dense_pool() if args.reselect else None

    rows = []
    for asset in assets:
        _, prof = P.get_profile(asset)
        risk = C.RISK_MAP.get(asset, X.profile_risk_pct(prof))
        mc = C.MAX_CONCURRENT_PER_ASSET
        try:
            sym = P.resolve_symbol_for_profile(asset, mt5)
            spread = P.live_spread(sym, prof, mt5)
            _, d, m1, _ = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
            htf_tfs = tuple(_opt(prof).get("htf_tfs", ("H4", "H1")))
            _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=htf_tfs)
        except RuntimeError as e:
            print(f"  {asset}: skip ({e})")
            continue

        B = S.Bars(d)
        ctx = S.M1Ctx(m1, d.index)
        htf = S.prepare_htf_context(htf_dfs)

        lo, hi = d.index[0], d.index[-1]
        split_ts = lo + (hi - lo) * args.train_frac
        min_sl = train_min_sl(d, split_ts, prof)

        cur_rules = list(prof.get("rules_override") or _opt(prof).get("enabled", []))
        spec = X.get_spec(sym, mt5)

        all_tr = run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, cur_rules, mc)
        tr_train, tr_test = split_trades(all_tr, split_ts)
        s_train = acc_stats(tr_train, risk, spread)
        s_test = acc_stats(tr_test, risk, spread)

        # tag for portfolio combine (out-of-sample)
        for t in tr_train:
            t["_asset"] = asset; t["_risk_pct"] = risk
            t["_spread"] = spread; t["_max_concurrent"] = mc; t["_spec"] = spec
        for t in tr_test:
            t["_asset"] = asset; t["_risk_pct"] = risk
            t["_spread"] = spread; t["_max_concurrent"] = mc; t["_spec"] = spec
        A_train_all += tr_train
        A_test_all += tr_test

        row = dict(asset=asset, label=prof.get("label", asset),
                   train=s_train, test=s_test, split=split_ts, min_sl=min_sl)

        if args.reselect:
            sel = greedy_select(d, m1, B, ctx, htf, prof, spread, min_sl,
                                split_ts, pool, mc, risk)
            re_tr = run_rules(d, m1, B, ctx, htf, prof, spread, min_sl, sel, mc) if sel else []
            re_train, re_test = split_trades(re_tr, split_ts)
            row["resel_rules"] = sel
            row["resel_train"] = acc_stats(re_train, risk, spread)
            row["resel_test"] = acc_stats(re_test, risk, spread)
            for t in re_test:
                t["_asset"] = asset; t["_risk_pct"] = risk
                t["_spread"] = spread; t["_max_concurrent"] = mc; t["_spec"] = spec
            B_test_all += re_test
        rows.append(row)

    M.shutdown(mt5)

    # ---- Test A: current live rules, in-sample vs out-of-sample ----
    print(f"\n  TEST A — CURRENT LIVE RULES  (in-sample TRAIN  →  out-of-sample TEST)")
    print(f"  {'asset':<10}  {'TRAIN: n  WR%   PF   ret%':<30}   {'TEST: n  WR%   PF   ret%':<30}")
    print("  " + "-" * 86)
    for r in rows:
        tr, te = r["train"], r["test"]
        print(f"  {r['label']:<10}  "
              f"{tr['n']:>3} {tr['wr']:>5.1f} {tr['pf']:>5.2f} {tr['ret']:>+7.1f}      "
              f"     {te['n']:>3} {te['wr']:>5.1f} {te['pf']:>5.2f} {te['ret']:>+7.1f}")

    def port(trades, tag):
        if not trades:
            print(f"  {tag}: no trades")
            return
        trades.sort(key=lambda t: (t["time"], t.get("exit_time")))
        res = simulate_portfolio(trades, BAL, max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK)
        print(f"  {tag}:  trades={res['n']}  WR={res['wr']:.1f}%  PF={res['pf']:.2f}  "
              f"DD={res['max_dd']:.1f}%  ${BAL:,.0f}→${res['final']:,.0f} "
              f"({res['ret_pct']:+.1f}%)")

    print(f"\n  PORTFOLIO (current rules):")
    port(A_train_all, "  IN-SAMPLE  TRAIN")
    port(A_test_all, "  OUT-SAMPLE TEST ")

    # ---- Test B: reselected rules ----
    if args.reselect:
        print(f"\n  TEST B — RE-SELECTED ON TRAIN, applied OUT-OF-SAMPLE on TEST")
        print(f"  {'asset':<10}  {'TRAIN(in): WR%  PF  ret%':<26}  {'TEST(oos): WR%  PF  ret%':<26}")
        print("  " + "-" * 80)
        for r in rows:
            if "resel_train" not in r:
                continue
            a, b = r["resel_train"], r["resel_test"]
            print(f"  {r['label']:<10}  "
                  f"{a['wr']:>5.1f} {a['pf']:>5.2f} {a['ret']:>+7.1f}  (n={a['n']:<3})   "
                  f"{b['wr']:>5.1f} {b['pf']:>5.2f} {b['ret']:>+7.1f}  (n={b['n']:<3})")
            print(f"             selected: {tuple(r['resel_rules'])}")
        print(f"\n  PORTFOLIO (reselected rules, OUT-OF-SAMPLE):")
        port(B_test_all, "  OUT-SAMPLE TEST ")

    print(f"\n  HOW TO READ:")
    print(f"   • If TEST WR/PF are close to TRAIN → the edge is real & stable.")
    print(f"   • If TEST collapses (WR«TRAIN, PF<1) → the rules are overfit to history.")
    print(f"   • Compare Test A vs B: if reselected TEST also collapses, the whole")
    print(f"     pipeline overfits; if it holds, only the frozen rules were stale.")
    print(sep)


if __name__ == "__main__":
    main()
