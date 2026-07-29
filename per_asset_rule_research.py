"""Rank each DENSE rule per asset — find which rules hurt on BTC/Brent/EUR."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import strategy as S
import mt5_data as M
import symbol_profiles as P

DAYS = 90
BAL, RISK, SPREAD_KEY = 1000.0, 0.01, "spread"
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
        "fib_spike_exempt", "spike_params")
ALL_RULES = list(S.OPT_DENSE_RULES)


def _stats(trades, spread):
    if not trades:
        return None
    R = np.array([t["R"] - spread / t["risk"] for t in trades])
    w, l = R[R > 0], R[R <= 0]
    pf = w.sum() / (-l.sum()) if l.sum() < 0 else 999.0
    acc = S.simulate_account(trades, BAL, RISK, spread)
    return dict(n=len(R), wr=(R > 0).mean() * 100, pf=pf, totR=R.sum(), ret=acc["ret_pct"])


def run_asset(asset, mt5, opt_base):
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    spread = P.live_spread(sym, prof, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    def bt(rules):
        o = P.apply_profile_to_settings(opt_base, prof, spread)
        o["enabled"] = list(rules)
        k = {x: o[x] for x in o if x not in META}
        k["htf_context"] = htf
        k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]
        return tr

    full = bt(ALL_RULES)
    base = _stats(full, spread)
    print(f"\n  === {asset} ({sym})  full DENSE-WIDE  n={base['n']} WR={base['wr']:.1f}% "
          f"PF={base['pf']:.2f} ret={base['ret']:+.1f}% ===")
    print(f"  {'rule':<12}{'n':>5}{'WR%':>7}{'PF':>7}{'totR':>8}")
    rows = []
    for rule in ALL_RULES:
        sub = [t for t in full if S.trades_for_enabled_rule(rule, [t])]
        s = _stats(sub, spread)
        if not s or s["n"] == 0:
            print(f"  {rule:<12}    -")
            continue
        rows.append((rule, s))
        print(f"  {rule:<12}{s['n']:>5}{s['wr']:>6.1f}%{s['pf']:>7.2f}{s['totR']:>+8.1f}")

    # greedy keep: start empty, add rules that improve score
    kept = []
    best_score = -1
    best_ret = -999
    for _ in range(len(ALL_RULES)):
        best_add = None
        for rule in ALL_RULES:
            if rule in kept:
                continue
            trial = kept + [rule]
            s = _stats(bt(trial), spread)
            if not s:
                continue
            score = s["n"] * max(s["pf"], 0.05) * (s["wr"] / 100.0)
            if score > best_score or (abs(score - best_score) < 0.01 and s["ret"] > best_ret):
                best_add = (rule, s, score)
        if best_add is None:
            break
        rule, s, score = best_add
        if score <= best_score and kept:
            break
        kept.append(rule)
        best_score = score
        best_ret = s["ret"]
    rec_s = _stats(bt(kept), spread)
    print(f"\n  RECOMMENDED subset ({len(kept)} rules): {kept}")
    if rec_s:
        print(f"  -> n={rec_s['n']} WR={rec_s['wr']:.1f}% PF={rec_s['pf']:.2f} ret={rec_s['ret']:+.1f}%")
    return kept, rec_s


def main():
    global DAYS
    DAYS = 90
    mt5 = M.connect()
    opt = S.dense_plus_settings(wide=True)
    print("=" * 72)
    print("  PER-ASSET RULE RESEARCH  |  DENSE-WIDE base  |  90d")
    print("=" * 72)
    recs = {}
    for asset in P.ASSET_KEYS:
        try:
            kept, st = run_asset(asset, mt5, opt)
            recs[asset] = kept
        except Exception as e:
            print(f"\n  {asset}: ERROR {e}")
    M.shutdown(mt5)
    print("\n" + "=" * 72)
    print("  COPY TO symbol_profiles.py rules_override:")
    for a, rules in recs.items():
        print(f'    "{a}": {tuple(rules)},')
    print("=" * 72)


if __name__ == "__main__":
    main()
