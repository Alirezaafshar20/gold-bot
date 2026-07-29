"""Finish greedy Brent subset from top DENSE-WIDE research picks."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X

META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)

CANDIDATES = [
    "ADX_L", "AB_IB_BO", "NW_W4", "FL", "ICT_MIT_S", "ICT_OTE_L",
    "AB_DBOT", "WYCK_SOW", "NDS_FRESH", "ICT_SB_S",
]


def main():
    _, prof = P.get_profile("BRENT")
    mt5 = M.connect()
    sym = P.resolve_symbol_for_profile("BRENT", mt5)
    spread = P.live_spread(sym, prof, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=90)
    _, htf_dfs = M.fetch_htf_bars(sym, days=90, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    min_sl = X.calibrate_min_sl(B, cutoff, prof)
    risk = X.profile_risk_pct(prof)

    def combo(enabled):
        o = P.apply_profile_to_settings(
            S.dense_plus_settings(wide=True), prof, spread, min_sl_override=min_sl)
        o["enabled"] = list(enabled)
        k = {x: o[x] for x in o if x not in META}
        k["htf_context"] = htf
        k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]
        if not tr:
            return None
        acc = S.simulate_account(tr, 1000.0, risk, spread)
        return dict(n=len(tr), wr=acc["wr"], pf=acc["pf"], ret=acc["ret_pct"])

    kept = []
    best = -1.0
    for _ in range(len(CANDIDATES)):
        best_add = None
        for rule in CANDIDATES:
            if rule in kept:
                continue
            st = combo(kept + [rule])
            if not st:
                continue
            score = st["n"] * max(st["pf"], 0.05) * (st["wr"] / 100.0)
            if score > best:
                best_add = (rule, st, score)
        if best_add is None:
            break
        rule, st, score = best_add
        if score <= best and kept:
            break
        kept.append(rule)
        best = score
        print(f"  + {rule} -> n={st['n']} WR={st['wr']:.1f}% PF={st['pf']:.2f} ret={st['ret']:+.1f}%")

    final = combo(kept)
    print(f"\n  RECOMMENDED rules_override: {tuple(kept)}")
    if final:
        print(f"  Combined: n={final['n']} WR={final['wr']:.1f}% PF={final['pf']:.2f} "
              f"ret={final['ret']:+.1f}%")
    print(f"  Current BRENT profile: {prof.get('rules_override')}")


if __name__ == "__main__":
    main()
