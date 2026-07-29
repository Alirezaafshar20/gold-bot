"""Compare Brent rule portfolios on DENSE-WIDE."""
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

SETS = {
    "Current profile": ("OB", "NDS", "MA_X_S", "ICT_SB_L"),
    "Research A (return)": ("ADX_L", "AB_IB_BO", "NW_W4", "FL"),
    "Research B (ICT+core)": ("OB", "ICT_SB_S", "ICT_MIT_S", "FL", "NDS_FRESH"),
    "Research C (volume)": ("FVG", "NW_W4", "EW_W4", "ADX_L", "OB"),
    "Hybrid SMC+research": ("OB", "ADX_L", "AB_IB_BO", "ICT_MIT_S", "FL"),
}


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

    print(f"BRENT {sym} | min_sl={min_sl:.4f} | spread={spread:.5f}\n")
    for label, rules in SETS.items():
        o = P.apply_profile_to_settings(
            S.dense_plus_settings(wide=True), prof, spread, min_sl_override=min_sl)
        o["enabled"] = list(rules)
        k = {x: o[x] for x in o if x not in META}
        k["htf_context"] = htf
        k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]
        acc = S.simulate_account(tr, 1000.0, risk, spread)
        print(f"  {label:28s} n={acc['n']:3d}  WR={acc['wr']:5.1f}%  "
              f"PF={acc['pf']:5.2f}  ret={acc['ret_pct']:+6.1f}%")


if __name__ == "__main__":
    main()
