"""Compare DENSE before vs after extended tier-B rules."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M

DAYS, SPREAD, BAL, RISK = 180, 0.28, 1000, 0.01
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")
OLD_RULES = ("OB", "NDS", "DEMAND", "HARM_BAT")


def run(d, m1, B, ctx, htf, cutoff, rules):
    opt = S.dense_settings()
    opt["enabled"] = list(rules)
    k = {x: opt[x] for x in opt if x not in META}
    k["htf_context"] = htf
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
          if t["time"] >= cutoff]
    if not tr:
        return None, {}
    r = S.simulate_account(tr, BAL, RISK, SPREAD)
    rules_ct = {}
    for t in tr:
        rk = t.get("setup") or t.get("rule", "?")
        rules_ct[rk] = rules_ct.get(rk, 0) + 1
    return r, rules_ct


def main():
    mt5 = M.connect()
    sym = M.resolve_symbol(M.DEFAULT_SYMBOL, mt5)
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    M.shutdown(mt5)
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)

    print("=" * 72)
    print("  DENSE COMPARE — before vs + MA_X_S SQZ_S ICT_SB_L ICT_MIT_S")
    print(f"  XAUUSD M15 | {DAYS}d requested (~70d M1 window)")
    print("=" * 72)

    for label, rules in [
        ("DENSE (before)", OLD_RULES),
        ("DENSE+ (extended)", S.OPT_DENSE_RULES),
    ]:
        r, rc = run(d, m1, B, ctx, htf, cutoff, rules)
        if r:
            print(f"\n  {label}")
            print(f"    rules: {', '.join(rules)}")
            print(f"    trades={r['n']}  WR={r['wr']:.1f}%  ret={r['ret_pct']:+.1f}%  "
                  f"PF={r['pf']:.2f}  DD={r['max_dd']:.1f}%")
            print(f"    by setup: {rc}")
        else:
            print(f"\n  {label}: 0 trades")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
