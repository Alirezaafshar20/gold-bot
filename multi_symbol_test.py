"""Multi-symbol with asset-appropriate min SL."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import strategy as S
import mt5_data as M

DAYS = 180
BAL, RISK = 1000.0, 0.01
META = ("spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
        "dense", "medium", "dense_plus", "dense_wide")

TESTS = [
    ("XAUUSD", ["XAUUSD@", "XAUUSD"], 0.28, 3.0, "gold tuned"),
    ("BTCUSD", ["BTCUSD@", "BTCUSD"], 30.0, 3.0, "gold minSL $3"),
    ("BRENT", ["BRENTCASH", "UKBRENT.V26", "WTICASH"], 0.05, 0.05, "oil adapted"),
    ("EURUSD", ["EURUSD@", "EURUSD"], 0.00012, 0.0003, "forex adapted"),
    ("EURUSD", ["EURUSD@", "EURUSD"], 0.00012, 3.0, "gold minSL $3 (broken)"),
]


def resolve(candidates, mt5):
    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        if not info.visible and not mt5.symbol_select(sym, True):
            continue
        if mt5.symbol_info_tick(sym) is None:
            continue
        return sym
    return None


def run(sym, spread, min_sl, opt, mt5):
    _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=DAYS)
    _, htf_dfs = M.fetch_htf_bars(sym, days=DAYS, mt5=mt5, tfs=("H4", "H1"))
    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    o = dict(opt)
    o["min_risk_usd"] = min_sl
    k = {x: o[x] for x in o if x not in META}
    k["htf_context"] = htf
    tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
          if t["time"] >= cutoff]
    if not tr:
        return None
    r = S.simulate_account(tr, BAL, RISK, spread)
    rules = {}
    for t in tr:
        rk = t.get("setup") or t.get("rule", "?")
        rules[rk] = rules.get(rk, 0) + 1
    return dict(n=r["n"], wr=r["wr"], ret=r["ret_pct"], pf=r["pf"], dd=r["max_dd"], rules=rules)


def main():
    mt5 = M.connect()
    opt = S.dense_settings()
    print("=" * 88)
    print("  MULTI-SYMBOL — DENSE + HARM_BAT | M15 | 180d")
    print("=" * 88)
    print(f"  {'asset':<8} {'symbol':<14} {'minSL':>8} {'note':<22} {'n':>4} {'WR':>6} {'ret':>7} {'DD':>5}")
    print("  " + "-" * 78)

    for label, cands, spread, min_sl, note in TESTS:
        sym = resolve(cands, mt5)
        if not sym:
            print(f"  {label:<8} {'NOT FOUND':<14} {'—':>8} {note:<22}    —      —       —     —")
            continue
        tick = mt5.symbol_info_tick(sym)
        if tick and tick.ask and tick.bid:
            spread = max(spread, tick.ask - tick.bid)
        try:
            s = run(sym, spread, min_sl, opt, mt5)
            if not s:
                print(f"  {label:<8} {sym:<14} {min_sl:>8} {note:<22}    0      —       —     —")
            else:
                print(f"  {label:<8} {sym:<14} {min_sl:>8} {note:<22} {s['n']:>4} {s['wr']:>5.1f}% "
                      f"{s['ret']:>+6.1f}% {s['dd']:>4.1f}%")
                print(f"           rules: {s['rules']}")
        except Exception as e:
            print(f"  {label:<8} {sym:<14}  ERROR: {e}")

    M.shutdown(mt5)
    print("=" * 88)


if __name__ == "__main__":
    main()
