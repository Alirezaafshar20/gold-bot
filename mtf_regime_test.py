"""
Compare hybrid (old) vs MTF v2 regime on the same 30d Gold portfolio path.

  python mtf_regime_test.py
  python mtf_regime_test.py --days 60
"""
import argparse
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import strategy as S
import mt5_data as M
import symbol_profiles as P
import portfolio_config as C
import floating_config as FC
import meta_gate as MG
import regime as RG
from multi_symbol_calibrate import _run
from portfolio_backtest import (
    prepare_portfolio_trades, simulate_portfolio, DEFAULT_BAL, RISK_MAP,
)

HYBRID = {
    "regime_detector": "hybrid",
    "regime_er_hi": 0.40,
    "regime_er_lo": 0.25,
}

MTF = dict(FC.REGIME)


def bucket_stats(trades, spread=0.28):
    if not trades:
        return dict(n=0, wr=0.0, avg_r=0.0, tot_r=0.0)
    Rs = np.array([t["R"] - spread / max(t.get("risk", 1), 1e-9) for t in trades])
    return dict(
        n=len(Rs),
        wr=100.0 * np.mean(Rs > 0.05),
        avg_r=float(Rs.mean()),
        tot_r=float(Rs.sum()),
    )


def build_rmap(sym, prof, mt5, days, det_cfg):
    opt = P.preset_for_profile(prof)
    opt = P.apply_profile_to_settings(opt, prof, P.live_spread(sym, prof, mt5))
    opt.update(det_cfg)
    _, htf_dfs = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=("H4", "H1"))
    htf = S.prepare_htf_context(htf_dfs)
    return S.build_regime_map(
        htf, tf=opt.get("regime_tf", "H1"),
        detector=opt.get("regime_detector", "hybrid"),
        win=opt.get("regime_win", 40),
        er_hi=opt.get("regime_er_hi", 0.40),
        er_lo=opt.get("regime_er_lo", 0.25),
        confirm=opt.get("regime_confirm", 2),
        h4_win=opt.get("regime_h4_win", 30),
        h4_lookback=opt.get("regime_h4_lookback", 60),
    )


def run_variant(asset, mt5, days, oos_mg, det_cfg, label):
    _, prof = P.get_profile(asset)
    sym = P.resolve_symbol_for_profile(asset, mt5)
    risk = RISK_MAP[asset]
    stats = _run(
        sym, prof, mt5, days=days, asset_key=asset, risk_pct=risk,
        max_concurrent=C.max_concurrent(asset),
        meta_gate_override=oos_mg,
        opt_extra=det_cfg,
    )
    rmap = build_rmap(sym, prof, mt5, days, det_cfg)
    raw = []
    for t in stats.get("trades") or []:
        tc = dict(t)
        tc["_asset"] = asset
        tc["_risk_pct"] = risk
        tc["_spread"] = stats["spread"]
        tc["_max_concurrent"] = C.max_concurrent(asset)
        raw.append(tc)
    import weekly_adaptive as WA
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
        raw, wf = WA.apply_walkforward_filter(raw, mt5=mt5, assets=(asset,))
    else:
        wf = {"before": len(raw), "after": len(raw)}
    trades = prepare_portfolio_trades(raw, {asset: rmap}, oos_mg)
    sim = simulate_portfolio(trades, DEFAULT_BAL, size_compound=False)
    return dict(
        label=label,
        wf=wf,
        raw_n=len(raw),
        filt_n=len(trades),
        sim=sim,
        trades=trades,
        rmap=rmap,
    )


def print_regime_mix(trades, title):
    c = Counter(t.get("_regime", "RANGE") for t in trades)
    m = Counter(t.get("_macro", 0) for t in trades)
    tot = len(trades) or 1
    print(f"\n  {title}")
    print("    regime: " + "  ".join(f"{k} {100*c[k]/tot:.0f}%" for k in RG.REGIMES))
    if any(k != 0 for k in m):
        bull = sum(v for k, v in m.items() if k > 0)
        bear = sum(v for k, v in m.items() if k < 0)
        neut = m.get(0, 0)
        print(f"    macro : BULL {100*bull/tot:.0f}%  BEAR {100*bear/tot:.0f}%"
              f"  NEUTRAL {100*neut/tot:.0f}%")


def print_wr_buckets(trades, title):
    buckets = {"WITH-TREND": [], "COUNTER": [], "RANGE": []}
    for t in trades:
        reg = t.get("_regime", "RANGE")
        d = str(t.get("dir", "long")).lower()
        if reg == "TREND_UP":
            buckets["WITH-TREND" if d == "long" else "COUNTER"].append(t)
        elif reg == "TREND_DOWN":
            buckets["WITH-TREND" if d == "short" else "COUNTER"].append(t)
        else:
            buckets["RANGE"].append(t)
    print(f"\n  {title}")
    print(f"    {'bucket':<14}{'n':>5}{'WR%':>7}{'avgR':>8}")
    print("    " + "-" * 34)
    for b in ("WITH-TREND", "COUNTER", "RANGE"):
        st = bucket_stats(buckets[b])
        if st["n"]:
            print(f"    {b:<14}{st['n']:>5}{st['wr']:>7.1f}{st['avg_r']:>+8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--asset", default="XAUUSD")
    args = ap.parse_args()
    asset = args.asset.upper()

    oos_mg = FC.load_meta_gate(assets=[asset])
    mt5 = M.connect()
    try:
        print("=" * 80)
        print(f"  MTF REGIME v2 TEST  |  {asset}  |  {args.days}d  |  fixed-risk tiered")
        print("=" * 80)

        base = run_variant(asset, mt5, args.days, oos_mg, HYBRID, "hybrid (old)")
        mtf = run_variant(asset, mt5, args.days, oos_mg, MTF, "MTF v2")

        print(f"\n  {'mode':<18}{'raw':>5}{'filt':>5}{'taken':>6}{'WR%':>7}"
              f"{'PF':>7}{'DD%':>7}{'ret%':>8}")
        print("  " + "-" * 62)
        for r in (base, mtf):
            s = r["sim"]
            print(f"  {r['label']:<18}{r['raw_n']:>5}{r['filt_n']:>5}{s['n']:>6}"
                  f"{s['wr']:>7.1f}{s['pf']:>7.2f}{s['max_dd']:>7.1f}"
                  f"{s['ret_pct']:>+8.1f}")

        for r in (base, mtf):
            print_regime_mix(r["trades"], f"Trade mix — {r['label']}")
            if r["rmap"] is not None:
                cov = r["rmap"].coverage()
                print(f"    H1 coverage: UP {cov['TREND_UP']:.0f}%"
                      f"  DOWN {cov['TREND_DOWN']:.0f}%  RANGE {cov['RANGE']:.0f}%")
                if hasattr(r["rmap"], "macro_coverage"):
                    mc = r["rmap"].macro_coverage()
                    print(f"    macro cov  : BULL {mc['BULL']:.0f}%"
                          f"  BEAR {mc['BEAR']:.0f}%  NEUTRAL {mc['NEUTRAL']:.0f}%")

        for r in (base, mtf):
            print_wr_buckets(r["trades"], f"WR buckets — {r['label']}")

        # rule WR for MTF
        print(f"\n  WR by rule (MTF v2):")
        rules = {}
        for t in mtf["trades"]:
            rk = t.get("rule", "?")
            rules.setdefault(rk, []).append(t)
        print(f"    {'rule':<12}{'n':>4}{'WR%':>7}{'avgR':>8}")
        print("    " + "-" * 32)
        for rk, ts in sorted(rules.items()):
            st = bucket_stats(ts)
            print(f"    {rk:<12}{st['n']:>4}{st['wr']:>7.1f}{st['avg_r']:>+8.2f}")

        ds = mtf["sim"]
        bs = base["sim"]
        dw = ds["wr"] - bs["wr"]
        print(f"\n  خلاصه: WR {bs['wr']:.1f}% → {ds['wr']:.1f}% ({dw:+.1f}pp)"
              f"  |  trades {bs['n']} → {ds['n']}"
              f"  |  ret {bs['ret_pct']:+.1f}% → {ds['ret_pct']:+.1f}%")
        print("=" * 80)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
