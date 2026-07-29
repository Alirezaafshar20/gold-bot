"""
Rank ALL gold-tested rules on any calibrated asset (Brent, Dow, BTC, ...).

Does NOT modify strategy.py or XAUUSD gold defaults.
Reads asset profile from symbol_profiles (read-only).

Run:
  python asset_rule_research.py --asset US30 --days 90
  python asset_rule_research.py --asset BRENT --scope dense --preset DENSE-WIDE
  python asset_rule_research.py --asset US30 --out reports/us30_rule_ranking.txt
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
import extended_strategies as EX
import wave_styles as WS
import wyckoff as WY
import al_brooks as AB
import spike_strategies as SP

META = (
    "spread_usd", "nds_mode", "nds_extended", "balanced", "balanced_plus",
    "dense", "medium", "dense_plus", "dense_wide", "spike_mode",
    "fib_spike_exempt", "spike_params",
)

BAL = 1000.0
MIN_N = 3

FILTER_PRESETS = (
    ("STABLE", lambda: S.stable_settings()),
    ("DENSE", lambda: S.dense_settings()),
    ("DENSE-WIDE", lambda: S.dense_plus_settings(wide=True)),
)


def collect_rules(scope):
    rules = {}

    def add(name, family):
        if name and name not in rules:
            rules[name] = family

    for k in S.DETECTORS:
        add(k, "SMC Core")
    for k in S.NDS_FAMILY:
        add(k, "NDS Family")
    for k in S.OPT_DENSE_RULES:
        add(k, "DENSE Live")
    for k in EX.EXT_ALL_RULES:
        meta = EX.EXT_RULE_META.get(k, {})
        add(k, meta.get("label", "Extended"))
    for k in WS.STYLE_ALL_RULES:
        meta = WS.STYLE_RULE_META.get(k, {})
        add(k, meta.get("label", "Wave Styles"))
    for k in WY.WYCK_ALL_RULES:
        meta = WY.WYCK_RULE_META.get(k, {})
        add(k, meta.get("label", "Wyckoff"))
    for k in AB.AB_ALL_RULES:
        add(k, "Al Brooks")
    for k in SP.SPIKE_DETECTORS:
        add(k, "Spike")

    if scope == "core":
        keep = {"SMC Core", "NDS Family", "DENSE Live"}
        return {k: v for k, v in rules.items() if v in keep}
    if scope == "dense":
        drop = {"Al Brooks", "Spike"}
        return {k: v for k, v in rules.items() if v not in drop and v != "Wave Styles"
                and "Wyckoff" not in v}
    return rules


def tier(row):
    if row["n"] < MIN_N:
        return "D"
    if row["wr"] >= 70 and row["pf"] >= 2.0:
        return "A"
    if row["wr"] >= 55 and row["pf"] >= 1.3:
        return "B"
    if row["wr"] >= 45:
        return "C"
    return "F"


def asset_kw(preset, prof, spread, min_sl, htf, rule):
    o = P.apply_profile_to_settings(preset, prof, spread, min_sl_override=min_sl)
    o["enabled"] = [rule]
    k = {x: o[x] for x in o if x not in META}
    k["htf_context"] = htf
    k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
    return k


def test_rule(rule, d, m1, B, ctx, htf, cutoff, preset, prof, spread, min_sl, risk):
    try:
        k = asset_kw(preset, prof, spread, min_sl, htf, rule)
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]
    except Exception as exc:
        return {"error": str(exc)}
    if not tr:
        return None
    acc = S.simulate_account(tr, BAL, risk, spread)
    Rs = np.array([t["R"] - spread / t["risk"] for t in tr])
    w, l = Rs[Rs > 0], Rs[Rs <= 0]
    pf = float(w.sum() / (-l.sum())) if l.sum() < 0 else 999.0
    return {
        "n": len(tr), "wr": acc["wr"], "ret": acc["ret_pct"], "pf": pf,
        "dd": acc["max_dd"], "tot_r": float(Rs.sum()),
    }


def greedy_subset(rule_stats, min_tier=("A", "B")):
    picks = []
    for rule, row in rule_stats.items():
        if row and row.get("tier") in min_tier and row["n"] >= MIN_N:
            score = row["n"] * max(row["pf"], 0.05) * (row["wr"] / 100.0)
            picks.append((rule, row, score))
    picks.sort(key=lambda x: (-x[2], -x[1]["wr"], -x[1]["ret"]))
    return picks


def main():
    ap = argparse.ArgumentParser(description="Per-asset rule ranking (gold model untouched)")
    ap.add_argument("--asset", required=True, choices=P.ASSET_KEYS)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--scope", choices=("core", "dense", "full"), default="full")
    ap.add_argument("--preset", default=None,
                    help="Run only one filter preset (STABLE, DENSE, DENSE-WIDE)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--greedy-only", metavar="JSON")
    args = ap.parse_args()

    asset = args.asset.upper()
    if args.out is None:
        args.out = f"reports/{asset.lower()}_rule_ranking.txt"

    _, prof = P.get_profile(asset)
    rules = collect_rules(args.scope)
    presets = FILTER_PRESETS
    if args.preset:
        key = args.preset.upper().replace("_", "-")
        presets = [(n, fn) for n, fn in FILTER_PRESETS if n == key]
        if not presets:
            print(f"Unknown preset {args.preset!r}")
            sys.exit(1)

    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    log("=" * 78)
    log(f"  {prof.get('label', asset).upper()} RULE RESEARCH — asset {asset}")
    log(f"  Scope: {args.scope} | {len(rules)} rules | {args.days}d | presets: "
        + ", ".join(n for n, _ in presets))
    log("  Gold (XAUUSD) model: READ-ONLY")
    log("=" * 78)

    mt5 = M.connect()
    try:
        sym = P.resolve_symbol_for_profile(asset, mt5)
        spread = P.live_spread(sym, prof, mt5)
        _, d, m1, cutoff = M.fetch_pair(sym, "M15", mt5=mt5, days=args.days)
        _, htf_dfs = M.fetch_htf_bars(sym, days=args.days, mt5=mt5, tfs=("H4", "H1"))
    finally:
        M.shutdown(mt5)

    B = S.Bars(d)
    ctx = S.M1Ctx(m1, d.index)
    htf = S.prepare_htf_context(htf_dfs)
    min_sl = X.calibrate_min_sl(B, cutoff, prof)
    risk = X.profile_risk_pct(prof)

    log(f"\n  Symbol: {sym}  spread={spread:.5f}  min_sl(ATR)={min_sl:.4f}  risk={risk*100:.2f}%")
    log(f"  Test window from: {cutoff}")

    if args.greedy_only:
        with open(args.greedy_only, encoding="utf-8") as f:
            cached = json.load(f)
        all_results = cached.get("results", {})
        sym = cached.get("symbol", sym)
        spread = cached.get("spread", spread)
        min_sl = cached.get("min_sl", min_sl)
        risk = cached.get("risk_pct", risk)
        log(f"\n  Loaded cached results from {args.greedy_only}")
    else:
        all_results = {}
        for preset_name, preset_fn in presets:
            preset = preset_fn()
            log(f"\n{'=' * 78}")
            log(f"  FILTER: {preset_name}")
            log(f"{'=' * 78}")
            log(f"  {'RULE':<16} {'family':<22} {'n':>4} {'WR%':>6} {'PF':>6} {'ret%':>7} tier")
            log("  " + "-" * 72)

            rows = []
            n_rules = len(rules)
            for idx, (rule, family) in enumerate(sorted(rules.items())):
                print(f"  [{preset_name}] {idx + 1}/{n_rules} {rule}...", end="\r", flush=True)
                row = test_rule(rule, d, m1, B, ctx, htf, cutoff, preset, prof, spread, min_sl, risk)
                if row and "error" in row:
                    log(f"  {rule:<16} {family[:22]:<22}  ERR {row['error'][:40]}")
                    continue
                if row:
                    row["tier"] = tier(row)
                    row["family"] = family
                rows.append((rule, family, row))
                all_results.setdefault(preset_name, {})[rule] = row

            print(" " * 60)
            rows.sort(key=lambda x: (-(x[2]["wr"] if x[2] else 0),
                                     -(x[2]["pf"] if x[2] else 0),
                                     -(x[2]["n"] if x[2] else 0)))
            for rule, family, row in rows:
                if row:
                    log(f"  {rule:<16} {family[:22]:<22} {row['n']:>4} {row['wr']:>5.1f}% "
                        f"{row['pf']:>6.2f} {row['ret']:>+6.1f}%  {row['tier']}")
                else:
                    log(f"  {rule:<16} {family[:22]:<22}    0      —      —       —   —")

    dw = all_results.get("DENSE-WIDE", all_results.get(list(all_results.keys())[-1], {}))
    log(f"\n{'=' * 78}")
    log(f"  TOP PICKS FOR {asset} (DENSE-WIDE, tier A or B, n>={MIN_N})")
    log("=" * 78)
    enriched = {r: {**row, "tier": tier(row)} for r, row in dw.items() if row and "error" not in row}
    picks = greedy_subset(enriched)
    if picks:
        for rule, row, score in picks[:25]:
            fam = rules.get(rule, "?")
            log(f"    {rule:<16} n={row['n']:>3}  WR={row['wr']:5.1f}%  PF={row['pf']:5.2f}  "
                f"ret={row['ret']:+6.1f}%  tier={row['tier']}  ({fam})")
    else:
        log("    No rules reached tier A/B on DENSE-WIDE.")

    log(f"\n  GREEDY {asset} SUBSET (top 10 tier A/B, max_concurrent from profile):")

    def combo_stats(enabled):
        o = P.apply_profile_to_settings(S.dense_plus_settings(wide=True), prof, spread,
                                        min_sl_override=min_sl)
        o["enabled"] = list(enabled)
        k = {x: o[x] for x in o if x not in META}
        k["htf_context"] = htf
        k["detector_params"] = P.merge_params(S.DEFAULT_PARAMS, prof)
        tr = [t for t in S.run_backtest(d, m1, B=B, ctx=ctx, tf_min=15, **k)
              if t["time"] >= cutoff]
        if not tr:
            return None
        acc = S.simulate_account(tr, BAL, risk, spread)
        return dict(n=len(tr), wr=acc["wr"], pf=acc["pf"], ret=acc["ret_pct"],
                    dd=acc["max_dd"])

    candidates = [r for r, row, _ in picks[:10]] if picks else []
    kept = []
    best_score = -1.0
    for _ in range(len(candidates)):
        best_add = None
        for rule in candidates:
            if rule in kept:
                continue
            print(f"  greedy try {kept + [rule]}...", flush=True)
            st = combo_stats(kept + [rule])
            if not st:
                continue
            score = st["n"] * max(st["pf"], 0.05) * (st["wr"] / 100.0)
            if score > best_score:
                best_add = (rule, st, score)
        if best_add is None:
            break
        rule, st, score = best_add
        if score <= best_score and kept:
            break
        kept.append(rule)
        best_score = score
        log(f"    + {rule} -> n={st['n']} WR={st['wr']:.1f}% PF={st['pf']:.2f} "
            f"ret={st['ret']:+.1f}% DD={st['dd']:.1f}%")
    final = combo_stats(kept) if kept else None
    log(f"    Recommended rules_override: {tuple(kept)}")
    if final:
        log(f"    Combined: n={final['n']} WR={final['wr']:.1f}% PF={final['pf']:.2f} "
            f"ret={final['ret']:+.1f}% DD={final['dd']:.1f}%")
    log(f"\n  Current {asset} profile rules: {prof.get('rules_override')}")
    log(f"  To apply: edit ONLY {asset} rules_override in symbol_profiles.py")
    log(f"  Generated: {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 78)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "asset": asset, "symbol": sym, "spread": spread, "min_sl": min_sl,
            "risk_pct": risk, "scope": args.scope, "days": args.days,
            "results": all_results, "recommended": kept, "combined": final,
        }, f, indent=2, default=str)

    log(f"\n  Saved: {args.out}")
    log(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
