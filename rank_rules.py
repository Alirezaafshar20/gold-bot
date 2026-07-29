"""
Turn calibrate_long.json (+ optional broker json) into a ranked markdown report,
good -> bad per symbol, with a composite quality grade and a recent-regime
cross-check. Read-only analysis; changes nothing in the strategy.

  python rank_rules.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

LONG = "reports/calibrate_long.json"
BROKER = "reports/calibrate_broker.json"
OUT = "reports/rule_ranking.md"

LABEL = {"XAUUSD": "Gold", "US500": "S&P 500", "US30": "Dow Jones",
         "BRENT": "Brent Oil", "BTCUSD": "Bitcoin"}


REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE")


def score(r):
    """Composite ranking score: rewards consistency, edge, sample size, regime
    breadth; penalises DD and collapse in any single regime."""
    n = r["n"]
    pf_c = min(r["pf"], 3.0)                       # cap tiny-sample PF blow-ups
    n_factor = min(n / 50.0, 1.5)                  # need enough trades to trust
    dd_factor = 1.0 / (1.0 + r["dd"] / 15.0)
    pos = 1.0 if r["median_fold_ret"] > 0 else 0.45
    base = (r["robust_pct"] / 100.0) * pf_c * n_factor * dd_factor * pos
    # regime breadth: reward rules that profit across UP/DOWN/RANGE, not just a
    # bull tape. 0 good regimes -> 0.6x ... 3 good regimes -> 1.2x.
    rp = r.get("regimes_pos")
    if rp is not None:
        base *= 0.6 + 0.2 * rp
        if r.get("worst_regime_ret", 0.0) < -3.0:
            base *= 0.7
    return base


def regime_cell(r):
    """Compact per-regime sign string, e.g. 'U+ D- R+' (blank if no data)."""
    br = r.get("by_regime")
    if not br:
        return "–"
    tags = {"TREND_UP": "U", "TREND_DOWN": "D", "RANGE": "R"}
    out = []
    for reg in REGIMES:
        st = br.get(reg)
        if not st or st.get("n", 0) < 3:
            out.append(f"{tags[reg]}·")
        else:
            out.append(f"{tags[reg]}{'+' if st['tot_r'] > 0 else '-'}")
    return " ".join(out)


def grade(r):
    n, pf, rob, dd = r["n"], r["pf"], r["robust_pct"], r["dd"]
    if n < 15:
        return "D"      # too few trades over 4 years to trust
    if n >= 40 and rob >= 75 and pf >= 1.6 and dd <= 12:
        return "A"
    if n >= 25 and rob >= 67 and pf >= 1.4 and dd <= 16:
        return "B"
    if rob >= 55 and pf >= 1.2:
        return "C"
    return "D"


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {d["asset"]: d for d in data}


def main():
    long = load(LONG)
    broker = load(BROKER)
    if not long:
        print(f"Missing {LONG}"); return

    lines = ["# Rule ranking — walk-forward calibration (2022-04 → 2026-06)", ""]
    lines += [
        "Primary source: **Dukascopy M15, 4 years, 17–18 folds**.  ",
        "`recent` column = same rule on the **broker's last ~90 days** "
        "(sanity check; ✓ = still profitable, ✗ = losing now, – = too few trades).",
        "",
        "**Grade:** A = strong & consistent · B = good · C = marginal · "
        "D = weak / too few trades.",
        "",
        "**Columns:** n = trades over 4y · WR = win-rate · PF = profit factor · "
        "ret% = total return · DD% = max drawdown · robust = % of 90-day folds "
        "that were profitable · medFold = median fold return · "
        "regime = sign of edge in trend-Up / trend-Down / Range "
        "(`+` profitable, `-` losing, `·` too few trades).",
        "",
    ]

    summary = {}
    for asset in long:
        res = long[asset]["results"]
        brk = broker.get(asset, {}).get("results", {}) if broker else {}
        rows = []
        for rule, r in res.items():
            if "error" in r or r.get("n", 0) < 5:
                continue
            r2 = dict(r, rule=rule, score=score(r), grade=grade(r))
            b = brk.get(rule)
            if b and "error" not in b and b.get("n", 0) >= 5:
                r2["recent"] = "✓" if b["pf"] >= 1.0 else "✗"
                r2["recent_pf"] = b["pf"]
            else:
                r2["recent"] = "–"
                r2["recent_pf"] = None
            rows.append(r2)
        rows.sort(key=lambda x: -x["score"])
        summary[asset] = rows

        lines.append(f"## {LABEL.get(asset, asset)}  ({asset})")
        lines.append("")
        lines.append("| # | rule | family | grade | n | WR% | PF | ret% | DD% | "
                     "robust | medFold | regime | recent |")
        lines.append("|---|------|--------|:-----:|--:|----:|---:|-----:|----:|"
                     "------:|--------:|:------:|:------:|")
        for i, r in enumerate(rows, 1):
            pf = min(r["pf"], 99.99)
            lines.append(
                f"| {i} | `{r['rule']}` | {str(r['family'])[:16]} | "
                f"**{r['grade']}** | {r['n']} | {r['wr']:.0f} | {pf:.2f} | "
                f"{r['ret']:+.0f} | {r['dd']:.1f} | {r['robust_pct']:.0f}% | "
                f"{r['median_fold_ret']:+.1f} | {regime_cell(r)} | {r['recent']} |")
        lines.append("")

    # recommendation = grade A/B, not failing recently, top by score, max 6
    lines.append("---")
    lines.append("## Recommended rules_override (my pick)")
    lines.append("")
    lines.append("Grade A/B, still profitable on recent broker data where testable, "
                 "ranked by score (which now rewards rules that profit across "
                 "trend-Up / trend-Down / Range, not just a bull tape). "
                 "Compare with the current live set.")
    lines.append("")
    rec_out = {}
    for asset in long:
        rows = summary[asset]
        picks = [r["rule"] for r in rows
                 if r["grade"] in ("A", "B") and r["recent"] != "✗"][:6]
        if len(picks) < 3:  # fallback: best by score regardless of recent
            picks = [r["rule"] for r in rows if r["grade"] in ("A", "B", "C")][:5]
        rec_out[asset] = picks
        reg_pick = long[asset].get("regime_robust")
        line = f"- **{LABEL.get(asset, asset)} ({asset})**: `{tuple(picks)}`"
        if reg_pick:
            line += f"  ·  regime-robust pick: `{tuple(reg_pick)}`"
        lines.append(line)
    lines.append("")

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {OUT}")
    for a, p in rec_out.items():
        print(f"  {a}: {p}")


if __name__ == "__main__":
    main()
