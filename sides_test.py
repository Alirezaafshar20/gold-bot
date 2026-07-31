"""Replay with side restrictions applied in-process only.

Both restriction tables in portfolio_config can be exercised here without
writing anything to disk, so a live bot reading that module keeps its current
behaviour while a variant is measured:

    RULE_SIDES     which sides a named rule may trade
    REGIME_SIDES   which sides anything may trade inside a regime label

Both go through portfolio_config.rule_allowed, which live_portfolio.register_zones
consults for every candidate. That function is the shared engine, so whatever is
measured here is what live would do with the same table.

Usage
  python sides_test.py M15 60 --regime RANGE=short
  python sides_test.py M15 60 --rules NDS,FL,NDS_BOS,NDS_FVG=short
  python sides_test.py M15 60 --regime RANGE=short --rules NDS=short --end 2026-01-30

Rule and regime tags accept either separator (NDS-FVG or NDS_FVG).
"""
import argparse
import sys

import portfolio_config as C

ap = argparse.ArgumentParser()
ap.add_argument("tf")
ap.add_argument("days")
ap.add_argument("--regime", action="append", default=[],
                help="LABEL=side[,side]  e.g. RANGE=short")
ap.add_argument("--rules", action="append", default=[],
                help="RULE[,RULE]=side[,side]  e.g. NDS,FL=short")
ap.add_argument("--end", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--spread", default=None,
                help="USD spread charged on every fill; see spread_probe.py")
args = ap.parse_args()


def _sides(raw):
    return tuple(s.strip().lower() for s in raw.split(",") if s.strip())


parts = []

if args.regime:
    table = {}
    for spec in args.regime:
        label, _, sides = spec.partition("=")
        label = label.strip().upper().replace("-", "_")
        table[label] = _sides(sides or "short")
        parts.append(f"{label.lower()}-{'+'.join(table[label])}")
    C.REGIME_SIDES = {"XAUUSD": table}
    for k, v in table.items():
        print(f"  regime {k:<12} -> {', '.join(s.upper() for s in v)} only")

if args.rules:
    table = {}
    for spec in args.rules:
        names, _, sides = spec.partition("=")
        s = _sides(sides or "short")
        for n in names.split(","):
            n = n.strip().upper().replace("-", "_")
            if n:
                table[n] = s
    C.RULE_SIDES = {"XAUUSD": table}
    parts.append("rules-" + "_".join(sorted(k.lower() for k in table)))
    print(f"  rules  {', '.join(sorted(table))} -> "
          f"{', '.join(s.upper() for s in next(iter(table.values())))} only")

if not parts:
    parts.append("baseline")

tag = args.tag or "_".join(parts)
if args.end:
    tag += f"_to{args.end.replace('-', '')}"

sys.argv = ["live_replay.py", "--tf", args.tf, "--days", args.days,
            "--asset", "XAUUSD", "--balance", "1000", "--flat", "0.02",
            "--journal-path", f"reports/rp_j_{args.tf}_{tag}.jsonl"]
if args.end:
    sys.argv += ["--end", args.end]
if args.spread:
    sys.argv += ["--spread", args.spread]

import live_replay  # noqa: E402  (import after the patch so load() sees it)

live_replay.main()
