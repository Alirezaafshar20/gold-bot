"""Replay with per-rule side restrictions applied in-process only.

Answers "is the long side broken, or are a few long rules broken?" without
touching anything on disk, so a live bot reading portfolio_config is unaffected
while the variant is measured.

The restriction goes through portfolio_config.rule_allowed, which live,
live_replay and portfolio_backtest all consult, so the same table would govern
all three if it were later made permanent.

Usage:
  python rule_sides_test.py M5 60 NDS,FL,NDS_BOS,NDS_FVG=short
  python rule_sides_test.py M15 60 NDS,FL,NDS_BOS,NDS_FVG=short --end 2026-01-30

Rule tags accept either separator (NDS-FVG or NDS_FVG).
"""
import sys

import portfolio_config as C

TF = sys.argv[1]
DAYS = sys.argv[2]
SPEC = sys.argv[3]
END = None
if "--end" in sys.argv:
    END = sys.argv[sys.argv.index("--end") + 1]

rules, _, side = SPEC.partition("=")
side = side or "short"
names = [r.strip().upper().replace("-", "_") for r in rules.split(",") if r.strip()]
C.RULE_SIDES = {"XAUUSD": {n: (side,) for n in names}}
print(f"  restricted to {side.upper()} only: {', '.join(names)}")

tag = f"sides_{side}_{'_'.join(n.lower() for n in names)}"
if END:
    tag += f"_to{END.replace('-', '')}"

sys.argv = ["live_replay.py", "--tf", TF, "--days", DAYS, "--asset", "XAUUSD",
            "--balance", "1000", "--flat", "0.02",
            "--journal-path", f"reports/rp_j_{TF}_{tag}.jsonl"]
if END:
    sys.argv += ["--end", END]

import live_replay  # noqa: E402  (import after the patch)

live_replay.main()
