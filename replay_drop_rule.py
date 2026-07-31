"""Replay one timeframe with a rule removed, without touching floating_config.

The OOS rule set per timeframe lives in FC.OOS_RULES_TF and is re-applied by
apply_tf_paths() on every run, so patching that dict in-process is enough to
answer "what does this book look like without rule X" while the on-disk config
(and any live bot reading it) stays untouched.

Usage: python replay_drop_rule.py M15 ADX_L [--days 60]
"""
import sys

import floating_config as FC

tf = sys.argv[1]
drop = sys.argv[2]
days = sys.argv[4] if len(sys.argv) > 4 else "60"

before = FC.OOS_RULES_TF[tf]
if drop not in before:
    raise SystemExit(f"{drop} is not in the {tf} set {before}")
after = tuple(r for r in before if r != drop)
FC.OOS_RULES_TF[tf] = after
FC.OOS_RULES["XAUUSD"] = after
print(f"  {tf} rules: {','.join(before)}  ->  {','.join(after)}")

sys.argv = ["live_replay.py", "--tf", tf, "--days", days, "--asset", "XAUUSD",
            "--balance", "1000", "--flat", "0.02",
            "--journal-path", f"reports/rp_j_{tf}_no_{drop}.jsonl"]

import live_replay  # noqa: E402  (import after the patch so load() sees it)

live_replay.main()
