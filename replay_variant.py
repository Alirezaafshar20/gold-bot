"""Replay one timeframe with config overrides applied in-process only.

Nothing on disk changes, so a live bot reading floating_config.py is unaffected
while a variant is being measured. Overrides are applied before live_replay is
imported so the profile picks them up through floating_opt_overrides().

Usage:
  python replay_variant.py M15 60 arm_priority=rr
  python replay_variant.py M5  60 wait=16
  python replay_variant.py M15 60 arm_priority=rr wait=24 tag=rr_w24

Recognised keys:
  wait=N        patch strategy.WAIT_BARS (bars the armed limit waits for a fill)
  end=DATE      replay the window ENDING on DATE instead of today
  tag=NAME      suffix for the journal filename
  <other>=V     merged into floating_config.ENTRY, e.g. arm_priority, max_same_dir
"""
import sys

import floating_config as FC
import strategy as S

TF = sys.argv[1]
DAYS = sys.argv[2]
OVERRIDES = [a.split("=", 1) for a in sys.argv[3:] if "=" in a]


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


tag_parts = []
end = None
for key, raw in OVERRIDES:
    val = _coerce(raw)
    if key == "tag":
        tag_parts.append(str(val))
        continue
    if key == "end":
        end = str(raw)
        tag_parts.append(f"to{str(raw).replace('-', '')}")
        continue
    if key == "wait":
        S.WAIT_BARS = int(val)
        print(f"  strategy.WAIT_BARS -> {S.WAIT_BARS}")
        tag_parts.append(f"w{val}")
        continue
    # floating_opt_overrides() merges REGIME, META and ENTRY into one opt dict,
    # so patch whichever of them already owns the key rather than assuming ENTRY.
    target, name = next(
        ((d, n) for d, n in ((FC.REGIME, "REGIME"), (FC.META, "META"),
                             (FC.ENTRY, "ENTRY")) if key in d),
        (FC.ENTRY, "ENTRY"))
    before = target.get(key, "<unset>")
    target[key] = val
    print(f"  {name}[{key}] {before!r} -> {val!r}")
    tag_parts.append(f"{key}_{val}")

tag = "_".join(tag_parts) or "base"
sys.argv = ["live_replay.py", "--tf", TF, "--days", DAYS, "--asset", "XAUUSD",
            "--balance", "1000", "--flat", "0.02",
            "--journal-path", f"reports/rp_j_{TF}_{tag}.jsonl"]
if end:
    sys.argv += ["--end", end]

import live_replay  # noqa: E402  (import after the patches)

live_replay.main()
