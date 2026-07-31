"""Trade-level diff: calibration engine vs live_replay on the same window.

Runs strategy.run_backtest through multi_symbol_calibrate._run, then matches its
trades against a live_replay log by entry timestamp, so the two engines can be
compared setup by setup. When both engines take the same setup, any difference
in R has to come from the entry price they booked or the exit they resolved —
which is precisely what the calibration engine has no broker model for.

Usage: python engine_diff.py M15 reports/rp_m15.log [days]
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import floating_config as FC
import mt5_data as M
import portfolio_config as C
import symbol_profiles as P
from multi_symbol_calibrate import _run

TF = sys.argv[1] if len(sys.argv) > 1 else "M15"
LOG = sys.argv[2] if len(sys.argv) > 2 else "reports/rp_m15.log"
DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ASSET = "XAUUSD"

ROW = re.compile(
    r"^\s*\d+\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(\S+)\s+(LONG|SHORT)\s+"
    r"entry=([\d.]+)\s+R=([+-][\d.]+)\s+\$([+-][\d.]+)\s+(LIMIT|TOUCH)\s+(\S+)\s*$")


def _open_text(path):
    """PowerShell '>' redirection writes UTF-16LE, python writes UTF-8."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    enc = "utf-16" if head[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return open(path, encoding=enc, errors="replace")


def replay_trades(path):
    out = []
    with _open_text(path) as fh:
        for line in fh:
            m = ROW.match(line.rstrip("\n"))
            if m:
                ts, rule, side, entry, R, usd, via, ex = m.groups()
                out.append({"t": pd.Timestamp(ts), "rule": rule,
                            "dir": side.lower(), "entry": float(entry),
                            "R": float(R), "via": via, "exit": ex})
    return out


mt5 = M.connect()
FC.apply_tf_paths(TF)
_, prof = P.get_profile(ASSET)
sym = P.resolve_symbol_for_profile(ASSET, mt5)
res = _run(sym, prof, mt5, days=DAYS, asset_key=ASSET,
           risk_pct=C.RISK_MAP[ASSET], max_concurrent=C.max_concurrent(ASSET),
           tf=TF)
cal = [{"t": pd.Timestamp(t["time"]), "dir": t["dir"], "entry": float(t["entry"]),
        "R": float(t["R"]), "exit": t.get("exit_reason"),
        "fill": t.get("fill_kind"), "tag": t.get("setup") or t.get("tag") or "?"}
       for t in (res.get("trades") or [])]
rep = replay_trades(LOG)

print("=" * 100)
print(f"  {ASSET} {TF} {DAYS}d   calibration engine: {len(cal)} trades   "
      f"live_replay: {len(rep)} trades")
print("=" * 100)

# Match on entry timestamp; the calibration engine stamps the signal-TF bar it
# filled on, replay stamps the M1 minute, so allow one signal bar of slack.
slack = pd.Timedelta(minutes=M.tf_minutes(TF))
used = set()
pairs = []
for c in cal:
    best = None
    for i, r in enumerate(rep):
        if i in used or r["dir"] != c["dir"]:
            continue
        d = abs((r["t"] - c["t"]).total_seconds())
        if d <= slack.total_seconds() and (best is None or d < best[0]):
            best = (d, i)
    if best is not None:
        used.add(best[1])
        pairs.append((c, rep[best[1]]))

cal_only = [c for c in cal if not any(c is p[0] for p in pairs)]
rep_only = [r for i, r in enumerate(rep) if i not in used]

print(f"  matched setups        : {len(pairs)}")
print(f"  calibration only      : {len(cal_only)}")
print(f"  live_replay only      : {len(rep_only)}")
print()

if pairs:
    dR = [p[0]["R"] - p[1]["R"] for p in pairs]
    dpx = [p[0]["entry"] - p[1]["entry"] for p in pairs]
    better = sum(1 for x in dR if x > 0.01)
    worse = sum(1 for x in dR if x < -0.01)
    same = len(dR) - better - worse
    print(f"  On matched setups the calibration engine booked:")
    print(f"    better R : {better:>4}   worse R : {worse:>4}   same R : {same:>4}")
    print(f"    total R  : cal {sum(p[0]['R'] for p in pairs):+8.2f}   "
          f"replay {sum(p[1]['R'] for p in pairs):+8.2f}   "
          f"delta {sum(dR):+8.2f}")
    print(f"    mean entry-price difference : {sum(dpx)/len(dpx):+.3f} USD")
    print(f"    setups where the entry price differs by >0.05 : "
          f"{sum(1 for x in dpx if abs(x) > 0.05)} of {len(dpx)}")
    print()
    print("  -- how each engine exited the same setup --")
    from collections import Counter
    cnt = Counter((str(p[0]["exit"]), str(p[1]["exit"])) for p in pairs)
    print(f"  {'cal exit':<16} {'replay exit':<16} {'n':>4}   {'delta R':>9}")
    for (ce, re_), n in cnt.most_common():
        d = sum(p[0]["R"] - p[1]["R"] for p in pairs
                if str(p[0]["exit"]) == ce and str(p[1]["exit"]) == re_)
        print(f"  {ce:<16} {re_:<16} {n:>4}   {d:>+9.2f}")
    print()
    print("  -- 15 largest R disagreements --")
    print(f"  {'entry time':<17} {'dir':<6} {'cal R':>7} {'rep R':>7} "
          f"{'cal px':>9} {'rep px':>9}  {'cal exit':<10} {'rep exit':<10} via")
    for c, r in sorted(pairs, key=lambda p: -abs(p[0]["R"] - p[1]["R"]))[:15]:
        print(f"  {str(c['t']):<17} {c['dir']:<6} {c['R']:>+7.2f} {r['R']:>+7.2f} "
              f"{c['entry']:>9.2f} {r['entry']:>9.2f}  "
              f"{str(c['exit']):<10} {str(r['exit']):<10} {r['via']}")

if rep_only:
    print()
    print(f"  -- {len(rep_only)} trades only live_replay took "
          f"(net {sum(r['R'] for r in rep_only):+.2f}R) --")
    for r in rep_only[:15]:
        print(f"  {str(r['t']):<17} {r['rule']:<10} {r['dir']:<6} "
              f"R={r['R']:+.2f} {r['via']} {r['exit']}")

if cal_only:
    print()
    print(f"  -- {len(cal_only)} trades only the calibration engine took "
          f"(net {sum(c['R'] for c in cal_only):+.2f}R) --")
    for c in cal_only[:15]:
        print(f"  {str(c['t']):<17} {c['dir']:<6} R={c['R']:+.2f} "
              f"fill={c['fill']} {c['exit']}")
