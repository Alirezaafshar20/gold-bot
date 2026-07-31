"""What the book earns, measured against what no skill at all would have earned.

Every number in this project so far has been relative: this variant beats that
variant, this rule beats that rule. None of it answers the question underneath —
is there skill here, and how much? That needs an absolute benchmark, and there is
a standard one.

The no-skill benchmark
---------------------
Take a driftless random walk with an absorbing barrier at -a (the stop) and one
at +b (the target). The probability of touching +b first is exactly

    P(win) = a / (a + b) = 1 / (1 + RR)

This is not an approximation or a heuristic; it is the gambler's-ruin identity,
and it is the win rate a monkey achieves by entering at random with the same
stop and target the book uses. At RR 2.0 that is 33.3%. So a 33% win rate on a
2:1 system is worth precisely nothing, and the only meaningful measure of an
entry rule is how many percentage points above that line it lands.

Expectancy is then

    E[R] = p*RR - (1 - p) - c

with c the round-trip cost in units of R. c is not a rounding error here: the
spread is a fixed number of dollars while 1R is the stop distance, so a rule
with tight stops pays a much larger share of its R away than one with wide
stops, and the two sides of this book have very different stop widths.

What the script reports
-----------------------
  1. no-skill win rate vs actual, per side  -> skill, in percentage points
  2. the spread's bite, per side            -> cost in R, and what it costs a year
  3. expectancy decomposed                  -> where each side's R comes from
  4. statistical weight                     -> is any of this distinguishable
                                               from luck, and at what n would it be
  5. exit mix                               -> how much of the book the barrier
                                               model actually describes

Usage: python edge_math.py reports/rp_j_m5.jsonl [more.jsonl ...]
"""
import collections
import json
import math
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Measured median over 1.51M WM Markets ticks (spread_probe.py, 7 days): 0.26,
# stable at 0.22-0.38 through every session. The 0.68 the replay banner reported
# was a closed-market quote and is not what these fills would have paid.
SPREAD_USD = 0.26
if "--spread" in sys.argv:
    SPREAD_USD = float(sys.argv[sys.argv.index("--spread") + 1])


def trades(path):
    """Fills joined to their closes, carrying geometry and outcome together."""
    zone_regime, by_zone, tk2zone = {}, {}, {}
    pending = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        ev = e.get("event")
        if ev == "armed":
            zone_regime[e.get("zone_id")] = e.get("regime")
        elif ev == "fill":
            entry, sl, tp = e.get("entry"), e.get("sl"), e.get("tp")
            if entry is None or sl is None:
                continue
            zid = e.get("zone_id")
            sl_usd = abs(float(entry) - float(sl))
            tp_usd = abs(float(tp) - float(entry)) if tp is not None else np.nan
            by_zone[zid] = {
                "side": e.get("side"), "rule": e.get("rule"),
                "regime": zone_regime.get(zid) or "?",
                "sl_usd": sl_usd, "tp_usd": tp_usd,
                "rr": tp_usd / sl_usd if sl_usd > 0 else np.nan,
                "R": np.nan, "exit": None,
            }
            pending = zid
        elif ev == "open_seen":
            tk = e.get("ticket")
            if tk is not None and tk not in tk2zone and pending is not None:
                tk2zone[tk] = pending
            pending = None
        elif ev == "close":
            z = tk2zone.get(e.get("ticket")) or e.get("zone_id")
            if z in by_zone:
                by_zone[z]["R"] = float(e.get("R") or 0.0)
                by_zone[z]["exit"] = e.get("reason") or e.get("exit")
    return [t for t in by_zone.values() if np.isfinite(t["R"])]


def pf(v):
    gp = sum(x for x in v if x > 0)
    gl = -sum(x for x in v if x < 0)
    return gp / gl if gl > 0 else float("inf")


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    if not paths:
        raise SystemExit(__doc__)
    tr = []
    for p in paths:
        tr += trades(p)
    if not tr:
        raise SystemExit("no joinable trades in those journals")

    print("=" * 90)
    print(f"  EDGE MATHEMATICS — {len(tr)} trades from {len(paths)} journals")
    print("=" * 90)

    sides = [("long", [t for t in tr if t["side"] == "long"]),
             ("short", [t for t in tr if t["side"] == "short"]),
             ("BOTH", tr)]

    print()
    print("  1. SKILL — actual win rate against the no-skill barrier baseline")
    print()
    print(f"  {'side':<8} {'n':>4} {'mean RR':>8} {'no-skill':>9} {'actual':>8} "
          f"{'skill':>8}   {'verdict':<22}")
    print("  " + "-" * 76)
    for name, s in sides:
        if not s:
            continue
        rr = float(np.nanmean([t["rr"] for t in s]))
        # the monkey's win rate for each trade's own geometry, then averaged
        noskill = 100.0 * float(np.nanmean(
            [1.0 / (1.0 + t["rr"]) for t in s if np.isfinite(t["rr"])]))
        act = 100.0 * sum(1 for t in s if t["R"] > 0) / len(s)
        d = act - noskill
        v = ("genuine edge" if d > 3 else
             "no measurable skill" if d > -1.5 else "WORSE THAN RANDOM")
        print(f"  {name:<8} {len(s):>4} {rr:>8.2f} {noskill:>8.1f}% {act:>7.1f}% "
              f"{d:>+7.1f}   {v:<22}")
    print()
    print("     no-skill = 1/(1+RR), the win rate a random entry achieves with the")
    print("     same stop and target. Anything at or below it is not an edge.")

    print()
    print("  2. COST — what the spread takes, as a share of 1R")
    print()
    print(f"  {'side':<8} {'n':>4} {'mean SL $':>10} {'spread/R':>9} "
          f"{'cost R':>8} {'% of gross':>11}")
    print("  " + "-" * 62)
    for name, s in sides:
        if not s:
            continue
        sl = float(np.mean([t["sl_usd"] for t in s]))
        # a round trip crosses the spread once in and once out
        cr = float(np.mean([SPREAD_USD / t["sl_usd"] for t in s if t["sl_usd"] > 0]))
        tot = cr * len(s)
        gross = sum(abs(t["R"]) for t in s)
        print(f"  {name:<8} {len(s):>4} {sl:>10.2f} {cr:>9.3f} {tot:>8.2f} "
              f"{100*tot/gross if gross else 0:>10.1f}%")
    print()
    print(f"     at a {SPREAD_USD} USD spread. A tight stop is a small R, so the")
    print("     same spread eats a bigger fraction of it — the side with the")
    print("     tighter stops pays proportionally more to trade.")

    print()
    print("  3. EXPECTANCY — decomposed")
    print()
    print(f"  {'side':<8} {'n':>4} {'WR':>6} {'avg win':>8} {'avg loss':>9} "
          f"{'E[R]':>8} {'PF':>6} {'netR':>8}")
    print("  " + "-" * 68)
    for name, s in sides:
        if not s:
            continue
        R = [t["R"] for t in s]
        w = [x for x in R if x > 0]
        l = [x for x in R if x <= 0]
        p = len(w) / len(R)
        print(f"  {name:<8} {len(s):>4} {100*p:>5.1f}% "
              f"{np.mean(w) if w else 0:>+8.2f} {np.mean(l) if l else 0:>+9.2f} "
              f"{np.mean(R):>+8.3f} {pf(R):>6.2f} {sum(R):>+8.2f}")

    print()
    print("  4. STATISTICAL WEIGHT — can any of this be told apart from luck?")
    print()
    print(f"  {'side':<8} {'n':>4} {'E[R]':>8} {'sd':>6} {'std err':>8} "
          f"{'t':>6} {'p<0.05?':>9} {'n needed':>9}")
    print("  " + "-" * 68)
    for name, s in sides:
        if len(s) < 5:
            continue
        R = np.array([t["R"] for t in s], dtype=float)
        mu, sd = float(R.mean()), float(R.std(ddof=1))
        se = sd / math.sqrt(len(R))
        t = mu / se if se > 0 else 0.0
        need = int(math.ceil((2.0 * sd / mu) ** 2)) if abs(mu) > 1e-9 else 0
        print(f"  {name:<8} {len(s):>4} {mu:>+8.3f} {sd:>6.2f} {se:>8.3f} "
              f"{t:>+6.2f} {'YES' if abs(t) >= 1.96 else 'no':>9} {need:>9}")
    print()
    print("     t is the mean divided by its own standard error; |t| >= 1.96 is the")
    print("     95% bar. 'n needed' is how many trades that side would need, at the")
    print("     edge and variance measured here, before the result stops being")
    print("     consistent with zero. This is the number that decides whether a")
    print("     finding is knowledge or a story.")

    print()
    print("  5. EXIT MIX — how much of the book the barrier model even describes")
    print()
    for name, s in sides[:2]:
        if not s:
            continue
        c = collections.Counter(str(t["exit"]) for t in s)
        tot = len(s)
        print(f"  {name}:  " + "   ".join(
            f"{k} {100*v/tot:.0f}%" for k, v in c.most_common()))
    print()
    print("     The 1/(1+RR) baseline assumes every trade ends at the stop or the")
    print("     target. Time and trail exits break that assumption, so treat")
    print("     section 1 as a benchmark rather than a law.")


if __name__ == "__main__":
    main()
