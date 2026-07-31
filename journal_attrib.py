"""Attribute every replay trade to the regime that was live when it was armed.

Chains the decision journal written by live_flight_recorder:

    armed(zone_id, regime) -> fill(zone_id) -> open_seen(ticket) -> close(ticket, R)

so each closed trade carries the regime label the engine believed at arm time.
Also summarises what the gates rejected, which tells us whether a bad trade got
through because no gate covered it or because a gate mislabelled the market.

Usage: python journal_attrib.py reports/rp_j_m15.jsonl [more.jsonl ...]
"""
import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def trades_with_regime(events):
    """Walk the journal in order and stitch arm-time regime onto each close."""
    zone_regime = {}
    zone_rule = {}
    ticket_meta = {}
    pending_zone = None
    out = []
    for e in events:
        ev = e.get("event")
        if ev == "armed":
            zid = e.get("zone_id")
            zone_regime[zid] = e.get("regime")
            zone_rule[zid] = (e.get("rule"), e.get("side"))
        elif ev == "limit_armed":
            # carries both ids, so the ticket can be mapped straight away
            zid, tk = e.get("zone_id"), e.get("ticket")
            if tk is not None:
                ticket_meta[tk] = (zone_rule.get(zid, (e.get("rule"), e.get("side"))),
                                   zone_regime.get(zid))
        elif ev == "fill":
            pending_zone = e.get("zone_id")
        elif ev == "open_seen":
            tk = e.get("ticket")
            if tk is not None and tk not in ticket_meta:
                ticket_meta[tk] = (
                    zone_rule.get(pending_zone, (e.get("rule"), e.get("side"))),
                    zone_regime.get(pending_zone))
            pending_zone = None
        elif ev == "close":
            tk = e.get("ticket")
            (rule, side), reg = ticket_meta.get(tk, ((e.get("rule"), e.get("side")), None))
            out.append({
                "ts": e.get("ts"), "rule": rule or e.get("rule"),
                "side": side or e.get("side"), "regime": reg or "?",
                "R": float(e.get("R") or 0.0),
                "usd": float(e.get("pnl_usd") or 0.0),
                "exit": e.get("reason"),
            })
    return out


def _blame_same_dir(events, skips):
    """Was the blocked slot held by an OPEN position or only by an ARMED zone?

    max_same_dir counts both, but they are governed by different settings:
    an open position holds the slot until it exits (MAX_HOLD), an armed zone
    only until the limit expires (WAIT_BARS). Knowing which one dominates says
    which setting is actually rationing the book.
    """
    open_iv = {}
    for e in events:
        tk = e.get("ticket")
        if e.get("event") == "open_seen" and tk is not None:
            open_iv[tk] = [e.get("ts"), None, e.get("side")]
        elif e.get("event") == "close" and tk in open_iv:
            open_iv[tk][1] = e.get("ts")
    live = [(a, b, s) for a, b, s in open_iv.values() if a and b]

    by_open = 0
    for e in skips:
        ts, side = e.get("ts"), e.get("side")
        if any(s == side and a <= ts <= b for a, b, s in live):
            by_open += 1
    n = len(skips)
    print(f"  blocked by an OPEN position : {by_open:>5}  ({100*by_open/n:4.1f}%)"
          f"   <- governed by MAX_HOLD")
    print(f"  blocked by an ARMED zone    : {n-by_open:>5}  "
          f"({100*(n-by_open)/n:4.1f}%)   <- governed by WAIT_BARS")


def pf(vals):
    gp = sum(v for v in vals if v > 0)
    gl = -sum(v for v in vals if v < 0)
    return gp / gl if gl > 0 else float("inf")


def line(name, rows, width=26):
    if not rows:
        return
    v = [r["R"] for r in rows]
    wins = sum(1 for x in v if x > 0)
    p = pf(v)
    ps = "  inf" if p == float("inf") else f"{p:5.2f}"
    print(f"  {name:<{width}} n={len(rows):<4} WR={100*wins/len(rows):5.1f}%  "
          f"PF={ps}  netR={sum(v):+7.2f}  avgR={sum(v)/len(rows):+6.3f}  "
          f"${sum(r['usd'] for r in rows):+9.2f}")


def report(path):
    events = list(load(path))
    tr = trades_with_regime(events)
    print("=" * 96)
    print(f"  {path}   {len(tr)} closed trades")
    print("=" * 96)

    line("ALL", tr)
    print()
    print("  -- by side --")
    for side in ("long", "short"):
        line(side, [t for t in tr if t["side"] == side])
    print()
    print("  -- by arm-time regime --")
    for reg in sorted({t["regime"] for t in tr}):
        line(reg, [t for t in tr if t["regime"] == reg])
    print()
    print("  -- by side x arm-time regime --")
    keys = sorted({(t["side"], t["regime"]) for t in tr})
    for side, reg in keys:
        line(f"{side} in {reg}",
             [t for t in tr if t["side"] == side and t["regime"] == reg])
    print()
    print("  -- by rule x side (every cell, worst first) --")
    rs = collections.defaultdict(list)
    for t in tr:
        rs[(t["rule"], t["side"])].append(t)
    for k in sorted(rs, key=lambda k: sum(x["R"] for x in rs[k])):
        line(f"{k[0]} {k[1]}", rs[k], width=30)
    print()
    print("  -- by rule x side x regime (losing cells only) --")
    grp = collections.defaultdict(list)
    for t in tr:
        grp[(t["rule"], t["side"], t["regime"])].append(t)
    for k in sorted(grp, key=lambda k: sum(x["R"] for x in grp[k])):
        rows = grp[k]
        if sum(x["R"] for x in rows) >= 0:
            continue
        line(f"{k[0]} {k[1]} {k[2]}", rows, width=30)

    cands = [e for e in events if e.get("event") == "cand"]
    print()
    print(f"  -- what the gates rejected ({len(cands)} candidate setups) --")
    rc = collections.Counter(e.get("reason") for e in cands)
    for reason, n in rc.most_common():
        print(f"  {reason:<26} {n:>5}")
    print()
    print("  -- 'regime' rejections by side and label --")
    rr = collections.Counter(
        (e.get("side"), e.get("regime")) for e in cands if e.get("reason") == "regime")
    for (side, reg), n in rr.most_common():
        print(f"  {str(side):<8} {str(reg):<14} {n:>5}")
    mg = collections.Counter(
        (e.get("side"), e.get("regime")) for e in cands if e.get("reason") == "meta_gate")
    if mg:
        print()
        print("  -- 'meta_gate' rejections by side and label --")
        for (side, reg), n in mg.most_common():
            print(f"  {str(side):<8} {str(reg):<14} {n:>5}")

    skips = [e for e in events if e.get("event") == "skip_register"]
    if skips:
        print()
        print(f"  -- {len(skips)} setups skipped before reaching the gates --")
        sc = collections.Counter(e.get("reason") for e in skips)
        for reason, n in sc.most_common():
            print(f"  {str(reason):<26} {n:>5}")
        sd = [e for e in skips if e.get("reason") == "same_dir"]
        if sd:
            print(f"  same_dir skips by side: "
                  + ", ".join(f"{k}={v}" for k, v in
                              collections.Counter(e.get("side") for e in sd).items()))
            _blame_same_dir(events, sd)

    bars = [e for e in events if e.get("event") == "bar"]
    if bars:
        print()
        print(f"  -- regime label distribution across {len(bars)} decision bars --")
        bc = collections.Counter(e.get("regime") for e in bars)
        for reg, n in bc.most_common():
            print(f"  {str(reg):<14} {n:>5}  ({100*n/len(bars):4.1f}%)")
        print()
        print("  -- bar gate outcomes --")
        gc = collections.Counter(e.get("gate") for e in bars)
        for g, n in gc.most_common():
            print(f"  {str(g):<20} {n:>5}  ({100*n/len(bars):4.1f}%)")
    return tr


all_tr = []
for p in sys.argv[1:]:
    all_tr += report(p)
    print()

if len(sys.argv) > 2:
    print("=" * 96)
    print(f"  MERGED  {len(all_tr)} trades")
    print("=" * 96)
    line("ALL", all_tr)
    for side in ("long", "short"):
        line(side, [t for t in all_tr if t["side"] == side])
    # Pooling every window is the only sample big enough to judge one rule on
    # one side, so this is the table that decides which rules get retired.
    # The regime audit measures what each label is worth as a pure directional
    # bias. This table says which labels the book actually trades in, so the two
    # can be compared: a side that loses inside a label the audit scores well is
    # a rule problem, not a detector problem.
    print()
    print("  -- MERGED by side x arm-time regime --")
    msr = collections.defaultdict(list)
    for t in all_tr:
        msr[(t["side"], t["regime"])].append(t)
    for k in sorted(msr, key=lambda k: (k[0], -len(msr[k]))):
        line(f"{k[0]} in {k[1]}", msr[k], width=30)
    print()
    print("  -- MERGED by rule x side (worst first) --")
    mrs = collections.defaultdict(list)
    for t in all_tr:
        mrs[(t["rule"], t["side"])].append(t)
    for k in sorted(mrs, key=lambda k: sum(x["R"] for x in mrs[k])):
        line(f"{k[0]} {k[1]}", mrs[k], width=30)
