"""
parity_diff — where did live and replay stop agreeing?

Both engines write the same decision journal from the shared register_zones:
one `bar` event per closed candle (with a fingerprint of the exact window they
looked at) and one `cand` event per rejected setup (with the gate that killed
it). Diffing the two tapes answers the only question that matters when a trade
shows up on one side and not the other:

  win_sha differs   -> they were never looking at the same candles (data bug)
  win_sha matches   -> same candles, different verdict (logic/state bug), and
                       the `cand` reason on the silent side names the gate

Usage:
  # 1. live is already writing reports/flight_recorder_M15.jsonl
  # 2. replay the same window into a separate tape
  python live_replay.py --tf M15 --days 3 --journal-path reports/replay_M15.jsonl
  # 3. diff them
  python parity_diff.py --tf M15 --live reports/flight_recorder_M15.jsonl \
                        --replay reports/replay_M15.jsonl --day 2026-07-30

  # or let it run the replay for you
  python parity_diff.py --tf M15 --replay-days 3 --day 2026-07-30
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import live_flight_recorder as FR

DEFAULT_REPLAY = "reports/replay_journal_{tf}.jsonl"


def _px(v, nd=2):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _load(path: str, tf: str, day: str | None, asset: str | None) -> list[dict]:
    evs = FR.read_events(path=path, tf=tf, day=day)
    out = []
    for e in evs:
        if str(e.get("tf", "")).upper() != tf:
            continue
        if asset and str(e.get("asset", "")).upper() != asset:
            continue
        out.append(e)
    return out


def _bars(events: list[dict]) -> dict[str, dict]:
    """ts -> last bar event for that ts (a live restart re-logs the same bar)."""
    out: dict[str, dict] = {}
    for e in events:
        if e.get("event") == "bar":
            out[str(e.get("ts"))] = e
    return out


def _zone_key(e: dict) -> tuple:
    return (str(e.get("ts")), str(e.get("rule")), str(e.get("side")),
            _px(e.get("proximal")))


def _by_key(events: list[dict], name: str) -> dict[tuple, dict]:
    return {_zone_key(e): e for e in events if e.get("event") == name}


def _cand_index(events: list[dict]) -> dict[tuple, list[dict]]:
    """(ts, side) -> everything the engine looked at and turned down."""
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("event") in ("cand", "skip_register"):
            idx[(str(e.get("ts")), str(e.get("side")))].append(e)
    return idx


def _explain(other_bars: dict, other_cands: dict, ts: str, rule: str,
             side: str) -> str:
    """Why the other tape has no zone here."""
    bar = other_bars.get(ts)
    if bar is None:
        return "no bar logged at this timestamp"
    gate = str(bar.get("gate", ""))
    if gate not in ("", "ok"):
        return f"whole bar skipped: {gate}"
    any_side = other_cands.get((ts, side), [])
    hits = [c for c in any_side if str(c.get("rule")) == rule]
    if hits:
        reasons = sorted({str(c.get("reason")) for c in hits})
        return "candidate rejected: " + ", ".join(reasons)
    if any_side:
        rejected = sorted({f"{c.get('rule')}:{c.get('reason')}" for c in any_side})
        return (f"this rule produced no setup; other candidates on the same "
                f"side were rejected as {', '.join(rejected[:4])}")
    n_c = bar.get("cands")
    if n_c:
        return (f"bar produced {n_c} candidate(s) but none for this rule/side "
                f"(armed {bar.get('armed', 0)})")
    return "detector produced no setup at all"


def _fmt_ev(e: dict) -> str:
    bits = [str(e.get("ts")), str(e.get("rule")), str(e.get("side", "")).upper()]
    for k, lbl in (("proximal", "prox"), ("entry", "entry"), ("sl", "SL"),
                   ("tp", "TP")):
        if e.get(k) is not None:
            bits.append(f"{lbl}={_px(e[k])}")
    if e.get("regime"):
        bits.append(str(e["regime"]))
    return "  ".join(bits)


def compare(live: list[dict], rep: list[dict], *, max_rows: int) -> int:
    problems = 0
    sep = "=" * 74

    print(sep)
    print("  SESSION")
    print(sep)
    for tag, evs in (("live", live), ("replay", rep)):
        s = [e for e in evs if e.get("event") == "session"]
        if not s:
            print(f"  {tag:<7} no session marker (older tape?)")
            continue
        s = s[-1]
        print(f"  {tag:<7} fp={s.get('fp')}  sig_bars={s.get('sig_bars')}  "
              f"htf_bars={s.get('htf_bars')}  risk={s.get('risk')}")
    lfp = next((e.get("fp") for e in reversed(live)
                if e.get("event") == "session"), None)
    rfp = next((e.get("fp") for e in reversed(rep)
                if e.get("event") == "session"), None)
    if lfp and rfp and lfp != rfp:
        problems += 1
        print("  MISMATCH: the two runs used different rules/settings. "
              "Nothing below is meaningful until this matches.")

    lb, rb = _bars(live), _bars(rep)
    common = sorted(set(lb) & set(rb))
    print()
    print(sep)
    print(f"  BARS   live={len(lb)}  replay={len(rb)}  common={len(common)}")
    print(sep)
    only_l = sorted(set(lb) - set(rb))
    only_r = sorted(set(rb) - set(lb))
    if only_l:
        print(f"  {len(only_l)} bar(s) only in live   "
              f"(first {only_l[0]}, last {only_l[-1]})")
    if only_r:
        print(f"  {len(only_r)} bar(s) only in replay "
              f"(first {only_r[0]}, last {only_r[-1]})")
    if not common:
        print("  No overlapping bars — check --day / --days.")
        return 1

    sha_bad, regime_bad, gate_bad = [], [], []
    htf_bad: dict[str, list[str]] = {"h4": [], "h1": []}
    for ts in common:
        a, b = lb[ts], rb[ts]
        if a.get("win_sha") != b.get("win_sha"):
            sha_bad.append(ts)
        if a.get("regime") != b.get("regime"):
            regime_bad.append(ts)
        if a.get("gate") != b.get("gate"):
            gate_bad.append(ts)
        for k in htf_bad:
            # The newest HTF candle is still forming, so its bytes can never
            # match tick for tick. Its open stamp and bar count must.
            if a.get(f"{k}_n") is None and b.get(f"{k}_n") is None:
                continue
            if (a.get(f"{k}_n") != b.get(f"{k}_n")
                    or a.get(f"{k}_last") != b.get(f"{k}_last")):
                htf_bad[k].append(ts)

    print()
    print(f"  window fingerprint identical : {len(common) - len(sha_bad)}/{len(common)}")
    print(f"  regime label identical       : {len(common) - len(regime_bad)}/{len(common)}")
    print(f"  bar-level gate identical     : {len(common) - len(gate_bad)}/{len(common)}")
    for k, bad in htf_bad.items():
        if not any(lb[ts].get(f"{k}_n") is not None for ts in common):
            continue
        print(f"  {k.upper()} context aligned        : "
              f"{len(common) - len(bad)}/{len(common)}")

    for k, bad in htf_bad.items():
        if not bad:
            continue
        problems += 1
        print(f"\n  {k.upper()} CONTEXT MISMATCH — one side is reading a "
              f"different higher-timeframe candle:")
        for ts in bad[:max_rows]:
            a, b = lb[ts], rb[ts]
            print(f"    {ts}  live n={a.get(f'{k}_n')} last={a.get(f'{k}_last')}"
                  f"  |  replay n={b.get(f'{k}_n')} last={b.get(f'{k}_last')}")
        if len(bad) > max_rows:
            print(f"    … {len(bad) - max_rows} more")

    if sha_bad:
        problems += 1
        print("\n  INPUT MISMATCH — the engines saw different candles:")
        for ts in sha_bad[:max_rows]:
            a, b = lb[ts], rb[ts]
            print(f"    {ts}")
            print(f"      live   n={a.get('n_bars')} {a.get('win_first')} → "
                  f"{a.get('win_last')} sha={a.get('win_sha')}")
            print(f"      replay n={b.get('n_bars')} {b.get('win_first')} → "
                  f"{b.get('win_last')} sha={b.get('win_sha')}")
            for tf_key in ("h4", "h1"):
                if a.get(f"{tf_key}_sha") or b.get(f"{tf_key}_sha"):
                    if a.get(f"{tf_key}_sha") != b.get(f"{tf_key}_sha"):
                        print(f"      {tf_key.upper():<6} live n={a.get(f'{tf_key}_n')} "
                              f"last={a.get(f'{tf_key}_last')} | replay "
                              f"n={b.get(f'{tf_key}_n')} last={b.get(f'{tf_key}_last')}")
        if len(sha_bad) > max_rows:
            print(f"    … {len(sha_bad) - max_rows} more")

    if regime_bad:
        problems += 1
        print("\n  REGIME MISMATCH:")
        for ts in regime_bad[:max_rows]:
            print(f"    {ts}  live={lb[ts].get('regime')}  "
                  f"replay={rb[ts].get('regime')}")
        if len(regime_bad) > max_rows:
            print(f"    … {len(regime_bad) - max_rows} more")

    lo, hi = common[0], common[-1]

    def _in_window(e):
        return lo <= str(e.get("ts")) <= hi

    lcand, rcand = _cand_index(live), _cand_index(rep)
    for name, label in (("armed", "ZONES ARMED"), ("fill", "FILLS")):
        la = {k: v for k, v in _by_key(live, name).items() if lo <= k[0] <= hi}
        ra = {k: v for k, v in _by_key(rep, name).items() if lo <= k[0] <= hi}
        both = set(la) & set(ra)
        print()
        print(sep)
        print(f"  {label}   live={len(la)}  replay={len(ra)}  matched={len(both)}")
        print(sep)
        for key in sorted(set(la) - set(ra)):
            problems += 1
            e = la[key]
            print(f"  LIVE ONLY   {_fmt_ev(e)}")
            print(f"              replay: "
                  f"{_explain(rb, rcand, key[0], key[1], key[2])}")
        for key in sorted(set(ra) - set(la)):
            problems += 1
            e = ra[key]
            print(f"  REPLAY ONLY {_fmt_ev(e)}")
            print(f"              live:   "
                  f"{_explain(lb, lcand, key[0], key[1], key[2])}")
        for key in sorted(both):
            a, b = la[key], ra[key]
            deltas = []
            for f in ("sl", "tp", "entry"):
                if a.get(f) is not None and b.get(f) is not None:
                    d = float(a[f]) - float(b[f])
                    if abs(d) > 0.005:
                        deltas.append(f"{f} {_px(a[f])} vs {_px(b[f])}")
            if deltas:
                problems += 1
                print(f"  DIFFERS     {_fmt_ev(a)}")
                print(f"              {'; '.join(deltas)}")

    lc = [e for e in live if e.get("event") == "close" and _in_window(e)]
    rc = [e for e in rep if e.get("event") == "close" and _in_window(e)]
    print()
    print(sep)
    print(f"  CLOSES   live={len(lc)}  replay={len(rc)}")
    print(sep)
    for tag, rows in (("live  ", lc), ("replay", rc)):
        for e in rows:
            print(f"  {tag}  {e.get('ts')}  {e.get('rule')} "
                  f"{str(e.get('side','')).upper()}  entry={_px(e.get('entry'))} "
                  f"exit={_px(e.get('exit'))}  R={_px(e.get('R'), 2)}  "
                  f"{e.get('reason')}")

    print()
    print(sep)
    if problems == 0:
        print("  VERDICT: live and replay agree on every logged decision.")
    else:
        print(f"  VERDICT: {problems} divergence(s). Fix window/fingerprint "
              f"mismatches first — they cause the rest.")
    print(sep)
    return 0 if problems == 0 else 2


def main():
    ap = argparse.ArgumentParser(
        description="Diff the live decision journal against a replay of the "
                    "same window")
    ap.add_argument("--tf", default="M15", choices=["M5", "M15"])
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--day", default=None,
                    help="Restrict to one YYYY-MM-DD (recommended)")
    ap.add_argument("--live", default=None,
                    help="Live tape (default reports/flight_recorder[_TF].jsonl)")
    ap.add_argument("--replay", default=None,
                    help=f"Replay tape (default {DEFAULT_REPLAY})")
    ap.add_argument("--replay-days", type=int, default=None,
                    help="Run live_replay over the last N days first and use "
                         "its tape")
    ap.add_argument("--balance", type=float, default=1000.0,
                    help="Start balance for --replay-days")
    ap.add_argument("--max-rows", type=int, default=12,
                    help="Cap per-section detail lines")
    args = ap.parse_args()

    tf = args.tf.upper()
    live_path = args.live or FR.recorder_path(tf)
    rep_path = args.replay or DEFAULT_REPLAY.format(tf=tf)

    if args.replay_days:
        import live_replay as LR
        if os.path.isfile(rep_path):
            os.remove(rep_path)
        os.makedirs(os.path.dirname(rep_path) or ".", exist_ok=True)
        FR.set_path(rep_path)
        print(f"[PARITY] replaying {args.replay_days}d {tf} → {rep_path}")
        LR.LiveReplay(args.asset, tf, args.replay_days, args.balance,
                      record=True).run()
        FR.set_path(None)
        print()

    for p in (live_path, rep_path):
        if not os.path.isfile(p):
            print(f"[PARITY] missing tape: {p}")
            print("         run the live bot and live_replay --journal-path "
                  "first (or pass --replay-days)")
            sys.exit(1)

    asset = args.asset.upper()
    live = _load(live_path, tf, args.day, asset)
    rep = _load(rep_path, tf, args.day, asset)
    print(f"  live tape   : {live_path}  ({len(live)} events)")
    print(f"  replay tape : {rep_path}  ({len(rep)} events)")
    if args.day:
        print(f"  day filter  : {args.day}")
    print()
    sys.exit(compare(live, rep, max_rows=args.max_rows))


if __name__ == "__main__":
    main()
