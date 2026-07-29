"""
Score live_replay vs a ground-truth CSV of real VPS / MT5 trades.

Usage:
  python live_parity.py --csv data/parity_live_2026-07-20.csv --days 2
  python live_parity.py --csv data/parity_live_2026-07-20.csv --days 2 --dual
  python live_parity.py --csv data/parity_live_2026-07-20.csv --days 2 --mirror-vps

Primary metric: timed trade F1 (tf + side + rule, Δt + Δentry).
Idea-bag recall is diagnostic only.
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)

import pandas as pd

import live_replay as LR


def _norm_rule(raw: str) -> str:
    s = str(raw or "").upper().strip()
    s = s.replace("SMC-", "").split("@", 1)[0]
    s = s.replace("_", "-")
    aliases = {
        "WYCK-SOW": "WYCK-SOW",
        "WYCKSOW": "WYCK-SOW",
        "CH-REV": "CH-REV",
        "CHREV": "CH-REV",
        "CH-REV-L": "CH-REV",
        "DEMAND": "DEMAND",
        "VWAP": "VWAP",
        "NDS": "NDS",
    }
    return aliases.get(s, s)


def _norm_side(raw: str) -> str:
    s = str(raw or "").lower().strip()
    if s in ("buy", "long", "0"):
        return "long"
    if s in ("sell", "short", "1"):
        return "short"
    return s


def load_ground_truth(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        out.append({
            "tf": str(row["tf"]).upper().strip(),
            "rule": _norm_rule(row["rule"]),
            "side": _norm_side(row["side"]),
            "entry_time": pd.Timestamp(row["entry_time"]),
            "entry": float(row["entry"]),
            "sl": float(row["sl"]) if pd.notna(row.get("sl")) else None,
            "tp": float(row["tp"]) if pd.notna(row.get("tp")) else None,
            "outcome": str(row.get("outcome", "")),
            "source": str(row.get("source", "")),
            "_matched": False,
        })
    return out


def filter_day(trades: list[dict], day: str) -> list[dict]:
    d0 = pd.Timestamp(day).normalize()
    d1 = d0 + pd.Timedelta(days=1)
    out = []
    for t in trades:
        et = pd.Timestamp(t["entry_time"])
        if d0 <= et < d1:
            tt = dict(t)
            tt["rule"] = _norm_rule(tt.get("rule") or tt.get("tag") or "")
            tt["side"] = _norm_side(tt.get("dir") or tt.get("side") or "")
            tt["tf"] = str(tt.get("tf") or "").upper()
            tt["entry_time"] = et
            tt["entry"] = float(tt["entry"])
            tt["_matched"] = False
            out.append(tt)
    return out


def _idea_key(t: dict) -> tuple:
    return (t["tf"], _norm_rule(t.get("rule", "")), _norm_side(t.get("side") or t.get("dir")))


def score_bag(gt: list[dict], replay: list[dict],
              ideas: list[dict] | None = None) -> dict:
    """Multiset overlap on (tf, rule, side) — diagnostic only."""
    from collections import Counter
    cg = Counter(_idea_key(t) for t in gt)
    cf = Counter(_idea_key(t) for t in replay)
    if ideas:
        ci = Counter(_idea_key(t) for t in ideas)
        cr: Counter = Counter()
        for k in set(cf) | set(ci):
            cr[k] = max(cf[k], ci[k])
    else:
        cr = cf
    keys = set(cg) | set(cr)
    overlap = sum(min(cg[k], cr[k]) for k in keys)
    n_gt = len(gt)
    n_rep = sum(cr.values())
    recall = (overlap / n_gt * 100.0) if n_gt else 0.0
    precision = (overlap / n_rep * 100.0) if n_rep else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return {
        "overlap": overlap,
        "recall": recall,
        "precision": precision,
        "score": f1,
        "gt_counts": dict(cg),
        "replay_counts": dict(cr),
    }


def score_parity(gt: list[dict], replay: list[dict], *,
                 time_tol_min: float = 30.0,
                 entry_tol: float = 8.0) -> dict:
    """Greedy 1:1 timed match. Primary score = timed F1."""
    matched = []
    miss = []
    cand = sorted(replay, key=lambda t: t["entry_time"])
    used = set()

    for g in sorted(gt, key=lambda x: x["entry_time"]):
        best_i = None
        best_score = None
        for i, r in enumerate(cand):
            if i in used:
                continue
            if r["tf"] != g["tf"] or r["side"] != g["side"]:
                continue
            if r["rule"] != g["rule"]:
                continue
            dt_min = abs((r["entry_time"] - g["entry_time"]).total_seconds()) / 60.0
            if dt_min > time_tol_min:
                continue
            dpx = abs(r["entry"] - g["entry"])
            if dpx > entry_tol:
                continue
            sc = (dt_min / max(time_tol_min, 1e-9)) + (dpx / max(entry_tol, 1e-9))
            if best_score is None or sc < best_score:
                best_score = sc
                best_i = i
        if best_i is None:
            miss.append(g)
            continue
        used.add(best_i)
        r = cand[best_i]
        matched.append({
            "gt": g,
            "replay": r,
            "dt_min": abs((r["entry_time"] - g["entry_time"]).total_seconds()) / 60.0,
            "d_entry": r["entry"] - g["entry"],
        })

    extra = [cand[i] for i in range(len(cand)) if i not in used]
    n_gt = len(gt)
    n_rep = len(cand)
    n_m = len(matched)
    recall = (n_m / n_gt * 100.0) if n_gt else 0.0
    precision = (n_m / n_rep * 100.0) if n_rep else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    bag = score_bag(gt, cand)
    return {
        "n_gt": n_gt,
        "n_replay": n_rep,
        "n_matched": n_m,
        "recall": recall,
        "precision": precision,
        "strict_f1": f1,
        "bag": bag,
        "score": f1,  # primary = timed trade F1
        "matched": matched,
        "miss": miss,
        "extra": extra,
    }


def _fmt_trade(t: dict) -> str:
    et = pd.Timestamp(t["entry_time"]).strftime("%H:%M")
    return (f"{t['tf']:<4} {_norm_rule(t.get('rule', '?')):<10} "
            f"{_norm_side(t.get('side') or t.get('dir')):<5} "
            f"{et}  @{float(t['entry']):.2f}")


def print_report(res: dict, day: str):
    bag = res.get("bag") or {}
    print(f"\n  === PARITY {day} ===")
    print(f"  Ground truth : {res['n_gt']}  |  Replay : {res['n_replay']}  |  "
          f"Timed matches : {res['n_matched']}")
    print(f"  Timed trade F1        : {res.get('strict_f1', res['score']):.1f}%  "
          f"(recall {res['recall']:.1f}% / prec {res['precision']:.1f}%)  ← primary")
    if bag:
        print(f"  Idea-bag F1 (diag)    : {bag['score']:.1f}%  "
              f"(overlap {bag.get('overlap', 0)} / "
              f"recall {bag['recall']:.1f}% / prec {bag['precision']:.1f}%)")
    target = 85.0
    status = "PASS" if res["score"] >= target else "BELOW TARGET"
    print(f"  Target timed F1 ≥ {target:.0f}% → {status}")

    if res["matched"]:
        print(f"\n  --- MATCHED ({len(res['matched'])}) ---")
        for m in res["matched"]:
            g, r = m["gt"], m["replay"]
            print(f"  ✓ {_fmt_trade(g)}  ↔  replay@{r['entry']:.2f}  "
                  f"Δt={m['dt_min']:.0f}m  Δpx={m['d_entry']:+.2f}")
    if res["miss"]:
        print(f"\n  --- MISS live not in replay ({len(res['miss'])}) ---")
        for g in res["miss"]:
            print(f"  ✗ {_fmt_trade(g)}  ({g.get('outcome', '')})")
    if res["extra"]:
        print(f"\n  --- EXTRA replay not in live ({len(res['extra'])}) ---")
        for r in res["extra"]:
            print(f"  + {_fmt_trade(r)}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Score live_replay vs live ground truth")
    ap.add_argument("--csv", default="data/parity_live_2026-07-20.csv")
    ap.add_argument("--days", type=int, default=2,
                    help="History window for replay (must cover the GT day)")
    ap.add_argument("--asset", default="XAUUSD")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--mirror-vps", action="store_true", default=False,
                    help="Legacy loose VPS profile (OFF by default — identity mode)")
    ap.add_argument("--dual", action="store_true", default=True,
                    help="Run M5+M15 (default ON)")
    ap.add_argument("--single-tf", default="",
                    help="If set (M5/M15), run one TF instead of dual")
    ap.add_argument("--time-tol", type=float, default=30.0,
                    help="Max |Δentry_time| minutes for a match")
    ap.add_argument("--entry-tol", type=float, default=8.0,
                    help="Max |Δentry price| for a match (XAU points)")
    args = ap.parse_args()

    mirror = bool(args.mirror_vps)
    gt = load_ground_truth(args.csv)
    if not gt:
        raise SystemExit(f"No ground-truth rows in {args.csv}")
    day = pd.Timestamp(gt[0]["entry_time"]).strftime("%Y-%m-%d")

    fill = "market" if mirror else "live"
    ideas: list[dict] = []
    print(f"  Parity mode: {'mirror-vps (legacy)' if mirror else 'identity (fill=live, shared process_once)'}")
    if args.single_tf:
        tf = LR.normalize_tf(args.single_tf)
        eng = LR.LiveReplay(
            args.asset, tf, args.days, args.balance, fill,
            mirror_vps=mirror, record=False)
        trades = eng.run()
        ideas = list(eng.ideas)
    else:
        trades, _, ideas = LR.run_dual(
            args.asset, args.days, args.balance,
            mirror_vps=mirror, fill_mode=fill, record=False)

    replay_day = filter_day(trades, day)
    idea_day = filter_day(ideas, day)
    res = score_parity(
        gt, replay_day,
        time_tol_min=args.time_tol,
        entry_tol=args.entry_tol)
    res["bag"] = score_bag(gt, replay_day, ideas=idea_day)
    # Primary gate remains timed F1 from score_parity
    res["n_ideas"] = len(idea_day)
    print_report(res, day)
    print(f"  Armed ideas that day : {len(idea_day)}")
    if res["bag"].get("gt_counts"):
        print("  GT idea counts   :", res["bag"]["gt_counts"])
        print("  Replay idea max  :", res["bag"]["replay_counts"])
    if res["score"] < 85.0:
        sys.exit(2)


if __name__ == "__main__":
    main()
