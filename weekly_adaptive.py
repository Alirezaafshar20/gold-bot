"""
Weekly adaptive gate — regime report + rolling rule×regime health + next-week schedule.

Causal only: uses closed trades and past H1 bars. Never tunes on future data.

  python weekly_adaptive.py              # build reports/weekly_gate.json
  python weekly_adaptive.py --analyze    # print report only (no file write)
  python weekly_adaptive.py --build --weeks 8

Output: reports/weekly_gate.json  (read by floating_config.load_meta_gate)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import floating_config as FC
import meta_gate as MG
import mt5_data as M
import portfolio_config as C
import regime as RG
import strategy as S
import symbol_profiles as P

DEFAULT_CSV = "reports/portfolio_trades.csv"
DEFAULT_OUT = "reports/weekly_gate.json"
REGIMES = MG.REGIMES


def normalize_rule(display: str, side: str) -> str:
    """Map trade-list label + side -> internal meta-gate rule name."""
    d = str(display or "?").strip().upper().replace(" ", "")
    side = str(side or "").lower()
    if d == "VWAP":
        return "VWAP_L" if side == "long" else "VWAP_S"
    if d in ("CH-REV", "CH_REV"):
        return "CH_REV_L" if side == "long" else "CH_REV_S"
    if d in ("CH-BO", "CH_BO"):
        return "CH_BO_L" if side == "long" else "CH_BO_S"
    if d in ("WYCK-SOW", "WYCK_SOW"):
        return "WYCK_SOW"
    if d in ("WYCK-SOS", "WYCK_SOS"):
        return "WYCK_SOS"
    if d in ("ICT-SB", "ICT_SB"):
        return "ICT_SB_L" if side == "long" else "ICT_SB_S"
    return d.replace("-", "_")


def load_trades_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        ts = pd.Timestamp(r["entry_time"])
        rows.append({
            "ts": ts,
            "asset": r["asset"].upper(),
            "rule_display": r["rule"],
            "rule": normalize_rule(r["rule"], r.get("side", "")),
            "side": r.get("side", ""),
            "R": float(r["R"]),
            "pnl": float(r["pnl_usd"]),
            "risk_usd": float(r["risk_usd"]),
            "netR": float(r["pnl_usd"]) / float(r["risk_usd"]) if float(r["risk_usd"]) else 0.0,
        })
    rows.sort(key=lambda x: x["ts"])
    return rows


def _h1_from_m15_csv(path: str):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample("1h").agg(agg).dropna(subset=["open", "high", "low", "close"])


def load_regime_map(asset: str, days: int = 60, mt5=None):
    """H1 RegimeMap using same hybrid settings as floating_config."""
    reg = FC.REGIME
    prof = P.PROFILES.get(asset, {})
    htf = None
    if mt5 is not None:
        try:
            sym = P.resolve_symbol_for_profile(asset, mt5)
            _, htf_dfs = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=("H1", "H4"))
            htf = S.prepare_htf_context(htf_dfs)
        except Exception:
            htf = None
    if htf is None:
        for path in (f"data/{asset}@_M15.csv", f"data/{asset}_M15.csv",
                     f"data/XAUUSD@_M15.csv", f"data/XAUUSD_M15.csv"):
            h1 = _h1_from_m15_csv(path)
            if h1 is not None and len(h1) > 100:
                return RG.RegimeMap(
                    h1,
                    detector=reg["regime_detector"],
                    win=reg["regime_win"],
                    er_trend=reg["regime_er_trend"],
                    slope_k=reg["regime_slope_k"],
                    er_hi=reg["regime_er_hi"],
                    er_lo=reg["regime_er_lo"],
                    confirm=reg["regime_confirm"],
                )
        return None
    return S.build_regime_map(
        htf,
        tf=reg["regime_tf"],
        detector=reg["regime_detector"],
        win=reg["regime_win"],
        er_trend=reg["regime_er_trend"],
        slope_k=reg["regime_slope_k"],
        er_hi=reg["regime_er_hi"],
        er_lo=reg["regime_er_lo"],
        confirm=reg["regime_confirm"],
        h4_win=reg.get("regime_h4_win", 30),
        h4_lookback=reg.get("regime_h4_lookback", 60),
    )


def regime_weekly_report(rmap, report_days: int = 7) -> dict:
    """% time in each regime over last report_days + current label."""
    if rmap is None:
        return {"error": "no regime data"}
    t = rmap.t
    lab = rmap.labels
    if len(t) == 0:
        return {"error": "empty regime map"}
    end = pd.Timestamp(t[-1])
    start = end - pd.Timedelta(days=report_days)
    mask = (pd.to_datetime(t) >= start) & (pd.to_datetime(t) <= end)
    subset = lab[mask] if mask.any() else lab[-min(24 * report_days, len(lab)):]
    n = len(subset)
    counts = {r: int((subset == r).sum()) for r in REGIMES}
    pct = {r: round(100.0 * counts[r] / n, 1) if n else 0.0 for r in REGIMES}
    current = str(lab[-1])
    top = max(REGIMES, key=lambda r: pct[r])
    bias = top if pct[top] >= 40 else "MIXED"
    return {
        "week_end": str(end)[:19],
        "week_start": str(start)[:19],
        "bars": n,
        "pct": pct,
        "counts": counts,
        "current": current,
        "bias": bias,
    }


def tag_trades_with_regime(rows: list[dict], rmaps: dict) -> list[dict]:
    out = []
    for r in rows:
        rmap = rmaps.get(r["asset"])
        reg = MG.regime_at(rmap, r["ts"], "RANGE") if rmap else "RANGE"
        out.append({**r, "regime": reg})
    return out


def rolling_rule_regime_stats(
    tagged: list[dict],
    weeks: int = 8,
    since: pd.Timestamp | None = None,
) -> dict:
    """PF / totR per (asset, rule, regime) over rolling window."""
    if not tagged:
        return {}
    last = max(r["ts"] for r in tagged)
    win_lo = last - pd.Timedelta(weeks=weeks)
    if since is not None:
        win_lo = max(win_lo, since)
    window = [r for r in tagged if r["ts"] >= win_lo]
    cells = defaultdict(list)
    for r in window:
        cells[(r["asset"], r["rule"], r["regime"])].append(r["netR"])
    out = {}
    for key, nets in cells.items():
        R = np.array(nets, dtype=float)
        wins = R[R > 0]
        loss = R[R <= 0]
        gp = wins.sum()
        gl = -loss.sum()
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        out[key] = {
            "n": len(R),
            "wr": round(100.0 * (R > 0).mean(), 1),
            "pf": round(float(pf), 2),
            "tot_r": round(float(R.sum()), 2),
            "avg_r": round(float(R.mean()), 3),
        }
    return out


def build_weekly_table(
    assets: tuple,
    oos_mg: MG.MetaGate | None,
    rolling: dict,
    cfg: dict,
) -> tuple[dict, dict]:
    """
    Merge OOS allow-table with rolling health checks.
    Returns (table for MetaGate, schedule detail for humans).
    """
    min_n = cfg.get("min_rolling_n", 3)
    pause_pf = cfg.get("pause_pf", 0.70)
    table = {}
    schedule = {}

    for asset in assets:
        asset = asset.upper()
        rules = FC.OOS_RULES.get(asset, ())
        schedule[asset] = {}
        for rule in rules:
            rule = rule.upper()
            key = (asset, rule)
            oos_regs = oos_mg.table.get(key) if oos_mg else None
            if oos_regs is None:
                oos_regs = frozenset(REGIMES)
            if oos_mg and oos_mg.skip_empty and not oos_regs:
                table[key] = frozenset()
                schedule[asset][rule] = {
                    "allowed": [],
                    "status": "blocked",
                    "reason": "OOS weak — excluded",
                }
                continue
            final = set()
            notes = []
            for reg in sorted(oos_regs):
                st = rolling.get((asset, rule, reg))
                if st and st["n"] >= min_n:
                    if st["tot_r"] <= 0 or st["pf"] < pause_pf:
                        notes.append(
                            f"{reg}: PAUSE rolling n={st['n']} PF={st['pf']} totR={st['tot_r']}"
                        )
                        continue
                    notes.append(f"{reg}: OK rolling n={st['n']} PF={st['pf']}")
                else:
                    n = st["n"] if st else 0
                    notes.append(f"{reg}: OOS only (rolling n={n})")
                final.add(reg)
            if not final and oos_regs:
                # Don't whipsaw entire rule — keep OOS if rolling bad but thin sample
                thin_bad = all(
                    (rolling.get((asset, rule, reg)) or {}).get("n", 0) < min_n
                    for reg in oos_regs
                )
                if thin_bad:
                    final = set(oos_regs)
                    status = "active"
                    reason = "OOS kept — rolling sample too thin"
                else:
                    status = "paused"
                    reason = "; ".join(notes)
            else:
                status = "active" if final else "blocked"
                reason = "; ".join(notes) if notes else "OOS"
            table[key] = frozenset(final)
            schedule[asset][rule] = {
                "allowed": sorted(final),
                "status": status,
                "reason": reason,
            }
    return table, schedule


def ledger_to_trade_rows(ledger: list[dict]) -> list[dict]:
    """Convert portfolio_backtest ledger rows -> rolling analysis rows."""
    rows = []
    for r in ledger:
        risk = float(r.get("risk_usd") or 0)
        net = float(r.get("net") or r.get("pnl_usd") or 0)
        rows.append({
            "ts": pd.Timestamp(r["entry_time"]),
            "asset": str(r["asset"]).upper(),
            "rule_display": r.get("rule", "?"),
            "rule": normalize_rule(r.get("rule", "?"), r.get("dir", "")),
            "side": r.get("dir", ""),
            "R": float(r.get("R", 0)),
            "pnl": net,
            "risk_usd": risk,
            "netR": net / risk if risk else 0.0,
        })
    rows.sort(key=lambda x: x["ts"])
    return rows


def backtest_trades_to_rows(trades: list[dict], default_asset: str = "XAUUSD") -> list[dict]:
    """Strategy trade dicts -> rolling rows (R-multiple as netR)."""
    rows = []
    for t in trades:
        rows.append({
            "ts": pd.Timestamp(t["time"]),
            "asset": str(t.get("_asset", default_asset)).upper(),
            "rule_display": t.get("setup") or t.get("rule", "?"),
            "rule": normalize_rule(t.get("setup") or t.get("rule", "?"), t.get("dir", "")),
            "side": t.get("dir", ""),
            "R": float(t.get("R", 0)),
            "netR": float(t.get("R", 0)),
        })
    rows.sort(key=lambda x: x["ts"])
    return rows


def meta_gate_from_prior_trades(
    prior_trades: list[dict],
    rmaps: dict,
    assets: tuple | None = None,
) -> MG.MetaGate:
    """Causal gate: OOS + rolling on trades closed before the current week."""
    assets = tuple(a.upper() for a in (assets or FC.ACTIVE_ASSETS))
    oos_path = FC.META.get("meta_gate_json", MG.DEFAULT_JSON)
    oos_mg = MG.load_meta_gate(
        oos_path,
        assets=assets,
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )
    rows = backtest_trades_to_rows(prior_trades)
    tagged = tag_trades_with_regime(rows, rmaps)
    rolling = rolling_rule_regime_stats(
        tagged, weeks=FC.WEEKLY.get("rolling_weeks", 8))
    table, _ = build_weekly_table(assets, oos_mg, rolling, FC.WEEKLY)
    return MG.meta_gate_from_table(table)


def apply_walkforward_filter(
    trades: list[dict],
    mt5=None,
    assets: tuple | None = None,
) -> tuple[list[dict], dict]:
    """
    Replay trades chronologically; each ISO week uses a fresh gate built only
    from trades that CLOSED before that week started (causal walk-forward).
    """
    assets = tuple(a.upper() for a in (assets or FC.ACTIVE_ASSETS))
    if not trades:
        return [], {"before": 0, "after": 0, "removed": []}

    own_mt5 = mt5 is None
    if own_mt5:
        mt5 = M.connect()
    rmaps = {}
    try:
        for a in assets:
            rmaps[a] = load_regime_map(a, days=120, mt5=mt5)
    finally:
        if own_mt5:
            M.shutdown(mt5)

    ordered = sorted(trades, key=lambda t: (pd.Timestamp(t["time"]), t.get("exit_time")))
    kept: list[dict] = []
    removed: list[dict] = []
    mg_cache = None
    cache_week = None

    for t in ordered:
        entry = pd.Timestamp(t["time"])
        week_start = entry.to_period("W-MON").start_time
        if week_start != cache_week:
            cache_week = week_start
            prior = [
                k for k in kept
                if pd.Timestamp(k.get("exit_time", k["time"])) < week_start
            ]
            mg_cache = meta_gate_from_prior_trades(prior, rmaps, assets)

        asset = str(t.get("_asset", assets[0])).upper()
        rule = normalize_rule(t.get("setup") or t.get("rule", "?"), t.get("dir", ""))
        reg = MG.regime_at(rmaps.get(asset), entry, "RANGE")

        if mg_cache is not None and not mg_cache.allows(asset, rule, reg):
            removed.append({
                "time": str(entry)[:16],
                "asset": asset,
                "rule": rule,
                "regime": reg,
                "week": str(week_start.date()),
                "display": t.get("setup") or t.get("rule"),
                "dir": t.get("dir"),
                "R": t.get("R"),
            })
            continue
        kept.append(t)

    return kept, {
        "before": len(ordered),
        "after": len(kept),
        "removed": removed,
    }


def auto_refresh(
    csv_path: str | None = None,
    out_path: str | None = None,
    mt5=None,
    ledger: list[dict] | None = None,
    silent: bool = True,
    write: bool = True,
) -> dict | None:
    """
    Silent entry point for portfolio_backtest / live — rebuild weekly_gate.json.
    Uses ledger rows when provided, else previous CSV on disk.
    """
    if not FC.WEEKLY.get("enabled", True):
        return None
    csv_path = csv_path or FC.WEEKLY.get("trades_csv", DEFAULT_CSV)
    out_path = out_path or FC.WEEKLY.get("json", DEFAULT_OUT)
    trade_rows = ledger_to_trade_rows(ledger) if ledger else None
    if trade_rows is None and not os.path.isfile(csv_path):
        trade_rows = []  # no journal yet — still refresh regime report + OOS table
    doc = build_gate(
        csv_path=csv_path,
        out_path=out_path,
        weeks=FC.WEEKLY.get("rolling_weeks", 8),
        report_days=FC.WEEKLY.get("regime_report_days", 7),
        mt5=mt5,
        trade_rows=trade_rows,
        write=write,
    )
    if not silent:
        print_report(doc)
    return doc


def build_gate(
    csv_path: str = DEFAULT_CSV,
    out_path: str = DEFAULT_OUT,
    weeks: int = 8,
    report_days: int = 7,
    assets: tuple | None = None,
    write: bool = True,
    mt5=None,
    trade_rows: list[dict] | None = None,
) -> dict:
    cfg = FC.WEEKLY
    assets = tuple(a.upper() for a in (assets or FC.ACTIVE_ASSETS))
    oos_path = FC.META.get("meta_gate_json", MG.DEFAULT_JSON)
    oos_mg = MG.load_meta_gate(
        oos_path,
        assets=assets,
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )

    rows = trade_rows if trade_rows is not None else load_trades_csv(csv_path)
    own_mt5 = mt5 is None
    if own_mt5:
        mt5 = M.connect()
    rmaps = {}
    regime_reports = {}
    try:
        for a in assets:
            rmaps[a] = load_regime_map(a, days=max(60, report_days * 10), mt5=mt5)
            regime_reports[a] = regime_weekly_report(rmaps[a], report_days)
    finally:
        if own_mt5:
            M.shutdown(mt5)

    tagged = tag_trades_with_regime(rows, rmaps)
    # Parity with the backtest walk-forward: each week's gate must be built
    # only from trades that closed BEFORE the current ISO week. Without this,
    # a mid-week refresh feeds this week's trades into the rolling stats and
    # the live gate drifts away from the replayed (backtested) one.
    if cfg.get("causal_week_cutoff", True) and tagged:
        week_start = pd.Timestamp(datetime.now()).to_period("W-MON").start_time
        tagged = [r for r in tagged if r["ts"] < week_start]
    rolling = rolling_rule_regime_stats(tagged, weeks=weeks)

    table, schedule = build_weekly_table(assets, oos_mg, rolling, cfg)

    # Nested rolling dict for JSON
    rolling_json = defaultdict(dict)
    for (asset, rule, reg), st in rolling.items():
        rolling_json[asset].setdefault(rule, {})[reg] = st

    now = datetime.now()
    week_id = now.strftime("%G-W%V")
    doc = {
        "meta": {
            "built_at": now.isoformat(timespec="seconds"),
            "week_id": week_id,
            "rolling_weeks": weeks,
            "regime_report_days": report_days,
            "oos_json": oos_path,
            "trades_csv": csv_path,
            "trades_in_window": sum(
                1 for r in tagged
                if tagged and r["ts"] >= max(x["ts"] for x in tagged) - pd.Timedelta(weeks=weeks)
            ),
        },
        "regime_report": regime_reports,
        "rolling": dict(rolling_json),
        "schedule": schedule,
        "table": {
            a: {rule: sorted(table.get((a, rule), frozenset()))
                for rule in FC.OOS_RULES.get(a, ())}
            for a in assets
        },
    }

    if write:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    return doc


def print_report(doc: dict):
    meta = doc.get("meta", {})
    print("=" * 72)
    print(f"  WEEKLY ADAPTIVE REPORT  |  {meta.get('week_id', '?')}")
    print(f"  Rolling: {meta.get('rolling_weeks', '?')}w  |  "
          f"Trades CSV: {meta.get('trades_csv', '?')}")
    print("=" * 72)

    for asset, rep in doc.get("regime_report", {}).items():
        if rep.get("error"):
            print(f"\n  {asset} regime: {rep['error']}")
            continue
        print(f"\n  {asset} — REGIME (last {meta.get('regime_report_days', 7)}d)")
        pct = rep.get("pct", {})
        print(f"    RANGE {pct.get('RANGE', 0):>5.1f}%  |  "
              f"TREND_UP {pct.get('TREND_UP', 0):>5.1f}%  |  "
              f"TREND_DOWN {pct.get('TREND_DOWN', 0):>5.1f}%")
        print(f"    Current: {rep.get('current')}  |  Bias: {rep.get('bias')}")

    print(f"\n  RULE SCHEDULE (OOS + rolling {meta.get('rolling_weeks')}w)")
    print(f"  {'asset':<8} {'rule':<12} {'status':<8} allowed")
    print("  " + "-" * 68)
    for asset, rules in doc.get("schedule", {}).items():
        for rule, info in sorted(rules.items()):
            allowed = ",".join(info.get("allowed") or []) or "—"
            print(f"  {asset:<8} {rule:<12} {info.get('status', '?'):<8} {allowed}")

    print(f"\n  ROLLING detail (cells with trades)")
    print(f"  {'asset':<8} {'rule':<12} {'regime':<12} {'n':>4} {'WR':>6} {'PF':>6} {'totR':>7}")
    print("  " + "-" * 68)
    for asset, rules in sorted(doc.get("rolling", {}).items()):
        for rule, regs in sorted(rules.items()):
            for reg, st in sorted(regs.items()):
                print(f"  {asset:<8} {rule:<12} {reg:<12} {st['n']:>4} "
                      f"{st['wr']:>5.1f}% {st['pf']:>6.2f} {st['tot_r']:>+7.2f}")

    pauses = []
    for asset, rules in doc.get("schedule", {}).items():
        for rule, info in rules.items():
            if info.get("status") == "paused":
                pauses.append((asset, rule, info.get("reason", "")))
    if pauses:
        print(f"\n  PAUSED this week:")
        for a, r, why in pauses:
            print(f"    {a} {r}: {why}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Weekly regime + rule schedule builder")
    ap.add_argument("--analyze", action="store_true",
                    help="Print report only — do not write weekly_gate.json")
    ap.add_argument("--build", action="store_true",
                    help="Write reports/weekly_gate.json (default if neither flag)")
    ap.add_argument("--weeks", type=int, default=None,
                    help=f"Rolling window weeks (default {FC.WEEKLY['rolling_weeks']})")
    ap.add_argument("--days", type=int, default=None,
                    help=f"Regime report days (default {FC.WEEKLY['regime_report_days']})")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--assets", nargs="*", default=None)
    args = ap.parse_args()

    weeks = args.weeks if args.weeks is not None else FC.WEEKLY["rolling_weeks"]
    report_days = args.days if args.days is not None else FC.WEEKLY["regime_report_days"]
    write = not args.analyze
    if not args.analyze and not args.build:
        write = True

    if not os.path.isfile(args.csv):
        print(f"WARNING: no trades at {args.csv} — rolling stats will be empty (OOS only).")

    doc = build_gate(
        csv_path=args.csv,
        out_path=args.out,
        weeks=weeks,
        report_days=report_days,
        assets=tuple(args.assets) if args.assets else None,
        write=write,
    )
    print_report(doc)
    if write:
        print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
