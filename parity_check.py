"""
Phase 4 — Execution parity check.

Compare today's (or last N days) portfolio backtest trades vs what live
should have taken. Run AFTER portfolio_backtest.py:

  python portfolio_backtest.py --days 7
  python parity_check.py
  python parity_check.py --days 3
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

import floating_config as FC


def load_backtest_trades(path, since):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                et = str(r.get("entry_time", ""))[:10]
                if et >= since:
                    rows.append(r)
    except FileNotFoundError:
        pass
    return rows


def main():
    ap = argparse.ArgumentParser(description="Backtest vs live parity (Phase 4)")
    ap.add_argument("--days", type=int, default=FC.PARITY.get("compare_days", 1))
    ap.add_argument("--csv", default=FC.PARITY.get("backtest_csv"))
    args = ap.parse_args()

    since = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    trades = load_backtest_trades(args.csv, since)
    active = set(a.upper() for a in FC.ACTIVE_ASSETS)

    print("=" * 72)
    print(f"  PARITY CHECK  |  backtest trades since {since}  |  window {args.days}d")
    print(f"  Active assets : {', '.join(FC.ACTIVE_ASSETS)}")
    print(f"  CSV           : {args.csv}")
    print("=" * 72)

    if not trades:
        print("  No backtest trades in window. Run:")
        print("    python portfolio_backtest.py --days 7")
        return

    by_day = defaultdict(list)
    by_asset = defaultdict(int)
    for t in trades:
        asset = t.get("asset", "?").upper()
        if asset not in active:
            continue
        day = str(t.get("entry_time", ""))[:10]
        by_day[day].append(t)
        by_asset[asset] += 1

    print(f"\n  Backtest trades (active assets only): {sum(by_asset.values())}")
    for a in FC.ACTIVE_ASSETS:
        print(f"    {a}: {by_asset.get(a, 0)}")

    print("\n  Per day:")
    for day in sorted(by_day):
        ts = [t for t in by_day[day] if t.get("asset", "").upper() in active]
        print(f"    {day}: {len(ts)} trades")
        for t in ts:
            side = "LONG" if t.get("side", t.get("dir")) in ("long", "LONG") else "SHORT"
            print(f"      {t.get('entry_time','')[:16]}  {t.get('asset')}  "
                  f"{t.get('rule','?'):<10}  {side}  R={float(t.get('R',0)):+.2f}")

    print("\n  LIVE CHECKLIST (manual):")
    print("    1. Open MT5 Trade History for same dates")
    print("    2. Count trades per asset — should match above (±1 timing)")
    print("    3. Same rule names in order comments (SMC-RULENAME)")
    print("    4. If backtest has trade live missed → check limit fill / spread")
    print("    5. If live has trade backtest missed → check data / regime gate")
    print("=" * 72)


if __name__ == "__main__":
    main()
