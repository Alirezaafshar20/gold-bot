"""
Merge one asset's trades from a fresh single-asset OOS run into the main ledger.

Use case: the main reports/oos_trades.csv came from the strong PC where one
asset (e.g. gold) had incomplete data. Re-run just that asset locally with full
data, then splice its rows in:

  python oos_test.py --asset XAUUSD --regime-detector hybrid --confirm 2 \
         --csv reports/oos_gold.csv --out reports/oos_gold.txt
  python merge_asset.py --asset XAUUSD --into reports/oos_trades.csv \
         --from-csv reports/oos_gold.csv

It removes all rows of --asset from --into, appends every row of that asset
from --from-csv, re-sorts by entry_time, renumbers the 'n' column, and writes
the result back (a .bak backup is kept). oos_analyze reads this fine because it
recomputes everything per-trade from pnl_usd / risk_usd.
"""
import argparse
import csv
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, help="asset key to replace, e.g. XAUUSD")
    ap.add_argument("--into", default="reports/oos_trades.csv",
                    help="main ledger to splice into (modified in place)")
    ap.add_argument("--from-csv", dest="src", required=True,
                    help="fresh single-asset ledger to take the asset from")
    ap.add_argument("--out", default=None,
                    help="optional separate output (default: overwrite --into)")
    args = ap.parse_args()

    main_rows, fields = read_rows(args.into)
    src_rows, _ = read_rows(args.src)

    asset = args.asset.upper()
    kept = [r for r in main_rows if r["asset"].upper() != asset]
    removed = len(main_rows) - len(kept)
    incoming = [r for r in src_rows if r["asset"].upper() == asset]

    merged = kept + incoming
    merged.sort(key=lambda r: r["entry_time"])
    for i, r in enumerate(merged, 1):
        if "n" in r:
            r["n"] = i

    out_path = args.out or args.into
    if out_path == args.into and os.path.exists(args.into):
        shutil.copyfile(args.into, args.into + ".bak")
        print(f"  backup: {args.into}.bak")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"  {asset}: removed {removed} old rows, added {len(incoming)} fresh rows")
    print(f"  total now: {len(merged)}  ->  {out_path}")
    # quick date sanity for the spliced asset
    a_rows = sorted((r["entry_time"] for r in incoming))
    if a_rows:
        print(f"  {asset} range now: {a_rows[0]}  ->  {a_rows[-1]}")


if __name__ == "__main__":
    main()
