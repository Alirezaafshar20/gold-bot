"""
Robust Dukascopy M1 downloader — month by month, with retries.

Dukascopy's server intermittently throws "Unknown error" on large ranges, so we
request ONE MONTH at a time (small, reliable), retrying each month a few times.
Files are saved as data/dukascopy/{KEY}_M1_{YYYY-MM}.csv and auto-merged later
by dukascopy_loader.py. Already-downloaded months are skipped, so you can re-run
this any time to fill gaps.

  python duka_download.py                      # all symbols, 2022-04 -> now
  python duka_download.py --from 2022-04 --to 2026-06
  python duka_download.py --symbols BTCUSD,XAUUSD
  python duka_download.py --retries 12
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data/dukascopy"

# project key -> dukascopy instrument id
SYMBOLS = {
    "XAUUSD": "xauusd",
    "US500":  "usa500idxusd",
    "US30":   "usa30idxusd",
    "BRENT":  "brentcmdusd",
    "BTCUSD": "btcusd",
}

NPX = "npx.cmd" if os.name == "nt" else "npx"


def month_iter(start_ym, end_ym):
    """Yield (from_date, to_date, tag) for each month in [start, end]."""
    sy, sm = start_ym
    ey, em = end_ym
    y, m = sy, sm
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        frm = f"{y:04d}-{m:02d}-01"
        to = f"{ny:04d}-{nm:02d}-01"
        yield frm, to, f"{y:04d}-{m:02d}"
        y, m = ny, nm


def parse_ym(s):
    parts = s.split("-")
    return int(parts[0]), int(parts[1])


def download_month(inst, frm, to, out_path, retries, pause):
    fn = os.path.splitext(os.path.basename(out_path))[0]
    cmd = [NPX, "--yes", "dukascopy-node", "-i", inst, "-from", frm, "-to", to,
           "-t", "m1", "-f", "csv", "-v", "-dir", DATA_DIR, "-fn", fn,
           "-r", str(retries), "-bs", "4", "-bp", "1000", "-ch"]
    for attempt in range(1, retries + 1):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"      timeout (try {attempt}/{retries})", flush=True)
            time.sleep(pause)
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        # server hiccup -> wait and retry
        err = (res.stderr or res.stdout or "").strip().splitlines()
        msg = err[-1][:70] if err else "no file produced"
        print(f"      retry {attempt}/{retries}: {msg}", flush=True)
        time.sleep(pause)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--from", dest="frm", default="2022-04")
    ap.add_argument("--to", dest="to", default=None, help="YYYY-MM (default: now)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--pause", type=float, default=3.0,
                    help="seconds between retries / months")
    args = ap.parse_args()

    start_ym = parse_ym(args.frm)
    if args.to:
        end_ym = parse_ym(args.to)
    else:
        now = datetime.utcnow()
        end_ym = (now.year, now.month)

    keys = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 76)
    print(f"  DUKASCOPY M1 DOWNLOAD (month by month)  {args.frm} -> "
          f"{end_ym[0]:04d}-{end_ym[1]:02d}")
    print(f"  symbols: {', '.join(keys)}")
    print("=" * 76)

    total_fail = []
    for key in keys:
        inst = SYMBOLS.get(key)
        if not inst:
            print(f"  {key}: unknown symbol, skipping")
            continue
        print(f"\n  --- {key} ({inst}) ---", flush=True)
        ok = skip = fail = 0
        for frm, to, tag in month_iter(start_ym, end_ym):
            out_path = os.path.join(DATA_DIR, f"{key}_M1_{tag}.csv")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skip += 1
                continue
            print(f"    {key} {tag}: downloading ...", flush=True)
            if download_month(inst, frm, to, out_path, args.retries, args.pause):
                size = os.path.getsize(out_path) / 1024.0
                print(f"    {key} {tag}: OK ({size:.0f} KB)", flush=True)
                ok += 1
                time.sleep(args.pause)
            else:
                print(f"    {key} {tag}: FAILED after {args.retries} tries", flush=True)
                fail += 1
                total_fail.append(f"{key} {tag}")
        print(f"    {key}: {ok} downloaded, {skip} already had, {fail} failed",
              flush=True)

    print("\n" + "=" * 76)
    if total_fail:
        print(f"  {len(total_fail)} month(s) failed — just re-run to retry only these:")
        for f in total_fail:
            print(f"    {f}")
    else:
        print("  All months downloaded.")
    print("  Next: python dukascopy_loader.py   (verify coverage)")
    print("=" * 76)


if __name__ == "__main__":
    main()
