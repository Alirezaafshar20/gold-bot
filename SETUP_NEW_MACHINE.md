# Setup on the new (stronger) machine — long calibration 2022→now

Goal: re-pick the trading rules on M15 from **2022-04-01 to now** using a clean
multi-year feed, so the live model is no longer overfit to the last 90 days.
Live logic and the gold (XAUUSD) defaults are **not** changed by any of this.

---

## 0. Copy the project
Copy the whole `old model` folder to the new PC (USB / zip / git). Keep the
folder structure intact (it contains `strategy.py`, `symbol_profiles.py`,
`calibrate_long.py`, `dukascopy_loader.py`, etc.).

## 1. Install Python 3.11+
https://www.python.org/downloads/  → tick **"Add Python to PATH"** during install.

## 2. Create a virtual env + install Python deps
```powershell
cd "path\to\old model"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install MetaTrader5            # only needed for live + broker validation
```

## 3. Install MetaTrader 5 (only for live trading + broker re-check)
- Install the WM Markets MT5 terminal, log into the **same demo/live account**.
- Open it once and let Market Watch show: `XAUUSD@`, `US500CASH`, `US30CASH`,
  `BRENTCASH`, `BTCUSD@`.
- Not required for the calibration itself (that runs on Dukascopy files offline).

## 4. Install Node.js (for the Dukascopy downloader)
https://nodejs.org → LTS version. Verify: `node -v`.

## 5. Download 1-minute history from Dukascopy (2022 → now)
Run in the project folder (creates `data/dukascopy/`):
```bash
npx dukascopy-node -i xauusd       -from 2022-04-01 -to now -t m1 -f csv -v -dir ./data/dukascopy -fn XAUUSD_M1
npx dukascopy-node -i usa500idxusd -from 2022-04-01 -to now -t m1 -f csv -v -dir ./data/dukascopy -fn US500_M1
npx dukascopy-node -i usa30idxusd  -from 2022-04-01 -to now -t m1 -f csv -v -dir ./data/dukascopy -fn US30_M1
npx dukascopy-node -i brentcmdusd  -from 2022-04-01 -to now -t m1 -f csv -v -dir ./data/dukascopy -fn BRENT_M1
npx dukascopy-node -i btcusd       -from 2022-04-01 -to now -t m1 -f csv -v -dir ./data/dukascopy -fn BTCUSD_M1
```
Each download is large (4 years of M1) and may take several minutes — that is fine.

Then verify the files are healthy:
```powershell
python dukascopy_loader.py
```
You should see row counts, start/end dates, and gap counts for all 5 symbols.

## 6. Run the long walk-forward calibration
Full run (all assets, 2022→now). Takes a while — that is expected:
```powershell
python calibrate_long.py --scope dense
```
Single asset (faster, to test one):
```powershell
python calibrate_long.py --asset BTCUSD --scope dense
```
Wider rule search (slower, more rules):
```powershell
python calibrate_long.py --scope full
```
Output → `reports/calibrate_long.txt` (+ `.json`). For each asset it prints two
leaderboards:
- **MOST ROBUST** — rules profitable across the most folds (anti-overfit). Recommended.
- **MOST PROFITABLE** — biggest raw return over the whole period.

## 7. (Optional) quick smoke test before the big run
Runs on the broker's recent ~90 days (no Dukascopy needed) just to confirm the
pipeline works on this machine:
```powershell
python calibrate_long.py --source mt5 --asset XAUUSD --days 90 --scope core --fold-days 30
```

## 8. Send the results back
Send me `reports/calibrate_long.txt` and `reports/calibrate_long.json`.
We then decide together (robust vs profit) and I edit only the per-asset
`rules_override` in `symbol_profiles.py` — nothing else.

---

### Notes on data limits (important, honest)
- The broker (WM Markets) keeps only ~100 days of M1, so it **cannot** drive a
  2022 backtest — that is why we use Dukascopy.
- Dukascopy prices differ slightly from the broker (different feed). For *rule
  selection* this is fine (the edge is relative: R / ATR based). Live still runs
  on broker data; after we pick rules we re-validate on the broker's recent data.
- BTC on this broker only goes back to 2023 on M15, but Dukascopy has BTC from
  2017 — so BTC is also covered from 2022.
