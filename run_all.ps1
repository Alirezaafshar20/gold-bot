# =====================================================================
#  One-click: setup + Dukascopy download + OUT-OF-SAMPLE test
#  Run from the project folder:
#      powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#  Re-running is safe: it skips steps already done (venv, existing data).
#
#  The honest question this answers:
#    rules are chosen ONLY on data up to $CUTOFF, then FROZEN and judged on
#    the unseen period after it. If the profit survives -> real edge.
#    If not -> it was overfit. No symbol is dropped; nothing is hand-tuned.
#
#  This run tests ALL edges together (the original rules AND the new ones -
#  Regression Channel: CH_*, and Prior Day/Week Levels: PD*/PW*) because they
#  are all part of the "dense" rule scope. They are re-judged under the NEW
#  low-lag + anti-whipsaw regime detector (hybrid: ER + slope + hysteresis).
#
#  Note on "last few days": Dukascopy publishes recent days with a small lag,
#  so the very last ~1-7 days may be missing. That is fine here - TRAIN ends at
#  $CUTOFF and the TEST window just ends a few days earlier. Nothing breaks.
# =====================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Section($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

# ---- config -----------------------------------------------------------
$FROM   = "2022-04-01"     # calibration data start
$CUTOFF = "2024-12-31"     # TRAIN = before this, TEST = unseen data after this
$SCOPE  = "dense"          # core | dense | full  (dense INCLUDES the new edges)
$PICK   = "regime"         # regime | robust | profit  (how rules are chosen on TRAIN)
$TOP    = 5                # rules kept per asset
# Regime detector MUST match floating_config.REGIME["regime_detector"], which is
# what the live bot uses. If they differ, the meta-gate is keyed on regime labels
# live never produces (mtf and hybrid only agree ~71% of the time).
$REGDET  = "mtf"           # mtf | hybrid | kaufman | slope | donchian | ma
$CONFIRM = 2               # bars a regime change must persist (anti-noise)
$DataDir = "data/dukascopy"

# ---- 1. checks --------------------------------------------------------
Section "1/6  Checking Python and Node.js"
try { $py = (python --version) 2>&1; Write-Host "  Python: $py" }
catch { Write-Host "  Python NOT found. Install from https://www.python.org/downloads/ (tick 'Add to PATH')." -ForegroundColor Red; exit 1 }
try { $nd = (node --version) 2>&1; Write-Host "  Node:   $nd" }
catch { Write-Host "  Node.js NOT found. Install LTS from https://nodejs.org then re-run." -ForegroundColor Red; exit 1 }

# ---- 2. venv + deps ---------------------------------------------------
Section "2/6  Python virtual environment + dependencies"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Created .venv"
} else { Write-Host "  .venv already exists - reusing" }
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
Write-Host "  Dependencies installed"
$PY = ".\.venv\Scripts\python.exe"

# ---- 3. download Dukascopy M1 ----------------------------------------
Section "3/6  Downloading Dukascopy 1-minute data (month by month, robust)"
Write-Host "  A few most-recent days may be missing (Dukascopy lag) - that is OK."
& $PY duka_download.py --from "2022-04"

# ---- 4. verify data ---------------------------------------------------
Section "4/6  Verifying Dukascopy data coverage"
& $PY dukascopy_loader.py

# ---- 5. regime-detector noise check ----------------------------------
Section "5/6  Regime detector noise check (hybrid vs kaufman vs ma)"
Write-Host "  Shows the new '$REGDET' detector flips far less than the old one,"
Write-Host "  without the moving-average lag. Lower flips/1000 = less whipsaw."
& $PY regime_compare.py

# ---- 6. OUT-OF-SAMPLE test (the real verdict) ------------------------
Section "6/6  OUT-OF-SAMPLE test  (ALL edges, regime=$REGDET, TRAIN < $CUTOFF)"
Write-Host "  Tests every rule (original + new CH_*/PD*/PW* edges) chosen on TRAIN"
Write-Host "  only, then frozen and judged on unseen data. This can take a while."
Write-Host "  No symbol is removed; nothing is hand-tuned."
Write-Host ""
Write-Host "  --- M15 -> reports/oos_test.json ---" -ForegroundColor Cyan
& $PY oos_test.py --tf M15 --from $FROM --cutoff $CUTOFF --scope $SCOPE --pick $PICK --top $TOP `
    --regime-detector $REGDET --confirm $CONFIRM
if ($LASTEXITCODE -ne 0) { throw "OOS M15 failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "  --- M5  -> reports/oos_test_M5.json ---" -ForegroundColor Cyan
& $PY oos_test.py --tf M5 --from $FROM --cutoff $CUTOFF --scope $SCOPE --pick $PICK --top $TOP `
    --regime-detector $REGDET --confirm $CONFIRM
if ($LASTEXITCODE -ne 0) { throw "OOS M5 failed (exit $LASTEXITCODE)" }

# ---- done -------------------------------------------------------------
Section "Done"
Write-Host "  Results:" -ForegroundColor Green
Write-Host "    reports/oos_test.txt     (M15: TRAIN vs TEST per rule/asset + VERDICT)"
Write-Host "    reports/oos_test.json"
Write-Host "    reports/oos_test_M5.txt  (M5: same, for the 5-minute bot)"
Write-Host "    reports/oos_test_M5.json"
Write-Host "    reports/oos_equity.png   (account curve on UNSEEN data)"
Write-Host "    reports/oos_trades.csv   (every test-period trade)"
Write-Host ""
Write-Host "  Then run:  $PY oos_analyze.py   (fixed-risk + regime split of the above)" -ForegroundColor Green
Write-Host "  Send reports/oos_test.txt back so we can read the verdict together." -ForegroundColor Green
