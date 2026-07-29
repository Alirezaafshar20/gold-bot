# =====================================================================
#  Test ONLY the two NEW edge families, out-of-sample.
#  Safe to run WHILE run_all.ps1 is still going: it does NOT download
#  anything - it just reads the Dukascopy data already on disk and
#  writes to its OWN report files (won't touch run_all's outputs).
#
#      powershell -ExecutionPolicy Bypass -File .\test_new_edges.ps1
#
#  New edges being judged on unseen data:
#    Regression channel  : CH_REV_L CH_REV_S CH_BO_L CH_BO_S
#    Prior day/week levels: PDL_SW_L PDH_SW_S PWL_SW_L PWH_SW_S PDC_RC_L PDC_RC_S
# =====================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# ---- config -----------------------------------------------------------
$FROM   = "2022-04-01"
$CUTOFF = "2024-12-31"     # TRAIN = before this, TEST = unseen data after this
$RULES  = "CH_REV_L,CH_REV_S,CH_BO_L,CH_BO_S,PDL_SW_L,PDH_SW_S,PWL_SW_L,PWH_SW_S,PDC_RC_L,PDC_RC_S"

# use the venv python if it exists, otherwise system python
if (Test-Path ".\.venv\Scripts\python.exe") { $PY = ".\.venv\Scripts\python.exe" }
else { $PY = "python" }

# fewer workers so it doesn't starve the run_all job that's already running
$Cores = [Environment]::ProcessorCount
$Workers = [Math]::Max(1, [Math]::Floor($Cores / 2))

Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  OUT-OF-SAMPLE test of the NEW edges only" -ForegroundColor Cyan
Write-Host "  (reads existing data, no download; separate report files)" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  python : $PY"
Write-Host "  workers: $Workers  (of $Cores cores - leaves room for run_all)"
Write-Host "  rules  : $RULES"
Write-Host ""

& $PY oos_test.py `
    --from $FROM `
    --cutoff $CUTOFF `
    --scope full `
    --rules $RULES `
    --top 10 `
    --workers $Workers `
    --out   reports/oos_new_edges.txt `
    --chart reports/oos_new_edges.png `
    --csv   reports/oos_new_edges.csv

Write-Host ""
Write-Host "  Done. Read: reports/oos_new_edges.txt" -ForegroundColor Green
Write-Host "  (chart: reports/oos_new_edges.png, trades: reports/oos_new_edges.csv)" -ForegroundColor Green
