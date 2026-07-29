# Run honest OOS for both M15 and M5 (XAUUSD). Use on the strong PC overnight.
# Usage:
#   .\run_oos_both.ps1
#   .\run_oos_both.ps1 -Workers 12
#   .\run_oos_both.ps1 -Asset XAUUSD -Scope dense

param(
    [string]$Asset = "XAUUSD",
    [string]$Scope = "dense",
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if ($Workers -le 0) {
    $Workers = [Math]::Max(1, ((Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors) - 1)
}

Write-Host "======================================================================"
Write-Host "  OOS M15  -> reports\oos_test.json"
Write-Host "  python=$py  workers=$Workers  asset=$Asset  scope=$Scope"
Write-Host "======================================================================"
& $py oos_test.py --tf M15 --asset $Asset --scope $Scope --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "OOS M15 failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "======================================================================"
Write-Host "  OOS M5   -> reports\oos_test_M5.json"
Write-Host "======================================================================"
& $py oos_test.py --tf M5 --asset $Asset --scope $Scope --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "OOS M5 failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "  Done. Live bots pick gates via floating_config.apply_tf_paths(tf)."
Write-Host "  M15: reports\oos_test.json"
Write-Host "  M5 : reports\oos_test_M5.json"
