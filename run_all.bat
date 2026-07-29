@echo off
REM ====================================================================
REM  One-click setup + Dukascopy download + long calibration (CMD)
REM  Just double-click this file, or run:  run_all.bat
REM  Safe to re-run: skips venv + already-downloaded data.
REM ====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "FROM=2022-04-01"
set "SCOPE=dense"
set "DATADIR=data\dukascopy"

echo ======================================================================
echo   1/7  Checking Python and Node.js
echo ======================================================================
python --version
if errorlevel 1 (
    echo   Python NOT found. Install from https://www.python.org/downloads/
    echo   ^(tick "Add Python to PATH"^) then re-run this file.
    pause & exit /b 1
)
node --version
if errorlevel 1 (
    echo   Node.js NOT found. Install LTS from https://nodejs.org then re-run.
    pause & exit /b 1
)

echo.
echo ======================================================================
echo   2/7  Python virtual environment + dependencies
echo ======================================================================
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists - reusing
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r requirements.txt
"%PY%" -m pip install --quiet MetaTrader5
echo   Dependencies installed (incl. MetaTrader5)

echo.
echo ======================================================================
echo   3/7  Downloading Dukascopy 1-minute data (month by month, robust)
echo ======================================================================
if not exist "%DATADIR%" mkdir "%DATADIR%"
"%PY%" duka_download.py --from 2022-04

echo.
echo ======================================================================
echo   4/7  Verifying Dukascopy data coverage
echo ======================================================================
"%PY%" dukascopy_loader.py

echo.
echo ======================================================================
echo   5/8  Long walk-forward calibration on Dukascopy (regime-aware)
echo ======================================================================
"%PY%" calibrate_long.py --from %FROM% --scope %SCOPE% --regime-detector kaufman --regime-tf H1

echo.
echo ======================================================================
echo   6/8  Cross-check on the live broker's recent data (MT5)
echo ======================================================================
echo   Make sure the MT5 terminal is OPEN and logged in.
"%PY%" calibrate_long.py --source mt5 --days 120 --scope %SCOPE% --regime-detector kaufman --regime-tf H1 --out reports/calibrate_broker.txt
if errorlevel 1 echo   Broker cross-check skipped (open MT5 and re-run if needed).

echo.
echo ======================================================================
echo   7/8  Building ranked regime-aware report (reports\rule_ranking.md)
echo ======================================================================
"%PY%" rank_rules.py

echo.
echo ======================================================================
echo   8/8  Done
echo ======================================================================
echo   Results:
echo     reports\calibrate_long.txt     (Dukascopy 2022 -^> now, main result)
echo     reports\calibrate_long.json
echo     reports\calibrate_broker.txt   (broker recent data cross-check)
echo     reports\rule_ranking.md        (ranked, regime-aware - read this one)
echo.
echo   Send these files back to review and pick the new rules.
pause
exit /b 0
