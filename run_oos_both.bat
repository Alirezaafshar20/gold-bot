@echo off
REM Run honest OOS for both M15 and M5 (XAUUSD). Use on the strong PC overnight.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ======================================================================
echo   OOS M15  -^> reports\oos_test.json
echo ======================================================================
"%PY%" oos_test.py --tf M15 --asset XAUUSD --scope dense --workers %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto :fail

echo.
echo ======================================================================
echo   OOS M5   -^> reports\oos_test_M5.json
echo ======================================================================
"%PY%" oos_test.py --tf M5 --asset XAUUSD --scope dense --workers %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto :fail

echo.
echo   Done. Live bots pick gates via floating_config.apply_tf_paths(tf).
echo   M15: reports\oos_test.json
echo   M5 : reports\oos_test_M5.json
pause
exit /b 0

:fail
echo   FAILED.
pause
exit /b 1
