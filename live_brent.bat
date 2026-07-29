@echo off
REM LIVE BRENT OIL - M15  (keep this window open; different magic number)
cd /d "%~dp0"
python live_smc.py --symbol UKBRENT.Q26 --tf M15 --magic 778902
pause
