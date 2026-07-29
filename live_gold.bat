@echo off
REM LIVE GOLD - M5  (keep this window open)
cd /d "%~dp0"
python live_smc.py --symbol XAUUSD@ --tf M5 --magic 778901
pause
