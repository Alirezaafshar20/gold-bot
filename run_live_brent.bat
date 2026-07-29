@echo off
title LIVE BRENT M15
cd /d "%~dp0"
python live_smc.py --symbol UKBRENT.Q26 --tf M15 --magic 778802
pause
