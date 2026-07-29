@echo off
REM Live portfolio — ONE process, all symbols, slice sizing, max 2 trades/symbol
REM Test first:
REM   python live_portfolio.py --dry-run
REM Live:
REM   python live_portfolio.py

cd /d "%~dp0"
python live_portfolio.py %*
