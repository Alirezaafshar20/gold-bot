"""What does the spread on this account actually cost?

Every backtest number in this project is net of a spread assumption, and that
assumption is currently taken from one instantaneous quote: symbol_profiles.
live_spread returns max(profile_spread, ask - bid) at the moment the script
starts, and live_replay then applies that single number to every fill across the
whole historical window. Start a replay while the market is thin and the entire
backtest pays a thin-market spread.

This probe replaces the guess with the distribution, from the broker's own tick
history, split by trading session — because an average is the wrong summary when
the book does not trade uniformly through the day.

Usage: python spread_probe.py [days]
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import MetaTrader5 as mt5_mod
import mt5_data as M
import strategy as S
import symbol_profiles as P

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ASSET = "XAUUSD"

mt5 = M.connect()
sym = P.resolve_symbol_for_profile(ASSET, mt5)
_, prof = P.get_profile(ASSET)

print("=" * 84)
print(f"  SPREAD PROBE — {sym}   last {DAYS} days of broker ticks")
print("=" * 84)

info = mt5.symbol_info(sym)
tick = mt5.symbol_info_tick(sym)
print()
print(f"  broker              : {mt5.account_info().company if mt5.account_info() else '?'}")
print(f"  account             : {mt5.account_info().login if mt5.account_info() else '?'}"
      f"   {mt5.account_info().server if mt5.account_info() else ''}")
print(f"  point size          : {info.point}")
print(f"  digits              : {info.digits}")
print(f"  spread now (points) : {info.spread}"
      f"   -> {info.spread * info.point:.3f} USD")
print(f"  spread floating     : {'yes' if info.spread_float else 'NO (fixed)'}")
if tick:
    print(f"  live tick           : bid {tick.bid}  ask {tick.ask}  "
          f"-> {tick.ask - tick.bid:.3f} USD")
print()
print(f"  what the code assumes")
print(f"    strategy.SPREAD_USD        : {S.SPREAD_USD:.3f} USD")
print(f"    profile['spread']          : {prof['spread']:.3f} USD")
print(f"    profile['spread_fixed']    : {prof.get('spread_fixed')}")
print(f"    live_spread() returns now  : {P.live_spread(sym, prof, mt5):.3f} USD"
      f"   <- what a replay started now would use")

to = datetime.now(timezone.utc)
frm = to - timedelta(days=DAYS)
ticks = mt5.copy_ticks_range(sym, frm, to, mt5_mod.COPY_TICKS_INFO)
if ticks is None or len(ticks) == 0:
    print()
    print(f"  no tick history returned ({mt5.last_error()}). "
          f"Cannot measure the distribution.")
    M.shutdown(mt5)
    raise SystemExit

df = pd.DataFrame(ticks)
df["t"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
df = df[(df["bid"] > 0) & (df["ask"] > 0)]
df["spread"] = df["ask"] - df["bid"]
# broker server time is what the session boundaries below refer to
off = M.broker_utc_offset_hours(mt5) if hasattr(M, "broker_utc_offset_hours") else 0.0
df["hour_utc"] = df["t"].dt.hour

print()
print(f"  {len(df):,} ticks   {df['t'].min()} .. {df['t'].max()}")
print()
q = df["spread"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print("  spread distribution (USD)")
print(f"    median            : {q[0.5]:.3f}")
print(f"    25th / 75th pct   : {q[0.25]:.3f}  /  {q[0.75]:.3f}")
print(f"    5th / 95th pct    : {q[0.05]:.3f}  /  {q[0.95]:.3f}")
print(f"    99th pct          : {q[0.99]:.3f}")
print(f"    mean              : {df['spread'].mean():.3f}")

# The book trades London and New York. Asian-session and rollover spreads are
# real but the engine rarely fills there, so a flat average overstates cost.
bands = [("Asia      00-06 UTC", range(0, 7)),
         ("London    07-12 UTC", range(7, 13)),
         ("NY overlap 13-17 UTC", range(13, 18)),
         ("NY late   18-20 UTC", range(18, 21)),
         ("rollover  21-23 UTC", range(21, 24))]
print()
print(f"  {'session':<22} {'ticks':>9} {'median':>8} {'mean':>8} {'95th':>8}")
print("  " + "-" * 60)
for name, hrs in bands:
    s = df[df["hour_utc"].isin(list(hrs))]["spread"]
    if len(s) == 0:
        continue
    print(f"  {name:<22} {len(s):>9,} {s.median():>8.3f} {s.mean():>8.3f} "
          f"{s.quantile(0.95):>8.3f}")

print()
print("  -- what this does to the cost figure --")
MEAN_SL_USD = 21.32     # measured over the 182-trade sample, both sides pooled
N = 182
for label, sp in (("replay assumption (0.68)", 0.68),
                  ("code constant (strategy.SPREAD_USD)", S.SPREAD_USD),
                  ("measured median", float(q[0.5])),
                  ("measured London+NY median",
                   float(df[df["hour_utc"].isin(list(range(7, 18)))]["spread"].median()))):
    cr = sp / MEAN_SL_USD
    print(f"  {label:<38} {sp:>6.3f} USD  ->  {cr:.4f} R/trade  "
          f"{cr*N:>6.2f} R over {N} trades")

print()
print("  A backtest that assumed a wider spread than the account really pays is")
print("  pessimistic, not optimistic: fills are harder to get and each one is")
print("  booked at a worse price. The measured column is the one to trust.")

M.shutdown(mt5)
