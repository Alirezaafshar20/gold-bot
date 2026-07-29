"""
Stop-loss timing analysis — portfolio 90d.

  python sl_stop_analysis.py
  python sl_stop_analysis.py --days 30
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import mt5_data as M
import meta_gate as MG
import portfolio_config as C
import symbol_profiles as P
import floating_config as FC
import weekly_adaptive as WA
from multi_symbol_calibrate import _run
from portfolio_backtest import simulate_portfolio, RISK_MAP

WEEKDAY_FA = {
    "Monday": "دوشنبه",
    "Tuesday": "سه‌شنبه",
    "Wednesday": "چهارشنبه",
    "Thursday": "پنج‌شنبه",
    "Friday": "جمعه",
    "Saturday": "شنبه",
    "Sunday": "یکشنبه",
}

HOUR_BANDS = (
    (0, 6, "00–06 شب/صبح"),
    (6, 9, "06–09 صبح"),
    (9, 13, "09–13"),
    (13, 17, "13–17"),
    (17, 21, "17–21"),
    (21, 24, "21–24"),
)


def hour_band(h: int) -> str:
    for lo, hi, label in HOUR_BANDS:
        if lo <= h < hi:
            return label
    return "?"


def load_trades(days: int) -> list[dict]:
    assets = list(C.PORTFOLIO_ASSETS)
    oos_mg = MG.load_meta_gate(
        FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
        assets=assets,
        min_regime_n=FC.META.get("meta_min_regime_n", 3),
        require_positive_r=FC.META.get("meta_require_positive_r", True),
        exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
    )
    mt5 = M.connect()
    all_trades = []
    try:
        for asset in assets:
            _, prof = P.get_profile(asset)
            sym = P.resolve_symbol_for_profile(asset, mt5)
            risk = RISK_MAP[asset]
            stats = _run(sym, prof, mt5, days=days, asset_key=asset, risk_pct=risk,
                         max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
                         meta_gate_override=oos_mg)
            for t in stats.get("trades") or []:
                tc = dict(t)
                tc["_asset"] = asset
                tc["_risk_pct"] = risk
                all_trades.append(tc)
        all_trades.sort(key=lambda t: (t["time"], t.get("exit_time")))
        if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
            all_trades, _ = WA.apply_walkforward_filter(
                all_trades, mt5=mt5, assets=tuple(assets))
    finally:
        M.shutdown(mt5)
    live = simulate_portfolio(all_trades, 1000.0, size_compound=False)
    return live.get("ledger") or []


def ledger_to_df(ledger: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(ledger)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df.get("exit_time", df["entry_time"]))
    df["is_sl"] = (df["exit"].astype(str).str.lower() == "sl") | (df["R"] <= -0.95)
    df["weekday"] = df["entry_time"].dt.day_name()
    df["weekday_fa"] = df["weekday"].map(WEEKDAY_FA)
    df["hour"] = df["entry_time"].dt.hour
    df["hour_band"] = df["hour"].map(hour_band)
    df["weekday_num"] = df["entry_time"].dt.dayofweek  # Mon=0 .. Sun=6
    return df


def print_table(title, series, total_sl, sort_col="count"):
    print(f"\n  {title}")
    print(f"  {'':20} {'SL':>5}  {'% of SL':>8}  {'% all trades':>12}")
    print("  " + "-" * 50)
    for idx, row in series.iterrows():
        pct_sl = row["count"] / total_sl * 100 if total_sl else 0
        pct_all = row.get("pct_all", 0)
        print(f"  {str(idx):<20} {int(row['count']):>5}  {pct_sl:>7.1f}%  {pct_all:>11.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--csv", default=None, help="Use existing CSV instead of backtest")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        df = df.rename(columns={"side": "dir", "pnl_usd": "net", "exit_reason": "exit"})
        ledger = df.to_dict("records")
        df = ledger_to_df(ledger)
    else:
        print(f"Running portfolio backtest {args.days}d (walk-forward)...")
        ledger = load_trades(args.days)
        df = ledger_to_df(ledger)

    total = len(df)
    sl = df[df["is_sl"]]
    n_sl = len(sl)
    n_all = total

    print("=" * 72)
    print(f"  STOP-LOSS TIMING  |  {args.days}d  |  portfolio {', '.join(C.PORTFOLIO_ASSETS)}")
    print(f"  Total trades: {n_all}  |  Stop-loss: {n_sl}  ({n_sl/n_all*100:.1f}% of all)")
    print("=" * 72)

    # By weekday (entry time)
    wd = sl.groupby("weekday_fa").agg(count=("R", "count")).sort_values("count", ascending=False)
    wd_all = df.groupby("weekday_fa").size()
    wd["pct_all"] = (wd["count"] / wd_all.reindex(wd.index).fillna(1) * 100).round(1)
    print_table("روز هفته (زمان ورود) — بیشترین SL", wd, n_sl)

    best_wd = wd.index[0] if len(wd) else "?"
    sat_n = int(wd.loc["شنبه", "count"]) if "شنبه" in wd.index else 0
    print(f"\n  → بیشترین SL: **{best_wd}** ({int(wd.iloc[0]['count'])} استاپ)")
    print(f"  → شنبه: {sat_n} SL ({sat_n/n_sl*100:.1f}% از کل استاپ‌ها)" if n_sl else "")

    # SL rate by weekday (quality view)
    print("\n  نرخ استاپ به ازای هر روز (SL / کل معاملات همان روز):")
    rate = df.groupby("weekday_fa").agg(
        trades=("R", "count"), sl=("is_sl", "sum"))
    rate["sl_rate"] = (rate["sl"] / rate["trades"] * 100).round(1)
    rate = rate.sort_values("sl_rate", ascending=False)
    for day, row in rate.iterrows():
        print(f"    {day:<12} {int(row['sl']):>3}/{int(row['trades']):>3}  →  {row['sl_rate']:.0f}% SL rate")

    # Hour bands
    hb = sl.groupby("hour_band").agg(count=("R", "count")).sort_values("count", ascending=False)
    hb_all = df.groupby("hour_band").size()
    hb["pct_all"] = (hb["count"] / hb_all.reindex(hb.index).fillna(1) * 100).round(1)
    print_table("بازه ساعتی (زمان ورود)", hb, n_sl)

    # Top hours
    hr = sl.groupby("hour").agg(count=("R", "count")).sort_values("count", ascending=False)
    print("\n  ساعات با بیشترین SL (ورود):")
    for h, row in hr.head(8).iterrows():
        flag = " *" if h < 9 or h >= 21 else ""
        print(f"    {int(h):02d}:00  {int(row['count']):>3} SL{flag}")
    print("  (* = خارج سشن 9–21 برای بیت؛ طلا 24/7)")

    # By asset
    print("\n  به تفکیک دارایی:")
    for asset in C.PORTFOLIO_ASSETS:
        sub = df[df["asset"] == asset]
        sl_a = sub[sub["is_sl"]]
        print(f"    {asset}: {len(sl_a)}/{len(sub)} SL ({len(sl_a)/len(sub)*100:.0f}% rate)" if len(sub) else f"    {asset}: —")

    # Saturday detail
    if sat_n:
        sat = sl[sl["weekday_fa"] == "شنبه"]
        print(f"\n  جزئیات شنبه ({sat_n} SL):")
        by_h = sat.groupby("hour").size().sort_values(ascending=False)
        for h, c in by_h.head(5).items():
            print(f"    {int(h):02d}:00  {int(c)} SL")
        by_a = sat.groupby("asset").size()
        for a, c in by_a.items():
            print(f"    {a}: {int(c)} SL")

    print("=" * 72)


if __name__ == "__main__":
    main()
