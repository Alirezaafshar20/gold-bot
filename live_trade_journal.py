"""
Append closed live trades to CSV for weekly_adaptive rolling gate.

Uses MT5 deal history when a position ticket disappears.
"""
from __future__ import annotations

import csv
import datetime as dt
import os

import floating_config as FC

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

FIELDS = (
    "n", "entry_time", "exit_time", "asset", "rule", "side", "entry", "exit",
    "R", "tp_r", "want_risk_pct", "use_risk_pct", "risk_usd",
    "pnl_usd", "balance_after", "exit_reason",
)


def journal_path() -> str:
    return FC.WEEKLY.get("trades_csv", "reports/portfolio_trades.csv")


def _next_row_num(path: str) -> int:
    if not os.path.isfile(path):
        return 1
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) + 1


def append_trade(row: dict, path: str | None = None) -> str:
    path = path or journal_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(FIELDS)
        w.writerow([
            row.get("n", _next_row_num(path)),
            row["entry_time"], row["exit_time"], row["asset"], row["rule"],
            row["side"], row["entry"], row["exit"], row["R"],
            row.get("tp_r", ""), row.get("want_risk_pct", ""), row.get("use_risk_pct", ""),
            row["risk_usd"], row["pnl_usd"], row.get("balance_after", ""),
            row.get("exit_reason", "unknown"),
        ])
    return path


def _deal_exit_reason(deal) -> str:
    if mt5 is None:
        return "unknown"
    reason = getattr(deal, "reason", None)
    if reason == mt5.DEAL_REASON_SL:
        return "sl"
    if reason == mt5.DEAL_REASON_TP:
        return "tp"
    if reason == mt5.DEAL_REASON_SO:
        return "so"
    return "time"


def fetch_exit_deal(ticket: int, symbol: str, magic: int, since: dt.datetime):
    if mt5 is None:
        return None
    deals = mt5.history_deals_get(since, dt.datetime.now() + dt.timedelta(minutes=5))
    if not deals:
        return None
    outs = [
        d for d in deals
        if d.symbol == symbol and d.magic == magic
        and d.entry == mt5.DEAL_ENTRY_OUT
        and (d.position_id == ticket or d.order == ticket)
    ]
    if not outs:
        return None
    return max(outs, key=lambda d: d.time)


def log_closed_position(
    st,
    ticket: int,
    pos_state: dict,
    balance: float,
    since: dt.datetime | None = None,
) -> dict | None:
    """Build journal row from pos_state + MT5 exit deal."""
    since = since or (dt.datetime.now() - dt.timedelta(days=14))
    deal = fetch_exit_deal(ticket, st.sym, st.magic, since)
    if deal is None:
        return None
    entry = float(pos_state.get("entry", deal.price))
    risk = max(float(pos_state.get("risk", 1e-6)), 1e-6)
    direction = pos_state.get("dir", "long")
    if direction == "long":
        R = (float(deal.price) - entry) / risk
    else:
        R = (entry - float(deal.price)) / risk
    risk_usd = balance * st.risk_pct
    pnl = R * risk_usd
    rule = str(pos_state.get("tag", "?")).replace("SMC-", "")
    row = {
        "entry_time": pos_state.get("entry_time", since).strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(pos_state.get("entry_time", since), "strftime")
        else str(pos_state.get("entry_time", since))[:19],
        "exit_time": dt.datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
        "asset": st.key,
        "rule": rule,
        "side": direction,
        "entry": round(entry, 5),
        "exit": round(float(deal.price), 5),
        "R": round(R, 4),
        "risk_usd": round(risk_usd, 2),
        "pnl_usd": round(pnl, 2),
        "balance_after": round(balance + pnl, 2),
        "exit_reason": _deal_exit_reason(deal),
        "want_risk_pct": st.risk_pct,
        "use_risk_pct": st.risk_pct,
    }
    append_trade(row)
    return row
