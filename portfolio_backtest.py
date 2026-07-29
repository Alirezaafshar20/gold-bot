"""
Portfolio backtest — ONE account, compound on current balance (same as live).

  risk$ at entry = balance_at_that_moment × RISK_MAP[symbol]
  P&L on exit updates balance → next trade uses new balance

Usage:
  python portfolio_backtest.py
  python portfolio_backtest.py --no-cap
  python portfolio_backtest.py --days 90 --chart reports/portfolio_equity.png
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import strategy as S
import mt5_data as M
import symbol_profiles as P
import symbol_specs as X
from multi_symbol_calibrate import _run, SEP, THIN
import portfolio_config as C
import floating_config as FC
import weekly_adaptive as WA
import meta_gate as MG

DEFAULT_BAL = 1000.0
DEFAULT_DAYS = 90
MIN_EFFECTIVE_RISK = 0.005
RISK_MAP = C.RISK_MAP


def _build_regime_map(sym, prof, mt5, days):
    opt = P.preset_for_profile(prof)
    opt = P.apply_profile_to_settings(opt, prof, P.live_spread(sym, prof, mt5))
    if not opt.get("regime_gate"):
        return None
    htf_tfs = list(opt.get("htf_tfs", ("H4", "H1")))
    rtf = opt.get("regime_tf", "H1")
    if rtf not in htf_tfs:
        htf_tfs.append(rtf)
    _, htf_dfs = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=tuple(htf_tfs))
    htf = S.prepare_htf_context(htf_dfs)
    det = opt.get("regime_detector", "hybrid")
    if det == "mtf" and "H4" not in htf_tfs:
        _, htf_dfs2 = M.fetch_htf_bars(sym, days=days, mt5=mt5, tfs=("H4", "H1"))
        htf = S.prepare_htf_context(htf_dfs2)
    return S.build_regime_map(
        htf, tf=rtf,
        detector=det,
        win=opt.get("regime_win", 40),
        er_trend=opt.get("regime_er_trend", 0.35),
        slope_k=opt.get("regime_slope_k", 0.0006),
        er_hi=opt.get("regime_er_hi", 0.40),
        er_lo=opt.get("regime_er_lo", 0.25),
        confirm=opt.get("regime_confirm", 2),
        h4_win=opt.get("regime_h4_win", 30),
        h4_lookback=opt.get("regime_h4_lookback", 60),
        structure_break=opt.get("regime_structure_break", False),
        brk_win=opt.get("regime_brk_win", 40),
        shock_win=opt.get("regime_shock_win", 24),
        shock_baseline=opt.get("regime_shock_baseline", 720),
    )


def prepare_portfolio_trades(all_trades, rmaps, meta_gate):
    """Tag golden/base tier + drop CH-REV outside TREND_UP."""
    out = []
    for t in all_trades:
        asset = t["_asset"]
        rmap = rmaps.get(asset)
        regime = rmap.at(t["time"]) if rmap is not None else "RANGE"
        rule = str(t.get("rule", "")).upper()
        macro = rmap.macro_at(t["time"]) if rmap is not None and hasattr(rmap, "macro_at") else 0
        direction = t.get("dir", "long")
        if not C.rule_allowed(asset, rule, regime, direction=direction, macro=macro):
            continue
        tc = dict(t)
        tc["_regime"] = regime
        tc["_macro"] = macro
        tc["_golden"] = C.is_golden_signal(asset, rule, regime, meta_gate)
        out.append(tc)
    return out


def fetch_portfolio_trades(days, mt5, oos_mg, tol_override=None, tf="M15"):
    """Load walk-forward trade list + regime maps for comparisons."""
    assets = list(C.PORTFOLIO_ASSETS)
    all_trades = []
    rmaps = {}
    for asset in assets:
        _, prof = P.get_profile(asset)
        sym = P.resolve_symbol_for_profile(asset, mt5)
        risk = RISK_MAP[asset]
        rmaps[asset] = _build_regime_map(sym, prof, mt5, days)
        stats = _run(
            sym, prof, mt5, days=days, asset_key=asset, risk_pct=risk,
            max_concurrent=C.max_concurrent(asset),
            tol_override=tol_override,
            meta_gate_override=oos_mg,
            tf=tf,
        )
        spec = X.get_spec(sym, mt5)
        for t in stats.get("trades") or []:
            tc = dict(t)
            tc["_asset"] = asset
            tc["_risk_pct"] = risk
            tc["_spread"] = stats["spread"]
            tc["_max_concurrent"] = C.max_concurrent(asset)
            tc["_spec"] = spec
            all_trades.append(tc)
    all_trades.sort(key=lambda t: (t["time"], t.get("exit_time")))
    wf = {"before": len(all_trades), "after": len(all_trades)}
    if FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True):
        all_trades, wf = WA.apply_walkforward_filter(
            all_trades, mt5=mt5, assets=tuple(assets))
    return all_trades, wf, rmaps


def simulate_portfolio(
    trades,
    start_balance,
    enforce_mc=True,
    max_portfolio_risk_pct=C.MAX_PORTFOLIO_RISK,
    scale_to_cap=True,
    max_concurrent=C.MAX_CONCURRENT_PER_ASSET,
    size_compound=True,
    fixed_lot=None,
):
    """One balance pool; risk% × balance at each entry (matches live_portfolio.py).

    size_compound=False sizes every trade off the STARTING balance (fixed risk$),
    removing the compounding snowball — a more honest expectancy view.

    fixed_lot: if set (e.g. 0.10), every trade uses that lot size; P&L comes from
    broker $/unit × lot × price move (via R × SL). Risk% cap is skipped.
    """
    if not trades:
        return _empty_result(start_balance)
    fixed_lot = float(fixed_lot) if fixed_lot is not None else None
    if fixed_lot is not None and fixed_lot <= 0:
        raise ValueError("fixed_lot must be > 0")

    ev = []
    for k, t in enumerate(trades):
        ev.append((t["exit_time"], 0, k))
        ev.append((t["time"], 1, k))
    # At equal timestamps, process ENTRIES (kind 1) before EXITS (kind 0) so that a
    # zero-duration trade (exit_time == entry_time) opens before it closes. Otherwise
    # its exit is skipped (position not yet open) and the slot leaks forever, which
    # starves max_concurrent and silently drops every later trade for that asset.
    ev.sort(key=lambda e: (e[0], -e[1]))

    balance = start_balance
    peak = start_balance
    max_dd = 0.0
    min_balance = start_balance
    min_balance_time = None
    sizes = {}
    riskd = {}
    eff_pct = {}
    taken = set()
    skipped_mc = set()
    skipped_cap = set()
    scaled = 0
    open_count_by_asset = {}
    open_base_by_asset = {}
    open_premium_by_asset = {}
    trade_tier = {}
    open_count = 0
    open_risk_usd = 0.0

    n_taken = n_wins = 0
    gp = gl = 0.0
    monthly = {}
    by_asset = {}
    ledger = []
    max_open = 0
    max_risk_open_pct = 0.0

    equity_curve = []
    first_ts = None

    for ts, kind, k in ev:
        t = trades[k]
        asset = t["_asset"]
        rp = t.get("risk", 0)

        if first_ts is None:
            first_ts = ts

        if kind == 1:
            if k in taken or k in skipped_mc or k in skipped_cap:
                continue
            mc = t.get("_max_concurrent", max_concurrent)
            tiers = C.tiered(asset)
            if enforce_mc and tiers:
                golden = bool(t.get("_golden", False))
                nb = open_base_by_asset.get(asset, 0)
                np = open_premium_by_asset.get(asset, 0)
                if not C.slot_available(asset, nb, np, golden):
                    skipped_mc.add(k)
                    continue
            elif enforce_mc and open_count_by_asset.get(asset, 0) >= mc:
                skipped_mc.add(k)
                continue
            if rp <= 0:
                skipped_mc.add(k)
                continue

            want = t.get("_risk_pct", 0.0) or 0.0
            use = want
            spec = t.get("_spec")

            if fixed_lot is not None:
                # Fixed lot: no risk% scaling, no portfolio risk cap.
                lot = fixed_lot
                if spec is not None:
                    step = spec.get("vol_step") or 0.01
                    vmin = spec.get("vol_min") or 0.01
                    vmax = spec.get("vol_max") or 100.0
                    lot = max(vmin, min(vmax, round(lot / step) * step))
                    rd = lot * X.loss_per_lot_usd(spec, rp)
                    if rd <= 0:
                        upu = spec.get("usd_per_unit") or spec.get("contract_size") or 100.0
                        rd = lot * upu * rp
                    sizes[k] = rd / rp
                else:
                    # No spec: assume $100 / price-unit / lot (XAU-like CFD).
                    sizes[k] = lot * 100.0
                    rd = sizes[k] * rp
                riskd[k] = rd
                eff_pct[k] = (rd / balance) if balance > 0 else 0.0
            else:
                if max_portfolio_risk_pct is not None and balance > 0:
                    projected = open_risk_usd + use * balance
                    if projected / balance > max_portfolio_risk_pct:
                        room_usd = max_portfolio_risk_pct * balance - open_risk_usd
                        if room_usd <= balance * MIN_EFFECTIVE_RISK:
                            skipped_cap.add(k)
                            continue
                        if scale_to_cap:
                            use = room_usd / balance
                            if use < MIN_EFFECTIVE_RISK:
                                skipped_cap.add(k)
                                continue
                            scaled += 1
                        else:
                            skipped_cap.add(k)
                            continue

                # Real lot from broker spec (respects vol_min/step/max) — same as live.
                size_bal = balance if size_compound else start_balance
                if spec is not None:
                    lot = X.calc_lot_from_spec(spec, size_bal, use, rp)
                    rd = lot * X.loss_per_lot_usd(spec, rp)
                    if rd <= 0:
                        rd = use * size_bal
                        sizes[k] = rd / rp
                    else:
                        sizes[k] = rd / rp  # $ per price unit (spread cost basis)
                else:
                    rd = use * size_bal
                    sizes[k] = rd / rp
                riskd[k] = rd
                eff_pct[k] = rd / size_bal if size_bal > 0 else use
            taken.add(k)
            if tiers:
                golden = bool(t.get("_golden", False))
                trade_tier[k] = "golden" if golden else "base"
                if golden:
                    open_premium_by_asset[asset] = open_premium_by_asset.get(asset, 0) + 1
                else:
                    open_base_by_asset[asset] = open_base_by_asset.get(asset, 0) + 1
            equity_curve.append({"time": ts, "balance": balance, "equity": balance})
            open_count_by_asset[asset] = open_count_by_asset.get(asset, 0) + 1
            open_count += 1
            open_risk_usd += rd
            max_open = max(max_open, open_count)
            if balance > 0:
                max_risk_open_pct = max(max_risk_open_pct, open_risk_usd / balance)

        else:
            if k not in taken:
                continue
            spread = t.get("_spread", 0.0)
            net = t["R"] * riskd[k] - sizes[k] * spread
            balance += net
            n_taken += 1
            if net > 0:
                n_wins += 1
                gp += net
            else:
                gl += -net

            ba = by_asset.setdefault(asset, {"n": 0, "net": 0.0, "wins": 0, "scaled": 0})
            ba["n"] += 1
            ba["net"] += net
            if eff_pct[k] < t["_risk_pct"] - 1e-9:
                ba["scaled"] += 1
            if net > 0:
                ba["wins"] += 1

            mkey = f"{ts.year}-{ts.month:02d}"
            monthly[mkey] = monthly.get(mkey, 0.0) + net
            peak = max(peak, balance)
            if peak > 0:
                max_dd = max(max_dd, (peak - balance) / peak)
            # MT5-style Absolute Drawdown tracks the lowest closed balance.
            if balance < min_balance:
                min_balance = balance
                min_balance_time = ts

            direction = t.get("dir", "?")
            entry = t.get("entry", 0.0)
            risk_px = t.get("risk", 0.0)
            if direction == "long":
                exit_px = entry + t["R"] * risk_px
            else:
                exit_px = entry - t["R"] * risk_px

            ledger.append({
                "asset": asset,
                "entry_time": t["time"],
                "exit_time": t["exit_time"],
                "rule": t.get("setup", "?"),
                "dir": direction,
                "entry": entry,
                "exit_px": exit_px,
                "R": t["R"],
                "tp_r": t.get("tp_r"),
                "want_pct": t["_risk_pct"],
                "use_pct": eff_pct[k],
                "risk_usd": riskd[k],
                "net": net,
                "balance": balance,
                "exit": t.get("exit_reason", "?"),
                "fill_kind": t.get("fill_kind", "?"),
                "entry_tol_atr": t.get("entry_tol_atr"),
                "entry_tol_usd": t.get("entry_tol_usd"),
                "fill_gap": t.get("fill_gap"),
                "entry_confirm_mode": t.get("entry_confirm_mode"),
            })
            equity_curve.append({"time": ts, "balance": balance, "equity": balance})

            open_risk_usd -= riskd[k]
            tier = trade_tier.pop(k, None)
            if tier == "golden":
                open_premium_by_asset[asset] = max(0, open_premium_by_asset.get(asset, 1) - 1)
            elif tier == "base":
                open_base_by_asset[asset] = max(0, open_base_by_asset.get(asset, 1) - 1)
            open_count_by_asset[asset] = max(0, open_count_by_asset.get(asset, 1) - 1)
            open_count -= 1
            del eff_pct[k]
            del sizes[k]
            del riskd[k]

    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    if first_ts is not None:
        equity_curve.insert(0, {"time": first_ts, "balance": start_balance, "equity": start_balance})
    abs_dd = max(0.0, start_balance - min_balance)
    return {
        "final": balance,
        "ret_pct": (balance / start_balance - 1) * 100,
        "profit": balance - start_balance,
        "n": n_taken,
        "scaled": scaled,
        "skipped": len(skipped_mc) + len(skipped_cap),
        "skipped_mc": len(skipped_mc),
        "skipped_cap": len(skipped_cap),
        "wr": (n_wins / n_taken * 100 if n_taken else 0),
        "pf": pf,
        "max_dd": max_dd * 100,  # relative peak→trough % (MT5 Relative DD)
        "min_balance": min_balance,
        "min_balance_time": min_balance_time,
        "abs_dd": abs_dd,  # MT5 Absolute Drawdown ($) = deposit - lowest balance
        "abs_dd_pct": (abs_dd / start_balance * 100) if start_balance > 0 else 0.0,
        "monthly": monthly,
        "by_asset": by_asset,
        "ledger": ledger,
        "equity_curve": equity_curve,
        "max_open": max_open,
        "max_risk_open_pct": max_risk_open_pct * 100,
    }


def _empty_result(start_balance):
    return dict(
        final=start_balance, ret_pct=0.0, profit=0.0, n=0, scaled=0, skipped=0,
        skipped_mc=0, skipped_cap=0, wr=0.0, pf=0.0, max_dd=0.0,
        min_balance=start_balance, min_balance_time=None,
        abs_dd=0.0, abs_dd_pct=0.0,
        monthly={}, by_asset={}, ledger=[], equity_curve=[], max_open=0, max_risk_open_pct=0.0,
    )


def _pd_ts(ts):
    return str(ts)[:16]


def summarize_by_asset(ledger, assets, start_balance=1000.0):
    """Per-asset stats from portfolio ledger (shared account, sequential PnL attribution)."""
    out = {}
    for asset in assets:
        rows = [r for r in ledger if r.get("asset") == asset]
        if not rows:
            out[asset] = None
            continue
        nets = [float(r["net"]) for r in rows]
        rs = [float(r["R"]) for r in rows]
        n = len(rows)
        wins = sum(1 for x in nets if x > 0)
        gp = sum(x for x in nets if x > 0)
        gl = sum(-x for x in nets if x < 0)
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        sl = sum(
            1 for r in rows
            if str(r.get("exit", "")).lower() == "sl" or float(r["R"]) <= -0.95
        )
        # Attribution equity: start_balance + only this symbol's realized PnL (in exit order).
        # Avoids >100% DD when cumulative PnL briefly dips below $0 (old formula bug).
        eq = peak = float(start_balance)
        max_dd = 0.0
        for r in sorted(rows, key=lambda x: x["exit_time"]):
            eq += float(r["net"])
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)
        out[asset] = {
            "n": n,
            "wins": wins,
            "wr": wins / n * 100,
            "pf": pf,
            "net": sum(nets),
            "tot_r": sum(rs),
            "avg_r": sum(rs) / n,
            "max_dd": max_dd * 100,
            "sl": sl,
        }
    return out


def print_per_asset_breakdown(ledger, assets, risk_map, rows, start_balance=1000.0):
    """Print Gold / BTC (etc.) stats side-by-side from shared portfolio ledger."""
    stats = summarize_by_asset(ledger, assets, start_balance=start_balance)
    labels = {}
    for row in rows:
        if not row.get("error"):
            labels[row["asset"]] = row.get("label", row["asset"])

    print(f"\n  PER-ASSET BREAKDOWN (shared account — PnL & DD attributed per symbol)")
    print(f"  {'Symbol':<12} {'Risk':>5}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  "
          f"{'MaxDD%':>7}  {'SL':>4}  {'totR':>7}  {'avgR':>6}  {'Net $':>11}")
    print("  " + THIN)
    for asset in assets:
        s = stats.get(asset)
        label = labels.get(asset, asset)
        risk = risk_map.get(asset, 0) * 100
        if s is None:
            print(f"  {label:<12} {risk:>4.0f}%  {'—':>6}  {'—':>6}  {'—':>6}  "
                  f"{'—':>7}  {'—':>4}  {'—':>7}  {'—':>6}  {'—':>11}")
            continue
        print(f"  {label:<12} {risk:>4.0f}%  {s['n']:>6}  {s['wr']:>5.1f}%  {s['pf']:>6.2f}  "
              f"{s['max_dd']:>6.1f}%  {s['sl']:>4}  {s['tot_r']:>+7.1f}  {s['avg_r']:>+6.2f}  "
              f"{s['net']:>+11.2f}")
    print("  (MaxDD% = peak-to-trough on ${:,.0f} + that symbol's Net $ only — attribution curve)".format(
        start_balance))


def _fmt_fill(r):
    kind = str(r.get("fill_kind", "?")).lower()
    mode = str(r.get("entry_confirm_mode", "")).lower()
    if kind == "tol":
        return "TOL"
    if kind in ("reject", "confirm") or mode in ("rejection", "m15_close"):
        return "REJ" if kind == "reject" or mode == "rejection" else "CONF"
    if kind == "exact":
        return "EXACT"
    return kind.upper()[:5]


def _fmt_etol(r):
    etol = r.get("entry_tol_usd")
    if etol is None:
        return "   -"
    return f"{float(etol):>5.2f}"


def print_full_ledger(ledger, title="PORTFOLIO TRADES"):
    if not ledger:
        return
    print(f"\n  --- {title} ({len(ledger)} trades) ---")
    print(f"  {'#':>3}  {'entry':<17}  {'asset':<8}  {'rule':<10}  {'side':<5}  "
          f"{'entry$':>10}  {'exit$':>10}  {'R':>6}  {'risk$':>8}  {'P&L$':>9}  "
          f"{'bal$':>11}  {'fill':>5}  {'etol$':>5}  exit")
    print("  " + THIN)
    tot = 0.0
    n_tol = 0
    for i, r in enumerate(ledger, 1):
        tot += r["net"]
        if str(r.get("fill_kind", "")).lower() == "tol":
            n_tol += 1
        side = "LONG" if r["dir"] == "long" else "SHORT"
        print(f"  {i:>3}  {_pd_ts(r['entry_time'])}  {r['asset']:<8}  {r['rule']:<10}  "
              f"{side:<5}  {r['entry']:>10.2f}  {r['exit_px']:>10.2f}  {r['R']:>+6.2f}  "
              f"{r['risk_usd']:>8.2f}  {r['net']:>+9.2f}  {r['balance']:>11.2f}  "
              f"{_fmt_fill(r):>5}  {_fmt_etol(r):>5}  {r['exit']}")
    print("  " + THIN)
    print(f"  {'TOTAL':>3}  {'':17}  {'':8}  {'':10}  {'':5}  {'':10}  {'':10}  "
          f"{'':6}  {'':8}  {tot:>+9.2f}")
    if ledger:
        print(f"  Fill mix: {len(ledger) - n_tol} exact, {n_tol} tolerance-only")


def export_equity_chart(curve, start_balance, stats, path, days=90):
    """Save MT5 Strategy Tester-style balance/equity chart."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if not curve:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    times = [p["time"] for p in curve]
    balance = [p["balance"] for p in curve]
    equity = [p["equity"] for p in curve]

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="#1e1e1e")
    ax.set_facecolor("#252525")

    ax.plot(times, balance, color="#4caf50", linewidth=1.8, label="Balance", drawstyle="steps-post")
    ax.plot(times, equity, color="#81c784", linewidth=1.0, alpha=0.85,
            label="Equity", drawstyle="steps-post")
    ax.axhline(start_balance, color="#757575", linestyle="--", linewidth=0.9, label="Deposit")
    min_bal = stats.get("min_balance")
    if min_bal is not None:
        ax.axhline(min_bal, color="#ef5350", linestyle=":", linewidth=0.9,
                   label=f"Lowest ${min_bal:,.0f}")

    ax.fill_between(times, start_balance, balance,
                    where=[b >= start_balance for b in balance],
                    color="#4caf50", alpha=0.12, step="post")
    ax.fill_between(times, start_balance, balance,
                    where=[b < start_balance for b in balance],
                    color="#ef5350", alpha=0.12, step="post")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=25, ha="right")

    ax.set_title("Portfolio Backtest — Balance / Equity", color="#e0e0e0", fontsize=12, pad=10)
    ax.set_xlabel("Date", color="#bdbdbd")
    ax.set_ylabel("USD", color="#bdbdbd")
    ax.tick_params(colors="#bdbdbd")
    ax.grid(True, color="#404040", linestyle="-", linewidth=0.4, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    profit = stats.get("profit", balance[-1] - start_balance)
    ret = stats.get("ret_pct", 0)
    dd = stats.get("max_dd", 0)
    wr = stats.get("wr", 0)
    n = stats.get("n", 0)
    mb = stats.get("min_balance", min(balance) if balance else start_balance)
    abs_dd = stats.get("abs_dd", max(0.0, start_balance - mb))
    txt = (
        f"Profit: ${profit:+,.0f}  ({ret:+.1f}%)\n"
        f"Max DD: {dd:.1f}%  |  Abs DD: ${abs_dd:,.0f}\n"
        f"Lowest: ${mb:,.0f}  |  WR: {wr:.1f}%  |  Trades: {n}\n"
        f"Start: ${start_balance:,.0f}  →  Final: ${balance[-1]:,.0f}"
    )
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            color="#e0e0e0", bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333", edgecolor="#555555"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.78), framealpha=0.85,
              facecolor="#333333", edgecolor="#555555", labelcolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def export_ledger_csv(ledger, path):
    import csv
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = (
        "n", "entry_time", "exit_time", "asset", "rule", "side", "entry", "exit",
        "R", "tp_r", "want_risk_pct", "use_risk_pct", "risk_usd",
        "pnl_usd", "balance_after", "fill_kind", "entry_tol_atr", "entry_tol_usd",
        "fill_gap", "exit_reason",
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, r in enumerate(ledger, 1):
            w.writerow([
                i, r["entry_time"], r["exit_time"], r["asset"], r["rule"],
                r["dir"], r["entry"], r["exit_px"], r["R"],
                r.get("tp_r"), r["want_pct"], r["use_pct"], r["risk_usd"],
                r["net"], r["balance"], r.get("fill_kind"), r.get("entry_tol_atr"),
                r.get("entry_tol_usd"), r.get("fill_gap"), r["exit"],
            ])
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", type=float, default=DEFAULT_BAL)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--max-portfolio-risk", type=float, default=C.MAX_PORTFOLIO_RISK)
    ap.add_argument("--no-trades", action="store_true")
    ap.add_argument("--export", default="reports/portfolio_trades.csv")
    ap.add_argument("--chart", default="reports/portfolio_equity.png",
                    help="Save MT5-style balance/equity chart (PNG)")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--fixed-risk", action="store_true",
                    help="Size every trade off starting balance (Phase 1 honest view)")
    ap.add_argument("--strict", action="store_true",
                    help="Zero tolerance — exact rule matching (legacy behavior)")
    ap.add_argument("--tf", default="M15", choices=["M5", "M15"],
                    help="Signal timeframe (must match live_portfolio --tf)")
    args = ap.parse_args()
    # Per-TF gate/journal isolation — M5 backtest must not overwrite the M15
    # weekly gate or journal (and vice versa) when both bots share an account.
    FC.apply_tf_paths(args.tf)
    if args.tf != "M15" and args.export == "reports/portfolio_trades.csv":
        args.export = FC.WEEKLY["trades_csv"]
    bal = args.balance
    days = args.days
    risk_cap = None if args.no_cap else args.max_portfolio_risk
    tol_override = {k: 0.0 for k in FC.TOLERANCE} if args.strict else None
    assets = list(C.PORTFOLIO_ASSETS)

    walkforward = FC.WEEKLY.get("enabled") and FC.WEEKLY.get("walkforward", True)
    oos_mg = None
    if walkforward and FC.META.get("meta_gate"):
        oos_mg = MG.load_meta_gate(
            FC.META.get("meta_gate_json", MG.DEFAULT_JSON),
            assets=assets,
            min_regime_n=FC.META.get("meta_min_regime_n", 3),
            require_positive_r=FC.META.get("meta_require_positive_r", True),
            exclude_weak_rules=FC.META.get("meta_exclude_weak_rules", True),
        )

    mt5 = M.connect()
    try:
        WA.auto_refresh(csv_path=args.export, mt5=mt5, silent=True)

        print(SEP)
        print(f"  PORTFOLIO BACKTEST  |  ONE account ${bal:,.0f}  |  {days}d  |  tf={args.tf}")
        print(f"  Sizing: real lot via calc_lot (vol_min/step/max enforced — same as live)")
        print(f"  Risk map: " + ", ".join(f"{a} {RISK_MAP[a]*100:.0f}%" for a in assets))
        mc_desc = ", ".join(C.describe_concurrent(a) for a in assets)
        print(f"  Max open/symbol: {mc_desc}", end="")
        if risk_cap is not None:
            print(f"  |  portfolio cap {risk_cap*100:.0f}% total open risk")
        else:
            print()
        print(SEP)
        FC.print_status()
        if walkforward:
            print(f"  Weekly replay   : ON — gate rebuilds each ISO week (causal rolling)")
        print(SEP)

        rows = []
        all_trades = []
        rmaps = {}
        for asset in assets:
            _, prof = P.get_profile(asset)
            risk = RISK_MAP[asset]
            try:
                sym = P.resolve_symbol_for_profile(asset, mt5)
            except RuntimeError as e:
                rows.append(dict(asset=asset, error=str(e)))
                continue
            rmaps[asset] = _build_regime_map(sym, prof, mt5, days)
            stats = _run(sym, prof, mt5, days=days, asset_key=asset, risk_pct=risk,
                         max_concurrent=C.max_concurrent(asset),
                         tol_override=tol_override,
                         meta_gate_override=oos_mg, tf=args.tf)
            spec = X.get_spec(sym, mt5)
            for t in stats.get("trades") or []:
                tc = dict(t)
                tc["_asset"] = asset
                tc["_risk_pct"] = risk
                tc["_spread"] = stats["spread"]
                tc["_max_concurrent"] = C.max_concurrent(asset)
                tc["_spec"] = spec
                all_trades.append(tc)
            rows.append(dict(
                asset=asset, label=prof.get("label", asset), sym=sym, risk=risk, stats=stats,
            ))

        all_trades.sort(key=lambda t: (t["time"], t.get("exit_time")))
        raw_n = len(all_trades)
        if walkforward and all_trades:
            all_trades, wf = WA.apply_walkforward_filter(
                all_trades, mt5=mt5, assets=tuple(assets))
            if wf["before"] != wf["after"]:
                print(f"\n  Weekly walk-forward filter: {wf['before']} → {wf['after']} trades "
                      f"({wf['before'] - wf['after']} removed)")
                for r in wf["removed"][:10]:
                    print(f"    - {r['time']}  {r['display']:<8} {r['dir']:<5}  "
                          f"{r['rule']} × {r['regime']}  R={r['R']:+.2f}")
                if len(wf["removed"]) > 10:
                    print(f"    ... +{len(wf['removed']) - 10} more")

        mg = FC.load_meta_gate(assets=list(assets))
        before_filter = len(all_trades)
        all_trades = prepare_portfolio_trades(all_trades, rmaps, mg)
        if before_filter != len(all_trades):
            print(f"\n  Tier/CH-REV filter: {before_filter} → {len(all_trades)} trades "
                  f"({before_filter - len(all_trades)} removed)")

        live = simulate_portfolio(all_trades, bal, max_portfolio_risk_pct=risk_cap,
                                  size_compound=not args.fixed_risk)

        ref_note = " (raw signals)" if walkforward and raw_n != live["n"] else ""
        print(f"\n  SINGLE-SYMBOL REFERENCE{ref_note} (one symbol, full ${bal:,.0f} — backtest.py)")
        print(f"  {'Symbol':<12} {'Risk':>5}  {'Trades':>6}  {'Return':>8}  {'DD':>6}")
        print("  " + THIN)
        for row in rows:
            if row.get("error"):
                continue
            s = row["stats"]
            print(f"  {row['label']:<12} {row['risk']*100:>4.0f}%  {s['n']:>6}  "
                  f"{s['ret']:>+7.1f}%  {s['dd']:>5.1f}%")

        print(f"\n  PORTFOLIO RESULT (shared balance, {'fixed risk' if args.fixed_risk else 'compound'})")
        print(f"  Trades: {live['n']}  |  scaled: {live['scaled']}  |  skipped: {live['skipped']}")
        print(f"  WR: {live['wr']:.1f}%  |  PF: {live['pf']:.2f}  |  Max DD: {live['max_dd']:.1f}%")
        print(f"  ${bal:,.2f}  →  ${live['final']:,.2f}  |  Profit ${live['profit']:+,.2f}  ({live['ret_pct']:+.1f}%)")

        print_per_asset_breakdown(live["ledger"], assets, RISK_MAP, rows, start_balance=bal)

        if live.get("monthly"):
            print(f"\n  Monthly P&L ($):")
            for m in sorted(live["monthly"]):
                print(f"    {m}:  {live['monthly'][m]:>+10.2f}")

        if live["ledger"]:
            ex, la = live["ledger"][0], live["ledger"][-1]
            print(f"\n  Example sizing:")
            print(f"    Trade #1:  bal before risk  → risk ${ex['risk_usd']:.2f}  ({ex['want_pct']*100:.0f}% target)")
            print(f"    Trade #{live['n']}: risk ${la['risk_usd']:.2f}  (balance had grown to compound)")

        if not args.no_trades and live["ledger"]:
            print_full_ledger(live["ledger"])

        if args.export and live["ledger"]:
            print(f"\n  Saved CSV: {export_ledger_csv(live['ledger'], args.export)}")
            WA.auto_refresh(csv_path=args.export, mt5=mt5,
                            ledger=live["ledger"], silent=True)

        if not args.no_chart and live.get("equity_curve"):
            chart_path = export_equity_chart(
                live["equity_curve"], bal, live, args.chart, days=days)
            if chart_path:
                print(f"  Saved chart: {chart_path}")
        print(SEP)
    finally:
        M.shutdown(mt5)


if __name__ == "__main__":
    main()
