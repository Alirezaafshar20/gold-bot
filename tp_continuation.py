"""
Post-TP continuation analysis — did price keep moving after take-profit?
Uses the same STABLE preset as backtest.py (--optimized, 180d).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import strategy as S
import mt5_data as M

LOOKAHEAD_BARS = (48, 96, 192)  # M15: 12h, 24h, 48h after exit
SPREAD = 0.28


def load():
    mt5 = M.connect()
    try:
        sym = M.resolve_symbol("XAUUSD@", mt5)
        _, d, m1, cutoff = M.fetch_pair(sym, "M15", days=180, mt5=mt5)
        _, htf_dfs = M.fetch_htf_bars(sym, days=180, mt5=mt5, tfs=("H4", "H1"))
    finally:
        M.shutdown(mt5)
    return d, m1, cutoff, S.prepare_htf_context(htf_dfs)


def run_trades(d, m1, htf):
    opt = S.stable_settings()
    skip = {"spread_usd", "nds_mode"}
    kw = {k: v for k, v in {**opt, "htf_context": htf, "tf_min": 15}.items()
          if k not in skip}
    return S.run_backtest(d, m1, **kw)


def find_sig_bar(B, t):
    """Bar index at or just before entry time."""
    idx = int(np.searchsorted(B.t, np.datetime64(t), "right")) - 1
    return max(3, min(idx, B.n - 2))


def htf_targets_all(signal_time, direction, entry, risk, htf_context,
                    min_r=S.OPT_MIN_TP_R, max_r=S.OPT_MAX_TP_R):
    """All HTF structural targets in range (not just nearest)."""
    out = []
    for tf_name in ("H4", "H1"):
        ctx = htf_context.get(tf_name)
        if ctx is None:
            continue
        B = ctx if isinstance(ctx, S.Bars) else S.Bars(ctx)
        idx = S.htf_bar_index(B, signal_time)
        targets = S.collect_htf_targets(B, idx, direction, entry,
                                        lookback=S.OPT_HTF_LOOKBACK)
        if direction == "long":
            min_px, max_px = entry + min_r * risk, entry + max_r * risk
            cands = sorted(t for t in targets if min_px <= t <= max_px)
        else:
            max_px, min_px = entry - min_r * risk, entry - max_r * risk
            cands = sorted((t for t in targets if min_px <= t <= max_px), reverse=True)
        for j, px in enumerate(cands):
            r = (px - entry) / risk if direction == "long" else (entry - px) / risk
            out.append({"tf": tf_name, "rank": j + 1, "px": px, "R": r})
    return out


def continuation_after(B, exit_idx, direction, entry, risk, bars):
    """Max favorable move in R for `bars` M15 candles after exit bar."""
    end = min(exit_idx + bars, B.n - 1)
    if exit_idx >= end:
        return 0.0, None
    seg_h = B.h[exit_idx + 1: end + 1]
    seg_l = B.l[exit_idx + 1: end + 1]
    if direction == "long":
        best = float(seg_h.max())
        extra_r = (best - entry) / risk
        best_t = B.t[exit_idx + 1 + int(np.argmax(seg_h))]
    else:
        best = float(seg_l.min())
        extra_r = (entry - best) / risk
        best_t = B.t[exit_idx + 1 + int(np.argmin(seg_l))]
    return extra_r, best_t


def trend_ok(htf_context, ts, direction, tfs=("H4", "H1")):
    ok = []
    for tf in tfs:
        ok.append((tf, S.htf_trend_pass(htf_context, ts, direction, tf=tf,
                                         ema_period=S.OPT_HTF_EMA)))
    return ok


def next_setups(B, htf, from_i, direction, enabled=("OB", "NDS", "DEMAND"),
                window=48):
    """Same-direction rule setups within `window` bars after exit."""
    found = []
    end = min(from_i + window, B.n - 2)
    for i in range(from_i + 1, end):
        for name in enabled:
            for d, prox, dist, tag in S.iter_rule_setups(
                    name, B, i, S.DEFAULT_PARAMS, htf_context=htf,
                    signal_time=B.t[i]):
                if d != direction:
                    continue
                found.append({
                    "time": str(B.t[i])[:16],
                    "rule": name,
                    "tag": tag,
                    "entry": prox,
                })
                break
    return found[:5]


def explain_continuation(t, B, htf, sig_i, exit_idx, targets, extra_r_96):
    """Why price may have continued (strategy-native reasons)."""
    reasons = []
    direction = t["dir"]
    entry, risk, tp = t["entry"], t["risk"], t.get("tp")
    tp_r = t.get("tp_r") or 0

    # 1) Further HTF targets beyond TP
    beyond = [x for x in targets if x["R"] > tp_r + 0.15]
    if beyond:
        nxt = beyond[0]
        reasons.append(
            f"Next HTF target ({nxt['tf']} #{nxt['rank']}) at {nxt['px']:.2f} "
            f"({nxt['R']:.1f}R) — TP hit 1st zone only"
        )

    # 2) HTF trend still aligned at exit + 24h later
    exit_ts = t["exit_time"]
    tr_exit = trend_ok(htf, exit_ts, direction)
    if all(x[1] for x in tr_exit):
        reasons.append("HTF trend (H4+H1 EMA) still aligned at exit")

    # 3) TP was 2R floor not full HTF run
    src = t.get("tp_source") or "?"
    if src in ("fixed_r", "htf_blend") and tp_r <= 2.05:
        reasons.append(f"TP capped near 2R floor ({src}) — room to next HTF level")

    # 4) New setups same direction after exit
    setups = next_setups(B, htf, exit_idx, direction)
    if setups:
        rules = ", ".join(f"{s['rule']}@{s['time']}" for s in setups[:3])
        reasons.append(f"New same-direction setups after exit: {rules}")

    # 5) Strong continuation magnitude
    if extra_r_96 >= tp_r + 1.0:
        reasons.append(
            f"Price ran {extra_r_96 - tp_r:.1f}R beyond TP within 24h "
            f"(total {extra_r_96:.1f}R from entry)"
        )
    elif extra_r_96 <= tp_r + 0.3:
        reasons.append("Little continuation — TP was near local extreme")

    return reasons


def main():
    d, m1, cutoff, htf = load()
    B = S.Bars(d)
    trades = run_trades(d, m1, htf)
    if cutoff is not None:
        trades = [t for t in trades if t["time"] >= cutoff]

    tp_trades = [t for t in trades if t.get("exit_reason") == "tp"]
    other = [t for t in trades if t.get("exit_reason") != "tp"]

    print("=" * 72)
    print("  POST-TP CONTINUATION ANALYSIS  |  STABLE 180d  |  XAUUSD@ M15")
    print("=" * 72)
    print(f"  Total trades: {len(trades)}  |  TP exits: {len(tp_trades)}  |  "
          f"SL/time: {len(other)}")
    print()

    rows = []
    for n, t in enumerate(tp_trades, 1):
        sig_i = find_sig_bar(B, t["time"])
        exit_idx = max(sig_i, int(t.get("exit_idx", sig_i)))
        direction = t["dir"]
        entry, risk = t["entry"], t["risk"]
        tp_r = t.get("tp_r") or 0

        targets = htf_targets_all(B.t[sig_i], direction, entry, risk, htf)
        cont = {}
        for lb in LOOKAHEAD_BARS:
            extra, _ = continuation_after(B, exit_idx, direction, entry, risk, lb)
            cont[lb] = extra

        extra_96 = cont[96]
        left_r = max(0, extra_96 - tp_r)
        continued = left_r >= 0.5

        reasons = explain_continuation(t, B, htf, sig_i, exit_idx, targets,
                                       extra_96)

        rows.append({
            "n": n, "time": str(t["time"])[:16], "rule": t.get("setup", "?"),
            "dir": direction, "tp_r": tp_r, "extra_24h": cont[96],
            "left_r": left_r, "continued": continued,
        })

        side = "LONG" if direction == "long" else "SHORT"
        print(f"  #{n}  {str(t['time'])[:16]}  {t.get('setup','?'):<4}  {side}")
        print(f"       TP @ {tp_r:.2f}R  |  max move after exit: "
              f"12h={cont[48]:.2f}R  24h={cont[96]:.2f}R  48h={cont[192]:.2f}R")
        print(f"       Left on table (24h): {left_r:+.2f}R  |  "
              f"Continued (>0.5R): {'YES' if continued else 'no'}")
        if targets:
            tps = "  ".join(f"{x['tf']}#{x['rank']}={x['R']:.1f}R" for x in targets[:4])
            print(f"       HTF targets at entry: {tps}")
        for r in reasons:
            print(f"       → {r}")
        print()

    if rows:
        cont_n = sum(1 for r in rows if r["continued"])
        avg_left = np.mean([r["left_r"] for r in rows])
        avg_tp = np.mean([r["tp_r"] for r in rows])
        avg_tot = np.mean([r["extra_24h"] for r in rows])
        print("-" * 72)
        print(f"  SUMMARY ({len(rows)} TP trades)")
        print(f"  Continued ≥0.5R beyond TP (24h): {cont_n}/{len(rows)} "
              f"({100*cont_n/len(rows):.0f}%)")
        print(f"  Avg TP taken: {avg_tp:.2f}R  |  Avg max move (24h): {avg_tot:.2f}R  |  "
              f"Avg missed: {avg_left:.2f}R")
        print()

    # Simulate alternative TP: 2nd HTF target instead of 1st / 2R floor
    print("  WHAT-IF: TP at 2nd HTF target (when exists) vs current")
    print("-" * 72)
    sim_cur_r = []
    sim_alt_r = []
    for t in tp_trades:
        sig_i = find_sig_bar(B, t["time"])
        direction, entry, risk = t["dir"], t["entry"], t["risk"]
        targets = htf_targets_all(B.t[sig_i], direction, entry, risk, htf)
        cur_r = t["R"]
        sim_cur_r.append(cur_r)
        if len(targets) >= 2:
            alt_r = targets[1]["R"]
            # Would we still win? Check if price reached 2nd target within hold
            exit_idx = max(sig_i, int(t.get("exit_idx", sig_i)))
            extra, _ = continuation_after(B, exit_idx, direction, entry, risk, 192)
            hit_2nd = extra >= alt_r - 0.05
            sim_alt_r.append(alt_r if hit_2nd else cur_r)
            if len(sim_alt_r) <= 12:
                print(f"  {str(t['time'])[:16]}  cur={cur_r:.2f}R  "
                      f"2nd HTF={alt_r:.1f}R  reached_48h={'yes' if hit_2nd else 'NO'}")
        else:
            sim_alt_r.append(cur_r)
    if sim_cur_r:
        print(f"\n  Avg R current TP exits: {np.mean(sim_cur_r):+.2f}")
        print(f"  Avg R if 2nd HTF (when reachable in 48h): {np.mean(sim_alt_r):+.2f}")
        print(f"  Potential gain: {np.mean(sim_alt_r) - np.mean(sim_cur_r):+.2f}R/trade")

    # Partial + trail simulation rough
    print()
    print("  WHAT-IF: 50% at current TP, rest trail 1R after +2R")
    print("-" * 72)
    partial_rs = []
    for t in tp_trades:
        sig_i = find_sig_bar(B, t["time"])
        exit_idx = max(sig_i, int(t.get("exit_idx", sig_i)))
        direction, entry, risk = t["dir"], t["entry"], t["risk"]
        tp_r = t.get("tp_r") or t["R"]
        extra_192, _ = continuation_after(B, exit_idx, direction, entry, risk, 192)
        # half at tp_r, half trails: exit at max(tp_r, extra_192 - 1) capped
        trail_r = max(tp_r, extra_192 - 1.0)
        blend = 0.5 * tp_r + 0.5 * min(trail_r, extra_192)
        partial_rs.append(blend)
    print(f"  Avg R partial+trail: {np.mean(partial_rs):+.2f}  "
          f"(vs full TP avg {np.mean(sim_cur_r):+.2f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
