"""
A/B test — V-shape regime guards + spike capture on XAUUSD.

Variants (config-gated features added 2026-07-14):
  struct_brk   H1 40-bar donchian break overrides slow H4 macro label
  shockR       block non-spike entries when 24h realized vol >= R x 30d median
  flipNh       block non-spike entries for N hours after a macro bias flip
  spike        add SPIKE_BRK/SQS momentum rules (exempt from shock/flip guards,
               meta_skip_if_no_rules off so TREND_DOWN bars are reachable)

Windows:
  recent 14d  -> the losing whipsaw window (Jul 1-14)
  90d         -> regression check (do the guards hurt the good period?)

Usage: python regime_guard_ab.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mt5_data as M
import symbol_profiles as P
import portfolio_config as C
from multi_symbol_calibrate import _run

ASSET = "XAUUSD"
RISK = C.RISK_MAP[ASSET]

mt5 = M.connect()
_, prof = P.get_profile(ASSET)
sym = P.resolve_symbol_for_profile(ASSET, mt5)


def _spike_extra():
    import spike_strategies as sp
    import floating_config as FC
    rules = list(FC.OOS_RULES[ASSET]) + ["SPIKE_BRK_L", "SPIKE_BRK_S",
                                         "SPIKE_SQS_L", "SPIKE_SQS_S"]
    return {
        "enabled": rules,
        "spike_mode": True,
        "spike_params": dict(sp.SPIKE_DEFAULT),
        "fib_spike_exempt": True,
        "meta_skip_if_no_rules": False,  # allow spike rules in TREND_DOWN bars
    }


VARIANTS = [
    ("baseline", {}),
    ("struct_brk", {"regime_structure_break": True}),
    ("shock1.6", {"regime_shock_gate": True, "regime_shock_ratio": 1.6}),
    ("shock1.8", {"regime_shock_gate": True, "regime_shock_ratio": 1.8}),
    ("flip12h", {"regime_flip_cooldown_h": 12.0}),
    ("flip24h", {"regime_flip_cooldown_h": 24.0}),
    ("brk+shock1.8", {"regime_structure_break": True,
                      "regime_shock_gate": True, "regime_shock_ratio": 1.8}),
    ("brk+shk+flip12", {"regime_structure_break": True,
                        "regime_shock_gate": True, "regime_shock_ratio": 1.8,
                        "regime_flip_cooldown_h": 12.0}),
    ("spike", _spike_extra()),
    ("brk+shk+spike", {**_spike_extra(),
                       "regime_structure_break": True,
                       "regime_shock_gate": True, "regime_shock_ratio": 1.8}),
]


def run_window(days, label):
    print("=" * 100)
    print(f"  {ASSET} {label} ({days}d)  risk={RISK*100:.1f}%  "
          f"confirm=rejection  meta-gate=weekly-off(oos)")
    print("=" * 100)
    print(f"  {'variant':<16} {'n':>4} {'WR%':>6} {'PF':>6} {'ret%':>8} "
          f"{'DD%':>6} {'SL':>4} {'netR':>7}  rules")
    print("-" * 100)
    rows = {}
    for name, extra in VARIANTS:
        r = _run(sym, prof, mt5, days=days, asset_key=ASSET, risk_pct=RISK,
                 max_concurrent=C.max_concurrent(ASSET), opt_extra=extra or None)
        tr = r.get("trades") or []
        net_r = sum(t["R"] for t in tr)
        sl = sum(1 for t in tr if t["R"] < -0.5)
        top = ",".join(f"{k}:{v}" for k, v in
                       sorted(r.get("rules", {}).items(), key=lambda x: -x[1])[:4])
        print(f"  {name:<16} {r['n']:>4} {r['wr']:>6.1f} {r['pf']:>6.2f} "
              f"{r['ret']:>+8.1f} {r['dd']:>6.1f} {sl:>4} {net_r:>+7.2f}  {top}")
        rows[name] = r
    return rows


recent = run_window(14, "WHIPSAW WINDOW")
long_w = run_window(90, "REGRESSION WINDOW")

print("\n  VERDICT GUIDE: a guard is worth enabling only if it improves the")
print("  whipsaw window WITHOUT degrading the 90d window materially.")
