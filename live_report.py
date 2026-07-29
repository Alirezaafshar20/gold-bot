"""
Parse an MT5 'Trade History Report' HTML and summarise what ACTUALLY happened
live (demo), so we can compare it against the backtest.

  python live_report.py "C:\\Users\\EMPART\\Desktop\\ReportHistory-20218355.html"
"""
import re
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8")

TAGRE = re.compile(r"<[^>]+>")


def cell_text(td):
    return TAGRE.sub("", td).replace("&nbsp;", " ").strip()


def _read(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


def parse(path):
    html = _read(path)
    # each position row: has a <tr ...> ... </tr> with the hidden rule cell
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    trades = []
    for r in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(tds) < 13:
            continue
        cells = [cell_text(t) for t in tds]
        # detect a position row: cells[0] looks like a datetime, type in buy/sell
        if not re.match(r"\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}", cells[0]):
            continue
        # find the type cell (buy/sell)
        typ = None
        for c in cells[:6]:
            if c in ("buy", "sell"):
                typ = c
                break
        if typ is None:
            continue
        # rule tag lives in the hidden cell (class="hidden")
        rule = ""
        m = re.search(r'class="hidden"[^>]*>(.*?)<', r, re.S)
        if m:
            rule = cell_text(m.group(1))
        try:
            profit = float(cells[-1].replace(" ", ""))
        except ValueError:
            continue
        trades.append({"time": cells[0], "symbol": cells[2], "side": typ,
                       "rule": rule, "profit": profit})
    return trades


def block(title, key, trades):
    agg = collections.defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in trades:
        a = agg[key(t)]
        a["n"] += 1
        a["pnl"] += t["profit"]
        if t["profit"] > 0:
            a["w"] += 1
    print(f"\n  {title}")
    print(f"    {'name':<16}{'n':>5}{'win':>5}{'WR%':>7}{'net$':>11}")
    print("    " + "-" * 44)
    for name, a in sorted(agg.items(), key=lambda kv: kv[1]["pnl"]):
        wr = 100.0 * a["w"] / a["n"] if a["n"] else 0
        print(f"    {str(name):<16}{a['n']:>5}{a['w']:>5}{wr:>7.1f}{a['pnl']:>+11.2f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\EMPART\Desktop\ReportHistory-20218355.html"
    trades = parse(path)
    if not trades:
        print("No trades parsed — check the file path/format.")
        return
    n = len(trades)
    wins = sum(1 for t in trades if t["profit"] > 0)
    pnl = sum(t["profit"] for t in trades)
    gross_w = sum(t["profit"] for t in trades if t["profit"] > 0)
    gross_l = sum(t["profit"] for t in trades if t["profit"] < 0)
    pf = gross_w / -gross_l if gross_l < 0 else 99.99

    print("=" * 54)
    print("  LIVE (DEMO) REPORT SUMMARY")
    print("=" * 54)
    print(f"  trades: {n}   wins: {wins}   WR: {100.0*wins/n:.1f}%")
    print(f"  net P&L: {pnl:+.2f}   PF: {pf:.2f}")
    print(f"  gross win: {gross_w:+.2f}   gross loss: {gross_l:+.2f}")
    print(f"  first: {trades[0]['time']}   last: {trades[-1]['time']}")

    block("BY SYMBOL", lambda t: t["symbol"], trades)
    block("BY RULE", lambda t: t["rule"] or "?", trades)
    block("BY SIDE", lambda t: t["side"], trades)


if __name__ == "__main__":
    main()
