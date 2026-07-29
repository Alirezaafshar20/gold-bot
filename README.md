# مدلِ معاملاتیِ قانون‌محور — SMC / ICT / RTM / Wyckoff

مدل **قانون‌محور** است — **نیازی به CSV یا ریترین ندارد**.
همه‌چیز مستقیم از **MetaTrader 5** خوانده می‌شود.

---

## فایل‌ها

| فایل | کار |
|---|---|
| `strategy.py` | کلِ مدل: قوانین + تریلینگ + شبیه‌ساز |
| `mt5_data.py` | خواندنِ دیتا از MT5 (مشترک بین live و backtest) |
| `backtest.py` | بک‌تست — **مستقیم از MT5** |
| `live_smc.py` | اجرای زنده — **مستقیم از MT5** |
| `download_data.py` | اختیاری: ذخیرهٔ CSV (آرشیو؛ لازم نیست) |

---

## پیش‌نیاز

1. **MetaTrader 5** باز و **لاگین** باشد
2. نماد طلا در Market Watch فعال باشد (معمولاً `XAUUSD@`)
3. `pip install MetaTrader5 pandas numpy`

---

## بک‌تست (پیشنهادی)

```powershell
python backtest.py --optimized --days 60 --spread 0.28
python backtest.py --optimized --compare-old --days 60
```

## لایو (پیشنهادی)

```powershell
python live_smc.py --optimized --symbol XAUUSD@ --dry-run
python live_smc.py --optimized --symbol XAUUSD@
```

**OPTIMIZED:** FVG+OB+DEMAND | M15 | BE@+1R + TP 2R | session 10-20 | min SL $2 | max 1 trade

---

## بک‌تست (قدیمی) (بدون CSV)

```powershell
# 30 روز اخیر — M5
python backtest.py --tf M5 --symbol XAUUSD@ --days 30

# 30 روز اخیر — M30 / H1 / H4
python backtest.py --tf M30 --days 30
python backtest.py --tf H1 --days 30
python backtest.py --tf H4 --days 30

# بازهٔ دقیق
python backtest.py --tf H1 --from 2026-01-01 --to 2026-02-01

# بدون فیلتر بازه → همهٔ تاریخچهٔ دانلودشده
python backtest.py --tf M15
```

**تایم‌فریم‌های پشتیبانی‌شده:** `M5` `M15` `M30` `H1` `H4`

---

## اجرای زنده

```powershell
# اول تمرینی
python live_smc.py --tf M5 --symbol XAUUSD@ --dry-run

# واقعی
python live_smc.py --tf M5 --symbol XAUUSD@
```

---

## دانلود CSV (اختیاری — فقط آرشیو)

```powershell
python download_data.py --timeframe M5 --symbol XAUUSD@
python download_data.py --timeframe M1 --symbol XAUUSD@
```

> **مهم:** اگر `XAUUSD` کار نکرد، حتماً `XAUUSD@` بزن (WM Markets و بروکرهای مشابه).

---

## خطای «No data / Could not enable symbol»

| علت | راه‌حل |
|---|---|
| نماد اشتباه | `--symbol XAUUSD@` (دقیقاً مثل MT5) |
| MT5 بسته است | MT5 را باز و لاگین کن |
| بازار بسته | صبر کن تا بازار باز شود |
| AutoTrading خاموش | در MT5: Tools → Options → Expert Advisors → Allow algo trading |

---

## قوانین فعال

`WYCK` `FVG` `OB` `BOS` `DEMAND` `SUPPLY` `FL` `QM` + فیلتر VP + تریلینگ R

## ریترین؟

**نه.** مدل قانون‌محور است — هیچ MLای برای ریترین وجود ندارد.
