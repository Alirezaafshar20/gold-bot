//+------------------------------------------------------------------+
//|                                           XGBotSignals.mq5       |
//|               XGBot Signal Visualizer for MetaTrader 5            |
//|                                                                    |
//|  ✔ حالت لایو: هر کندل جدید سیگنال‌ها آپدیت می‌شوند             |
//|  ✔ مولتی تایم‌فریم: وقتی TF عوض می‌شود، فایل متناسب لود می‌شود  |
//|                                                                    |
//|  نحوه استفاده:                                                    |
//|    1. python signal_exporter.py --live --all-tf                    |
//|    2. این indicator را به چارت الصاق کن                            |
//+------------------------------------------------------------------+
#property copyright   "XGBot"
#property version     "2.00"
#property description "XGBot Multi-TF Signal Visualizer"
#property description "Run: python signal_exporter.py --live --all-tf"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//--- Inputs
input bool   InpShowSLTP    = false;         // Show SL/TP lines (پیش‌فرض: خاموش)
input bool   InpShowLabel   = false;         // Show label on arrow (پیش‌فرض: خاموش — tooltip داره)
input color  InpBuyColor    = C'30,144,255'; // BUY color
input color  InpSellColor   = C'255,69,58';  // SELL color
input int    InpArrowSize   = 3;             // Arrow size (1-5)
input int    InpSLTPBars    = 8;             // SL/TP line length (bars)
input int    InpRefreshSec  = 15;            // Refresh interval (seconds)
input int    InpMaxSignals  = 60;            // Max signals to show (آخرین N سیگنال)

//--- State
const string PREFIX = "XGB_";
ENUM_PERIOD  g_last_period = PERIOD_CURRENT;
string       g_last_file   = "";

//+------------------------------------------------------------------+
string GetTFName(ENUM_PERIOD p)
{
    switch(p)
    {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        default:         return "M15";
    }
}

//--- فایل سیگنال مطابق تایم‌فریم فعلی چارت
string GetSignalFile()
{
    return "xgbot_signals_" + GetTFName(Period()) + ".csv";
}

//+------------------------------------------------------------------+
int OnInit()
{
    g_last_period = Period();
    g_last_file   = GetSignalFile();
    EventSetTimer(InpRefreshSec);
    DrawSignals();
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTimer()
{
    string cur_file = GetSignalFile();

    // اگر تایم‌فریم عوض شد، چارت را پاک کن و سیگنال‌های جدید بکش
    if(Period() != g_last_period || cur_file != g_last_file)
    {
        ClearObjects();
        g_last_period = Period();
        g_last_file   = cur_file;
    }

    DrawSignals();
    ChartRedraw(0);
}

//+------------------------------------------------------------------+
void DrawSignals()
{
    string filename = GetSignalFile();

    int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
    if(handle == INVALID_HANDLE)
    {
        string msg = "XGBot: No signals for " + GetTFName(Period()) +
                     " — run: python signal_exporter.py --live --tf " +
                     IntegerToString(PeriodSeconds() / 60);
        Comment(msg);
        return;
    }

    // حذف آبجکت‌های قبلی قبل از رسم جدید
    ClearObjects();

    // Skip header
    if(!FileIsEnding(handle))
        FileReadString(handle);

    // همه سطرها رو بخون، فقط آخرین InpMaxSignals رو رسم کن
    string all_lines[];
    int    total_lines = 0;
    while(!FileIsEnding(handle))
    {
        string line = FileReadString(handle);
        if(StringLen(line) < 10) continue;
        ArrayResize(all_lines, total_lines + 1);
        all_lines[total_lines++] = line;
    }
    FileClose(handle);

    // شروع از آخرین InpMaxSignals سطر
    int start_from = MathMax(0, total_lines - InpMaxSignals);

    int n_buy = 0, n_sell = 0, count = 0;

    for(int li = start_from; li < total_lines; li++)
    {
        string line = all_lines[li];
        if(StringLen(line) < 10) continue;

        string parts[];
        if(StringSplit(line, ',', parts) < 6) continue;

        string   dt_str  = parts[0];
        string   sig_str = parts[1];
        double   price   = StringToDouble(parts[2]);
        double   sl      = StringToDouble(parts[3]);
        double   tp      = StringToDouble(parts[4]);
        double   proba   = StringToDouble(parts[5]);

        datetime dt = StringToTime(dt_str);
        if(dt == 0 || price == 0.0) continue;

        bool   is_buy = (sig_str == "BUY");
        color  clr    = is_buy ? InpBuyColor : InpSellColor;
        string id     = PREFIX + IntegerToString(count);

        // offset پویا: ۰.۱۵٪ قیمت — برای طلا، بیتکوین و فارکس
        double offset = price * 0.0015;

        // ── فلش ──────────────────────────────────────────────────
        string arrow_nm = id + "_A";
        double arrow_y  = is_buy ? (price - offset) : (price + offset);
        int    arrow_cd = is_buy ? 233 : 234;  // 233=▲  234=▼

        if(ObjectCreate(0, arrow_nm, OBJ_ARROW, 0, dt, arrow_y))
        {
            ObjectSetInteger(0, arrow_nm, OBJPROP_ARROWCODE, arrow_cd);
            ObjectSetInteger(0, arrow_nm, OBJPROP_COLOR,     clr);
            ObjectSetInteger(0, arrow_nm, OBJPROP_WIDTH,     InpArrowSize);
            ObjectSetInteger(0, arrow_nm, OBJPROP_BACK,      false);
            ObjectSetInteger(0, arrow_nm, OBJPROP_SELECTABLE, false);
            double conf_pct = is_buy ? proba : (100.0 - proba);
            ObjectSetString (0, arrow_nm, OBJPROP_TOOLTIP,
                             sig_str + " | Conf: " + DoubleToString(conf_pct,1) + "%" +
                             (is_buy ? " (Rise)" : " (Fall)") +
                             "  Price: " + DoubleToString(price,2) +
                             "  SL: "   + DoubleToString(sl,2) +
                             "  TP: "   + DoubleToString(tp,2));
        }

        // ── لیبل احتمال ──────────────────────────────────────────
        if(InpShowLabel)
        {
            string lbl_nm  = id + "_L";
            double lbl_y   = is_buy ? (arrow_y - offset * 0.7) : (arrow_y + offset * 0.7);
            // BUY: نشان‌دادن احتمال صعود | SELL: نشان‌دادن احتمال نزول (واضح‌تر)
            double show_pct = is_buy ? proba : (100.0 - proba);
            string lbl_txt  = (is_buy ? "B " : "S ") + DoubleToString(show_pct, 0) + "%";

            if(ObjectCreate(0, lbl_nm, OBJ_TEXT, 0, dt, lbl_y))
            {
                ObjectSetString (0, lbl_nm, OBJPROP_TEXT,      lbl_txt);
                ObjectSetInteger(0, lbl_nm, OBJPROP_COLOR,     clr);
                ObjectSetInteger(0, lbl_nm, OBJPROP_FONTSIZE,  7);
                ObjectSetString (0, lbl_nm, OBJPROP_FONT,      "Arial Bold");
                ObjectSetInteger(0, lbl_nm, OBJPROP_ANCHOR,    is_buy ? ANCHOR_UPPER : ANCHOR_LOWER);
                ObjectSetInteger(0, lbl_nm, OBJPROP_BACK,      false);
                ObjectSetInteger(0, lbl_nm, OBJPROP_SELECTABLE, false);
            }
        }

        // ── خطوط SL / TP ─────────────────────────────────────────
        if(InpShowSLTP)
        {
            datetime dt_end = dt + (datetime)(PeriodSeconds() * InpSLTPBars);

            // SL خط قرمز نقطه‌چین
            string sl_nm = id + "_SL";
            if(ObjectCreate(0, sl_nm, OBJ_TREND, 0, dt, sl, dt_end, sl))
            {
                ObjectSetInteger(0, sl_nm, OBJPROP_COLOR,      clrTomato);
                ObjectSetInteger(0, sl_nm, OBJPROP_STYLE,      STYLE_DOT);
                ObjectSetInteger(0, sl_nm, OBJPROP_WIDTH,      1);
                ObjectSetInteger(0, sl_nm, OBJPROP_RAY_RIGHT,  false);
                ObjectSetInteger(0, sl_nm, OBJPROP_BACK,       true);
                ObjectSetInteger(0, sl_nm, OBJPROP_SELECTABLE, false);
                ObjectSetString (0, sl_nm, OBJPROP_TOOLTIP,    "SL: " + DoubleToString(sl,2));
            }

            // TP خط سبز نقطه‌چین
            string tp_nm = id + "_TP";
            if(ObjectCreate(0, tp_nm, OBJ_TREND, 0, dt, tp, dt_end, tp))
            {
                ObjectSetInteger(0, tp_nm, OBJPROP_COLOR,      C'50,205,50');
                ObjectSetInteger(0, tp_nm, OBJPROP_STYLE,      STYLE_DOT);
                ObjectSetInteger(0, tp_nm, OBJPROP_WIDTH,      1);
                ObjectSetInteger(0, tp_nm, OBJPROP_RAY_RIGHT,  false);
                ObjectSetInteger(0, tp_nm, OBJPROP_BACK,       true);
                ObjectSetInteger(0, tp_nm, OBJPROP_SELECTABLE, false);
                ObjectSetString (0, tp_nm, OBJPROP_TOOLTIP,    "TP: " + DoubleToString(tp,2));
            }
        }

        if(is_buy) n_buy++; else n_sell++;
        count++;
    }

    Comment("XGBot [" + GetTFName(Period()) + "]  " +
            IntegerToString(count) + " signals  " +
            "(B=" + IntegerToString(n_buy) + " S=" + IntegerToString(n_sell) + ")  " +
            "Updated: " + TimeToString(TimeCurrent(), TIME_MINUTES));
}

//+------------------------------------------------------------------+
void ClearObjects()
{
    int total = ObjectsTotal(0, 0, -1);
    for(int i = total - 1; i >= 0; i--)
    {
        string nm = ObjectName(0, i, 0, -1);
        if(StringFind(nm, PREFIX) == 0)
            ObjectDelete(0, nm);
    }
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
{
    return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    ClearObjects();
    Comment("");
}
//+------------------------------------------------------------------+
