//+------------------------------------------------------------------+
//|                                           CandleCountdown.mq5   |
//|                          Candle Timer — Professional Edition     |
//+------------------------------------------------------------------+
#property copyright   "XGBot"
#property version     "1.00"
#property indicator_chart_window
#property indicator_plots 0

//── Inputs ────────────────────────────────────────────────────────
input color  InpTextColor    = clrWhite;          // رنگ متن
input color  InpUrgentColor  = clrOrangeRed;      // رنگ وقتی < 30 ثانیه
input color  InpBarColor     = clrDodgerBlue;     // رنگ progress bar
input int    InpFontSize     = 11;                // اندازه فونت
input string InpFontName     = "Consolas";        // فونت (monospace)
input int    InpCorner       = CORNER_RIGHT_UPPER; // گوشه نمایش
input int    InpXOffset      = 10;                // فاصله افقی
input int    InpYOffset      = 20;                // فاصله عمودی
input bool   InpShowBar      = true;              // نمایش progress bar
input bool   InpShowOHLC     = true;              // نمایش OHLC کندل فعلی

//── Object names ──────────────────────────────────────────────────
#define OBJ_TIME    "CC_Time"
#define OBJ_BAR     "CC_Bar"
#define OBJ_OHLC    "CC_OHLC"
#define OBJ_LABEL   "CC_Label"

//+------------------------------------------------------------------+
int OnInit()
{
    EventSetMillisecondTimer(500);   // آپدیت هر 500ms
    DrawAll();
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    ObjectDelete(0, OBJ_TIME);
    ObjectDelete(0, OBJ_BAR);
    ObjectDelete(0, OBJ_OHLC);
    ObjectDelete(0, OBJ_LABEL);
}

//+------------------------------------------------------------------+
void OnTimer()   { DrawAll(); }
void OnChartEvent(const int id, const long& lparam,
                  const double& dparam, const string& sparam)
{ DrawAll(); }
int  OnCalculate(const int rates_total, const int prev_calculated,
                 const datetime& time[], const double& open[],
                 const double& high[], const double& low[],
                 const double& close[], const long& tick_volume[],
                 const long& volume[], const int& spread[])
{ DrawAll(); return rates_total; }

//+------------------------------------------------------------------+
void DrawAll()
{
    // ── زمان باقی‌مانده ──────────────────────────────────────────
    datetime now        = TimeCurrent();
    int      period_sec = PeriodSeconds();
    datetime bar_open   = (datetime)(MathFloor((double)now / period_sec) * period_sec);
    datetime bar_close  = bar_open + period_sec;
    int      elapsed    = (int)(now - bar_open);
    int      remaining  = (int)(bar_close - now);

    // ── درصد پیشرفت ──────────────────────────────────────────────
    double   pct        = (double)elapsed / period_sec;   // 0.0 → 1.0

    // ── فرمت متن countdown ───────────────────────────────────────
    string   rem_str    = FormatTime(remaining);
    string   elpsd_str  = FormatTime(elapsed);

    // ── رنگ بر اساس فوریت ────────────────────────────────────────
    color    main_color = (remaining <= 30) ? InpUrgentColor : InpTextColor;

    // ── خط اصلی: Countdown ───────────────────────────────────────
    string   line1 = StringFormat("⏱  -%s  |  +%s  [%d%%]",
                                  rem_str, elpsd_str, (int)(pct * 100));
    DrawLabel(OBJ_TIME, line1, InpFontSize + 1, main_color,
              InpXOffset, InpYOffset);

    // ── Progress Bar ─────────────────────────────────────────────
    if(InpShowBar)
    {
        int    bar_width = 20;
        int    filled    = (int)MathRound(pct * bar_width);
        string bar_fill  = StringRepeat("█", filled);
        string bar_empty = StringRepeat("░", bar_width - filled);
        string line2     = StringFormat("[%s%s] %s", bar_fill, bar_empty,
                                        TimeToString(bar_close, TIME_MINUTES));
        DrawLabel(OBJ_BAR, line2, InpFontSize, InpBarColor,
                  InpXOffset, InpYOffset + InpFontSize + 6);
    }

    // ── OHLC کندل فعلی ───────────────────────────────────────────
    if(InpShowOHLC)
    {
        double o = iOpen (_Symbol, PERIOD_CURRENT, 0);
        double h = iHigh (_Symbol, PERIOD_CURRENT, 0);
        double l = iLow  (_Symbol, PERIOD_CURRENT, 0);
        double c = iClose(_Symbol, PERIOD_CURRENT, 0);
        double chg = c - o;
        string arrow = (chg >= 0) ? "▲" : "▼";
        color  chg_color = (chg >= 0) ? clrLimeGreen : clrOrangeRed;

        string line3 = StringFormat("O:%.2f  H:%.2f  L:%.2f  C:%.2f  %s%.2f",
                                    o, h, l, c, arrow, MathAbs(chg));
        DrawLabel(OBJ_OHLC, line3, InpFontSize - 1, chg_color,
                  InpXOffset, InpYOffset + (InpFontSize + 6) * 2 + 4);
    }
}

//+------------------------------------------------------------------+
string FormatTime(int total_sec)
{
    int h = total_sec / 3600;
    int m = (total_sec % 3600) / 60;
    int s = total_sec % 60;

    if(h > 0)
        return StringFormat("%02d:%02d:%02d", h, m, s);
    return StringFormat("%02d:%02d", m, s);
}

//+------------------------------------------------------------------+
string StringRepeat(const string ch, const int count)
{
    string result = "";
    for(int i = 0; i < count; i++)
        result += ch;
    return result;
}

//+------------------------------------------------------------------+
void DrawLabel(const string name, const string text,
               const int font_size, const color clr,
               const int x, const int y)
{
    if(ObjectFind(0, name) < 0)
    {
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
        ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
        ObjectSetInteger(0, name, OBJPROP_BACK,       false);
    }
    ObjectSetInteger(0, name, OBJPROP_CORNER,    InpCorner);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
    ObjectSetString (0, name, OBJPROP_TEXT,      text);
    ObjectSetString (0, name, OBJPROP_FONT,      InpFontName);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE,  font_size);
    ObjectSetInteger(0, name, OBJPROP_COLOR,     clr);
    ChartRedraw(0);
}
//+------------------------------------------------------------------+
