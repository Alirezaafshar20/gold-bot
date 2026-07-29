//+------------------------------------------------------------------+
//| SMCZoneDrawer.mq5 — draw entry zones from Python live_portfolio  |
//|                                                                  |
//| Install: copy to MQL5/Experts/, compile, attach to XAUUSD chart |
//| Python writes: MQL5/Files/smc_zones.csv                          |
//+------------------------------------------------------------------+
#property copyright "SMC Portfolio"
#property version   "1.00"
#property strict

input int    TimerSec     = 2;
input string ZoneFile     = "smc_zones.csv";
input color  LongEntryClr = clrPaleGreen;
input color  ShortEntryClr= clrLightPink;
input color  StructClr    = clrGainsboro;
input color  FilledClr    = clrKhaki;   // zone that already produced a trade
input color  DeadClr      = clrSilver;  // zone invalidated before entry
input int    EntryAlpha   = 50;    // 0..255 fill transparency
input int    StructAlpha  = 25;
input int    FilledAlpha  = 40;
input int    DeadAlpha    = 30;

//+------------------------------------------------------------------+
string NormSym(const string sym)
  {
   string s = sym;
   StringReplace(s, "@", "");
   StringToUpper(s);
   return s;
  }

//+------------------------------------------------------------------+
bool SymMatch(const string chart_sym, const string row_sym)
  {
   return NormSym(chart_sym) == NormSym(row_sym);
  }

//+------------------------------------------------------------------+
void DeleteZoneObjects()
  {
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, "SMC_Z_") == 0)
         ObjectDelete(0, name);
     }
  }

//+------------------------------------------------------------------+
color WithAlpha(color base, int alpha)
  {
   return (color)((alpha << 24) | (base & 0xFFFFFF));
  }

//+------------------------------------------------------------------+
bool DrawRect(const string name, datetime t1, datetime t2,
              double p_top, double p_bot, color clr, int alpha, string tip)
  {
   if(p_top < p_bot)
     {
      double tmp = p_top;
      p_top = p_bot;
      p_bot = tmp;
     }
   if(t2 <= t1)
      t2 = t1 + 15 * 60;
   if(!ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p_top, t2, p_bot))
     {
      ObjectDelete(0, name);
      if(!ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p_top, t2, p_bot))
         return false;
     }
   ObjectSetInteger(0, name, OBJPROP_COLOR, WithAlpha(clr, alpha));
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
   return true;
  }

//+------------------------------------------------------------------+
void SyncFromFile()
  {
   string chart_sym = Symbol();
   DeleteZoneObjects();

   int handle = FileOpen(ZoneFile, FILE_READ|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
     {
      Comment("SMCZoneDrawer\nfile: ", ZoneFile,
              "\nERROR: file not found in MQL5\\Files (err ", GetLastError(), ")",
              "\nIs the Python bot running and writing this file?");
      return;
     }
   int rows_total = 0, rows_sym = 0, drawn = 0;

   // Skip the header LINE. FileReadString reads ONE FIELD in CSV mode,
   // not one line — skipping a single field here shifted every column
   // by one and broke the symbol match (nothing was ever drawn).
   if(!FileIsEnding(handle))
      do { FileReadString(handle); }
      while(!FileIsLineEnding(handle) && !FileIsEnding(handle));

   while(!FileIsEnding(handle))
     {
      string kind      = FileReadString(handle);
      if(kind != "entry" && kind != "struct" && kind != "filled" && kind != "dead")
        {
         // unknown first column — resync to the next line
         while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
            FileReadString(handle);
         continue;
        }
      string sym       = FileReadString(handle);
      string zone_id   = FileReadString(handle);
      string direction = FileReadString(handle);
      long   t_start   = StringToInteger(FileReadString(handle));
      long   t_end     = StringToInteger(FileReadString(handle));
      double p_lo      = StringToDouble(FileReadString(handle));
      double p_hi      = StringToDouble(FileReadString(handle));
      string p_distal  = FileReadString(handle); // unused, keeps column align
      string tag       = FileReadString(handle);

      rows_total++;
      if(!SymMatch(chart_sym, sym))
         continue;
      rows_sym++;

      datetime t1 = (datetime)t_start;
      datetime t2 = (datetime)t_end;
      bool is_long = (direction == "long");
      string prefix = "SMC_Z_" + zone_id + "_";

      if(kind == "struct")
        {
         string n = prefix + "STRUCT";
         if(DrawRect(n, t1, t2, p_hi, p_lo, StructClr, StructAlpha,
                     tag + " structural zone"))
            drawn++;
        }
      else if(kind == "entry")
        {
         string n = prefix + "ENTRY";
         color c = is_long ? LongEntryClr : ShortEntryClr;
         if(DrawRect(n, t1, t2, p_hi, p_lo, c, EntryAlpha,
                     tag + " entry zone [" + DoubleToString(p_lo, _Digits)
                     + " - " + DoubleToString(p_hi, _Digits) + "]"))
            drawn++;
        }
      else if(kind == "filled")
        {
         string n = prefix + "FILLED";
         if(DrawRect(n, t1, t2, p_hi, p_lo, FilledClr, FilledAlpha,
                     tag + " FILLED " + direction + " [" + DoubleToString(p_lo, _Digits)
                     + " - " + DoubleToString(p_hi, _Digits) + "]"))
            drawn++;
        }
      else if(kind == "dead")
        {
         string n = prefix + "DEAD";
         if(DrawRect(n, t1, t2, p_hi, p_lo, DeadClr, DeadAlpha,
                     tag + " INVALIDATED " + direction + " [" + DoubleToString(p_lo, _Digits)
                     + " - " + DoubleToString(p_hi, _Digits) + "]"))
            drawn++;
        }
     }
   FileClose(handle);
   Comment("SMCZoneDrawer  ", TimeToString(TimeLocal(), TIME_SECONDS),
           "\nfile: ", ZoneFile,
           "\nrows: ", rows_total, "  for ", chart_sym, ": ", rows_sym,
           "  drawn: ", drawn);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   Print("SMCZoneDrawer started on ", Symbol(), " — reading MQL5\\Files\\", ZoneFile);
   EventSetTimer(TimerSec);
   SyncFromFile();
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   DeleteZoneObjects();
   Comment("");
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   SyncFromFile();
  }
//+------------------------------------------------------------------+
