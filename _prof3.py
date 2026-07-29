import time, sys
sys.stdout.reconfigure(encoding='utf-8')
import strategy as S, mt5_data as M, symbol_profiles as P
mt5=M.connect()
sym=P.resolve_symbol_for_profile('XAUUSD',mt5)
_,prof=P.get_profile('XAUUSD')
sig=M.fetch_bars(sym,'M15',count=99999,mt5=mt5)
h4=M.fetch_bars(sym,'H4',count=99999,mt5=mt5); h1=M.fetch_bars(sym,'H1',count=99999,mt5=mt5)
M.shutdown(mt5)
B=S.Bars(sig); ctx=S.M1Ctx(sig,sig.index); H=S.prepare_htf_context({'H4':h4,'H1':h1})
o=P.apply_profile_to_settings(S.dense_plus_settings(wide=True),prof,2.0,min_sl_override=0.5)
META=('spread_usd','nds_mode','nds_extended','balanced','balanced_plus','dense','medium','dense_plus','dense_wide','spike_mode','fib_spike_exempt','spike_params')
for rule in ['HARM_BAT','WOLFE_B','WYCK','AB_2BAR_L','EW_W2']:
    o['enabled']=[rule]
    k={x:o[x] for x in o if x not in META}; k['htf_context']=H; k['detector_params']=P.merge_params(S.DEFAULT_PARAMS,prof); k['max_concurrent']=2
    t0=time.time()
    try:
        tr=S.run_backtest(sig,None,B=B,ctx=ctx,tf_min=15,**k); print(f'{rule:12s} {len(tr):4d} tr  {time.time()-t0:6.1f}s', flush=True)
    except Exception as e:
        print(f'{rule:12s} ERR {e}', flush=True)
