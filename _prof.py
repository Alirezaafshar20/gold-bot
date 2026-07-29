import time, sys, cProfile, pstats, io
sys.stdout.reconfigure(encoding='utf-8')
import strategy as S, mt5_data as M, symbol_profiles as P
mt5=M.connect()
sym=P.resolve_symbol_for_profile('XAUUSD',mt5)
_,prof=P.get_profile('XAUUSD')
sig=M.fetch_bars(sym,'M15',count=20000,mt5=mt5)
htf={tf:M.fetch_bars(sym,tf,count=20000,mt5=mt5) for tf in ('H4','H1')}
M.shutdown(mt5)
print('bars',len(sig))
B=S.Bars(sig); ctx=S.M1Ctx(sig,sig.index); H=S.prepare_htf_context(htf)
o=P.apply_profile_to_settings(S.dense_plus_settings(wide=True),prof,2.0,min_sl_override=0.5)
META=('spread_usd','nds_mode','nds_extended','balanced','balanced_plus','dense','medium','dense_plus','dense_wide','spike_mode','fib_spike_exempt','spike_params')
o['enabled']=['OB']
k={x:o[x] for x in o if x not in META}; k['htf_context']=H; k['detector_params']=P.merge_params(S.DEFAULT_PARAMS,prof); k['max_concurrent']=2
pr=cProfile.Profile(); pr.enable()
t0=time.time(); tr=S.run_backtest(sig,None,B=B,ctx=ctx,tf_min=15,**k); dt=time.time()-t0
pr.disable()
print(f'OB {len(tr)} trades  {dt:.1f}s for 20000 bars')
s=io.StringIO(); ps=pstats.Stats(pr,stream=s).sort_stats('cumulative'); ps.print_stats(18)
print(s.getvalue())
