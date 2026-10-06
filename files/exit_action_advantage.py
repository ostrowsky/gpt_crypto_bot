"""Causal concrete soft-SELL deferral; active strategy remains unchanged."""
from dataclasses import asdict
import numpy as np
import replay_backtest as rb
from impulse_entry_catboost import features,PARAMETERS,BAR
from datetime import datetime,time,timedelta,timezone
from capacity_catboost import TZ

MODES=('retest','breakout','trend','impulse_speed','strong_trend','impulse')
MODEL_PARAMETERS=dict(PARAMETERS,depth=3)
THRESHOLD=.05


def cohorts(clock,available,valid,at,start):
    first=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(at/1000,timezone.utc).astimezone(TZ).date()
    cut=int(datetime.combine(first+timedelta(days=int((last-first).days*.8)),time(),tzinfo=TZ).timestamp()*1000)
    return valid&(clock<cut)&(available<cut),valid&(clock>=cut)&(available<at),cut


def position_features(trade,raw15,clock,origin_price):
    x=features(raw15,clock,trade.tf)
    if x is None or origin_price<=0 or trade.entry_price<=0:return None
    closes=raw15['t']+BAR
    path=raw15[(closes>trade.entry_ts)&(closes<=clock)]
    if not len(path) or not np.array_equal(path['t']+BAR,np.arange(trade.entry_ts+BAR,clock+1,BAR)):
        return None
    if not np.isfinite(path['h']).all() or (path['h']<=0).any():return None
    peak=max(trade.entry_price,float(path['h'].max()))
    pnl=100*(origin_price/trade.entry_price-1);mfe=100*(peak/trade.entry_price-1)
    elapsed=(clock-trade.entry_ts)/60000
    remaining=max(0,trade.max_hold_bars*rb.BAR_MS[trade.tf]/60000-elapsed)
    distance=100*(origin_price-trade.trail_stop)/origin_price if trade.trail_stop>0 else 0.
    result=np.r_[x,pnl,mfe,max(0,mfe-pnl),elapsed,remaining,distance,[float(trade.mode==m) for m in MODES]]
    return result if np.isfinite(result).all() else None


def deadline(data,idx,tf):return int(data['t'][idx])+2*rb.BAR_MS[tf]


def valid_origin(trade,data,idx,clock,price):
    return (idx>=0 and int(data['t'][idx])+rb.BAR_MS[trade.tf]==clock
            and np.isclose(float(data['c'][idx]),price,rtol=1e-10,atol=1e-12))


def cash_advantage(now,later,fee,slip):
    if not now>0 or not later>0 or not 0<=fee<10000 or not 0<=slip<10000:
        raise ValueError('invalid action prices/costs')
    return 100*(later/now-1)*(1-fee/10000)*(1-slip/10000)


def label_action(trade,cache,end,fee,slip,progress):
    """Concrete cloned one-bar action, protected by original hard exits, no maxima."""
    data,feat=cache[trade.sym,trade.tf];origin=trade.exit_ts
    idx=rb._find_last_closed_candle_index(data['t'],origin,rb.BAR_MS[trade.tf])
    if idx is None or not valid_origin(trade,data,idx,origin,trade.exit_price):return None
    stop=deadline(data,idx,trade.tf)
    if stop>end:return None
    clone=rb.ReplayTrade(**asdict(trade));clone.exit_ts=0;clone.exit_price=0.;clone.exit_reason=''
    micro=cache.get((trade.sym,'15m')) if trade.tf=='1h' else None
    for at in range(origin+BAR,stop+1,BAR):
        j=rb._find_last_closed_candle_index(data['t'],at,rb.BAR_MS[trade.tf])
        if j is None:return None
        reason=progress(clone,data,feat,j,ts_ms=at,micro_pack=micro)
        if reason and not rb._is_weak_exit_reason(reason) or at>=stop:
            exit_at=clone.exit_ts or at;price=clone.exit_price or float(data['c'][j])
            if not origin<exit_at<=stop:return None
            return {'available_at':stop,'hold_exit_at':exit_at,'hold_exit_price':price,
                    'hold_reason':reason if reason and not rb._is_weak_exit_reason(reason) else 'one_bar_deadline',
                    'advantage':cash_advantage(trade.exit_price,price,fee,slip)}
    raise AssertionError('missing concrete action endpoint')


class ExitPolicy:
    """Process-local wrapper; model sees as-of features, never forward action labels."""
    def __init__(self,progress,cache,predictor):
        self.progress=progress;self.cache=cache;self.predictor=predictor
        self.active={};self.considered=set();self.decisions=[];self.outcomes=[]

    def __call__(self,trade,data,feat,idx,**kwargs):
        at=int(kwargs['ts_ms']);key=(trade.sym,trade.tf,trade.entry_ts)
        reason=self.progress(trade,data,feat,idx,**kwargs)
        state=self.active.get(key)
        if state is not None:
            if reason and not rb._is_weak_exit_reason(reason):
                self.outcomes.append({'key':list(key),'at':at,'kind':'HARD','reason':reason});del self.active[key]
                return reason
            if at>=state['deadline']:
                self.outcomes.append({'key':list(key),'at':at,'kind':'TIMEOUT'});del self.active[key]
                return state['origin_reason']+' [action one-bar deadline]'
            return None
        if not rb._is_weak_exit_reason(reason) or key in self.considered:return reason
        self.considered.add(key);price=float(data['c'][idx])
        x=(position_features(trade,self.cache[trade.sym,'15m'][0],at,price)
           if valid_origin(trade,data,idx,at,price) else None)
        score,model_id=self.predictor(at,x)
        defer=bool(score is not None and np.isfinite(score) and score>THRESHOLD)
        until=deadline(data,idx,trade.tf)
        self.decisions.append({'key':list(key),'at':at,'origin_price':price,'origin_reason':reason,
            'entry_price':trade.entry_price,'trail_stop':trade.trail_stop,'max_hold_bars':trade.max_hold_bars,
            'mode':trade.mode,'features':None if x is None else x.tolist(),'prediction':score,
            'model_id':model_id,'defer':defer,'deadline':until})
        if defer:self.active[key]={'deadline':until,'origin_reason':reason};return None
        return reason
