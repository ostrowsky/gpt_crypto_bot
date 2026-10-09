"""Versioned outcome contract shared by research and new forward scorecards."""
from __future__ import annotations
from datetime import datetime,time,timedelta,timezone
from zoneinfo import ZoneInfo
import math

TZ=ZoneInfo('Europe/Budapest')
CONTRACT=dict(id='mission-global20-watchlist-localday-v1',timezone='Europe/Budapest',top_n=20,
    minimum_day_quote_volume=1_000_000.,early_remaining_ratio=.35,capacity=10,
    cutoff='next_local_midnight',precision_unit='unique_day_symbol_first_BUY')
EXCLUDED={'USDCUSDT','BUSDUSDT','FDUSDUSDT','TUSDUSDT','DAIUSDT','USDPUSDT','USDSUSDT','EURUSDT','GBPUSDT','TRYUSDT','BRLUSDT'}


def target_symbol(symbol):
    return symbol.endswith('USDT') and symbol not in EXCLUDED and not symbol.endswith(('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT'))


def window(day):
    d=datetime.fromisoformat(day).date()
    return tuple(int(datetime.combine(x,time(),tzinfo=TZ).timestamp()*1000) for x in (d,d+timedelta(days=1)))


def local_day(clock):return datetime.fromtimestamp(clock/1000,timezone.utc).astimezone(TZ).date().isoformat()


def daily_label(day,bars,watchlist,available_at,coverage):
    """Future outcome only. Reject incomplete/invalid native aggregation or immature day."""
    lo,hi=window(day)
    if available_at<hi:return dict(day=day,state='PENDING',available_at=hi,leaders=None)
    values={}
    for symbol,row in bars.items():
        if not target_symbol(symbol):continue
        if row[0]!=lo or row[6]!=hi-1:raise ValueError('daily timezone/close boundary mismatch')
        opening,closing,volume=float(row[1]),float(row[4]),float(row[7])
        if not all(math.isfinite(x) for x in (opening,closing,volume)) or min(opening,closing)<=0 or volume<0:raise ValueError('invalid daily outcome')
        values[symbol]=dict(open=opening,close=closing,quote_volume=volume,return_pct=100*(closing/opening-1))
    eligible=[s for s,v in values.items() if v['quote_volume']>=CONTRACT['minimum_day_quote_volume']]
    ranked=sorted(eligible,key=lambda s:(values[s]['return_pct'],s),reverse=True)[:CONTRACT['top_n']]
    return dict(contract=CONTRACT['id'],day=day,state='OBSERVED_UNIVERSE' if coverage['missing'] else 'COMPLETE_REQUESTED_UNIVERSE',
        historical_PIT_certified=coverage['historical_PIT_certified'],available_at=hi,
        leaders=[s for s in ranked if s in watchlist],exchange_top=ranked,values=values,coverage=coverage)


def remaining(opening,closing,price):
    if not all(math.isfinite(x) and x>0 for x in (opening,closing,price)):return None
    return max(0.,min(1.5,(closing-price)/(closing-opening))) if closing>opening else None


def candidate_target(symbol,clock,price,label,fit_at=None):
    """Label explicitly absent until maturity; inference must not call this function."""
    if label['day']!=local_day(clock):raise ValueError('wrong label day')
    if fit_at is not None and label['available_at']>=fit_at:return None
    if label['state']=='PENDING' or symbol not in label['values']:return None
    row=label['values'][symbol];leader=symbol in label['leaders'];capture=remaining(row['open'],row['close'],price)
    return dict(leader=int(leader),early_leader=int(leader and capture is not None and capture>=.35),
        available_at=label['available_at'],capture=capture,contract=CONTRACT['id'])
