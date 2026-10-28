"""All-USDT scope, composite ownership and conservative portfolio accounting."""
from copy import deepcopy
from decimal import Decimal,Context,localcontext
import hashlib,re,time
from pathlib import Path

from .adapter import _sha,AdapterError
from .execution_pilot import PilotJournal,KNOWN,amount,text,execution_plan,fail,is_usdt_symbol
from .universe import PERMISSIONS


def usdt_catalog(metadata,permissions,*,can_trade):
    if not isinstance(metadata,dict) or not isinstance(metadata.get('symbols'),list):fail('USDT_METADATA_INVALID')
    if can_trade is not True or not permissions or not set(permissions)<=set(PERMISSIONS):fail('USDT_ACCOUNT_NOT_READY')
    result=[];seen={}
    for m in metadata['symbols']:
        symbol=m.get('symbol')
        if not isinstance(symbol,str):fail('USDT_METADATA_INVALID')
        if symbol in seen:
            if seen[symbol]!=m:fail('USDT_METADATA_CONFLICT')
            continue
        seen[symbol]=m;groups=m.get('permissionSets');reasons=[]
        structure=isinstance(groups,list) and bool(groups) and all(isinstance(g,list) and bool(g)
            and all(isinstance(p,str) for p in g) for g in groups)
        permitted=structure and all(any(p in permissions for p in g) for g in groups)
        monitored=(m.get('quoteAsset')=='USDT' and m.get('status')=='TRADING'
            and m.get('isSpotTradingAllowed') is True and permitted)
        if not monitored:reasons.append('OUTSIDE_AVAILABLE_USDT_SCOPE')
        if m.get('ocoAllowed') is not True:reasons.append('OCO_UNSUPPORTED')
        if m.get('quoteOrderQtyMarketAllowed') is not True:reasons.append('QUOTE_MARKET_UNSUPPORTED')
        if not {'MARKET','LIMIT_MAKER','STOP_LOSS'}<=set(m.get('orderTypes',[])):reasons.append('ORDER_TYPES_UNSUPPORTED')
        if not isinstance(m.get('filters'),list) or any(f.get('filterType') not in KNOWN for f in m['filters']):reasons.append('FILTER_UNSUPPORTED')
        if monitored and (not is_usdt_symbol(symbol) or m.get('baseAsset','')+'USDT'!=symbol):
            reasons.append('INSTRUMENT_SCHEMA_UNSUPPORTED')
        result.append(dict(symbol=symbol,base_asset=m.get('baseAsset'),monitored=bool(monitored),
            execution_supported=not reasons,reasons=reasons,metadata=deepcopy(m)))
    return sorted(result,key=lambda r:r['symbol'])


def portfolio_inventory(rows,owned_keys,base_by_symbol):
    with localcontext(Context(prec=50)):
        deltas={};fees={};seen={}
        for r in rows:
            symbol=r['symbol'];key=(symbol,r['trade_id'])
            if (symbol,r['order_id']) not in owned_keys or symbol not in base_by_symbol:fail('PILOT_FOREIGN_FILL')
            if key in seen:
                if seen[key]!=r:fail('PILOT_FILL_CONFLICT')
                continue
            seen[key]=r;base=base_by_symbol[symbol]
            q,quote,fee=(amount(r[k]) for k in ('quantity','quote_quantity','commission_quantity'))
            if q<=0 or quote<=0 or r['side'] not in ('BUY','SELL'):fail('PILOT_INVALID')
            sign=Decimal(1) if r['side']=='BUY' else Decimal(-1)
            deltas[base]=deltas.get(base,Decimal(0))+sign*q
            deltas['USDT']=deltas.get('USDT',Decimal(0))-sign*quote
            if fee:
                asset=r['commission_asset']
                if asset not in (base,'USDT'):fail('PILOT_FEE_ASSET_UNSUPPORTED')
                deltas[asset]=deltas.get(asset,Decimal(0))-fee;fees[asset]=fees.get(asset,Decimal(0))+fee
        if any(v<0 for k,v in deltas.items() if k!='USDT'):fail('PILOT_UNOWNED_SALE')
        return dict(deltas={k:text(v) for k,v in deltas.items()},fees={k:text(v) for k,v in fees.items()},fills_n=len(seen))


def portfolio_value(deltas,marks):
    with localcontext(Context(prec=50)):
        result=Decimal('100')
        for asset,value in deltas.items():
            value=Decimal(value)
            if not value.is_finite():fail('PILOT_INVALID')
            if not value:continue
            if asset=='USDT':result+=value;continue
            if asset not in marks:fail('PILOT_MARK_MISSING')
            bid,ask=(amount(v) for v in marks[asset])
            if bid<=0 or ask<bid:fail('PILOT_MARK_INVALID')
            result+=value*(bid+ask)/2
        return result


def select_round_robin(catalog,quotes,attempt_counts,*,validated):
    candidates=[]
    for r in catalog:
        symbol=r['symbol'];q=quotes.get(symbol)
        if not r['monitored'] or not r['execution_supported'] or symbol not in validated or q is None:continue
        try:
            bid,ask=amount(q['bidPrice']),amount(q['askPrice'])
            if bid<=0 or ask<bid or (ask/bid-1)*10000>20:continue
            if min(bid*amount(q['bidQty']),ask*amount(q['askQty']))<30:continue
            execution_plan(r['metadata'],q['bidPrice'],q['askPrice'],text((bid+ask)/2),symbol=symbol)
        except (AdapterError,KeyError,ValueError,TypeError):continue
        candidates.append((attempt_counts.get(symbol,0),symbol))
    return min(candidates)[1] if candidates else None


IDENTIFIERS={'symbol','base_asset','commission_asset','asset','commissionAsset','baseAsset','quoteAsset'}


def encode_identifiers(value):
    if isinstance(value,list):return [encode_identifiers(v) for v in value]
    if isinstance(value,dict):
        result={}
        for k,v in value.items():
            if k in IDENTIFIERS and isinstance(v,str) and 1<=len(v)<=24 and v.isalnum() and v==v.upper():
                result[k+'_codepoints']=[ord(c) for c in v]
            elif k=='fees' and isinstance(v,dict):
                result['fee_rows']=[encode_identifiers(dict(asset=a,amount=b)) for a,b in sorted(v.items())]
            else:result[k]=encode_identifiers(v)
        return result
    return value


def decode_identifiers(value):
    if isinstance(value,list):return [decode_identifiers(v) for v in value]
    if isinstance(value,dict):
        result={}
        for k,v in value.items():
            if k.endswith('_codepoints') and k[:-11] in IDENTIFIERS:
                try:s=''.join(chr(c) for c in v)
                except (TypeError,ValueError):fail('PILOT_INTEGRITY')
                if not 1<=len(s)<=24 or not s.isalnum() or s!=s.upper():fail('PILOT_INTEGRITY')
                result[k[:-11]]=s
            elif k=='fee_rows':
                rows=decode_identifiers(v);result['fees']={r['asset']:r['amount'] for r in rows}
            else:result[k]=decode_identifiers(v)
        return result
    return value


class UsdtPilotJournal(PilotJournal):
    def __init__(self,*args,clock_ms=None,**kwargs):
        self._usdt_scope=None;self._now_ms=clock_ms or (lambda:time.time_ns()//1000000)
        super().__init__(*args,**kwargs);self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def _save(self,op,value):return super()._save(op,encode_identifiers(value))
    def _apply(self,event):
        event=decode_identifiers(event)
        if event['operation']=='USDT_UNIVERSE':self._usdt_scope=deepcopy(event['value'])
        else:super()._apply(event)
    def authorize_universe(self,catalog,*,received_at_ms,source_sha256,validated_symbols):
        self._ready();now=self._now_ms()
        if not 0<=now-received_at_ms<=900000 or not re.fullmatch('[0-9a-f]{64}',source_sha256):fail('USDT_SCOPE_STALE')
        rows=[dict(symbol=r['symbol'],base_asset=r['base_asset'],eligible=r['monitored'] and r['execution_supported'],
            replay_complete=r['symbol'] in set(validated_symbols)) for r in catalog if r['monitored']]
        self._save('USDT_UNIVERSE',dict(received_at_ms=received_at_ms,source_sha256=source_sha256,rows=rows))
    def _symbol_allowed(self,symbol,*,closing=False):
        if not is_usdt_symbol(symbol):return False
        if closing:return any(s==symbol for s,_ in self.owned_order_keys())
        now=self._now_ms()
        if not self._binding['start_ms']<=now<self._binding['expires_ms'] or self._usdt_scope is None:return False
        if not 0<=now-self._usdt_scope['received_at_ms']<=900000:return False
        return any(r['symbol']==symbol and r['eligible'] and r['replay_complete'] for r in self._usdt_scope['rows'])
