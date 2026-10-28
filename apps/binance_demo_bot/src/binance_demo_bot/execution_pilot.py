"""Capped Spot Demo execution qualification, separate from ordinary entry authority."""
from copy import deepcopy
from decimal import Decimal,Context,localcontext,ROUND_DOWN,ROUND_UP
import hashlib
import json
from pathlib import Path
import re
import uuid

from .adapter import AdapterError,DemoHttpTransport,_sha,_json,sign_rest
from .interaction_audit import AuditJournal,sanitize
from .journal import WriterBusy

PAIRS={('POST','/api/v3/order'),('POST','/api/v3/orderList/oco'),
       ('DELETE','/api/v3/order'),('DELETE','/api/v3/orderList')}
KNOWN={'PRICE_FILTER','LOT_SIZE','MARKET_LOT_SIZE','NOTIONAL','MIN_NOTIONAL','PERCENT_PRICE_BY_SIDE',
       'PERCENT_PRICE','MAX_NUM_ORDERS','MAX_NUM_ALGO_ORDERS','MAX_NUM_ORDER_LISTS',
       'MAX_NUM_ORDER_AMENDS','ICEBERG_PARTS','TRAILING_DELTA','MAX_POSITION'}


def fail(reason):raise AdapterError(reason)
def amount(value):
    if not isinstance(value,(str,Decimal)):fail('PILOT_INVALID')
    try:
        d=Decimal(value)
        if not d.is_finite() or d<0 or len(d.as_tuple().digits)>38 or abs(d.as_tuple().exponent)>36:fail('PILOT_INVALID')
        return d
    except ArithmeticError:fail('PILOT_INVALID')
def text(value):
    value=format(value,'f')
    return value.rstrip('0').rstrip('.') if '.' in value else value
def rounded(value,step,up=False):
    return (value/step).to_integral_value(rounding=ROUND_UP if up else ROUND_DOWN)*step if step else value

def is_usdt_symbol(value):
    return (isinstance(value,str) and value.endswith('USDT') and 4<len(value)<=24
        and value.isalnum() and value==value.upper())


def execution_plan(info,bid,ask,average,*,owned='0',fill_price=None,symbol='BTCUSDT'):
    with localcontext(Context(prec=50)):
        bid,ask,avg,owned=(amount(x) for x in (bid,ask,average,owned))
        if bid<=0 or ask<bid or avg<=0:fail('PILOT_INVALID')
        if (not is_usdt_symbol(symbol) or not isinstance(info.get('baseAsset'),str)
            or info.get('symbol')!=symbol or info.get('quoteAsset')!='USDT'
            or info.get('baseAsset')+'USDT'!=symbol or info.get('status')!='TRADING'
            or info.get('isSpotTradingAllowed') is not True or info.get('ocoAllowed') is not True
            or info.get('quoteOrderQtyMarketAllowed') is not True
            or not {'MARKET','LIMIT_MAKER','STOP_LOSS'}<=set(info.get('orderTypes',[]))):fail('PILOT_FILTER_UNSUPPORTED')
        filters={r['filterType']:r for r in info['filters']}
        if len(filters)!=len(info['filters']) or set(filters)-KNOWN:fail('PILOT_FILTER_UNSUPPORTED')
        try:
            for name,key,required in (('MAX_NUM_ORDERS','maxNumOrders',2),
                    ('MAX_NUM_ALGO_ORDERS','maxNumAlgoOrders',1),('MAX_NUM_ORDER_LISTS','maxNumOrderLists',1)):
                if name in filters and int(filters[name][key])<required:fail('PILOT_FILTER_INVALID')
            lot,pf=filters['LOT_SIZE'],filters['PRICE_FILTER']
            step,tick=amount(lot['stepSize']),amount(pf['tickSize'])
            if step<=0 or tick<=0:fail('PILOT_FILTER_INVALID')
            price=amount(fill_price) if fill_price is not None else ask
            tp=rounded(price*Decimal('1.005'),tick,True);sl=rounded(price*Decimal('.995'),tick)
            qty=rounded(owned*Decimal('.998'),step) if owned else Decimal('0')
            buy_qty=rounded(Decimal('30')/ask,step)
            for q in (buy_qty,qty) if owned else (buy_qty,):
                if not amount(lot['minQty'])<=q<=amount(lot['maxQty']):fail('PILOT_FILTER_INVALID')
            for p in (tp,sl):
                if (amount(pf['minPrice']) and p<amount(pf['minPrice'])) or (amount(pf['maxPrice']) and p>amount(pf['maxPrice'])):fail('PILOT_FILTER_INVALID')
            if not sl<bid<=ask<tp:fail('PILOT_FILTER_INVALID')
            notional=filters.get('NOTIONAL',filters.get('MIN_NOTIONAL'))
            if not notional:fail('PILOT_FILTER_UNSUPPORTED')
            if Decimal('30')<amount(notional['minNotional']):fail('PILOT_FILTER_INVALID')
            if owned and qty*sl<amount(notional['minNotional']):fail('PILOT_FILTER_INVALID')
            if 'maxNotional' in notional and Decimal('30')>amount(notional['maxNotional']):fail('PILOT_FILTER_INVALID')
            ml=filters.get('MARKET_LOT_SIZE')
            if ml and (buy_qty<amount(ml['minQty']) or buy_qty>amount(ml['maxQty'])):fail('PILOT_FILTER_INVALID')
            side=filters.get('PERCENT_PRICE_BY_SIDE')
            if side:
                if not avg*amount(side['bidMultiplierDown'])<=ask<=avg*amount(side['bidMultiplierUp']):fail('PILOT_FILTER_INVALID')
                if not all(avg*amount(side['askMultiplierDown'])<=p<=avg*amount(side['askMultiplierUp']) for p in (tp,sl)):fail('PILOT_FILTER_INVALID')
            generic=filters.get('PERCENT_PRICE')
            if generic and not all(avg*amount(generic['multiplierDown'])<=p<=avg*amount(generic['multiplierUp']) for p in (ask,tp,sl)):fail('PILOT_FILTER_INVALID')
            if 'MAX_POSITION' in filters and owned+buy_qty>amount(filters['MAX_POSITION']['maxPosition']):fail('PILOT_FILTER_INVALID')
            return dict(quantity=text(qty),take_profit=text(tp),stop_loss=text(sl),quote='30')
        except (KeyError,TypeError):fail('PILOT_FILTER_UNSUPPORTED')


def fill_inventory(rows,owned_ids):
    with localcontext(Context(prec=50)):
        seen={};base=cash=Decimal('0');fees={}
        for row in rows:
            if row['symbol']!='BTCUSDT' or row['order_id'] not in owned_ids:fail('PILOT_FOREIGN_FILL')
            key=(row['symbol'],row['trade_id'])
            if key in seen:
                if seen[key]!=row:fail('PILOT_FILL_CONFLICT')
                continue
            seen[key]=row
            qty,quote,fee=(amount(row[k]) for k in ('quantity','quote_quantity','commission_quantity'))
            if qty<=0 or quote<=0 or row['side'] not in ('BUY','SELL'):fail('PILOT_INVALID')
            sign=Decimal('1') if row['side']=='BUY' else Decimal('-1')
            base+=sign*qty;cash-=sign*quote
            if fee:
                asset=row['commission_asset'];fees[asset]=fees.get(asset,Decimal('0'))+fee
                if asset=='BTC':base-=fee
                elif asset=='USDT':cash-=fee
                else:fail('PILOT_FEE_ASSET_UNSUPPORTED')
        return dict(base=text(base),cash_delta=text(cash),fills_n=len(seen),fees={k:text(v) for k,v in fees.items()})


class DemoMatchingTransport(DemoHttpTransport):
    def request(self,method,path,query,headers,timeout_s):
        if (method,path) not in PAIRS:fail('FORBIDDEN_ENDPOINT')
        return self._request_https(method,path,query,headers,timeout_s,matching_engine=True)


class MatchingGateway:
    def __init__(self,client):self.client=client;self.transport=DemoMatchingTransport()
    def mutate(self,method,path,params):
        c=self.client
        if c.audit is None or c.audit._state!='READY':fail('AUDIT_UNAVAILABLE')
        c._guard(1);timestamp=c.clock.timestamp_ms();deadline=c.clock.monotonic_ns()+15_000_000_000
        signed=dict(params,timestamp=timestamp,recvWindow=5000)
        query=sign_rest(signed,c._credentials.api_secret)
        response,*rest=c._transport_attempt(path,query,{'X-MBX-APIKEY':c._credentials.api_key},signed,
            True,deadline,uuid.uuid4().hex,0,method=method,transport=self.transport,matching_engine=True)
        c._observe_weight(response.headers,rest[3])
        if response.status in (418,429):c.cooldown_until_ns=c.clock.monotonic_ns()+60_000_000_000;fail('PILOT_QUERY_REQUIRED')
        data=_json(response.body)
        if response.status!=200:
            if isinstance(data,dict) and data.get('code') in (-1013,-2010,-2014,-2015,-1022,-1100,-1102,-1111):fail('PILOT_REJECTED')
            fail('PILOT_QUERY_REQUIRED')
        return data
    def query_action(self,method,path,params):
        if 'orderList' in path:
            try:raw,_=self.client._request('/api/v3/orderList',{'origClientOrderId':params['listClientOrderId']},signed=True,weight=4);return raw
            except AdapterError:return None
        try:
            query={'symbol':params['symbol']}
            if 'orderId' in params:query['orderId']=params['orderId']
            else:query['origClientOrderId']=params.get('newClientOrderId',params.get('origClientOrderId'))
            raw,_=self.client._request('/api/v3/order',query,signed=True,weight=4)
            return raw
        except AdapterError:return None
    def close(self):self.transport.close()


class PilotJournal:
    def __init__(self,paths,*,account_identity,start_ms,expires_ms,initial_balances,enabled=False,opening_sha256=None):
        self._audit=None;self._binding=None;self._actions={};self._state={};self._outcomes=[];self.enabled=enabled
        self._source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if type(enabled) is not bool or not re.fullmatch('[0-9a-f]{64}',account_identity):fail('PILOT_INVALID')
        if type(start_ms) is not int or type(expires_ms) is not int or not 0<expires_ms-start_ms<=86_400_000:fail('PILOT_INVALID')
        binding=dict(account_identity=account_identity,start_ms=start_ms,expires_ms=expires_ms,
            initial_balances=initial_balances,opening_sha256=opening_sha256,mode='EXECUTION_QUALIFICATION',
            quote='30',budget='100',max_cycles=8,loss_stop='1',max_exposure='32',max_dust='2')
        try:
            self._audit=AuditJournal(paths,account_alias='demo-primary',episode_id='opening-candidate-20261027',
                scope_id='whole-account-opening',database='pilot/journal.sqlite',max_bytes=256*1024*1024)
            for row in self._audit.read(limit=1000):
                if row['kind']=='DECISION':self._apply(row['data'])
            after=1000
            while self._audit._last_seq>after:
                for row in self._audit.read(after_seq=after,limit=1000):
                    if row['kind']=='DECISION':self._apply(row['data'])
                after+=1000
            if self._binding is None:self._save('BIND',binding)
            elif self._binding!=binding:fail('PILOT_IDENTITY')
            for action in list(self._actions):
                if self._actions[action]['state']=='SEND_STARTED':self._save('UNKNOWN',dict(action=action))
        except BaseException as error:
            self.close()
            if isinstance(error,WriterBusy):fail('PILOT_BUSY')
            raise
    def _ready(self):
        if self._audit is None or self._audit._closed or self._audit._state!='READY':fail('PILOT_UNAVAILABLE')
    def _apply(self,event):
        op,value=event['operation'],event['value']
        if op=='BIND':
            if self._binding is not None:fail('PILOT_INTEGRITY')
            self._binding=deepcopy(value)
        elif op=='PREPARED':self._actions[value['action']]=dict(value,state='PREPARED',response=None)
        elif op in ('START','UNKNOWN','REJECT','ACK'):
            item=self._actions[value['action']];item['state']={'START':'SEND_STARTED','UNKNOWN':'QUERY_REQUIRED','REJECT':'REJECTED','ACK':'ACKNOWLEDGED'}[op]
            if op=='ACK':item['response']=deepcopy(value['response'])
        elif op=='STATE':self._state=deepcopy(value)
        elif op=='OUTCOME':self._outcomes.append(deepcopy(value))
        elif op!='DECISION':fail('PILOT_INTEGRITY')
    def _save(self,op,value):
        event=dict(operation=op,value=value,source_code_sha256=self._source)
        if sanitize(event)!=event:fail('PILOT_INVALID')
        self._audit.append('DECISION',event);self._apply(event)
    def client_id(self,action,leg=''):return 'bq_'+_sha(dict(binding=self._binding,action=action,leg=leg))[:32]
    def _valid_response(self,item,data):
        if not isinstance(data,dict) or data.get('symbol')!=item['params']['symbol']:return False
        p=item['params']
        if item['method']=='DELETE' and item['path']=='/api/v3/order':
            return data.get('orderId')==p['orderId'] and data.get('status') in ('CANCELED','FILLED','EXPIRED')
        if 'orderList' in item['path']:
            if data.get('listClientOrderId')!=p['listClientOrderId'] or type(data.get('orderListId')) is not int:return False
            rows=data.get('orders',[])
            return len(rows)==2 and all(type(r.get('orderId')) is int and r.get('symbol',p['symbol'])==p['symbol'] for r in rows) and (item['method']=='DELETE' or
                {r.get('clientOrderId') for r in rows}=={p['aboveClientOrderId'],p['belowClientOrderId']})
        return (data.get('clientOrderId')==p.get('newClientOrderId',p.get('origClientOrderId'))
                and type(data.get('orderId')) is int and data.get('status') in {'NEW','PARTIALLY_FILLED','FILLED','CANCELED','EXPIRED','REJECTED'})
    def _symbol_allowed(self,symbol,*,closing=False):return symbol=='BTCUSDT'
    def call(self,action,method,path,params,broker):
        with self._audit._lock:
            self._ready()
            if not self.enabled:fail('PILOT_DISABLED')
            if (method,path) not in PAIRS:fail('PILOT_INVALID')
            old=self._actions.get(action)
            if old:
                if any(old[k]!=v for k,v in dict(method=method,path=path,params=params).items()):fail('PILOT_ACTION_CONFLICT')
                if old['state']=='ACKNOWLEDGED':return deepcopy(old['response'])
                fail('PILOT_QUERY_REQUIRED')
            if not self._symbol_allowed(params.get('symbol'),closing=method=='DELETE' or params.get('side')=='SELL'):fail('PILOT_INVALID')
            if method=='POST' and path=='/api/v3/order':
                if params.get('type')!='MARKET' or params.get('side') not in ('BUY','SELL'):fail('PILOT_INVALID')
                if params['side']=='BUY' and not 0<amount(params.get('quoteOrderQty'))<=30:fail('PILOT_RISK_LIMIT')
                if params['newClientOrderId']!=self.client_id(action):fail('PILOT_INVALID')
                if params['side']=='SELL' and not self.owned_order_ids(params['symbol']):fail('PILOT_NOT_OWNED')
            if method=='DELETE':
                if path=='/api/v3/order':
                    if (params['symbol'],params.get('orderId')) not in self.owned_order_keys():fail('PILOT_NOT_OWNED')
                else:
                    ids={a['params'].get('listClientOrderId') for a in self._actions.values() if a['state']=='ACKNOWLEDGED' and a['method']=='POST' and 'orderList' in a['path'] and a['params']['symbol']==params['symbol']}
                    if params.get('listClientOrderId') not in ids:fail('PILOT_NOT_OWNED')
            if method=='POST' and path=='/api/v3/orderList/oco' and not self.owned_order_ids(params['symbol']):fail('PILOT_NOT_OWNED')
            self._save('PREPARED',dict(action=action,method=method,path=path,params=params))
            self._save('START',dict(action=action))
            try:
                response=broker.mutate(method,path,params)
                if not self._valid_response(self._actions[action],response):fail('PILOT_QUERY_REQUIRED')
            except Exception as error:
                self._save('REJECT' if isinstance(error,AdapterError) and error.reason=='PILOT_REJECTED' else 'UNKNOWN',dict(action=action))
                fail('PILOT_QUERY_REQUIRED')
            self._save('ACK',dict(action=action,response=response));return deepcopy(response)
    def recover(self,action,broker):
        with self._audit._lock:
            self._ready();item=self._actions[action]
            response=broker.query_action(item['method'],item['path'],item['params'])
            if not self._valid_response(item,response):fail('PILOT_QUERY_REQUIRED')
            self._save('ACK',dict(action=action,response=response));return deepcopy(response)
    def owned_order_keys(self):
        result=set()
        for item in self._actions.values():
            if item['state']!='ACKNOWLEDGED' or item['method']!='POST':continue
            r=item['response']
            if 'orderList' in item['path']:result.update((r['symbol'],x['orderId']) for x in r['orders'])
            else:result.add((r['symbol'],r['orderId']))
        return sorted(result)
    def owned_order_ids(self,symbol=None):
        return sorted({oid for pair,oid in self.owned_order_keys() if symbol is None or pair==symbol})
    def abandon_reviewed_local_refusal(self,action,*,network_audit,snapshot,blocked_adapter_sha256):
        """Operator-only migration of the reviewed 05ac2d4 pre-network refusal.

        Source review is required in addition to the verified closed audit.
        Never retry the old ID or erase an attempt from the budget.
        """
        self._ready();item=self._actions.get(action,{})
        if (blocked_adapter_sha256!='e254e6f90b7523207aa7ee7944ddeb1bfab13a193ae0900287c30f3db77e174e'
            or not isinstance(network_audit,AuditJournal) or item.get('state')!='QUERY_REQUIRED'
            or action!='buy-1' or len(self._actions)!=1 or self._state.get('cycles')!=1
            or snapshot.get('account_identity')!=self._binding['account_identity']
            or snapshot.get('open_orders') or not self._binding['start_ms']<=snapshot.get('now_ms',0)<self._binding['expires_ms']):
            fail('PILOT_LOCAL_PROOF_INVALID')
        if any(network_audit.identity[k]!=self._audit.identity[k] for k in
                ('account_alias','account_episode_id','portfolio_scope_id')):fail('PILOT_LOCAL_PROOF_INVALID')
        actual={k:v for k,v in snapshot.get('balances',{}).items() if amount(v['free']) or amount(v['locked'])}
        if actual!=self._binding['initial_balances']:fail('PILOT_LOCAL_PROOF_INVALID')
        manifest=network_audit.manifest();closed=False
        for after in range(0,manifest['rows_n'],1000):
            for row in network_audit.read(after_seq=after,limit=1000):
                if row['kind']=='RUN_END':closed=True
                d=row['data']
                if row['kind']=='REST_BEGIN' and (d.get('method'),d.get('endpoint')) in PAIRS:
                    fail('PILOT_LOCAL_PROOF_INVALID')
        if not closed:fail('PILOT_LOCAL_PROOF_INVALID')
        self._save('DECISION',dict(action='ABANDON_NEVER_SENT',original_action=action,
            blocked_adapter_sha256=blocked_adapter_sha256,network_manifest_sha256=manifest['manifest_sha256'],
            cutoff_ms=snapshot['now_ms'],reason='OPERATOR_REVIEWED_LOCAL_ALLOWLIST_REFUSAL'))
        self._save('REJECT',dict(action=action))
        state=self.state();state.update(phase='IDLE',entry_killed=False,local_unsent_abandoned=True,
            reason='LOCAL_REFUSAL_ABANDONED',next_entry_ms=0)
        self.save_state(state)
    def actions(self):self._ready();return deepcopy(self._actions)
    def state(self):self._ready();return deepcopy(self._state)
    def save_state(self,state):self._ready();self._save('STATE',state)
    def decision(self,value):self._ready();self._save('DECISION',value)
    def outcome(self,value):self._ready();self._save('OUTCOME',value)
    def outcomes(self):self._ready();return deepcopy(self._outcomes)
    def close(self):
        if self._audit is not None:self._audit.close()
