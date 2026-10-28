"""One capped execution-qualification worker; no profit-policy promotion."""
import argparse,json,math,os,time,uuid
from decimal import Decimal,Context,localcontext
from pathlib import Path

from .adapter import AdapterError,BinanceDemoAdapter,_sha
from .account_opening import AccountOpeningManifest
from .account_evidence import AccountEvidenceStore
from .execution_pilot import PilotJournal,MatchingGateway,execution_plan,fill_inventory,amount,text,rounded,fail
from .interaction_audit import AuditJournal
from .isolation import AppPaths,load_demo_credentials
from .market_data import DemoMarketClient
from .private_stream import PrivateEventReader
from .pilot_research import learning_handoff
from .order_validation import DemoTradeValidator


class PilotService:
    def __init__(self,journal,gateway):self.book,self.gateway=journal,gateway
    def _rows(self):return self.gateway.fill_rows(self.book.owned_order_ids())
    def _inventory(self):return fill_inventory(self._rows(),set(self.book.owned_order_ids()))
    def _save(self,state):self.book.save_state(state);return self.book.state()
    def _account_agrees(self,snapshot,inventory):
        initial=self.book._binding['initial_balances'];current=snapshot['balances']
        for asset in set(initial)|set(current)|{'BTC','USDT'}:
            expected=sum((amount(initial.get(asset,{}).get(k,'0')) for k in ('free','locked')),Decimal('0'))
            if asset=='BTC':expected+=Decimal(inventory['base'])
            if asset=='USDT':expected+=Decimal(inventory['cash_delta'])
            if asset not in current:
                if expected!=0:return False
                continue
            actual=amount(current[asset]['free'])+amount(current[asset]['locked'])
            if actual!=expected:return False
        return True
    def _close(self,state,snapshot,reason,emergency=False):
        if state.get('protection_action'):
            action=state['protection_action'];item=self.book.actions()[action]
            if item['state']!='ACKNOWLEDGED':
                state.update(phase='RECOVERY_REQUIRED',entry_killed=True,reason='PROTECTION_UNKNOWN');return self._save(state)
            params=dict(symbol='BTCUSDT',listClientOrderId=item['params']['listClientOrderId'])
            result=self.gateway.query_action(item['method'],item['path'],item['params'])
            if result is None or result.get('listOrderStatus')!='ALL_DONE':
                result=self.book.call('cancel-'+str(state['cycles']),'DELETE','/api/v3/orderList',params,self.gateway)
            if result.get('listOrderStatus')!='ALL_DONE':fail('PILOT_QUERY_REQUIRED')
        rows=self._rows();inventory=fill_inventory(rows,set(self.book.owned_order_ids()))
        snapshot=self.gateway.snapshot();base=Decimal(inventory['base']);bid=amount(snapshot['bid'])
        if any(o['order_id'] not in self.book.owned_order_ids() for o in snapshot['open_orders']):fail('PILOT_FOREIGN_ORDER')
        if not self._account_agrees(snapshot,inventory):fail('PILOT_BALANCE_MISMATCH')
        if emergency and bid<Decimal(state['fill_price'])*Decimal('.99'):fail('PILOT_EMERGENCY_SLIPPAGE')
        lot=next(r for r in snapshot['metadata']['filters'] if r['filterType']=='LOT_SIZE')
        quantity=rounded(base*Decimal('.998'),amount(lot['stepSize']))
        minimum=next(r for r in snapshot['metadata']['filters'] if r['filterType'] in ('NOTIONAL','MIN_NOTIONAL'))
        free=amount(snapshot['balances'].get('BTC',{}).get('free','0'))
        if quantity*bid>=amount(minimum['minNotional']):
            if quantity>free:fail('PILOT_SELL_LOCKED')
            action='sell-'+str(state['cycles'])
            p=dict(symbol='BTCUSDT',side='SELL',type='MARKET',quantity=text(quantity),
                newClientOrderId=self.book.client_id(action),newOrderRespType='FULL')
            self.book.call(action,'POST','/api/v3/order',p,self.gateway)
        rows=self._rows();inventory=fill_inventory(rows,set(self.book.owned_order_ids()));snapshot=self.gateway.snapshot()
        if not self._account_agrees(snapshot,inventory):fail('PILOT_BALANCE_MISMATCH')
        base=Decimal(inventory['base']);mid=(amount(snapshot['bid'])+amount(snapshot['ask']))/2
        if base<0 or base*mid>2:fail('PILOT_DUST_OR_UNCLOSED_EXPOSURE')
        delivered=self.gateway.private_matches(rows)
        equity=Decimal('100')+Decimal(inventory['cash_delta'])+base*mid
        outcome=dict(state='CLOSED',cycle=state['cycles'],technical_probe=state['cycles']==1,
            origin_ms=state['origin_ms'],feature_end_ms=state['feature_end_ms'],features=state['features'],
            label_end_ms=snapshot['now_ms'],label_available_at_ms=snapshot['now_ms'],
            net_usdt=text(equity-Decimal(state['start_equity'])),exit_reason=reason,
            source_kind='ACTUAL_DEMO',private_delivery_matched=delivered,fees=inventory['fees'],
            fills_N=inventory['fills_n'],daily_objective_eligible=False,
            snapshot_id=state['snapshot_id'],opening_sha256=self.book._binding.get('opening_sha256'))
        if not any(o['cycle']==state['cycles'] for o in self.book.outcomes()):self.book.outcome(outcome)
        state.update(phase='IDLE',qualified=bool(state.get('qualified') or delivered),
            entry_killed=bool(state.get('entry_killed') or not delivered or emergency),
            last_equity=text(equity),peak_equity=text(max(equity,Decimal(state.get('peak_equity','100')))),
            next_entry_ms=state['origin_ms']+10800000,reason=reason,protection_action=None,
            dust_usdt=text(base*mid))
        return self._save(state)
    def tick(self):
        with localcontext(Context(prec=50)):
            state=self.book.state() or dict(phase='IDLE',cycles=0,qualified=False,entry_killed=False,
                peak_equity='100',last_equity='100',next_entry_ms=0)
            if state['phase'] in ('HALTED','FINISHED','RECOVERY_REQUIRED'):return state
            try:
                snapshot=self.gateway.snapshot();now=snapshot['now_ms'];rows=self._rows()
                inventory=fill_inventory(rows,set(self.book.owned_order_ids()))
                if snapshot['account_identity']!=self.book._binding['account_identity']:fail('PILOT_ACCOUNT_CHANGED')
                if not self._account_agrees(snapshot,inventory):fail('PILOT_BALANCE_MISMATCH')
                if any(o['order_id'] not in self.book.owned_order_ids() for o in snapshot['open_orders']):fail('PILOT_FOREIGN_ORDER')
                if state['phase']=='OPEN':
                    if now>=state['deadline_ms']:return self._close(state,snapshot,'PROBE_TIME' if state['cycles']==1 else 'TIME')
                    if not snapshot['open_orders'] and Decimal(inventory['base'])*amount(snapshot['bid'])<=2:
                        return self._close(state,snapshot,'EXCHANGE_EXIT')
                    return self._save(state)
                if state['phase'] in ('BUY_PENDING','PROTECTION_PENDING'):return self._entry(state,snapshot)
                if state['phase']!='IDLE':fail('PILOT_RECOVERY_REQUIRED')
                equity=Decimal('100')+Decimal(inventory['cash_delta'])+Decimal(inventory['base'])*(amount(snapshot['bid'])+amount(snapshot['ask']))/2
                if (now>=self.book._binding['expires_ms'] or state['cycles']>=8):
                    state.update(phase='FINISHED',reason='PILOT_LIMIT_REACHED');return self._save(state)
                if (equity<99 or Decimal(state.get('peak_equity','100'))-equity>=1):
                    state.update(phase='HALTED',entry_killed=True,reason='PILOT_LOSS_STOP');return self._save(state)
                stopped=self.book._audit.path.parent.joinpath('stop-entries').exists()
                if (stopped or state.get('entry_killed') or (state['cycles'] and not state.get('qualified'))
                        or now<state['next_entry_ms'] or not snapshot['stream_ok']):
                    self.book.decision(dict(action='WAIT',reason='ENTRY_GUARD',cutoff_ms=now,snapshot_id=snapshot['snapshot_id']));return self._save(state)
                if Decimal(inventory['base'])*amount(snapshot['bid'])>2:fail('PILOT_DUST_OR_UNCLOSED_EXPOSURE')
                if amount(snapshot['balances']['USDT']['free'])<31:fail('PILOT_INSUFFICIENT_CASH')
                execution_plan(snapshot['metadata'],snapshot['bid'],snapshot['ask'],snapshot['average'])
                if not self.gateway.preflight(snapshot):fail('PILOT_TRADE_PERMISSION')
                now=snapshot['now_ms']
                if now>=self.book._binding['expires_ms']:fail('PILOT_EXPIRED')
                state.update(phase='BUY_PENDING',cycles=state['cycles']+1,origin_ms=now,
                    feature_end_ms=snapshot['feature_end_ms'],features=snapshot['features'],
                    snapshot_id=snapshot['snapshot_id'],start_equity=text(equity))
                if state['feature_end_ms']>=now:fail('PILOT_FUTURE_FEATURE')
                self._save(state)
                self.book.decision(dict(action='BUY',reason='EXECUTION_QUALIFICATION',cutoff_ms=now,
                    snapshot_id=snapshot['snapshot_id'],features=state['features'],feature_end_ms=state['feature_end_ms']))
                return self._entry(state,snapshot)
            except AdapterError as error:
                buy=self.book.actions().get('buy-'+str(state['cycles']),{})
                if (buy.get('state')=='ACKNOWLEDGED' and not state.get('protection_action')
                        and state.get('fill_price') and error.reason not in ('PILOT_PARTIAL_BUY','PILOT_QUERY_REQUIRED')):
                    try:
                        state['entry_killed']=True
                        return self._close(state,self.gateway.snapshot(),error.reason,emergency=True)
                    except AdapterError:pass
                state['recovery_phase']=state['phase']
                state.update(phase='RECOVERY_REQUIRED' if state['phase'] not in ('IDLE','HALTED') else 'HALTED',
                    entry_killed=True,reason=error.reason)
                return self._save(state)

    def _entry(self,state,snapshot):
        action='buy-'+str(state['cycles'])
        response=self.book.call(action,'POST','/api/v3/order',dict(symbol='BTCUSDT',side='BUY',type='MARKET',
            quoteOrderQty='30',newClientOrderId=self.book.client_id(action),newOrderRespType='FULL'),self.gateway)
        rows=self._rows();inventory=fill_inventory(rows,set(self.book.owned_order_ids()))
        if response.get('status')=='PARTIALLY_FILLED':
            canceled=self.book.call('cancel-entry-'+str(state['cycles']),'DELETE','/api/v3/order',
                dict(symbol='BTCUSDT',orderId=response['orderId']),self.gateway)
            if canceled.get('status') not in ('CANCELED','FILLED','EXPIRED'):fail('PILOT_QUERY_REQUIRED')
            rows=self._rows();inventory=fill_inventory(rows,set(self.book.owned_order_ids()));state['entry_killed']=True
        elif response.get('status')!='FILLED':fail('PILOT_PARTIAL_BUY')
        buys=[r for r in rows if r['order_id']==response['orderId'] and r['side']=='BUY']
        gross=sum((amount(str(r['quantity'])) for r in buys),Decimal('0'))
        quote=sum((amount(str(r['quote_quantity'])) for r in buys),Decimal('0'))
        if gross<=0:fail('PILOT_PARTIAL_BUY')
        state['fill_price']=text(quote/gross)
        snapshot=self.gateway.snapshot()
        if not self._account_agrees(snapshot,inventory):fail('PILOT_BALANCE_MISMATCH')
        if Decimal(inventory['base'])*amount(snapshot['ask'])>32:fail('PILOT_EXPOSURE_LIMIT')
        plan=execution_plan(snapshot['metadata'],snapshot['bid'],snapshot['ask'],snapshot['average'],
            owned=inventory['base'],fill_price=state['fill_price'])
        paction='protect-'+str(state['cycles'])
        params=dict(symbol='BTCUSDT',side='SELL',quantity=plan['quantity'],listClientOrderId=self.book.client_id(paction),
            aboveClientOrderId=self.book.client_id(paction,'tp'),belowClientOrderId=self.book.client_id(paction,'sl'),
            aboveType='LIMIT_MAKER',abovePrice=plan['take_profit'],belowType='STOP_LOSS',belowStopPrice=plan['stop_loss'])
        state.update(phase='PROTECTION_PENDING',protection_action=paction);self._save(state)
        try:protection=self.book.call(paction,'POST','/api/v3/orderList/oco',params,self.gateway)
        except AdapterError:
            if self.book.actions().get(paction,{}).get('state')=='REJECTED':
                state.update(protection_action=None,entry_killed=True)
                return self._close(state,snapshot,'PROTECTION_REJECTED',emergency=True)
            raise
        if not self.gateway.protection_ok(protection,plan['quantity']):fail('PILOT_PROTECTION_UNVERIFIED')
        snapshot=self.gateway.snapshot()
        state.update(phase='OPEN',deadline_ms=state['origin_ms']+(60000 if state['cycles']==1 else 900000),
            protected_quantity=plan['quantity'],take_profit=plan['take_profit'],stop_loss=plan['stop_loss'],
            unprotected_duration_ms=snapshot['now_ms']-state['origin_ms'])
        if state['unprotected_duration_ms']>10000:state.update(entry_killed=True,reason='UNPROTECTED_DEADLINE_BREACH')
        return self._save(state)

    def recover_once(self):
        state=self.book.state()
        if state.get('phase')!='RECOVERY_REQUIRED':return state
        try:
            for action,item in self.book.actions().items():
                if item['state'] in ('QUERY_REQUIRED','SEND_STARTED'):self.book.recover(action,self.gateway)
            state.update(phase=state.get('recovery_phase','OPEN'),entry_killed=True)
            self._save(state);return self.tick()
        except AdapterError:return self.book.state()


class LiveGateway(MatchingGateway):
    def __init__(self,client,evidence,reader):
        super().__init__(client);self.evidence=evidence;self.reader=reader;self.market=DemoMarketClient(client)
        self.info=None;self.info_at=0;self.cache={}
    def preflight(self,snapshot):
        c=self.client
        if c.read_open_orders()['data'] or c.read_open_order_lists()['data']:fail('PILOT_FOREIGN_ORDER')
        lot=next(r for r in snapshot['metadata']['filters'] if r['filterType']=='LOT_SIZE')
        pf=next(r for r in snapshot['metadata']['filters'] if r['filterType']=='PRICE_FILTER')
        validator=DemoTradeValidator(c,allow_order_validation=True)
        try:
            validator.validate_limit('BTCUSDT',side='BUY',
                quantity=text(rounded(Decimal('30')/amount(snapshot['ask']),amount(lot['stepSize']))),
                price=text(rounded(amount(snapshot['ask']),amount(pf['tickSize']))))
        finally:validator.close()
        fresh=self.snapshot();snapshot.clear();snapshot.update(fresh)
        execution_plan(snapshot['metadata'],snapshot['bid'],snapshot['ask'],snapshot['average'])
        return all(c.capabilities()[k]['state']=='SUPPORTED' for k in
            ('USER_DATA_READ','ACCOUNT_SPOT_ALLOWED','KEY_TRADE_VALIDATED')) and snapshot['stream_ok']
    def snapshot(self):
        c=self.client;now=time.time_ns()//1000000
        if self.info is None or now-self.info_at>900000:
            raw,meta=c.request_public('/api/v3/exchangeInfo',{'symbol':'BTCUSDT'},weight=20)
            self.info=raw['symbols'][0];self.info_at=meta['received_at_ms']
        account=c.read_account();caps=c.capabilities()
        if any(caps[k]['state']!='SUPPORTED' for k in ('USER_DATA_READ','ACCOUNT_SPOT_ALLOWED')):fail('PILOT_ACCOUNT_NOT_READY')
        quotes=self.market.read_book_tickers(['BTCUSDT']);book=quotes['data'][0]
        avg,avgmeta=c.request_public('/api/v3/avgPrice',{'symbol':'BTCUSDT'},weight=2)
        bars,barmeta=c.request_public('/api/v3/klines',{'symbol':'BTCUSDT','interval':'15m','limit':5},weight=2)
        now=time.time_ns()//1000000;closed=[r for r in bars if int(r[6])<now]
        if len(closed)<3:fail('PILOT_FEATURE_GAP')
        last,prev,third=closed[-1],closed[-2],closed[-3]
        if int(last[0])-int(prev[0])!=900000 or int(prev[0])-int(third[0])!=900000:fail('PILOT_FEATURE_GAP')
        lastc=float(last[4]);features=[lastc/float(prev[4])-1,lastc/float(third[4])-1,
            (float(last[2])-float(last[3]))/lastc,math.log1p(float(last[5]))]
        orders=c.read_open_orders('BTCUSDT')['data'];now=time.time_ns()//1000000
        if now-quotes['received_at_ms']>5000 or now-account['received_at_ms']>30000:fail('PILOT_STALE_SNAPSHOT')
        balances={r['asset']:{k:text(r[k]) for k in ('free','locked')} for r in account['data']['balances']}
        return dict(now_ms=now,bid=book['bidPrice'],ask=book['askPrice'],average=avg['price'],metadata=self.info,
            balances=balances,stream_ok=self.reader.health()['transport_healthy'],open_orders=orders,
            features=features,feature_end_ms=int(last[6]),snapshot_id=_sha(dict(account=account['source_sha256'],
                book=quotes['source_sha256'],bars=_sha(bars),at=now)),account_identity=account['data']['account_identity'])
    def fill_rows(self,ids):
        result=[]
        for oid in ids:
            if oid in self.cache:result.extend(self.cache[oid]);continue
            order=self.client.read_order('BTCUSDT',oid)['data']
            raw,meta=self.client._request('/api/v3/myTrades',{'symbol':'BTCUSDT','orderId':oid,'limit':1000},signed=True,weight=20)
            rows=[self.client._fill(r,'BTCUSDT') for r in raw]
            if any(r['order_id']!=oid for r in rows):fail('PILOT_FOREIGN_FILL')
            if sum((r['quantity'] for r in rows),Decimal('0'))!=order['executed_quantity']:fail('PILOT_FILL_COVERAGE')
            if order['status'] in ('FILLED','CANCELED','EXPIRED','REJECTED'):self.cache[oid]=rows
            result.extend(rows)
        return result
    def protection_ok(self,response,quantity):
        rows=[self.client.read_order('BTCUSDT',o['orderId'])['data'] for o in response['orders']]
        return (len(rows)==2 and {r['type'] for r in rows}=={'LIMIT_MAKER','STOP_LOSS'} and
            all(r['status']=='NEW' and r['original_quantity']==Decimal(quantity)
                and r['order_list_id']==response['orderListId'] for r in rows))
    def private_matches(self,rows):
        wire={(r['data']['symbol'],r['data']['trade_id']):r['data'] for r in self.evidence.fills()}
        return bool(rows) and all((r['symbol'],r['trade_id']) in wire and all(
            Decimal(wire[r['symbol'],r['trade_id']][k])==Decimal(str(r[k]))
            for k in ('quantity','quote_quantity','commission_quantity')) for r in rows)


def write_json(path,value):
    temporary=path.with_suffix('.next');temporary.write_text(json.dumps(value,default=str,indent=2),encoding='utf-8');os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-demo-qualification',action='store_true')
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if not args.enable_demo_qualification:raise SystemExit('EXPLICIT_DEMO_QUALIFICATION_FLAG_REQUIRED')
    paths=AppPaths(Path(__file__).resolve().parents[2]);folder=paths.runtime_path('pilot');folder.mkdir(parents=True,exist_ok=True)
    research=json.loads(paths.runtime_path('pilot-research/manifest.json').read_text(encoding='utf-8'))
    if research['state']!='MAXIMUM_AVAILABLE_FETCH_COMPLETE' or research['bars_N']<=0:raise SystemExit('REPLAY_REQUIRED')
    namespace=dict(account_alias='demo-primary',episode_id='opening-candidate-20261027',scope_id='whole-account-opening')
    audit=AuditJournal(paths,**namespace,database='pilot/network-audit.sqlite',max_bytes=512*1024*1024)
    c=BinanceDemoAdapter(load_demo_credentials(paths),**namespace,credential_generation='pilot-'+str(time.time_ns()),
        identity_key=paths.runtime_path('account-identity.key').read_bytes(),audit=audit)
    opening=AccountOpeningManifest(paths,**namespace);reader=evidence=book=gateway=None
    try:
        c.sync_clock();initial=c.read_account();identity=initial['data']['account_identity']
        if identity!=opening.summary()['attestation']['account_identity']:fail('PILOT_ACCOUNT_CHANGED')
        profile_path=folder/'profile.json'
        if profile_path.exists():profile=json.loads(profile_path.read_text(encoding='utf-8'))
        else:
            if (c.read_open_orders()['data'] or c.read_open_order_lists()['data']
                or any(r['locked'] for r in initial['data']['balances'])
                or any(r['asset']=='BTC' and r['free'] for r in initial['data']['balances'])):fail('PILOT_FOREIGN_INVENTORY')
            balances={r['asset']:{k:text(r[k]) for k in ('free','locked')} for r in initial['data']['balances'] if r['free'] or r['locked']}
            seed=opening.summary()['candidates']['initial-observation-1793135919330516500']
            now=time.time_ns()//1000000
            profile=dict(account_identity=identity,start_ms=now,expires_ms=now+86400000,
                initial_balances=balances,opening_sha256=seed['manifest_sha256'])
            write_json(profile_path,profile)
        book=PilotJournal(paths,**profile,enabled=True)
        evidence=AccountEvidenceStore(paths,**namespace,account_identity=identity,database='pilot/account-evidence.sqlite')
        reader=PrivateEventReader(c,evidence,process_instance_id=uuid.uuid4().hex,background=True);reader.start()
        gateway=LiveGateway(c,evidence,reader);service=PilotService(book,gateway)
        while True:
            state=service.tick();report=dict(mode='EXECUTION_QUALIFICATION',pid=os.getpid(),checked_at_ms=time.time_ns()//1000000,
                state=state,expires_ms=profile['expires_ms'],actual_outcomes_n=len(book.outcomes()),
                daily_net_pnl_usdt=None,financial_status='PENDING',ordinary_strategy_enabled=False,
                replay_verdict=research['replay_verdict'],stream=reader.health())
            write_json(folder/'status.json',report);write_json(folder/'outcomes.json',book.outcomes())
            write_json(folder/'learning-handoff.json',learning_handoff(book.outcomes(),fit_cutoff_ms=time.time_ns()//1000000))
            if state['phase']=='RECOVERY_REQUIRED':
                for _ in range(6):
                    state=service.recover_once()
                    if state['phase']!='RECOVERY_REQUIRED':break
                    time.sleep(1)
                write_json(folder/'status.json',dict(report,state=state))
            if args.once or state['phase'] in ('HALTED','RECOVERY_REQUIRED','FINISHED'):break
            time.sleep(10 if state['phase']=='OPEN' else 30)
    except AdapterError as error:
        write_json(folder/'status.json',dict(state={'phase':'RECOVERY_REQUIRED','reason':error.reason},checked_at_ms=time.time_ns()//1000000,
            mode='EXECUTION_QUALIFICATION',ordinary_strategy_enabled=False));raise SystemExit(error.reason) from None
    finally:
        if gateway is not None:gateway.close()
        if reader is not None:reader.close()
        if book is not None:book.close()
        if evidence is not None:evidence.close()
        opening.close();c.close();audit.close()


if __name__=='__main__':main()
