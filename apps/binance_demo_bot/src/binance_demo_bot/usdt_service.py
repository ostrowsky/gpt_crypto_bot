"""All permitted USDT monitoring with one account-wide protected Demo pilot."""
import argparse,json,math,os,time,uuid
from copy import deepcopy
from collections import Counter
from decimal import Decimal
from pathlib import Path
from .adapter import BinanceDemoAdapter,AdapterError,_sha
from .account_opening import AccountOpeningManifest
from .account_evidence import AccountEvidenceStore
from .execution_pilot import amount,text,fail
from .interaction_audit import AuditJournal
from .isolation import AppPaths,load_demo_credentials
from .market_store import MarketSnapshotStore
from .pilot_service import PilotService,LiveGateway,RATE_REASONS,pilot_sleep_seconds,write_json
from .pilot_research import learning_handoff
from .private_stream import PrivateEventReader
from .universe import PERMISSIONS
from .usdt_scope import UsdtPilotJournal,usdt_catalog,portfolio_inventory,portfolio_value,select_round_robin
from .usdt_history import read_ready,verify_reference


def forecast_features(state,bars,received_at_ms,*,now_ms):
    if state.get('phase') in ('BUY_PENDING','PROTECTION_PENDING','OPEN') and state.get('features') is not None:
        return list(state['features']),state['feature_end_ms'],state.get('feature_available_at_ms',state.get('origin_ms',now_ms))
    if type(received_at_ms) is not int or received_at_ms>now_ms:fail('PILOT_FUTURE_FEATURE')
    closed=[r for r in bars if r[6]<now_ms and r[6]<=received_at_ms]
    if len(closed)<3 or any(closed[-i][0]-closed[-i-1][0]!=900000 for i in (1,2)):fail('PILOT_FEATURE_GAP')
    last,prev,third=closed[-1],closed[-2],closed[-3]
    try:
        price=float(last[4]);p=float(prev[4]);p3=float(third[4]);volume=float(last[5])
        if min(price,p,p3)<=0 or volume<0:fail('PILOT_FEATURE_INVALID')
        features=[price/p-1,price/p3-1,(float(last[2])-float(last[3]))/price,math.log1p(volume)]
        if not all(math.isfinite(v) for v in features):fail('PILOT_FEATURE_INVALID')
    except (ValueError,ArithmeticError):fail('PILOT_FEATURE_INVALID')
    return features,last[6],received_at_ms


def operational_report(state,monitor,*,pid,now_ms,expires_ms,actual_outcomes_n,stream,error=None):
    monitor=deepcopy(monitor);received=monitor.get('quotes_received_at_ms')
    fresh=type(received) is int and 0<=now_ms-received<=30000
    if not fresh:monitor['quote_ready_n']=None
    monitor['quote_status']='FRESH' if fresh else 'UNKNOWN_OR_STALE'
    return dict(mode='ALL_USDT_EXECUTION_QUALIFICATION',pid=pid,checked_at_ms=now_ms,state=deepcopy(state),
        monitoring=monitor,monitoring_enabled=True,operational_status='RUNNING' if fresh and error is None else 'PAUSED',
        last_error=error,entry_enabled=bool(error is None and fresh and now_ms<expires_ms and state.get('phase')=='IDLE'
            and not state.get('entry_killed') and monitor.get('selected_symbol')),
        expires_ms=expires_ms,actual_outcomes_n=actual_outcomes_n,daily_net_pnl_usdt=None,financial_status='PENDING',
        ordinary_strategy_enabled=False,stream=stream)


class UsdtPilotService(PilotService):
    def tick(self):
        before=self.book.state();actions=set(self.book.actions());state=super().tick()
        pair_errors={'PILOT_FILTER_INVALID','PILOT_FEATURE_GAP','PILOT_FEATURE_INVALID','PILOT_FUTURE_FEATURE','USDT_MARKET_NOT_READY'}
        if (before.get('phase')=='IDLE' and not before.get('entry_killed') and state.get('phase')=='HALTED'
            and state.get('reason') in pair_errors and set(self.book.actions())==actions):
            now=self.book._now_ms();symbol=state.get('symbol','BTCUSDT')
            blocks=[r for r in state.get('pair_blocks',[]) if r['until_ms']>now and r['symbol']!=symbol]
            blocks.append(dict(symbol=symbol,until_ms=now+60000,reason=state['reason']))
            self.book.decision(dict(action='BLOCK_PAIR',symbol=symbol,reason=state['reason'],cutoff_ms=now))
            state.update(phase='IDLE',entry_killed=False,reason='PAIR_DATA_WAIT',pair_blocks=blocks)
            return self._save(state)
        return state
    def _rows(self):return self.gateway.fill_rows(self.book.owned_order_keys())
    def _inventory(self,rows=None):
        rows=self._rows() if rows is None else rows
        bases={s:s[:-4] for s,_ in self.book.owned_order_keys()}
        result=portfolio_inventory(rows,set(self.book.owned_order_keys()),bases)
        symbol=self.book.state().get('symbol','BTCUSDT')
        result.update(base=result['deltas'].get(symbol[:-4],'0'),cash_delta=result['deltas'].get('USDT','0'))
        return result
    def _equity(self,inventory,snapshot):return portfolio_value(inventory['deltas'],snapshot['marks'])
    def _position_value(self,inventory,snapshot,*,ask=False):
        result=Decimal(0)
        for asset,value in inventory['deltas'].items():
            if asset=='USDT' or Decimal(value)==0:continue
            if asset not in snapshot['marks']:fail('PILOT_MARK_MISSING')
            price=amount(snapshot['marks'][asset][1 if ask else 0]);result+=Decimal(value)*price
        return result
    def _account_agrees(self,snapshot,inventory):
        initial=self.book._binding['initial_balances'];current=snapshot['balances'];deltas=inventory['deltas']
        for asset in set(initial)|set(current)|set(deltas):
            expected=sum((amount(initial.get(asset,{}).get(k,'0')) for k in ('free','locked')),Decimal(0))+Decimal(deltas.get(asset,'0'))
            actual=sum((amount(current.get(asset,{}).get(k,'0')) for k in ('free','locked')),Decimal(0))
            if expected!=actual:return False
        return True
    def _foreign_orders(self,snapshot):
        return any((o['symbol'],o['order_id']) not in self.book.owned_order_keys() for o in snapshot['open_orders'])
    def _is_probe(self,state,snapshot):
        return not any(o.get('symbol','BTCUSDT')==snapshot['metadata']['symbol'] and o.get('private_delivery_matched')
            for o in self.book.outcomes())


class UsdtGateway(LiveGateway):
    def __init__(self,client,evidence,reader,book,paths):
        super().__init__(client,evidence,reader);self.book=book;self.paths=paths
        self.symbol=book.state().get('symbol','BTCUSDT');self.catalog=[];self.metadata={}
        self.selected_metadata_at={}
        self.catalog_at=0;self.catalog_source=None;self.quotes={};self.quote_at=0;self.quote_source=None
        self.bar_cache={};self.last_snapshot=None;self.scope_signature=None
        self.verified_histories={}
        self.store=MarketSnapshotStore(paths,account_alias='demo-primary',episode_id='opening-candidate-20261027',
            scope_id='whole-account-opening',database='all-usdt/live-market.sqlite')
    def refresh_catalog(self):
        now=time.time_ns()//1000000
        if self.catalog and 0<=now-self.catalog_at<900000:return
        self.market._budget(20)
        raw,meta=self.client.request_public('/api/v3/exchangeInfo',
            {'permissions':json.dumps(PERMISSIONS,separators=(',',':')),'showPermissionSets':True},weight=20)
        account=self.client.read_account()
        if account['data']['account_identity']!=self.book._binding['account_identity']:fail('PILOT_ACCOUNT_CHANGED')
        if self.client.capabilities()['ACCOUNT_SPOT_ALLOWED']['state']!='SUPPORTED':fail('USDT_ACCOUNT_NOT_READY')
        catalog=usdt_catalog(raw,account['data']['permissions'],can_trade=account['data']['can_trade'])
        snapshot=self.store.append('UNIVERSE',dict(format_version='all-usdt-v1',rows=catalog,
            metadata_sha256=_sha(raw),metadata_received_at_ms=meta['received_at_ms'],account_sha256=account['source_sha256'],
            account_received_at_ms=account['received_at_ms']),received_at_ms=account['received_at_ms'],
            received_monotonic_ns=account['received_monotonic_ns'])
        self.catalog=catalog;self.metadata.update({r['symbol']:r['metadata'] for r in catalog})
        self.selected_metadata_at.update({r['symbol']:meta['received_at_ms'] for r in catalog})
        self.catalog_at=meta['received_at_ms'];self.catalog_source=snapshot['snapshot_id']
        write_json(self.paths.runtime_path('all-usdt/catalog.json'),catalog)
    def poll_quotes(self,*,force=False):
        now=time.time_ns()//1000000
        if self.quotes and not force and 0<=now-self.quote_at<30000:return
        self.market._budget(4)
        raw,meta=self.client.request_public('/api/v3/ticker/bookTicker',{},weight=4)
        if not isinstance(raw,list):fail('USDT_QUOTES_INVALID')
        quotes={}
        for q in raw:
            symbol=q.get('symbol')
            if not isinstance(symbol,str):fail('USDT_QUOTES_INVALID')
            if symbol in quotes and quotes[symbol]!=q:fail('USDT_QUOTES_CONFLICT')
            try:
                bid,ask,bq,aq=(amount(q[k]) for k in ('bidPrice','askPrice','bidQty','askQty'))
                if bid<=0 or ask<bid or bq<=0 or aq<=0:continue
            except (AdapterError,KeyError):continue
            quotes[symbol]=q
        snapshot=self.store.append('QUOTES',dict(time_basis='OBSERVED_RESPONSE',exchange_age='UNKNOWN',rows=list(quotes.values()),
            source_sha256=_sha(raw)),received_at_ms=meta['received_at_ms'],received_monotonic_ns=meta['received_monotonic_ns'])
        self.quotes=quotes;self.quote_at=meta['received_at_ms'];self.quote_source=snapshot['snapshot_id']
    def authorize(self):
        ready=read_ready(self.paths);signature=_sha([self.catalog_source,ready])
        if signature!=self.scope_signature:
            self.book.authorize_universe(self.catalog,received_at_ms=self.catalog_at,
                source_sha256=self.catalog_source,validated_symbols=ready);self.scope_signature=signature
        return ready
    def snapshot(self):
        c=self.client;state=self.book.state();now=time.time_ns()//1000000
        critical=state.get('phase') in ('BUY_PENDING','PROTECTION_PENDING','OPEN','RECOVERY_REQUIRED')
        if not critical:self.refresh_catalog();self.poll_quotes()
        elif self.symbol not in self.metadata or now-self.selected_metadata_at.get(self.symbol,0)>900000:
            raw,meta=c.request_public('/api/v3/exchangeInfo',{'symbol':self.symbol},weight=20)
            if len(raw.get('symbols',[]))!=1 or raw['symbols'][0]['symbol']!=self.symbol:fail('USDT_OWNED_METADATA_MISSING')
            self.metadata[self.symbol]=raw['symbols'][0];self.selected_metadata_at[self.symbol]=meta['received_at_ms']
        if self.symbol not in self.metadata:fail('USDT_OWNED_METADATA_MISSING')
        account=c.read_account();caps=c.capabilities()
        if any(caps[k]['state']!='SUPPORTED' for k in ('USER_DATA_READ','ACCOUNT_SPOT_ALLOWED')):fail('PILOT_ACCOUNT_NOT_READY')
        avg,_=c.request_public('/api/v3/avgPrice',{'symbol':self.symbol},weight=2)
        now=time.time_ns()//1000000;state=self.book.state()
        if state.get('phase') in ('BUY_PENDING','PROTECTION_PENDING','OPEN'):
            features,feature_end,feature_available=forecast_features(state,None,None,now_ms=now)
        else:
            cache=self.bar_cache.get(self.symbol)
            if cache is None or now//900000!=cache[0]:
                bars,meta=c.request_public('/api/v3/klines',{'symbol':self.symbol,'interval':'15m','limit':5},weight=2)
                self.bar_cache[self.symbol]=(now//900000,bars,meta)
            _,bars,bar_meta=self.bar_cache[self.symbol]
            features,feature_end,feature_available=forecast_features(state,bars,bar_meta['received_at_ms'],now_ms=time.time_ns()//1000000)
        orders=c.read_open_orders(self.symbol)['data'];now=time.time_ns()//1000000
        if not critical and now-self.quote_at>25000:self.poll_quotes(force=True)
        wanted=sorted({self.symbol}|({s for s,_ in self.book.owned_order_keys()} if critical else set()))
        quote_rows,quote_meta=c.request_public('/api/v3/ticker/bookTicker',{'symbols':json.dumps(wanted,separators=(',',':'))},weight=4)
        if not isinstance(quote_rows,list) or {r.get('symbol') for r in quote_rows}!=set(wanted):fail('PILOT_MARK_MISSING')
        selected=dict(data=quote_rows,source_sha256=_sha(quote_rows),**quote_meta)
        quote=next(r for r in quote_rows if r['symbol']==self.symbol)
        if time.time_ns()//1000000-account['received_at_ms']>30000:account=c.read_account()
        now=time.time_ns()//1000000
        if now-selected['received_at_ms']>5000 or now-account['received_at_ms']>30000:fail('PILOT_STALE_SNAPSHOT')
        marks={r['base_asset']:(self.quotes[r['symbol']]['bidPrice'],self.quotes[r['symbol']]['askPrice'])
            for r in self.catalog if r['metadata'].get('quoteAsset')=='USDT' and r['symbol'] in self.quotes}
        marks[self.metadata[self.symbol]['baseAsset']]=(quote['bidPrice'],quote['askPrice'])
        if critical:marks.update({r['symbol'][:-4]:(r['bidPrice'],r['askPrice']) for r in quote_rows})
        result=dict(now_ms=now,bid=quote['bidPrice'],ask=quote['askPrice'],average=avg['price'],metadata=self.metadata[self.symbol],
            balances={r['asset']:{k:text(r[k]) for k in ('free','locked')} for r in account['data']['balances']},
            marks=marks,stream_ok=self.reader.health()['transport_healthy'],open_orders=orders,features=features,
            bid_qty=quote['bidQty'],ask_qty=quote['askQty'],feature_end_ms=feature_end,
            snapshot_id=_sha([account['source_sha256'],selected['source_sha256'],_sha(features),feature_available,self.quote_source]),
            account_identity=account['data']['account_identity'],feature_available_at_ms=feature_available)
        self.last_snapshot=result;return result
    def preflight(self,snapshot):
        symbol=snapshot['metadata']['symbol']
        if symbol not in self.verified_histories:self.verified_histories[symbol]=verify_reference(self.paths,symbol)['manifest_sha256']
        if not super().preflight(snapshot):return False
        bid,ask=amount(snapshot['bid']),amount(snapshot['ask'])
        if (bid<=0 or ask<bid or (ask/bid-1)*10000>20
            or min(bid*amount(snapshot['bid_qty']),ask*amount(snapshot['ask_qty']))<30):fail('USDT_MARKET_NOT_READY')
        return True
    def fill_rows(self,keys):
        result=[]
        for symbol,oid in keys:
            key=(symbol,oid)
            if key in self.cache:result.extend(self.cache[key]);continue
            order=self.client.read_order(symbol,oid)['data']
            raw,_=self.client._request('/api/v3/myTrades',{'symbol':symbol,'orderId':oid,'limit':1000},signed=True,weight=20)
            rows=[self.client._fill(r,symbol) for r in raw]
            if any((r['symbol'],r['order_id'])!=key for r in rows):fail('PILOT_FOREIGN_FILL')
            if sum((r['quantity'] for r in rows),Decimal(0))!=order['executed_quantity']:fail('PILOT_FILL_COVERAGE')
            if order['status'] in ('FILLED','CANCELED','EXPIRED','REJECTED'):self.cache[key]=rows
            result.extend(rows)
        return result
    def monitoring_status(self,ready):
        monitored=[r for r in self.catalog if r['monitored']]
        counts=Counter(r['params']['symbol'] for r in self.book.actions().values() if r['method']=='POST'
            and r['path']=='/api/v3/order' and r['params'].get('side')=='BUY')
        blocked={r['symbol'] for r in self.book.state().get('pair_blocks',[]) if r['until_ms']>time.time_ns()//1000000}
        selected=select_round_robin(monitored,self.quotes,counts,validated=set(ready)-blocked)
        return dict(monitored_pairs_N=len(monitored),quote_ready_n=sum(r['symbol'] in self.quotes for r in monitored),
            execution_supported_n=sum(r['execution_supported'] for r in monitored),history_validated_n=len(ready),
            pair_quarantined_n=len(blocked),
            selected_symbol=selected,metadata_source=self.catalog_source,quote_source=self.quote_source,
            quotes_received_at_ms=self.quote_at,policy='technical-round-robin-v1',profit_strategy_enabled=False)
    def close(self):self.store.close();super().close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-all-usdt-demo',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not args.enable_all_usdt_demo:raise SystemExit('EXPLICIT_USDT_DEMO_SCOPE_REQUIRED')
    paths=AppPaths(Path(__file__).resolve().parents[2]);folder=paths.runtime_path('pilot')
    profile=json.loads((folder/'profile.json').read_text())
    ns=dict(account_alias='demo-primary',episode_id='opening-candidate-20261027',scope_id='whole-account-opening')
    audit=AuditJournal(paths,**ns,database='pilot/network-audit.sqlite',max_bytes=8*1024*1024*1024)
    c=BinanceDemoAdapter(load_demo_credentials(paths),**ns,credential_generation='all-usdt-'+str(time.time_ns()),
        identity_key=paths.runtime_path('account-identity.key').read_bytes(),audit=audit)
    book=reader=evidence=gateway=None
    try:
        c.sync_clock();account=c.read_account()
        if account['data']['account_identity']!=profile['account_identity']:fail('PILOT_ACCOUNT_CHANGED')
        book=UsdtPilotJournal(paths,**profile,enabled=True)
        evidence=AccountEvidenceStore(paths,**ns,account_identity=profile['account_identity'],database='pilot/account-evidence.sqlite')
        reader=PrivateEventReader(c,evidence,process_instance_id=uuid.uuid4().hex,background=True);reader.start()
        gateway=UsdtGateway(c,evidence,reader,book,paths);service=UsdtPilotService(book,gateway)
        while True:
            try:
                state=book.state();gateway.symbol=state.get('symbol','BTCUSDT')
                critical_processed=state.get('phase') in ('OPEN','BUY_PENDING','PROTECTION_PENDING','RECOVERY_REQUIRED')
                if critical_processed:
                    state=service.recover_once() if state['phase']=='RECOVERY_REQUIRED' else service.tick()
                gateway.refresh_catalog();gateway.poll_quotes();ready=gateway.authorize();monitor=gateway.monitoring_status(ready)
                state=book.state();gateway.symbol=state.get('symbol','BTCUSDT')
                # Explicit user-authorized scope migration, only from the old
                # flat rate halt. Preserve every monetary and temporal limit.
                if (state.get('phase')=='HALTED' and state.get('reason') in RATE_REASONS
                    and state.get('recovery_phase')=='IDLE' and state.get('protection_action') is None):
                    snapshot=gateway.snapshot();inventory=service._inventory()
                    if (c.read_open_orders()['data'] or c.read_open_order_lists()['data'] or snapshot['open_orders'] or not service._account_agrees(snapshot,inventory)
                        or service._position_value(inventory,snapshot,ask=True)>2
                        or any(a['state'] not in ('ACKNOWLEDGED','REJECTED') for a in book.actions().values())):fail('USDT_MIGRATION_NOT_READY')
                    book.decision(dict(action='USDT_SCOPE_MIGRATION',reason='USER_AUTHORIZED_ALL_USDT',cutoff_ms=snapshot['now_ms'],limits_changed=False))
                    state.update(phase='IDLE',entry_killed=False,reason='USDT_SCOPE_MIGRATED');book.save_state(state)
                now=time.time_ns()//1000000
                if not critical_processed and state.get('phase')=='IDLE' and not state.get('entry_killed') and now>=state.get('next_entry_ms',0):
                    candidate=monitor['selected_symbol']
                    if candidate is None:
                        book.decision(dict(action='WAIT',reason='NO_VALIDATED_USDT_CANDIDATE',cutoff_ms=now))
                    else:
                        state.update(symbol=candidate,base_asset=gateway.metadata[candidate]['baseAsset']);book.save_state(state)
                        gateway.symbol=candidate;state=service.tick()
                elif not critical_processed and state.get('phase') not in ('HALTED','FINISHED'):state=service.tick()
                if now>=profile['expires_ms'] and state.get('phase')=='IDLE':
                    state.update(phase='FINISHED',reason='PILOT_LIMIT_REACHED');book.save_state(state)
                report=operational_report(state,monitor,pid=os.getpid(),now_ms=time.time_ns()//1000000,
                    expires_ms=profile['expires_ms'],actual_outcomes_n=len(book.outcomes()),stream=reader.health())
                write_json(folder/'status.json',report);write_json(folder/'outcomes.json',book.outcomes())
                write_json(folder/'learning-handoff.json',learning_handoff(book.outcomes(),fit_cutoff_ms=time.time_ns()//1000000))
                write_json(paths.runtime_path('all-usdt/monitoring.json'),monitor)
                if args.once or (paths.runtime_path('all-usdt/stop-worker').exists() and state.get('phase') in ('IDLE','HALTED','FINISHED')):break
                time.sleep(pilot_sleep_seconds(state,time.time_ns()//1000000))
            except AdapterError as error:
                book.decision(dict(action='WAIT' if error.reason in RATE_REASONS else 'BLOCK',reason=error.reason,cutoff_ms=time.time_ns()//1000000))
                write_json(paths.runtime_path('all-usdt/incident.json'),dict(reason=error.reason,at_ms=time.time_ns()//1000000))
                monitor=dict(monitored_pairs_N=sum(r['monitored'] for r in gateway.catalog) if gateway.catalog else None,
                    quote_ready_n=None,quotes_received_at_ms=gateway.quote_at,metadata_source=gateway.catalog_source)
                write_json(folder/'status.json',operational_report(book.state(),monitor,pid=os.getpid(),now_ms=time.time_ns()//1000000,
                    expires_ms=profile['expires_ms'],actual_outcomes_n=len(book.outcomes()),stream=reader.health(),error=error.reason))
                write_json(folder/'outcomes.json',book.outcomes())
                write_json(folder/'learning-handoff.json',learning_handoff(book.outcomes(),fit_cutoff_ms=time.time_ns()//1000000))
                if args.once:raise
                delay=60 if error.reason in RATE_REASONS else 30
                if book.state().get('phase') in ('OPEN','RECOVERY_REQUIRED'):
                    delay=min(delay,pilot_sleep_seconds(book.state(),time.time_ns()//1000000))
                time.sleep(delay)
    finally:
        if gateway:gateway.close()
        if reader:reader.close()
        if book:book.close()
        if evidence:evidence.close()
        c.close();audit.close()


if __name__=='__main__':main()
