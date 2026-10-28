"""Finite native GET-only boundary observer. No OMS, training or promotion."""
import argparse,json,os,time
from pathlib import Path
from .account_opening import AccountOpeningManifest
from .adapter import BinanceDemoAdapter,AdapterError
from .financial_boundaries import FinancialBoundaryStore,ZONE
from .interaction_audit import AuditJournal
from .isolation import AppPaths,load_demo_credentials
from .market_data import DemoMarketClient

def capture_native(client,market,store,metadata,observation_id,now_ms):
    client.sync_clock()
    before=client.read_account();orders=client.read_open_orders();lists=client.read_open_order_lists()
    tickers=market.read_book_tickers();after=client.read_account()
    raw=dict(account_before=before,orders=orders,order_lists=lists,metadata=metadata,tickers=tickers,account_after=after,cutoff_ms=now_ms())
    return store.observe(observation_id=observation_id,**raw),raw

def run_schedule(boundaries,*,now_ms,sleep,warm,capture,emit,stopped):
    resumed_at=now_ms()
    def wait_until(target,boundary):
        announced=False
        while now_ms()<target:
            if stopped():emit(dict(state='STOPPED',boundary_ms=boundary));return False
            if not announced:emit(dict(state='WAITING',boundary_ms=boundary,next_capture_ms=target));announced=True
            sleep(min(5,max(.001,(target-now_ms())/1000)))
        if stopped():emit(dict(state='STOPPED',boundary_ms=boundary));return False
        return True
    for boundary in boundaries:
        if boundary+10000<resumed_at:
            emit(dict(state='BOUNDARY_MISSED',boundary_ms=boundary));continue
        if not wait_until(boundary-120000,boundary):return
        try:warm(boundary)
        except Exception as error:
            emit(dict(state='WARM_FAILED',boundary_ms=boundary,reason=error.reason if isinstance(error,AdapterError) else type(error).__name__))
        for offset in (-6000,1000,10000):
            target=boundary+offset
            if target<resumed_at:
                emit(dict(state='SKIPPED_PAST_SLOT',boundary_ms=boundary,target_ms=target));continue
            if not wait_until(target,boundary):return
            try:
                capture(boundary,target);emit(dict(state='CAPTURED',boundary_ms=boundary,target_ms=target))
            except Exception as error:
                emit(dict(state='CAPTURE_FAILED',boundary_ms=boundary,target_ms=target,
                    reason=error.reason if isinstance(error,AdapterError) else type(error).__name__))
        emit(dict(state='BOUNDARY_DONE',boundary_ms=boundary))
    emit(dict(state='COMPLETED'))

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,default=str,indent=2),encoding='utf-8');os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observe-next-two-midnights',action='store_true');args=parser.parse_args()
    if not args.observe_next_two_midnights:raise SystemExit('EXPLICIT_FINANCIAL_OBSERVER_REQUIRED')
    paths=AppPaths(Path(__file__).resolve().parents[2]);folder=paths.runtime_path('financial-boundaries');folder.mkdir(exist_ok=True)
    ns=dict(account_alias='demo-primary',episode_id='opening-candidate-20261027',scope_id='whole-account-opening')
    opening=store=audit=client=None;metadata={}
    now=lambda:time.time_ns()//1000000
    def emit(event):
        event=dict(event,checked_at_ms=now(),pid=os.getpid(),timezone=ZONE,daily_net_pnl_usdt=None,
            financial_status='PENDING',execution_authorized=False)
        write_json(folder/'status.json',event)
        with (folder/'events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(event,default=str)+'\n')
        if store and 'boundary_ms' in event:write_json(folder/('boundary-'+str(event['boundary_ms'])+'.json'),store.boundary_report(event['boundary_ms']))
    try:
        emit(dict(state='STARTING'))
        profile=json.loads(paths.runtime_path('pilot/profile.json').read_text(encoding='utf-8'))
        opening=AccountOpeningManifest(paths,**ns)
        candidates=[r['candidate_id'] for r in opening.summary()['candidates'].values() if r['manifest_sha256']==profile.get('opening_sha256')]
        if len(candidates)!=1:raise AdapterError('FBO_OPENING_INVALID')
        store=FinancialBoundaryStore(paths,opening_manifest=opening,candidate_id=candidates[0])
        opening.close() # Immutable validation basis is frozen; release the original writer lease while waiting.
        audit=AuditJournal(paths,**ns,database='financial-boundaries/network.sqlite',max_bytes=1024*1024*1024)
        client=BinanceDemoAdapter(load_demo_credentials(paths),**ns,credential_generation='boundary-'+str(now()),
            identity_key=paths.runtime_path('account-identity.key').read_bytes(),audit=audit)
        client.sync_clock();account=client.read_account()
        if account['data']['account_identity']!=store._binding['account_identity']:raise AdapterError('FBO_ACCOUNT_CHANGED')
        plan=store.schedule(now());write_json(folder/'plan.json',plan);market=DemoMarketClient(client)
        def warm(boundary):
            client.sync_clock();metadata[boundary]=market.read_exchange_info();write_json(folder/('metadata-'+str(boundary)+'.json'),metadata[boundary])
        def capture(boundary,target):
            if boundary not in metadata:warm(boundary)
            key='boundary-'+str(boundary)+'-'+str(target-boundary)
            try:
                observation,raw=capture_native(client,market,store,metadata[boundary],key,now)
                write_json(folder/(key+'-sources.json'),raw);write_json(folder/(key+'.json'),observation)
            except AdapterError:raise
        run_schedule(plan['boundaries_ms'],now_ms=now,sleep=time.sleep,warm=warm,capture=capture,emit=emit,
            stopped=lambda:(folder/'stop-observer').exists())
    except Exception as error:
        emit(dict(state='FAIL',reason=error.reason if isinstance(error,AdapterError) else type(error).__name__))
        raise SystemExit(1) from None
    finally:
        for item in (client,audit,store,opening):
            if item:item.close()

if __name__=='__main__':main()
