"""Transport repair: preserve clock/risk guards and all financial uncertainty."""
from decimal import Decimal
import time
from .adapter import AdapterError,_sha
from .execution_pilot import fail
from .pilot_service import CLOCK_REASONS,RATE_REASONS
from .private_stream import PrivateEventReader
from .usdt_scope import decode_identifiers

TRANSIENT_PRIVATE={'WS_UNAVAILABLE','WS_HEARTBEAT_FAILED','WS_PONG_TIMEOUT','WS_RATE_LIMITED','READ_UNAVAILABLE'}|CLOCK_REASONS|RATE_REASONS


class PrivateReaderSupervisor:
    def __init__(self,gateway,*,factory=PrivateEventReader,book=None):
        self.gateway,self.factory=gateway,factory;self.book=book or getattr(gateway,'book',None)
        self.next_attempt_ns=0;self.fatal=None
    def maintain(self):
        g=self.gateway;reader=g.reader
        if self.book and self.book.state().get('phase') in ('OPEN','BUY_PENDING','PROTECTION_PENDING','RECOVERY_REQUIRED'):
            return reader
        health=reader.health()
        if health.get('connected') and health.get('transport_healthy') and health.get('reader_alive'):return reader
        if self.fatal:fail(self.fatal)
        reason=health.get('error_reason')
        if reason is not None and reason not in TRANSIENT_PRIVATE:fail(reason)
        now=g.client.clock.monotonic_ns()
        if now<self.next_attempt_ns:fail('PILOT_PRIVATE_RETRY_WAIT')
        self.next_attempt_ns=now+60_000_000_000
        reader.close()
        if reader.health().get('reader_alive'):fail('PILOT_PRIVATE_READER_STUCK')
        fresh=None
        try:
            fresh=self.factory(g.client,g.evidence,process_instance_id=reader.process_id,background=True)
            fresh.start();health=fresh.health()
            if not (health.get('connected') and health.get('transport_healthy') and health.get('reader_alive')):
                fail('WS_UNAVAILABLE')
            if self.book:self.book.decision(dict(action='RECONNECT',reason='PRIVATE_TRANSPORT_RECOVERED',
                cutoff_ms=g.client.clock.wall_ms()))
        except BaseException as error:
            if fresh:fresh.close()
            if isinstance(error,AdapterError) and error.reason not in TRANSIENT_PRIVATE:self.fatal=error.reason
            raise
        g.reader=fresh
        return fresh


def _safe_halt_origin(book):
    book._audit.manifest();previous=None;safe=False
    for after in range(0,book._audit._last_seq,1000):
        for row in book._audit.read(after_seq=after,limit=1000):
            data=decode_identifiers(row['data'])
            if row['kind']!='DECISION' or data.get('operation')!='STATE':continue
            state=data['value']
            if state.get('phase')=='HALTED' and state.get('reason') in CLOCK_REASONS:
                safe=bool(previous and previous.get('phase')=='IDLE' and previous.get('entry_killed') is False
                    and previous.get('cycles')==state.get('cycles') and previous.get('protection_action') is None)
            previous=state
    return safe


def resume_clock_halt(service,*,reviewed=False):
    book,g=service.book,service.gateway;state=book.state()
    if (reviewed is not True or state.get('phase')!='HALTED' or state.get('reason') not in CLOCK_REASONS
        or state.get('recovery_phase')!='IDLE' or state.get('protection_action') is not None
        or not _safe_halt_origin(book)
        or any(a['state'] not in ('ACKNOWLEDGED','REJECTED') for a in book.actions().values())):
        fail('PILOT_REVIEW_NOT_READY')
    outcomes=book.outcomes()
    if state['cycles'] and (not outcomes or outcomes[-1]['cycle']!=state['cycles'] or outcomes[-1]['state']!='CLOSED'
        or outcomes[-1].get('private_delivery_matched') is not True):fail('PILOT_REVIEW_NOT_READY')
    probes=[]
    for _ in range(3):
        probe=g.client.sync_clock();uncertainty=probe.get('clock_uncertainty_ms')
        if (not isinstance(uncertainty,(int,Decimal)) or isinstance(uncertainty,bool)
            or not Decimal(uncertainty).is_finite() or not 0<=Decimal(uncertainty)<=250):fail('CLOCK_QUALITY')
        g.client.clock.timestamp_ms();probes.append(probe)
    health=g.reader.health()
    if not (health.get('connected') and health.get('transport_healthy') and health.get('reader_alive')):fail('PILOT_REVIEW_NOT_READY')
    snapshot=g.snapshot();inventory=service._inventory();equity=service._equity(inventory,snapshot)
    if (snapshot['account_identity']!=book._binding['account_identity'] or not snapshot['stream_ok']
        or g.client.read_open_orders()['data'] or g.client.read_open_order_lists()['data'] or snapshot['open_orders']
        or not service._account_agrees(snapshot,inventory) or service._position_value(inventory,snapshot,ask=True)>2
        or snapshot['now_ms']>=book._binding['expires_ms'] or state['cycles']>=8 or equity<99
        or Decimal(state.get('peak_equity','100'))-equity>=1):fail('PILOT_REVIEW_NOT_READY')
    g.client.clock.timestamp_ms()
    wall=g.client.clock.wall_ms();health=g.reader.health()
    if (not 0<=wall-snapshot['now_ms']<=5000
        or any(type(p.get('received_at_ms')) is not int or not 0<=wall-p['received_at_ms']<=60000 for p in probes)):
        fail('PILOT_REVIEW_SOURCE_STALE')
    if not (health.get('connected') and health.get('transport_healthy') and health.get('reader_alive')):fail('PILOT_REVIEW_NOT_READY')
    book.decision(dict(action='OPERATOR_RESUME',reason='FLAT_CLOCK_INCIDENT_REVIEWED',cutoff_ms=wall,
        snapshot_id=snapshot['snapshot_id'],clock_sources_sha256=_sha(probes),limits_changed=False))
    state.pop('data_wait_reason',None)
    state.update(phase='IDLE',entry_killed=False,reason='CLOCK_REVIEWED',last_equity=str(equity),
        peak_equity=str(max(equity,Decimal(state.get('peak_equity','100')))))
    book.save_state(state);return book.state()


def review_flat_clock_startup(service,supervisor,*,sleep=time.sleep):
    gateway=service.gateway
    gateway.refresh_catalog();gateway.poll_quotes();gateway.authorize()
    for attempt in range(3):
        try:
            reader=supervisor.maintain()
            resume_clock_halt(service,reviewed=True)
            return reader
        except AdapterError as error:
            if error.reason not in CLOCK_REASONS|RATE_REASONS|{'PILOT_REVIEW_SOURCE_STALE'} or attempt==2:raise
            sleep(5)
