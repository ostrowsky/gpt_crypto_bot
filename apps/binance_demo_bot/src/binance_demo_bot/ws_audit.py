"""WSS application calls/messages only; hidden protocol controls are not observed."""
from copy import deepcopy
import json
import uuid

from .adapter import AdapterError
from .interaction_audit import AuditJournal, response_payload

OPERATIONS=('CONNECT','SEND','RECV','PING','CLOSE')
SAFE_ERRORS=frozenset(('WS_UNAVAILABLE','WS_BUFFER_OVERFLOW','WS_RATE_LIMITED',
    'INVALID_PARAMETERS','FORBIDDEN_METHOD','HANDSHAKE_TARGET_MISMATCH',
    'FORBIDDEN_ENDPOINT','WS_ALREADY_CONNECTED','TLS_UNSAFE'))


def bind_ws_audit(transport,journal,clock,identity,*,private=False,credentials=None):
    if journal is None:return transport
    if not isinstance(journal,AuditJournal) or any(journal.identity.get(k)!=identity.get(k)
            for k in ('account_alias','account_episode_id','portfolio_scope_id')):
        raise AdapterError('AUDIT_IDENTITY_MISMATCH')
    return AuditedWsTransport(transport,journal,clock,private=private,credentials=credentials)


class AuditedWsTransport:
    def __init__(self,transport,journal,clock,*,private,credentials=None):
        if not isinstance(journal,AuditJournal) or type(private) is not bool:
            raise AdapterError('AUDIT_INVALID')
        self.transport,self.journal,self.clock=transport,journal,clock
        self.private,self.credentials=private,credentials
        self.connection_id=None
        self.last_receipt=None
        self.error=None
        self._signature=None
        self._closed=False
        self._sequence=0
        self._frames=0
        self._cleanup_closes=0
        self._counts={op:dict(attempted_n=0,completed_n=0,errors_n=0,recorded_n=0) for op in OPERATIONS}

    def _secrets(self):
        values=[]
        if self.credentials is not None:
            credential=self.credentials()
            if credential is not None:values.extend((credential.api_key,credential.api_secret))
        if self._signature:values.append(self._signature)
        return tuple(values)

    def _physical_close(self):
        if not self._closed:
            self._closed=True
            self._cleanup_closes+=1
            try:self.transport.close()
            except Exception:pass

    def _record(self,kind,data,wall,mono):
        try:
            return self.journal.append(kind,data,wall_ms=wall,monotonic_ns=mono,secrets=self._secrets())
        except AdapterError as error:
            self.error=error.reason
            self._physical_close()
            raise

    def _perform(self,operation,call,request=None):
        if operation=='CONNECT':
            self.connection_id=uuid.uuid4().hex
            self._closed=False
        if operation=='RECV':self.last_receipt=None
        self._sequence+=1
        common=dict(operation=operation,operation_id=uuid.uuid4().hex,connection_id=self.connection_id,
                    operation_sequence=self._sequence,private=self.private,
                    endpoint='DEMO_PRIVATE_WSS' if self.private else 'DEMO_MARKET_WSS')
        intent=dict(common)
        if request is not None:
            if self.private and isinstance(request,dict):
                signature=request.get('params',{}).get('signature')
                self._signature=signature if isinstance(signature,str) else None
            intent['request_method']=({'SUBSCRIBE':'SUBSCRIBE_MARKET',
                'userDataStream.subscribe.signature':'SUBSCRIBE_USER_DATA'}.get(request.get('method'),'OTHER')
                if isinstance(request,dict) else 'OTHER')
            body=json.dumps(request,separators=(',',':')).encode()
            intent['payload']=response_payload(body,private=self.private,secrets=self._secrets())
        parent=self._record('WS_BEGIN',intent,self.clock.wall_ms(),self.clock.monotonic_ns())
        sent_wall,sent_mono=self.clock.wall_ms(),self.clock.monotonic_ns()
        count=self._counts[operation]
        count['attempted_n']+=1
        result,error=None,None
        try:
            result=call()
            count['completed_n']+=1
            if operation=='RECV':self._frames+=1
            if operation=='CLOSE':self._closed=True
        except BaseException as caught:
            error=caught;count['errors_n']+=1
        received_wall,received_mono=self.clock.wall_ms(),self.clock.monotonic_ns()
        outcome=('NEGATIVE' if operation=='PING' and result is not True else 'COMPLETED') if error is None else (
            'TIMEOUT' if isinstance(error,TimeoutError) else 'CANCELLED'
            if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'TRANSPORT_ERROR')
        terminal=dict(common,parent_event_id=parent,outcome=outcome,sent_at_ms=sent_wall,
            sent_monotonic_ns=sent_mono,received_at_ms=received_wall,received_monotonic_ns=received_mono,
            latency_ns=received_mono-sent_mono if received_mono>=sent_mono else None)
        if operation=='RECV' and error is None:
            raw=result.encode() if isinstance(result,str) else result
            terminal['payload']=(response_payload(raw,private=self.private,secrets=self._secrets())
                if isinstance(raw,bytes) and len(raw)<=1024*1024 else
                dict(representation='OMITTED',bytes_n=len(raw) if isinstance(raw,bytes) else None))
        identifier=self._record('WS_END',terminal,received_wall,received_mono)
        count['recorded_n']+=1
        if error is not None:
            if isinstance(error,(TimeoutError,KeyboardInterrupt,SystemExit)):raise error
            safe=error.reason if isinstance(error,AdapterError) and error.reason in SAFE_ERRORS else 'WS_UNAVAILABLE'
            raise AdapterError(safe) from None
        if operation=='RECV':
            self.last_receipt=dict(event_id=identifier,wall_ms=received_wall,monotonic_ns=received_mono)
        return result

    def connect(self):return self._perform('CONNECT',self.transport.connect)
    def send(self,value):return self._perform('SEND',lambda:self.transport.send(value),value)
    def recv(self,timeout_s=1):return self._perform('RECV',lambda:self.transport.recv(timeout_s))
    def ping(self,timeout_s=5):return self._perform('PING',lambda:self.transport.ping(timeout_s))

    def close(self):
        if self._closed:return
        try:self._perform('CLOSE',self.transport.close)
        except Exception:self._physical_close()

    def processing(self,receipt_ids,*,snapshot_ids=(),outcome='PROCESSED',**extra):
        if not receipt_ids:return
        return self._record('WS_PROCESSING',dict(connection_id=self.connection_id,
            receipt_event_ids=list(receipt_ids),snapshot_ids=list(snapshot_ids),outcome=outcome,**extra),
            self.clock.wall_ms(),self.clock.monotonic_ns())

    def health(self):
        return dict(scope='WSS_APPLICATION_BOUNDARY',protocol_controls_observed=False,full_audit_complete=False,
            audit_state=self.journal.health()['state'],error=self.error,connection_id=self.connection_id,
            received_frames_n=self._frames,cleanup_closes_n=self._cleanup_closes,operations=deepcopy(self._counts))


def receipt_of(transport,clock):
    if isinstance(transport,AuditedWsTransport) and transport.last_receipt is not None:
        return dict(transport.last_receipt)
    return dict(event_id=None,wall_ms=clock.wall_ms(),monotonic_ns=clock.monotonic_ns())


def disposition(transport,identifier,outcome,*,if_healthy=False):
    if not isinstance(transport,AuditedWsTransport) or not identifier:return
    if if_healthy and transport.journal._state!='READY':return
    return transport.processing([identifier],outcome=outcome)
