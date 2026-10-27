"""Pinned parsed-control hooks. Local socket write is never a remote ACK."""
import socket
import threading
import websockets
from websockets.client import ClientProtocol
from websockets.frames import PING, PONG, CLOSE
from websockets.sync.client import ClientConnection

from .adapter import AdapterError
from .interaction_audit import response_payload

CONTROLS={PING:'PING',PONG:'PONG',CLOSE:'CLOSE'}


class ProtocolControlAudit:
    def __init__(self,journal,clock,connection_id,*,private,secrets=None,on_failure=None,max_pending=64):
        if type(private) is not bool or type(max_pending) is not int or not 1<=max_pending<=64:
            raise AdapterError('AUDIT_INVALID')
        self.journal,self.clock,self.connection_id=journal,clock,connection_id
        self.private,self.secrets,self.on_failure=private,secrets or (lambda:()),on_failure
        self.max_pending=max_pending
        self._lock=threading.RLock()
        self._socket=None
        self._installed=False
        self._error=None
        self._pending=[]
        self._parent=None
        self._last_rejected=None
        self._inbound=self._outbound=self._queued=0
        self._expected=self._acked=0
        self._attempted=self._returned=self._unknown=self._pressure=0

    def _fail(self,reason):
        with self._lock:
            if self._error:return
            self._error=reason
        self.journal._failed(reason)
        # Called under the native protocol mutex: never call connection.close().
        if self._socket is not None:
            try:self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            try:self._socket.close()
            except OSError:pass
        if self.on_failure is not None:
            try:self.on_failure(reason)
            except Exception:pass

    def _record(self,phase,*,opcode=None,parent=None,payload=None,control_ids=None):
        if self._error:return None
        with self._lock:self._expected+=1
        data=dict(connection_id=self.connection_id,phase=phase,opcode=opcode,
            direction='INBOUND' if phase in ('RECEIVED','REJECTED') else 'OUTBOUND',
            parent_control_event_id=parent,parsed_controls_only=True,serialized_wire_observed=False,
            remote_acknowledged=False)
        if payload is not None:
            data['payload']=(dict(representation='OMITTED',bytes_n=len(payload)) if opcode=='CLOSE'
                else response_payload(bytes(payload),private=self.private,secrets=self.secrets()))
        if control_ids is not None:data['control_event_ids']=list(control_ids)
        try:
            identifier=self.journal.append('WS_CONTROL',data,wall_ms=self.clock.wall_ms(),
                monotonic_ns=self.clock.monotonic_ns(),secrets=self.secrets())
            with self._lock:self._acked+=1
            return identifier
        except AdapterError as error:
            self._fail(self.journal._failure_reason or error.reason)
            return None

    def install(self,protocol,sock):
        if websockets.__version__!='17.1' or not isinstance(protocol,ClientProtocol):
            raise AdapterError('AUDIT_PROTOCOL_UNSUPPORTED')
        if self._installed or getattr(protocol,'_demo_control_owner',None) is not None:
            raise AdapterError('AUDIT_PROTOCOL_OWNERSHIP')
        self._socket=sock
        original_recv,original_send=protocol.recv_frame,protocol.send_frame
        protocol._demo_control_owner=self

        def receive(frame):
            if self._error:return
            if frame.opcode not in CONTROLS:return original_recv(frame)
            with self._lock:self._inbound+=1
            identifier=self._record('RECEIVED',opcode=CONTROLS[frame.opcode],payload=frame.data)
            if identifier is None:return
            previous,self._parent=self._parent,identifier
            try:return original_recv(frame)
            except Exception:
                self._last_rejected=identifier
                self._record('REJECTED',opcode=CONTROLS[frame.opcode],parent=identifier)
                raise
            finally:self._parent=previous

        def send(frame):
            if self._error:return
            if frame.opcode not in CONTROLS:return original_send(frame)
            with self._lock:
                self._outbound+=1
                full=len(self._pending)>=self.max_pending
                if full:
                    self._pressure+=1;self._expected+=1
            if full:
                self._fail('AUDIT_CONTROL_BACKPRESSURE');return
            parent=self._parent or (self._last_rejected if frame.opcode==CLOSE else None)
            identifier=self._record('INTENT',opcode=CONTROLS[frame.opcode],parent=parent,payload=frame.data)
            if identifier is None:return
            result=original_send(frame)
            with self._lock:
                self._pending.append(identifier);self._queued+=1
            self._record('QUEUED',opcode=CONTROLS[frame.opcode],parent=identifier)
            return result

        protocol.recv_frame,protocol.send_frame=receive,send
        self._installed=True

    def write_pending(self,send):
        if self._error:return None
        with self._lock:pending=list(self._pending)
        if not pending:return send()
        if self._record('WRITE_ATTEMPT',control_ids=pending) is None:return None
        with self._lock:self._attempted+=1
        try:
            result=send()
        except BaseException:
            with self._lock:self._unknown+=1
            self._record('WRITE_UNKNOWN',control_ids=pending)
            raise
        else:
            with self._lock:self._returned+=1
            self._record('WRITE_RETURNED',control_ids=pending)
            return result
        finally:
            with self._lock:self._pending.clear()

    def connection_factory(self):
        if websockets.__version__!='17.1':raise AdapterError('AUDIT_PROTOCOL_UNSUPPORTED')
        observer=self
        class AuditedConnection(ClientConnection):
            def __init__(self,sock,protocol,**kwargs):
                observer.install(protocol,sock)
                super().__init__(sock,protocol,**kwargs)

            def send_data(self):
                return observer.write_pending(super().send_data)
        return AuditedConnection

    def health(self):
        with self._lock:
            return dict(scope='PARSED_WSS_CONTROL_FRAMES',installed=self._installed,
                state='FAILED' if self._error else 'READY',error=self._error,
                connection_id=self.connection_id,inbound_n=self._inbound,outbound_n=self._outbound,
                queued_n=self._queued,pending_n=len(self._pending),pending_limit_N=self.max_pending,
                write_batches_attempted_n=self._attempted,write_batches_returned_n=self._returned,
                write_batches_unknown_n=self._unknown,backpressure_n=self._pressure,
                records_expected_N=self._expected,records_acked_n=self._acked,record_gap_n=self._expected-self._acked,
                capture_coverage='UNKNOWN',serialized_wire_observed=False,full_audit_complete=False)
