"""Causal operational decisions; no trading signals or inferred historical N."""
import threading
from .adapter import AdapterError
from .interaction_audit import AuditJournal


class OperationalAudit:
    def __init__(self,journal,clock,store,*,component):
        if journal is not None and (not isinstance(journal,AuditJournal) or
                any(journal.identity.get(k)!=store.identity.get(k) for k in
                    ('account_alias','account_episode_id','portfolio_scope_id'))):
            raise AdapterError('AUDIT_IDENTITY_MISMATCH')
        self.journal,self.clock,self.store,self.component=journal,clock,store,component
        self._lock=threading.Lock()
        self._expected=self._recorded=0
        self._error=None

    def decide(self,operation,action,reason,*,source_ids=(),population=(),context=None):
        with self._lock:self._expected+=1
        if self.journal is None:return None
        try:
            cutoff=self.clock.wall_ms()
            sources=[]
            for identifier in sorted(set(source_ids)):
                source=self.store.get(identifier)
                if source is None:raise AdapterError('AUDIT_SOURCE_INVALID')
                if source['available_at_ms']>cutoff:raise AdapterError('AUDIT_FUTURE_SOURCE')
                sources.append(dict(snapshot_id=identifier,available_at_ms=source['available_at_ms']))
            if self._error:raise AdapterError(self._error)
            if self.journal._state!='READY':raise AdapterError(self.journal._failure_reason or 'AUDIT_UNAVAILABLE')
            identifier=self.journal.append('DECISION',dict(decision_type='OPERATIONAL',component=self.component,
                operation=operation,action=action,reasons=[reason],cutoff_ms=cutoff,sources=sources,
                candidate_population=sorted(set(population)),population_state='KNOWN' if source_ids else 'NOT_DISCOVERED',
                policy_version='read-only-coordinator-v1',risk_version='read-only-v1',model_version=None,
                intent=None,context={} if context is None else context),
                wall_ms=cutoff,monotonic_ns=self.clock.monotonic_ns())
            with self._lock:self._recorded+=1
            return identifier
        except AdapterError as error:
            # Invalid source is denied too, but does not invent a persisted row.
            with self._lock:
                if self._error is None:self._error=error.reason
            raise
        except Exception:
            with self._lock:
                if self._error is None:self._error='AUDIT_SOURCE_UNAVAILABLE'
            raise AdapterError('AUDIT_SOURCE_UNAVAILABLE') from None

    def health(self):
        with self._lock:
            journal_failed=self.journal is not None and self.journal._state!='READY'
            return dict(component=self.component,state='FAILED' if self._error or journal_failed else
                'UNCONFIGURED' if self.journal is None else 'READY',error=self._error or
                (self.journal._failure_reason or 'AUDIT_UNAVAILABLE' if journal_failed else None),
                expected_N=self._expected,recorded_n=self._recorded,gap_n=self._expected-self._recorded,
                counter_basis='PROCESS_BRANCH_DECISIONS',full_audit_complete=False)
