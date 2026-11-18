"""RFD09 exact original-validator chunk transitions; bounded offline workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from .public_flow_dataset import FlowValidator,_sha
from .receipt_flow_direction import revalidate


class BoundaryState:
    """Cheap proposed boundary, accepted only after original transition replay."""
    def __init__(self):
        self.validator=FlowValidator();self.last_data={}

    def advance(self,record,data):
        v=self.validator;kind=record['kind'];wall=record['wall_ms'];mono=record['mono_ns']
        if kind=='SESSION_START':
            v.begin_session(record['session_id'],wall_ms=wall,mono_ns=mono)
            self.last_data.clear();return
        v.last_wall,v.last_mono=wall,mono
        if kind=='SESSION_END':return
        if kind!='MARKET':raise ValueError('UNKNOWN_RECORD')
        topic=record['stream']
        if topic=='!serverShutdown':return
        validation=record['validation']
        if not validation['valid']:raise ValueError('INVALID_FRAME')
        v.counts[topic]+=1
        v.gaps_n+=int(validation['continuity']=='GAP')
        v.duplicates_n+=int(validation['continuity']=='DUPLICATE')
        v.by_topic[topic]=dict(id=data.get('a',data.get('lastUpdateId')),digest=None,
            wall_ms=wall,last_trade_id=data.get('l'),event_time_ms=data.get('E'),trade_time_ms=data.get('T'))
        self.last_data[topic]=data
        v.first_receipt=wall if v.first_receipt is None else v.first_receipt
        v.last_receipt=wall

    def snapshot(self):
        state=deepcopy(self.validator.__dict__)
        for topic,row in state['by_topic'].items():row['digest']=_sha(self.last_data[topic])
        return state


def validate_chunk(records,before,after):
    validator=FlowValidator();validator.__dict__.update(deepcopy(before))
    for record in records:revalidate(validator,record)
    if validator.__dict__!=after:raise ValueError('RAW_VALIDATION_BOUNDARY_MISMATCH')
    return len(records)


class BoundedValidation:
    def __init__(self,workers=4):
        if workers!=4:raise ValueError('FROZEN_WORKER_COUNT')
        self.pool=ProcessPoolExecutor(max_workers=workers);self.pending=deque()
        self.records_n=0;self.chunks_n=0

    def submit(self,records,before,after):
        if len(self.pending)>=4:self._take()
        self.pending.append(self.pool.submit(validate_chunk,records,before,after))

    def _take(self):
        self.records_n+=self.pending.popleft().result();self.chunks_n+=1

    def finish(self):
        try:
            while self.pending:self._take()
        finally:self.pool.shutdown(wait=True,cancel_futures=True)
        return dict(verified_records_n=self.records_n,verified_chunks_n=self.chunks_n,
                    worker_processes=4,original_validator_all_records=True)
