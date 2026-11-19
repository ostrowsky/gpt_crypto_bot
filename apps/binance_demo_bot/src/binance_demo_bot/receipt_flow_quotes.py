"""RFP01 strict stored receipt quotes; never contacts an exchange."""
from bisect import bisect_left
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from .financial import decimal_amount
from .flow_capture_audit import _verified_chunk_lines
from .receipt_flow_direction import file_sha


class ReceiptQuotes:
    def __init__(self,folder,manifest,expected_hashes,*,cache_limit=16):
        self.folder=Path(folder).resolve();self.items=manifest['chunks']
        if type(cache_limit) is not int or not 1<=cache_limit<=16:raise ValueError('QUOTE_CACHE_LIMIT')
        self.limit=cache_limit;self.cache=OrderedDict();self.used={};self.lookups_n=0
        self.first=[];previous=None
        for i,item in enumerate(self.items):
            name=item['file'];first=item['first_receipt_ms'];last=item['last_receipt_ms']
            if (name!=f'chunk-{i:08d}.jsonl.gz' or item['sha256']!=expected_hashes.get(name)
                    or first>last or previous is not None and previous>first):raise ValueError('QUOTE_MANIFEST')
            self.first.append(first);previous=last

    def _load(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        item=self.items[index];raw=(self.folder/item['file']).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256'] or len(raw)!=item['compressed_bytes']:raise ValueError('QUOTE_CHUNK_HASH')
        meta=[];books={};sessions=set();walls=[]
        for seq,line in enumerate(_verified_chunk_lines(raw,item)):
            r=json.loads(line);at=r['wall_ms'];kind=r['kind'];sid=r['session_id'];topic=r.get('stream')
            if walls and at<walls[-1]:raise ValueError('QUOTE_RECEIPT_REGRESSION')
            walls.append(at);sessions.add(sid);meta.append((at,kind,sid,topic))
            if kind!='MARKET' or not topic.endswith('@depth20@100ms'):continue
            if not r['validation']['valid']:raise ValueError('QUOTE_INVALID_FRAME')
            if r['validation']['continuity']=='DUPLICATE':continue
            data=json.loads(r['raw'])['data'];bid,bq=data['bids'][0];ask,aq=data['asks'][0]
            if min(decimal_amount(v) for v in (bid,bq,ask,aq))<=0 or decimal_amount(bid)>=decimal_amount(ask):
                raise ValueError('QUOTE_TOP_LEVEL')
            symbol=topic.split('@')[0].upper()
            quote=dict(symbol=symbol,bid=bid,ask=ask,bid_qty=bq,ask_qty=aq,receipt_ms=at,
                session_id=sid,chunk=item['file'],chunk_sha256=digest,record_index=seq)
            books.setdefault(symbol,[]).append(quote)
        if (not walls or walls[0]!=item['first_receipt_ms'] or walls[-1]!=item['last_receipt_ms'] or
                sorted(sessions)!=item['session_ids']):raise ValueError('QUOTE_CHUNK_METADATA')
        result=dict(meta=meta,walls=walls,books=books,
                    book_walls={s:[q['receipt_ms'] for q in rows] for s,rows in books.items()})
        self.used[item['file']]=digest;self.cache[index]=result
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return result

    def quote_at(self,symbol,at_ms):
        self.lookups_n+=1
        i=bisect_left(self.first,at_ms)-1
        if i<0:return None
        current=self._load(i);j=bisect_left(current['walls'],at_ms)-1
        if j<0:raise ValueError('QUOTE_MANIFEST_CLOCK')
        _,kind,sid,topic=current['meta'][j]
        if kind=='SESSION_END' or topic=='!serverShutdown':return None
        for index in range(i,-1,-1):
            if self.items[index]['last_receipt_ms']<at_ms-1000:break
            chunk=self._load(index);rows=chunk['books'].get(symbol,[])
            j=bisect_left(chunk['book_walls'].get(symbol,[]),at_ms)-1
            if j>=0:
                found=rows[j]
                if found['session_id']!=sid or not 0<at_ms-found['receipt_ms']<=1000:return None
                return dict(found)
        return None

    def verify_used(self):
        for name,digest in self.used.items():
            if file_sha(self.folder/name)!=digest:raise ValueError('QUOTE_SOURCE_CHANGED')

    def evidence(self):
        return dict(used_chunks=dict(sorted(self.used.items())),cached_chunks_n=len(self.cache),
                    cache_limit=self.limit,lookups_n=self.lookups_n,
                    time_basis='RECEIPT_STRICT_BEFORE',exchange_age='UNKNOWN',
                    prior_raw_validation='RFD_FULL_ORIGINAL_VALIDATOR_PROOF_REUSED')
