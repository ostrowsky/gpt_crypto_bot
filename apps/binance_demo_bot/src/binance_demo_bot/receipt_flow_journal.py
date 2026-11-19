"""JR01/02 bounded gzip-member journals with verified prefix rollback."""
import errno
import gzip
import hashlib
import os
from pathlib import Path
import time
from .receipt_flow_direction import file_sha,write_json


def append_bytes(path,data):
    with Path(path).open('ab',buffering=0) as stream:
        cursor=0
        while cursor<len(data):
            n=stream.write(data[cursor:])
            if not n:raise OSError(errno.EIO,'short journal write')
            cursor+=n
        os.fsync(stream.fileno())


class DurableGzipText:
    def __init__(self,path,*,append=None,chunk_bytes=1024**2,compresslevel=3):
        self.path=Path(path).resolve();base=Path(__file__).resolve().parents[2]/'.runtime'
        if not self.path.is_relative_to(base.resolve()) or not self.path.name.endswith('.partial'):
            raise ValueError('JOURNAL_OUTSIDE_OWN_RUNTIME')
        if not 32<=chunk_bytes<=1024**2:raise ValueError('JOURNAL_BLOCK_LIMIT')
        with self.path.open('xb'):pass
        self.append=append or append_bytes;self.limit=chunk_bytes;self.level=compresslevel
        self.buffer=bytearray();self.body=hashlib.sha256();self.compressed=hashlib.sha256()
        self.offset=0;self.members=0;self.records=0;self.body_bytes=0
        self.recoveries=[];self.closed=False;self.complete=False

    def __enter__(self):return self

    def _restore_prefix(self):
        if self.path.resolve()!=self.path or self.path.stat().st_size<self.offset:raise ValueError('JOURNAL_PREFIX_CHANGED')
        actual=hashlib.sha256();remaining=self.offset
        with self.path.open('rb') as stream:
            while remaining:
                block=stream.read(min(65536,remaining))
                if not block:raise ValueError('JOURNAL_PREFIX_CHANGED')
                actual.update(block);remaining-=len(block)
        if actual.digest()!=self.compressed.digest():raise ValueError('JOURNAL_PREFIX_CHANGED')
        with self.path.open('r+b') as stream:
            stream.truncate(self.offset);stream.flush();os.fsync(stream.fileno())

    def _flush(self):
        if not self.buffer and self.members:return
        member=gzip.compress(bytes(self.buffer),compresslevel=self.level,mtime=0)
        for attempt in range(3):
            try:
                if self.path.resolve()!=self.path or self.path.stat().st_size!=self.offset:raise ValueError('JOURNAL_PREFIX_SIZE_CHANGED')
                self.append(self.path,member)
                if self.path.stat().st_size!=self.offset+len(member):raise OSError(errno.EIO,'short journal member')
                self.compressed.update(member);self.offset+=len(member);self.members+=1;self.buffer.clear();return
            except OSError as error:
                incident=dict(at_ms=time.time_ns()//1000000,errno=error.errno,attempt=attempt+1,
                    committed_offset=self.offset,bytes_before_rollback=self.path.stat().st_size)
                self.recoveries.append(incident)
                if error.errno not in (errno.EINVAL,errno.EIO) or attempt==2:
                    incident['recovered']=False
                    write_json(self.path.with_suffix(self.path.suffix+'.recovery.json'),self.recoveries)
                    raise
                try:self._restore_prefix()
                except BaseException:
                    incident['recovered']=False
                    write_json(self.path.with_suffix(self.path.suffix+'.recovery.json'),self.recoveries)
                    raise
                incident['recovered']=True
                write_json(self.path.with_suffix(self.path.suffix+'.recovery.json'),self.recoveries)

    def write(self,text):
        if self.closed:raise ValueError('JOURNAL_CLOSED')
        if not isinstance(text,str):raise TypeError('JOURNAL_TEXT_ONLY')
        data=text.replace('\n',os.linesep).encode('utf-8')
        if len(data)>self.limit:raise ValueError('JOURNAL_RECORD_LIMIT')
        if self.buffer and len(self.buffer)+len(data)>self.limit:self._flush()
        self.buffer.extend(data);self.body.update(data);self.body_bytes+=len(data);self.records+=data.count(b'\n')
        return len(text)

    def close(self):
        if self.closed:return
        self._flush()
        if file_sha(self.path)!=self.compressed.hexdigest():raise ValueError('JOURNAL_COMPRESSED_HASH')
        self.closed=True;self.complete=True
        write_json(self.path.with_suffix(self.path.suffix+'.storage.json'),self.proof())

    def _abort(self):
        self.complete=False;self.closed=True
        try:
            if self.buffer:self.path.with_suffix(self.path.suffix+'.aborted-buffer.jsonl').write_bytes(self.buffer)
            write_json(self.path.with_suffix(self.path.suffix+'.storage.json'),self.proof())
        except OSError:pass

    def __exit__(self,kind,error,tb):
        if kind is not None:self._abort();return False
        try:self.close()
        except BaseException:self._abort();raise
        return False

    def proof(self):
        return dict(complete=self.complete,codec='CONCATENATED_GZIP_MEMBERS',members_n=self.members,
            records_n=self.records,uncompressed_bytes=self.body_bytes,uncompressed_sha256=self.body.hexdigest(),
            compressed_bytes=self.offset,compressed_sha256=self.compressed.hexdigest(),
            recoveries_n=sum(e.get('recovered') is True for e in self.recoveries),io_incidents=self.recoveries)
