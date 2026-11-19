"""RFP01: hash-bound original raw standing quotes at strict receipt cuts."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from binance_demo_bot import receipt_flow_quotes as q
from binance_demo_bot.public_flow_dataset import FlowValidator


def raw_book(update,bid='100',ask='101',symbol='BTCUSDT'):
    return json.dumps(dict(stream=symbol.lower()+'@depth20@100ms',data=dict(
        lastUpdateId=update,bids=[[bid,'2']],asks=[[ask,'3']])) )


def chunks_at(folder,chunks):
    manifest={'chunks':[]};hashes={}
    for i,rows in enumerate(chunks):
        body=b''.join(json.dumps(r,sort_keys=True,separators=(',',':')).encode()+b'\n' for r in rows)
        raw=gzip.compress(body,mtime=0);name=f'chunk-{i:08d}.jsonl.gz'
        (folder/name).write_bytes(raw);digest=hashlib.sha256(raw).hexdigest();hashes[name]=digest
        manifest['chunks'].append(dict(file=name,sha256=digest,compressed_bytes=len(raw),
            uncompressed_bytes=len(body),uncompressed_sha256=hashlib.sha256(body).hexdigest(),
            records_n=len(rows),first_receipt_ms=min(r['wall_ms'] for r in rows),
            last_receipt_ms=max(r['wall_ms'] for r in rows),session_ids=sorted({r['session_id'] for r in rows})))
    return manifest,hashes


class ReceiptQuoteContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime')
        self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)

    def test_RFP01_strict_cut_and_equal_clock_preserves_record_order(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000)]
        for t,update,bid,ask in ((1000,1,'100','101'),(1100,2,'101','102'),(1100,3,'102','103')):
            rows.append(v.ingest(raw_book(update,bid,ask),wall_ms=t,mono_ns=t,session_id='s'))
        manifest,hashes=chunks_at(self.folder,[rows[:2],rows[2:]])
        source=q.ReceiptQuotes(self.folder,manifest,hashes)
        self.assertIsNone(source.quote_at('BTCUSDT',1000))
        self.assertEqual(source.quote_at('BTCUSDT',1100)['bid'],'100')
        found=source.quote_at('BTCUSDT',1101)
        self.assertEqual(found['bid'],'102');self.assertEqual(found['receipt_ms'],1100)
        self.assertEqual(found['record_index'],1)

    def test_RFP01_duplicate_does_not_refresh_age(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000)]
        for t in (1000,1500):rows.append(v.ingest(raw_book(1),wall_ms=t,mono_ns=t,session_id='s'))
        manifest,hashes=chunks_at(self.folder,[rows])
        source=q.ReceiptQuotes(self.folder,manifest,hashes)
        self.assertIsNotNone(source.quote_at('BTCUSDT',2000))
        self.assertIsNone(source.quote_at('BTCUSDT',2001))

    def test_RFP01_session_cannot_inherit_previous_book(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000),
            v.ingest(raw_book(1),wall_ms=1000,mono_ns=1000,session_id='s')]
        rows.append(dict(kind='SESSION_END',session_id='s',wall_ms=1500,mono_ns=1500))
        rows.append(v.begin_session('new',wall_ms=1600,mono_ns=1600))
        rows.append(v.ingest(raw_book(2,symbol='ETHUSDT'),wall_ms=1700,mono_ns=1700,session_id='new'))
        rows.append(v.ingest(raw_book(2),wall_ms=1800,mono_ns=1800,session_id='new'))
        manifest,hashes=chunks_at(self.folder,[rows])
        source=q.ReceiptQuotes(self.folder,manifest,hashes)
        self.assertIsNone(source.quote_at('BTCUSDT',1501))
        self.assertIsNone(source.quote_at('BTCUSDT',1701))
        self.assertEqual(source.quote_at('BTCUSDT',1801)['session_id'],'new')

    def test_RFP01_final_boundary_ignores_post_cut_frame(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000)]
        rows.append(v.ingest(raw_book(1),wall_ms=1900,mono_ns=1900,session_id='s'))
        rows.append(v.ingest(raw_book(2,'900','901'),wall_ms=2000,mono_ns=2000,session_id='s'))
        manifest,hashes=chunks_at(self.folder,[rows]);source=q.ReceiptQuotes(self.folder,manifest,hashes)
        self.assertEqual(source.quote_at('BTCUSDT',2000)['bid'],'100')

    def test_RFP01_tampered_raw_or_manifest_metadata_fails_closed(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000),
            v.ingest(raw_book(1),wall_ms=1100,mono_ns=1100,session_id='s')]
        manifest,hashes=chunks_at(self.folder,[rows]);bad=deepcopy(manifest)
        bad['chunks'][0]['last_receipt_ms']=1200
        with self.assertRaises(ValueError):q.ReceiptQuotes(self.folder,bad,hashes).quote_at('BTCUSDT',1201)
        path=self.folder/manifest['chunks'][0]['file'];path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaises(ValueError):q.ReceiptQuotes(self.folder,manifest,hashes).quote_at('BTCUSDT',1201)

    def test_RFP01_cache_bounded_and_final_used_hash_rechecked(self):
        v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000)]
        for i in range(4):rows.append(v.ingest(raw_book(i+1),wall_ms=1000+i*100,mono_ns=1000+i*100,session_id='s'))
        manifest,hashes=chunks_at(self.folder,[rows[:2],rows[2:3],rows[3:4],rows[4:]])
        source=q.ReceiptQuotes(self.folder,manifest,hashes,cache_limit=2)
        for t in (1001,1101,1201,1301):self.assertIsNotNone(source.quote_at('BTCUSDT',t))
        self.assertLessEqual(source.evidence()['cached_chunks_n'],2)
        source.verify_used()
        path=self.folder/manifest['chunks'][0]['file'];path.write_bytes(b'changed')
        with self.assertRaises(ValueError):source.verify_used()


if __name__=='__main__':unittest.main()
