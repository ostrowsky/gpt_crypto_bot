"""R2A03/04: immutable original lookup and exact native-candle compatibility."""
import copy
from contextlib import closing
from decimal import Decimal
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

import _support


ORIGIN = 900_000
PARENT = 900_000


def candle(at=ORIGIN, interval=60_000, *, volume='1.00000000', quote='100.00000000',
           bought='0.40000000', bought_quote='40.00000000', count=2, high='101', low='99'):
    return [at, '100', high, low, '100', volume, at + interval - 1,
            quote, count, bought, bought_quote, '0']


def fine(interval=60_000):
    return [candle(ORIGIN + i * interval, interval) for i in range(PARENT // interval)]


def original(interval=60_000):
    count = PARENT // interval
    return candle(interval=PARENT, volume=f'{count}.00000000', quote=f'{100*count}.00000000',
                  bought=format(Decimal('.4') * count, 'f'), bought_quote=f'{40*count}.00000000', count=count*2)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def archive(path, rows, *, symbol='AAAUSDT', source_sha=None, blob_sha=None, metadata=None):
    # Unicode is deliberately retained in the raw page, matching adapter._sha.
    blob = zlib.compress(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode())
    values = dict(start_ms=0, rows_n=len(rows), first_ms=rows[0][0] if rows else None,
                  last_ms=rows[-1][0] if rows else None, received_ms=ORIGIN+PARENT+10)
    values.update(metadata or {})
    with closing(sqlite3.connect(path)) as con, con:
        con.execute('CREATE TABLE pages (symbol TEXT,start_ms INTEGER,rows_n INTEGER,first_ms INTEGER,last_ms INTEGER,source_sha TEXT,blob_sha TEXT,body BLOB,received_ms INTEGER)')
        con.execute('INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)', (symbol, values['start_ms'], values['rows_n'],
            values['first_ms'], values['last_ms'], source_sha or canonical(rows),
            blob_sha or hashlib.sha256(blob).hexdigest(), blob, values['received_ms']))


class IntrabarValidationTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('binance_demo_bot.intrabar_data')

    def validate(self, rows=None, reference=None, interval=60_000, **kwargs):
        return self.api.validate_window(original(interval) if reference is None else reference,
            fine(interval) if rows is None else rows, interval_ms=interval,
            received_at_ms=ORIGIN+PARENT, **kwargs)

    def test_R2A04_exact_fifteen_minutes_and_ninehundred_seconds_match(self):
        for interval, expected in ((60_000, 15), (1_000, 900)):
            result = self.validate(interval=interval)
            self.assertEqual(result['state'], 'MATCH')
            self.assertEqual(result['reasons'], [])
            self.assertEqual(result['aggregate'][0], ORIGIN)
            self.assertEqual(result['aggregate'][6], ORIGIN+PARENT-1)
            self.assertEqual(result['aggregate'][8], expected*2)
            self.assertEqual(Decimal(result['aggregate'][5]), expected)
            self.assertTrue(all(v['exact_equal'] for v in result['differences'].values()))

    def test_R2A04_open_high_low_close_aggregate_in_temporal_order(self):
        rows = fine();rows[0][1]='100.001';rows[3][2]='103';rows[9][3]='97';rows[-1][4]='100.002'
        ref=original();ref[1]='100.001';ref[2]='103';ref[3]='97';ref[4]='100.002'
        result=self.validate(rows,ref)
        self.assertEqual(result['state'],'MATCH')
        self.assertEqual(list(map(Decimal,result['aggregate'][1:5])),list(map(Decimal,ref[1:5])))

    def test_R2A04_prices_and_trade_counts_have_zero_tolerance(self):
        for field,value,name in ((2,'101.0000000000000000000000001','high'),(8,31,'trade_count')):
            ref=original();ref[field]=value
            result=self.validate(reference=ref)
            self.assertEqual(result['state'],'MISMATCH')
            self.assertFalse(result['differences'][name]['within_bound'])
            self.assertEqual(Decimal(result['differences'][name]['bound']),0)

    def test_R2A04_volume_fixed_bound_accepts_exact_edge_and_rejects_above(self):
        ref=original();ref[5]='15.00000008'
        accepted=self.validate(reference=ref)
        self.assertEqual(accepted['state'],'MATCH')
        diff=accepted['differences']['base_volume']
        self.assertEqual(Decimal(diff['difference']),Decimal('0.00000008'))
        self.assertEqual(Decimal(diff['bound']),Decimal('0.00000008'))
        self.assertFalse(diff['exact_equal'])
        self.assertEqual(diff['rounding_assumption'],'ASSUMED_NEAREST_ROUNDING')
        ref[5]='15.00000009'
        self.assertEqual(self.validate(reference=ref)['state'],'MISMATCH')

    def test_R2A04_wire_precision_cannot_widen_volume_bound(self):
        ref=original();ref[5]='15.00000009'
        coarse=fine();precise=fine()
        for row in coarse:row[5]='1'
        for row in precise:row[5]='1.000000000000000000'
        a=self.validate(coarse,ref);b=self.validate(precise,ref)
        self.assertEqual(a['state'],b['state']);self.assertEqual(a['state'],'MISMATCH')
        self.assertEqual(a['differences']['base_volume']['bound'],b['differences']['base_volume']['bound'])

    def test_R2A04_finer_original_precision_freezes_finer_quantum(self):
        ref=original();ref[5]='15.0000000008'
        result=self.validate(reference=ref)
        self.assertEqual(result['state'],'MATCH')
        self.assertEqual(Decimal(result['differences']['base_volume']['bound']),Decimal('0.0000000008'))

    def test_R2A04_all_four_additive_volume_fields_are_checked(self):
        for field,name in ((5,'base_volume'),(7,'quote_volume'),(9,'taker_base_volume'),(10,'taker_quote_volume')):
            ref=original();ref[field]=format(Decimal(ref[field])+Decimal('.00001'),'f')
            result=self.validate(reference=ref)
            self.assertEqual(result['state'],'MISMATCH',name)
            self.assertFalse(result['differences'][name]['within_bound'])

    def test_R2A04_no_binary_float_rounding_or_default_decimal_context_loss(self):
        rows=fine();ref=original()
        price='123456789012345678901234567890.1234567890123456789'
        for row in rows:
            for field in (1,2,3,4):row[field]=price
        for field in (1,2,3,4):ref[field]=price
        result=self.validate(rows,ref)
        self.assertEqual(result['state'],'MATCH')
        self.assertEqual(Decimal(result['aggregate'][4]),Decimal(price))

    def test_R2A04_duplicate_gap_unsorted_and_wrong_count_are_incomplete(self):
        for change in ('duplicate','gap','unsorted','short','extra'):
            rows=fine()
            if change=='duplicate':rows[4]=copy.deepcopy(rows[3])
            if change=='gap':rows[4][0]+=1;rows[4][6]+=1
            if change=='unsorted':rows[3],rows[4]=rows[4],rows[3]
            if change=='short':rows.pop()
            if change=='extra':rows.append(candle(ORIGIN+PARENT))
            self.assertEqual(self.validate(rows)['state'],'INCOMPLETE',change)

    def test_R2A04_out_of_range_unclosed_and_bad_receipt_are_incomplete(self):
        for field,value in ((0,ORIGIN-60_000),(6,ORIGIN+60_000)):
            rows=fine();rows[0][field]=value
            self.assertEqual(self.validate(rows)['state'],'INCOMPLETE')
        self.assertEqual(self.api.validate_window(original(),fine(),received_at_ms=ORIGIN+PARENT-1)['state'],'INCOMPLETE')
        self.assertEqual(self.api.validate_window(original(),fine(),received_at_ms=True)['state'],'INCOMPLETE')

    def test_R2A04_malformed_fine_data_never_throws_or_becomes_mismatch(self):
        changes=((1,True),(2,'NaN'),(3,'Infinity'),(4,'0'),(5,'-1'),(7,None),
                 (8,True),(8,1.5),(9,'2'),(10,'101'),(11,'NaN'),(1,100.0))
        for field,value in changes:
            rows=fine();rows[0][field]=value
            self.assertEqual(self.validate(rows)['state'],'INCOMPLETE',(field,value))
        for rows in (None,{},['broken'],[r[:11] for r in fine()]):
            self.assertEqual(self.api.validate_window(original(),rows,
                received_at_ms=ORIGIN+PARENT)['state'],'INCOMPLETE')

    def test_R2A04_malformed_original_is_a_contract_error(self):
        for change in ('shape','timestamp','count','price','irregular','bool'):
            ref=original()
            if change=='shape':ref.pop()
            if change=='timestamp':ref[0]+=1
            if change=='count':ref[8]=-1
            if change=='price':ref[1]='NaN'
            if change=='irregular':ref[6]-=1
            if change=='bool':ref[0]=True
            with self.assertRaises(ValueError):self.validate(reference=ref)

    def test_R2A04_zero_volumes_are_valid_and_taker_cannot_exceed_total(self):
        rows=fine();ref=original()
        for row in rows+[ref]:
            for field in (5,7,9,10):row[field]='0.00000000'
            row[8]=0
        self.assertEqual(self.validate(rows,ref)['state'],'MATCH')
        rows[0][9]='0.00000001'
        self.assertEqual(self.validate(rows,ref)['state'],'INCOMPLETE')

    def test_R2A06_seconds_require_each_of_fifteen_minute_buckets(self):
        seconds=fine(1_000);minutes=[]
        for i in range(15):
            minutes.append(candle(ORIGIN+i*60_000,60_000,volume='60.00000000',quote='6000.00000000',
                bought='24.00000000',bought_quote='2400.00000000',count=120))
        ref=original(1_000)
        self.assertEqual(self.api.validate_seconds(ref,minutes,seconds,received_at_ms=ORIGIN+PARENT)['state'],'MATCH')
        minutes[0][2]='102';ref[2]='102';seconds[60][2]='102'
        self.assertEqual(self.validate(seconds,ref,interval=1_000)['state'],'MATCH')
        result=self.api.validate_seconds(ref,minutes,seconds,received_at_ms=ORIGIN+PARENT)
        self.assertEqual(result['state'],'MISMATCH')
        self.assertEqual(len(result['minute_checks']),15)
        self.assertEqual(sum(r['state']=='MISMATCH' for r in result['minute_checks']),2)

    def test_R2A06_missing_seconds_or_unaccepted_minutes_remain_incomplete(self):
        result=self.api.validate_seconds(original(),fine(),[],received_at_ms=ORIGIN+PARENT)
        self.assertEqual(result['state'],'INCOMPLETE')


class OriginalArchiveTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.intrabar_data')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_R2A03_exact_lookup_is_read_only_and_returns_page_hashes(self):
        path=self.root/'history.sqlite';rows=[original()];archive(path,rows)
        before=path.read_bytes();names={p.name for p in self.root.iterdir()}
        result=self.api.original_bar(path,'AAAUSDT',ORIGIN)
        self.assertEqual(result['row'],rows[0])
        self.assertEqual(result['page_source_sha256'],canonical(rows))
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual({p.name for p in self.root.iterdir()},names)

    def test_R2A03_unicode_symbol_and_canonical_page_hash_match_adapter(self):
        from binance_demo_bot.adapter import _sha
        path=self.root/'history.sqlite';rows=[original()];archive(path,rows,symbol='币安人生USDT')
        result=self.api.original_bar(path,'币安人生USDT',ORIGIN)
        self.assertEqual(result['page_source_sha256'],_sha(rows))
        value={'symbol':'币安人生USDT','rows':rows}
        self.assertEqual(self.api.canonical_sha256(value),_sha(value))

    def test_R2A03_missing_path_or_target_never_creates_archive(self):
        path=self.root/'missing.sqlite'
        with self.assertRaises(ValueError):self.api.original_bar(path,'AAAUSDT',ORIGIN)
        self.assertFalse(path.exists())
        archive(path,[original()])
        with self.assertRaises(ValueError):self.api.original_bar(path,'OTHERUSDT',ORIGIN)
        with self.assertRaises(ValueError):self.api.original_bar(path,'AAAUSDT',ORIGIN+PARENT)

    def test_R2A03_corrupt_compressed_and_source_hash_fail_closed(self):
        for field in ('source_sha','blob_sha'):
            path=self.root/f'{field}.sqlite';archive(path,[original()],**{field:'0'*64})
            with self.assertRaises(ValueError):self.api.original_bar(path,'AAAUSDT',ORIGIN)

    def test_R2A03_duplicate_irregular_or_page_metadata_mismatch_rejected(self):
        for variant in ('duplicate','irregular','metadata'):
            path=self.root/f'{variant}.sqlite';rows=[original()];metadata={}
            if variant=='duplicate':rows.append(copy.deepcopy(rows[0]))
            if variant=='irregular':rows[0][6]-=1
            if variant=='metadata':metadata={'rows_n':2}
            archive(path,rows,metadata=metadata)
            with self.assertRaises(ValueError):self.api.original_bar(path,'AAAUSDT',ORIGIN)

    def test_R2A03_duplicates_in_distinct_overlapping_pages_rejected(self):
        path=self.root/'history.sqlite';archive(path,[original()])
        with closing(sqlite3.connect(path)) as con, con:
            con.execute('INSERT INTO pages SELECT symbol,1,rows_n,first_ms,last_ms,source_sha,blob_sha,body,received_ms FROM pages')
        with self.assertRaises(ValueError):self.api.original_bar(path,'AAAUSDT',ORIGIN)

    def test_R2A03_invalid_lookup_arguments_fail_closed(self):
        for symbol,at in (('',ORIGIN),('AAAUSDT',True),('AAAUSDT',ORIGIN+1)):
            with self.assertRaises(ValueError):self.api.original_bar(self.root/'missing.sqlite',symbol,at)


if __name__=='__main__':unittest.main()
