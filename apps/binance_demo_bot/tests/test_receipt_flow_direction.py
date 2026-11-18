"""RFD01..08: causal receipt-clock research, before implementation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from binance_demo_bot import receipt_flow_direction as rfd


def frame(t, kind='book', *, price=100., identity=None, maker=False, gap=False,
          duplicate=False, sid='s'):
    identity=t if identity is None else identity
    topic='btcusdt@'+('depth20@100ms' if kind=='book' else 'aggTrade')
    data=(dict(lastUpdateId=identity,bids=[[str(price),'2'],[str(price-1),'3']],
               asks=[[str(price+1),'4'],[str(price+2),'5']]) if kind=='book' else
          dict(e='aggTrade',s='BTCUSDT',a=identity,f=identity,l=identity,
               E=t+1,T=t,p=str(price),q='2',m=maker))
    return dict(kind='MARKET',wall_ms=t,mono_ns=t*1000000,session_id=sid,
                stream=topic,raw=json.dumps(dict(stream=topic,data=data)),
                validation=dict(valid=True,fatal=False,reasons=[],
                    continuity='GAP' if gap else 'DUPLICATE' if duplicate else 'CONTIGUOUS'))


def records(end=70000, **changes):
    out=[dict(kind='SESSION_START',wall_ms=0,mono_ns=0,session_id='s')]
    for t in range(0,end,500):
        out += [frame(t,price=100+t/100000),frame(t,'trade',maker=t%1000==0)]
    for i,value in changes.items():out[int(i)]=value
    return out


def grids(source, end=70000):
    engine=rfd.ReceiptGrid(('BTCUSDT',),start_ms=0,end_ms=end)
    result=[]
    for record in source:result.extend(engine.feed(record))
    result.extend(engine.finish())
    return result


class ReceiptFlowContracts(unittest.TestCase):
    def test_RFD07_grid_gap_emits_lazily_with_bounded_memory(self):
        engine=rfd.ReceiptGrid(('BTCUSDT',),start_ms=0,end_ms=1000000)
        emitted=engine.feed(dict(kind='SESSION_START',wall_ms=999999,
                                 mono_ns=999999000000,session_id='s'))
        self.assertFalse(isinstance(emitted,list),'Receipt gap must be streamed')
        iterator=iter(emitted)
        self.assertEqual(next(iterator)['at_ms'],0)
        self.assertLess(engine.next_ms,10000)

    def test_RFD01_strict_cutoff_and_future_prefix(self):
        source=records(22000)
        original=grids(source,22000)
        changed=deepcopy(source)
        for record in changed:
            if record.get('kind')=='MARKET' and record['wall_ms']>=15000:
                record['raw']=frame(record['wall_ms'],price=900)['raw'] if 'depth' in record['stream'] else record['raw']
        revised=grids(changed,22000)
        self.assertEqual([r for r in original if r['at_ms']<=15000],
                         [r for r in revised if r['at_ms']<=15000])
        row=next(r for r in original if r['at_ms']==15000)
        self.assertAlmostEqual(row['mid'],100+.145+.5)
        self.assertEqual(row['feature_end_ms'],14999)
        self.assertIsNotNone(row['features'])

    def test_RFD02_gap_discards_context_and_future_label(self):
        source=records()
        i=next(i for i,r in enumerate(source) if r.get('stream','').endswith('aggTrade') and r['wall_ms']==25000)
        source[i]['validation']['continuity']='GAP'
        rows=grids(source)
        self.assertIsNone(next(r for r in rows if r['at_ms']==30000)['features'])
        issued=rfd.matched_examples(rows,0,70000)
        before=next(r for r in issued if r['at_ms']==20000)
        self.assertIsNone(before['targets'])
        self.assertIsNotNone(before['features'])

    def test_RFD02_stale_quotes_are_missing_not_flat_labels(self):
        source=[r for r in records() if r['wall_ms']<30000 or r['wall_ms']>=40000]
        rows=grids(source)
        self.assertFalse(next(r for r in rows if r['at_ms']==35000)['valid'])
        self.assertIsNone(next(r for r in rows if r['at_ms']==35000)['mid'])
        self.assertIsNone(next(r for r in rows if r['at_ms']==45000)['features'])

    def test_RFD02_session_boundary_resets_even_equal_price(self):
        source=records()
        source.insert(next(i for i,r in enumerate(source) if r['wall_ms']==25000),
                      dict(kind='SESSION_END',wall_ms=25000,mono_ns=25000000000,session_id='s'))
        rows=grids(source)
        self.assertIsNone(next(r for r in rows if r['at_ms']==26000)['features'])

    def test_RFD03_duplicates_do_not_double_trade_volume(self):
        source=records(22000)
        at=next(i for i,r in enumerate(source) if r.get('stream','').endswith('aggTrade') and r['wall_ms']==14000)
        duplicate=deepcopy(source[at]);duplicate['validation']['continuity']='DUPLICATE'
        changed=source[:at+1]+[duplicate]+source[at+1:]
        a=grids(source,22000);b=grids(changed,22000)
        self.assertEqual(a,b)

    def test_RFD03_trade_maker_sign_and_actual_aggregate_quantity(self):
        source=records(22000)
        for r in source:
            if r.get('stream','').endswith('aggTrade'):
                d=json.loads(r['raw']);d['data'].update(m=False,f=1,l=999)
                r['raw']=json.dumps(d)
        row=next(r for r in grids(source,22000) if r['at_ms']==15000)
        self.assertEqual(row['features']['F']['trade_imbalance_1'],1.)
        self.assertAlmostEqual(row['features']['F']['log_trade_qty_1'],math.log1p(4))

    def test_RFD03_ofi_supply_demand_signs(self):
        old=(100.,2.,101.,4.)
        self.assertEqual(rfd.book_ofi(old,(100.,3.,101.,4.)),1.)
        self.assertEqual(rfd.book_ofi(old,(100.,2.,101.,5.)),-1.)
        self.assertEqual(rfd.book_ofi(old,(101.,3.,102.,4.)),7.)
        self.assertEqual(rfd.book_ofi(old,(99.,3.,100.,5.)),-7.)

    def test_RFD04_common_30_second_maturity_purge(self):
        issued=rfd.matched_examples(grids(records()),0,70000)
        self.assertIsNotNone(next(r for r in issued if r['at_ms']==35000)['targets'])
        self.assertIsNone(next(r for r in issued if r['at_ms']==40000)['targets'])
        self.assertTrue(all(set(r['targets'])=={5,10,30} for r in issued if r['targets'] is not None))

    def test_RFD04_calendar_splits_use_calendar_not_rows(self):
        from datetime import datetime
        ms=lambda s:int(datetime.fromisoformat(s).timestamp()*1000)
        reg={'full_days_start_ms':ms('2026-11-06T00:00:00+01:00'),
             'planned_end_ms':ms('2026-11-13T00:00:00+01:00'),'timezone':'Europe/Budapest'}
        bounds=rfd.calendar_splits(reg)
        self.assertEqual(bounds['train'],(reg['full_days_start_ms'],ms('2026-11-09T00:00:00+01:00')))
        self.assertEqual(bounds['test'],(ms('2026-11-11T00:00:00+01:00'),reg['planned_end_ms']))

    def test_RFD05_metrics_correct_denominators_and_zero_base(self):
        result=rfd.metrics([1.,-2.,0.],[.5,.1,0.],majority_sign=-1)
        self.assertEqual((result['direction_correct_n'],result['direction_N']),(1,2))
        self.assertEqual(result['three_sign_correct_n'],2)
        self.assertEqual(result['majority_correct_n'],1)
        self.assertAlmostEqual(result['zero_mae_bps'],1.)
        self.assertEqual(result['positive_n'],1)
        self.assertEqual(result['unchanged_n'],1)
        self.assertIsNone(rfd.metrics([],[],majority_sign=1)['mae_bps'])
        self.assertIsNone(rfd.metrics([0.],[0.],majority_sign=1)['direction_accuracy'])

    def test_RFD05_mismatched_metrics_raise_not_zip_truncate(self):
        with self.assertRaises(ValueError):rfd.metrics([1.],[],majority_sign=1)

    def test_RFD06_finite_sample_interval_rank_and_train_majority(self):
        self.assertEqual(rfd.conformal_radius([4.,1.,3.,2.],coverage=.6),3.)
        self.assertTrue(math.isinf(rfd.conformal_radius([1.],coverage=.9)))
        with self.assertRaises(ValueError):rfd.conformal_radius([])
        self.assertEqual(rfd.majority_direction([1.,-1.,-2.,0.]),-1)
        self.assertEqual(rfd.majority_direction([1.,-1.]),1)

    def test_RFD07_freeze_rejects_rewrite_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime') as folder:
            path=Path(folder)/'registration.json'
            rfd.freeze(path,{'source':'a'})
            rfd.freeze(path,{'source':'a'})
            with self.assertRaises(ValueError):rfd.freeze(path,{'source':'b'})
            path.write_text('{}')
            with self.assertRaises(ValueError):rfd.read_frozen(path)

    def test_RFD07_revalidation_detects_forged_stored_validation(self):
        from binance_demo_bot.public_flow_dataset import FlowValidator
        validator=FlowValidator()
        rfd.revalidate(validator,dict(kind='SESSION_START',session_id='s',wall_ms=0,mono_ns=0))
        raw=frame(1000);raw['validation']['continuity']='DUPLICATE'
        with self.assertRaisesRegex(ValueError,'VALIDATION'):rfd.revalidate(validator,raw)

    def test_RFD08_charts_select_by_feature_availability_not_labels(self):
        rows=[dict(symbol='BTCUSDT',at_ms=t,features={'P':{}},targets=None) for t in (10,20,30)]
        self.assertEqual(rfd.chart_origins(rows,('BTCUSDT',),(15,25)),[('BTCUSDT',20),('BTCUSDT',30)])
        rows[1]['targets']={5:999}
        self.assertEqual(rfd.chart_origins(rows,('BTCUSDT',),(15,25)),[('BTCUSDT',20),('BTCUSDT',30)])


if __name__=='__main__':unittest.main()
