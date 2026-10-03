import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import replay_backtest as rb
import prospective_policy_portfolios as producer
import verify_policy_stream_parity as verifier
from replay_closed_trade_accounting import extend_index, reconcile
from validated_ranker_rollout import canonical

STEP = 900000


def trade(entry=26*STEP, exit=27*STEP):
    return rb.ReplayTrade(sym='BTCUSDT', tf='15m', mode='trend', entry_ts=entry,
                         exit_ts=exit, entry_price=100., exit_price=101.,
                         entry_i=25, trail_k=2., max_hold_bars=16, trail_stop=0.)


class AccountingTests(unittest.TestCase):
    def test_reconcile_cumulative_annotations_not_execution_fields(self):
        row = asdict(trade())
        index = {}
        extend_index(index, [row])
        snapshot = dict(row, cooldown_blocked_count=2, cooldown_positive_blocked_count=1,
                        cooldown_harm_pct=1.5, entry_price=999.)
        state = json.loads(canonical({'last_closed_by_symbol': {'BTCUSDT': snapshot}}))
        reconcile(index, state)
        reconcile(index, state)  # cumulative, not additive/double counting
        self.assertEqual(row['cooldown_blocked_count'], 2)
        self.assertEqual(row['cooldown_harm_pct'], 1.5)
        self.assertEqual(row['entry_price'], 100.)

    def test_unknown_duplicate_regressing_and_invalid_records_fail(self):
        row = asdict(trade())
        index = {}
        extend_index(index, [row])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            extend_index(index, [dict(row)])
        for change in ({'entry_ts':25*STEP}, {'cooldown_harm_pct':float('nan')},
                       {'cooldown_blocked_count':True}, {'cooldown_positive_blocked_count':1},
                       {'cooldown_blocked_count':-1}):
            with self.assertRaises(ValueError):
                reconcile(index, {'last_closed_by_symbol': {'BTCUSDT': dict(row, **change)}})
        row['cooldown_blocked_count'] = 3
        with self.assertRaisesRegex(ValueError, 'regressing'):
            reconcile(index, {'last_closed_by_symbol': {'BTCUSDT':dict(row, cooldown_blocked_count=2)}})

    def test_trade_identity_survives_newer_trade_for_same_symbol(self):
        old, new = trade(), trade(28*STEP,29*STEP)
        index = {}
        extend_index(index, [old,new])
        reconcile(index, {'last_closed_by_symbol': {'BTCUSDT': dict(asdict(new), cooldown_blocked_count=1)}})
        self.assertEqual(old.cooldown_blocked_count, 0)
        self.assertEqual(new.cooldown_blocked_count, 1)

    def test_real_simulator_close_cooldown_restart_parity(self):
        data = np.zeros(64, dtype=rb._KLINE_DTYPE)
        data['t'] = np.arange(64)*STEP
        for k in ('o','h','l','c'): data[k] = 100+np.arange(64)*.1
        data['v'] = 10
        cache = {('BTCUSDT','15m'):(data,{'atr':np.ones(64)})}
        cache['BTCUSDT','4h'] = cache['BTCUSDT','15m']
        def candidate(i):
            return rb.ReplayCandidate(sym='BTCUSDT',tf='15m',mode='trend',ts_ms=i*STEP,
                i=i-1,price=float(data['c'][i-1]),trail_k=2,max_hold_bars=16,
                score=100,top_gainer_score=100)
        def exit_trade(t,data,feat,idx,**kw):
            t.exit_ts=kw['ts_ms']; t.exit_price=float(data['c'][idx]); return 'WEAK'
        frames = [26*STEP,27*STEP,28*STEP]
        snapshot = ({frames[0]:[candidate(26)],frames[2]:[candidate(28)]},set(frames),2)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(rb,'_update_trade_progress',side_effect=exit_trade), \
             patch.object(rb,'_load_temporal_scout_events',return_value=({},{})), \
             patch.object(rb,'_cooldown_bars_after_exit',return_value=2):
            output = Path(tmp)/'differences.json'
            result = asyncio.run(verifier.compare(['BTCUSDT'],cache,None,snapshot,
                                                   difference_output=output))
            self.assertEqual(result['state'], 'PASS')
            self.assertEqual(result['closed_trades'], 1)
            self.assertEqual(result['bulk_sha256'], result['stream_sha256'])
            self.assertEqual(json.loads(output.read_bytes())['differences'], [])

    def test_difference_paths_missing_null_order_and_numeric_representation(self):
        left={'closed':[{'exit_price':1.0}], 'state':{'x':None,'n':0}}
        right={'closed':[{'exit_price':2.0},{}], 'state':{'n':0.0}}
        result=list(verifier.differences(left,right))
        by_path={tuple(r['path']):r for r in result}
        self.assertEqual(by_path['closed',0,'exit_price']['bulk'], 1.0)
        self.assertFalse(by_path['state','x']['stream_present'])
        self.assertFalse(by_path['closed',1]['bulk_present'])
        self.assertIn(('state','n'),by_path)
        self.assertTrue(list(verifier.differences([1,2],[2,1])))

    def test_unknown_result_and_existing_evidence_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json'
            with patch.object(verifier,'_run',side_effect=ValueError('source drift')):
                result=asyncio.run(verifier.run(Path(tmp),Path(tmp),p))
            self.assertEqual(result['state'],'UNKNOWN')
            self.assertFalse(result['runtime_eligible'])
            raw=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'overwrite'):
                asyncio.run(verifier.run(Path(tmp),Path(tmp),p))
            self.assertEqual(p.read_bytes(),raw)

    def test_real_execution_difference_is_saved_not_reconciled_away(self):
        left, right = trade(), trade()
        right.exit_price = 102.
        async def simulate(*args, candidate_snapshot, stream_state, **kwargs):
            value = left if len(candidate_snapshot[1]) == 2 else right
            stream_state.update(open_positions=[], last_closed_by_symbol={})
            return ([value] if 26*STEP in candidate_snapshot[1] else []), None
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb,'simulate_portfolio',side_effect=simulate):
            p=Path(tmp)/'diff.json'
            result=asyncio.run(verifier.compare([],{},None,({}, {26*STEP,27*STEP},0),
                                                difference_output=p))
            self.assertEqual(result['state'],'FAIL')
            records=json.loads(p.read_bytes())['differences']
            price=next(row for row in records if row['path']==['closed',0,'exit_price'])
            self.assertEqual([price['bulk'],price['stream']],[101.,102.])
            self.assertEqual(price['bulk_trade_identity']['sym'],'BTCUSDT')

    def test_validation_is_atomic_before_applying_updates(self):
        row=asdict(trade())
        index={}
        extend_index(index,[row])
        state={'last_closed_by_symbol':{
            'BTCUSDT':dict(row,cooldown_blocked_count=2),
            'OTHER':dict(row,sym='OTHER')}}
        with self.assertRaises(ValueError): reconcile(index,state)
        self.assertEqual(row['cooldown_blocked_count'],0)

    def test_producer_checkpoint_preserves_post_close_annotations(self):
        frame = 1800000000000 // producer.DAY * producer.DAY
        incoming = {}
        for tf in ('15m', '1h', '4h'):
            step = rb.BAR_MS[tf]
            incoming['BTCUSDT/'+tf] = [dict(t=frame-(100-i)*step,
                o=100., h=100., l=100., c=100., v=10.) for i in range(100)]
        closed = asdict(trade(frame-3*STEP, frame-2*STEP))
        initial = {'arms': {arm: {'closed_trades': [dict(closed)],
            'candidate_state': {}, 'state': {'open_positions': [],
                'last_closed_by_symbol': {'BTCUSDT': dict(closed)}}}
            for arm in ('champion', 'candidate')}}
        async def simulate(*args, stream_state, **kwargs):
            stream_state['last_closed_by_symbol']['BTCUSDT'].update(
                cooldown_blocked_count=2, cooldown_positive_blocked_count=1,
                cooldown_harm_pct=1.5)
            return [], {}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(producer, 'frozen_arm', return_value=nullcontext()), \
             patch.object(producer, 'verify_registration'), \
             patch.object(producer.time, 'time', return_value=(frame+1000)/1000), \
             patch.object(rb, 'build_replay_candidate_snapshot', return_value=({}, set(), 0)), \
             patch.object(rb, 'simulate_portfolio', side_effect=simulate):
            root = Path(tmp)
            asyncio.run(producer.advance(root, {'symbols': ['BTCUSDT']},
                json.loads(canonical(initial)), incoming, frame, frame+1000))
            stored = json.loads((root/'checkpoint.json').read_bytes())['body']
            for arm in ('champion', 'candidate'):
                row = stored['arms'][arm]['closed_trades'][0]
                self.assertEqual(row['cooldown_blocked_count'], 2)
                self.assertEqual(row['cooldown_harm_pct'], 1.5)
                self.assertEqual(row['exit_price'], closed['exit_price'])
            self.assertEqual(initial['arms']['champion']['closed_trades'][0]['cooldown_blocked_count'], 0)


if __name__ == '__main__': unittest.main()
