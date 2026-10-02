import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import replay_backtest as rb
import verify_candidate_population_parity as verifier
from forward_evidence_service import atomic

STEP=verifier.STEP


def data(tf,n=40):
    value=np.zeros(n,dtype=rb._KLINE_DTYPE)
    value['t']=np.arange(n)*rb.BAR_MS[tf]
    for key in ('o','h','l','c'): value[key]=100+np.arange(n)
    value['v']=10
    return value


def candidate(frame,score=40):
    return rb.ReplayCandidate(sym='BTCUSDT',tf='15m',mode='trend',ts_ms=frame,
        i=frame//STEP-1,price=100,trail_k=2,max_hold_bars=10,score=score)


class CandidateParityTests(unittest.TestCase):
    def test_prefix_contains_no_future_price(self):
        value=data('15m')
        with patch.object(rb,'compute_features',return_value={}):
            prefix,_=verifier.closed_prefix(value,'15m',3*STEP)
        self.assertEqual(list(prefix['t']),[0,STEP,2*STEP,3*STEP])
        self.assertEqual(list(prefix['c']),[100,101,102,102])
        self.assertEqual(value['c'][3],103)

    def test_no_closed_row_is_missing_not_fabricated(self):
        self.assertIsNone(verifier.closed_prefix(data('1h'),'1h',STEP))

    def test_fields_and_duplicates_are_compared(self):
        frame=26*STEP
        self.assertTrue(verifier.difference([candidate(frame)],[candidate(frame)],frame)['equal'])
        self.assertFalse(verifier.difference([candidate(frame)],[candidate(frame,41)],frame)['equal'])
        self.assertFalse(verifier.difference([candidate(frame)],[candidate(frame)]*2,frame)['equal'])

    def test_empty_and_bad_clock(self):
        self.assertEqual(verifier.difference([],[],STEP)['batch_count'],0)
        with self.assertRaisesRegex(ValueError,'clock'):
            verifier.difference([candidate(26*STEP)],[],27*STEP)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            verifier.difference([candidate(26*STEP,float('nan'))],[],26*STEP)

    def run_compare(self,mismatch=False,empty=False):
        start,end=26*STEP,30*STEP
        cache={('BTCUSDT',tf):(data(tf),{}) for tf in ('15m','1h','4h')}
        seen=[]
        async def builder(*args,**kwargs):
            frame=kwargs.get('frame_ms')
            if frame is None:
                return ({} if empty else {start:[candidate(start)]}),set(),0
            seen.append(frame)
            state=kwargs['candidate_stream_state']
            self.assertEqual(state.get('last_frame'),seen[-2] if len(seen)>1 else None)
            state['last_frame']=frame
            pack=args[2]['BTCUSDT','15m'][0]
            self.assertEqual(int(pack[-1]['t']),frame)
            raw={} if empty or frame!=start else {frame:[candidate(frame,41 if mismatch else 40)]}
            return raw,set(),sum(len(v) for v in raw.values())
        with patch.object(rb,'compute_features',return_value={}), \
             patch.object(rb,'_build_bull_day_context',return_value=None), \
             patch.object(rb,'build_replay_candidate_snapshot',side_effect=builder):
            result=asyncio.run(verifier.compare(['BTCUSDT'],cache,start,end))
        self.assertLess(seen[0],start)
        return result

    def test_complete_stream_warmup_restart_and_bounds(self):
        result=self.run_compare()
        self.assertEqual(result['state'],'PASS')
        self.assertEqual(result['frames'],4)
        self.assertEqual(result['batch_candidates'],1)
        self.assertEqual(result['stream_candidates'],1)
        self.assertFalse(result['runtime_eligible'])
        self.assertIn('not_monitor_buy_path',result['scope'])

    def test_actual_mismatch_is_fail_not_unknown(self):
        result=self.run_compare(mismatch=True)
        self.assertEqual(result['state'],'FAIL')
        self.assertEqual(result['mismatched_frames'],1)
        self.assertEqual(result['first_mismatch']['frame'],26*STEP)

    def test_zero_candidate_denominator_has_no_percentage(self):
        result=self.run_compare(empty=True)
        self.assertEqual(result['batch_candidates'],0)
        self.assertNotIn('candidate_match_fraction',result)

    def test_invalid_interval_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(verifier.compare([],{},1,STEP))

    def test_input_drift_unknown_and_not_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic(root/'archive'/'manifest.json',{'start_ms':STEP,'end_ms':2*STEP})
            atomic(root/'models'/'registration.json',{'archive_sha256':'0'*64})
            result=asyncio.run(verifier.run(root/'archive',root/'models',root/'result.json'))
            self.assertEqual(result['state'],'UNKNOWN')
            self.assertFalse(result['runtime_eligible'])
            self.assertFalse(result['closed_loop'])
            self.assertNotEqual(result['contract'],'independent-full-policy-validation-v1')

    def test_missing_inputs_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=asyncio.run(verifier.run(root/'missing',root/'models',root/'result.json'))
            self.assertEqual(result['state'],'UNKNOWN')
            self.assertFalse(result['runtime_eligible'])

    def test_shortened_archive_not_maximum_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic(root/'archive'/'manifest.json',{'start_ms':STEP,'end_ms':2*STEP})
            from validated_ranker_rollout import sha
            digest=sha((root/'archive'/'manifest.json').read_bytes())
            atomic(root/'models'/'registration.json',{'archive_sha256':digest,
                   'maximum_available_bounds':[0,3*STEP]})
            result=asyncio.run(verifier.run(root/'archive',root/'models',root/'result.json'))
            self.assertEqual(result['state'],'UNKNOWN')
            self.assertIn('maximum',result['reason'])


if __name__=='__main__': unittest.main()
