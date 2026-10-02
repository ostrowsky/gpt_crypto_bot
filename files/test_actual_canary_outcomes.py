import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import actual_canary_outcomes as canary
import policy_runtime_receipts as receipts
import validated_ranker_rollout as release
from forward_evidence_service import atomic
from logical_learning_authority import LogicalAuthority
from test_validated_ranker_rollout import KEY,AUTHORITY,CHAMPION,CANDIDATE,NOW,ticket


class CanaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.release=self.root/'release'
        self.authorization=ticket()
        self.sym=next('C'+str(i)+'USDT' for i in range(10000)
                      if release.assigned('C'+str(i)+'USDT',self.authorization['body']))
        self.entry=(NOW-900)*1000
        self.exit=NOW*1000
        self.positions=self.root/'positions.json'
        score={'sym':self.sym,'tf':'15m','bar_ts':self.entry,'bonus':.5,
               'candidate_sha256':self.authorization['body']['candidate_sha256'],
               'ticket_sha256':release.sha(release.canonical(self.authorization['body']))}
        atomic(receipts.context_path(self.release,self.sym,'15m',self.entry),
               {'score':score,'authorization':self.authorization})
        atomic(self.positions,{self.sym:{'entry_ts':self.entry,'tf':'15m','entry_price':100.}})
        self.admission=receipts.admission(self.release,KEY,CHAMPION,self.positions,
                                         self.sym,'15m',self.entry,NOW)
        self.request=lambda:canary.request_exit(self.release,KEY,CHAMPION,self.sym,'15m',
                                                self.entry,self.exit,105.,'ATR',NOW+1)
        self.registry=self.root/'evaluator'
        atomic(self.registry/'portfolios'/'champion.json',json.loads(CHAMPION))
        # Champion bytes are frozen exactly, not reformatted by JSON exporter.
        (self.registry/'portfolios'/'champion.json').write_bytes(CHAMPION)
        (self.registry/'portfolios'/'candidate.json').write_bytes(CANDIDATE)
        from learning_cohort_controller import CONTRACT
        atomic(self.registry/'portfolios'/'cohort.json',{'contract':CONTRACT,'phase':'canary',
               'state':'COLLECTING','start_ms':self.entry})
        self.deployment={'registry':str(self.registry),'release_root':str(self.release)}

    def collect(self):
        with patch.object(canary,'material',return_value=(LogicalAuthority(AUTHORITY),KEY)):
            return canary.tick(self.deployment,NOW+5)

    def test_sell_intent_is_not_closed_trade(self):
        self.request()
        self.assertEqual(canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+2),0)
        result=self.collect()
        self.assertEqual(result['closed'],0)
        self.assertEqual(result['open_or_missing'],1)
        self.assertIsNone(result['diagnostic_net_mean_pct'])

    def test_durable_close_and_restart_count_once(self):
        self.request()
        atomic(self.positions,{})
        self.assertEqual(canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+2),1)
        self.assertEqual(canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+3),0)
        self.assertTrue(self.request())
        result=self.collect()
        self.assertEqual(result['closed'],1)
        self.assertEqual(result['admissions'],1)
        self.assertEqual(result['open_or_missing'],0)
        self.assertAlmostEqual(result['diagnostic_net_mean_pct'],(1.05*(1-.00125)**2-1)*100)
        self.assertFalse(result['runtime_eligible'])
        self.assertFalse((self.registry/'portfolios'/'canary_unsigned.json').exists())

    def test_same_symbol_reentry_is_distinct(self):
        self.request()
        atomic(self.positions,{self.sym:{'entry_ts':self.exit,'tf':'15m','entry_price':105.}})
        self.assertEqual(canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+2),1)

    def test_conflicting_repeat_exit_rejected(self):
        self.request()
        with self.assertRaisesRegex(ValueError,'changed'):
            canary.request_exit(self.release,KEY,CHAMPION,self.sym,'15m',self.entry,self.exit,99.,'ATR',NOW+2)
        atomic(self.positions,{})
        with self.assertRaisesRegex(ValueError,'conflicting'):
            canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+3)
        self.assertEqual(self.collect()['state'],'BLOCKED')

    def test_invalid_or_future_exit_rejected(self):
        for ts,price in [(self.entry-1,100),(int((NOW+10)*1000),100),(self.exit,float('nan')),(self.exit,0)]:
            with self.subTest(ts=ts,price=price),self.assertRaises(ValueError):
                canary.request_exit(self.release,KEY,CHAMPION,self.sym,'15m',self.entry,ts,price,'x',NOW+1)

    def test_forged_exit_is_blocked_not_profit(self):
        self.request()
        atomic(self.positions,{})
        canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+2)
        path=next((self.release/'actual_canary'/'exits').glob('*.json'))
        value=json.loads(path.read_bytes())
        value['body']['request']['body']['exit_price']=500
        atomic(path,value)
        result=self.collect()
        self.assertEqual(result['state'],'BLOCKED')
        self.assertNotIn('diagnostic_net_mean_pct',result)

    def test_orphan_exit_is_blocked(self):
        atomic(self.release/'actual_canary'/'exits'/'orphan.json',{})
        self.assertEqual(self.collect()['state'],'BLOCKED')

    def test_unsafe_cost_assumption_rejected(self):
        self.request()
        atomic(self.positions,{})
        with self.assertRaises(ValueError):
            canary.finalize(self.release,KEY,CHAMPION,self.positions,fee_bps=0,now=NOW+2)

    def test_snapshot_still_contains_entry_invalidates_receipt(self):
        self.request()
        original=json.loads(self.positions.read_bytes())
        atomic(self.positions,{})
        canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+2)
        path=next((self.release/'actual_canary'/'exits').glob('*.json'))
        body=release.unseal(json.loads(path.read_bytes()),KEY)
        body['persisted_positions']=original
        body['positions_canonical_sha256']=release.sha(release.canonical(original))
        atomic(path,release.seal(body,KEY))
        self.assertEqual(self.collect()['state'],'BLOCKED')

    def test_failed_persistence_never_finalizes(self):
        import monitor
        with patch.object(monitor.config,'POSITIONS_FILE',str(self.positions),create=True), \
             patch.object(monitor,'_pos_to_dict',return_value={'entry_price':100}), \
             patch.object(monitor._os_pers,'replace',side_effect=OSError('disk failed')), \
             patch.object(canary,'runtime') as observer:
            monitor.save_positions({self.sym:object()})
            observer.assert_not_called()

    def test_successful_persistence_invokes_finalizer(self):
        import monitor
        with patch.object(monitor.config,'POSITIONS_FILE',str(self.positions),create=True), \
             patch.object(canary,'runtime') as observer:
            monitor.save_positions({})
            observer.assert_called_once_with('finalize',str(self.positions))

    def test_all_position_removals_have_exit_observation(self):
        import ast
        import monitor
        source=Path(monitor.__file__).read_text(encoding='utf-8')
        lines=source.splitlines()
        deletions=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Delete)
                   and any(ast.unparse(target).startswith('state.positions[') for target in node.targets)]
        self.assertEqual(len(deletions),8)
        for node in deletions:
            preceding='\n'.join(lines[max(0,node.lineno-24):node.lineno-1])
            self.assertTrue('_register_suspicious_reentry_watch(' in preceding
                            or "canary_observation('request'" in preceding,node.lineno)

    def test_previous_cohort_entries_not_counted(self):
        path=self.registry/'portfolios'/'cohort.json'
        cohort=json.loads(path.read_bytes())
        cohort['start_ms']=(NOW+1)*1000
        atomic(path,cohort)
        self.assertEqual(self.collect()['admissions'],0)

    def test_sealed_cohort_not_presented_as_canary(self):
        path=self.registry/'portfolios'/'cohort.json'
        cohort=json.loads(path.read_bytes())
        cohort['phase']='sealed'
        atomic(path,cohort)
        result=self.collect()
        self.assertEqual(result['state'],'WAITING_CANARY_COHORT')
        self.assertNotIn('closed',result)

    def test_delayed_save_not_a_causal_close(self):
        self.request()
        atomic(self.positions,{})
        with self.assertRaisesRegex(ValueError,'budget'):
            canary.finalize(self.release,KEY,CHAMPION,self.positions,now=NOW+122)


if __name__=='__main__': unittest.main()
