import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import paired_full_policy_replay as paired
from test_validated_ranker_rollout import bundle, NOW
import independent_portfolio_gate as gate


class PairedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('champion','candidate','general'):
            (self.root/(name+'.json')).write_text(json.dumps({'runtime_eligible':True}))

    def test_disabled_runtime_and_zero_weight_block_instead_of_fake_learning(self):
        with patch.object(paired.config,'ML_CANDIDATE_RANKER_RUNTIME_ENABLED',False,create=True), \
             patch.object(paired.config,'ML_CANDIDATE_RANKER_SCORE_WEIGHT',0,create=True):
            self.assertEqual(len(paired.preflight({})),3)

    def test_live_provenance_not_overridden(self):
        with patch.object(paired.config,'ML_CANDIDATE_RANKER_RUNTIME_ENABLED',True,create=True), \
             patch.object(paired.config,'ML_CANDIDATE_RANKER_SCORE_WEIGHT',1,create=True):
            self.assertEqual(paired.preflight({}),['champion rejected by live provenance loader'])
            self.assertEqual(paired.preflight({'runtime_eligible':True}),[])

    def test_arm_isolation_restores_flags_and_model_loader(self):
        original = paired.config.VALIDATED_RANKER_ROLLOUT_ENABLED
        loader = paired.monitor._RANKER_MODEL_FILE
        with paired.frozen_arm(self.root,'candidate'):
            self.assertTrue(paired.config.VALIDATED_RANKER_ROLLOUT_ENABLED)
            self.assertEqual(paired.monitor._RANKER_MODEL_FILE,self.root/'champion.json')
            self.assertEqual(paired.rollout.select()[1]['stage'],'OFFLINE_REPLAY')
        self.assertEqual(paired.config.VALIDATED_RANKER_ROLLOUT_ENABLED,original)
        self.assertEqual(paired.monitor._RANKER_MODEL_FILE,loader)
        with paired.frozen_arm(self.root,'champion'):
            self.assertIsNone(paired.rollout.select())

    def test_separate_full_candidate_population_and_full_exits_for_each_arm(self):
        cache = {('BTCUSDT',tf):(None,{}) for tf in ('15m','1h','4h')}
        builder = AsyncMock(return_value=({},set(),0))
        sim = AsyncMock(return_value=([],paired.rb.ReplayRunStats()))
        with patch.object(paired,'bounded_cache',return_value=cache), \
             patch.object(paired.rb,'_build_bull_day_context',return_value=None), \
             patch.object(paired.rb,'build_replay_candidate_snapshot',builder), \
             patch.object(paired.rb,'simulate_portfolio',sim):
            result = asyncio.run(paired.run_arms(cache,['BTCUSDT'],0,900000,self.root))
        self.assertEqual(builder.await_count,2)
        self.assertEqual(sim.await_count,2)
        self.assertEqual(set(result),{'champion','candidate'})
        for call in sim.call_args_list:
            self.assertEqual(call.kwargs['max_open_positions'],10)
            self.assertEqual(call.kwargs['variant'],'score_replace_cluster')

    def test_rule_family_changes_only_separate_bonus_switch(self):
        (self.root/'champion.json').write_text(json.dumps({'contract':paired.rule_score.CONTRACT}))
        (self.root/'base_ranker.json').write_text('{}')
        with patch.object(paired.config,'ML_CANDIDATE_RANKER_RUNTIME_ENABLED',False,create=True):
            with paired.frozen_arm(self.root,'candidate'):
                self.assertTrue(paired.config.CERTIFIED_RULE_SCORE_POLICY_ENABLED)
                self.assertFalse(paired.config.VALIDATED_RANKER_ROLLOUT_ENABLED)
                self.assertFalse(paired.config.ML_CANDIDATE_RANKER_RUNTIME_ENABLED)
                self.assertEqual(paired.monitor._RANKER_MODEL_FILE,self.root/'base_ranker.json')
            with paired.frozen_arm(self.root,'champion'):
                self.assertFalse(paired.config.CERTIFIED_RULE_SCORE_POLICY_ENABLED)
                self.assertIsNone(paired.rollout.select())

    def test_unsigned_comparison_has_no_release_authority(self):
        result = gate.compare_accounts(bundle(),NOW)
        self.assertTrue(result['passed'])
        self.assertNotIn('certification_sha256',result)
        self.assertNotIn('runtime_eligible',result)
        with self.assertRaises(ValueError):
            gate.authorize({},b'a'*32,b'e'*32,b'{}',b'{}',now=NOW)

    def test_progress_does_not_replace_or_modify_series_builder(self):
        cache = {('BTCUSDT',tf):(None,{}) for tf in ('15m','1h','4h')}
        series = AsyncMock(return_value=[])
        async def build(*args, **kwargs):
            await paired.rb._build_candidates_for_symbol('BTCUSDT','1h','data',
                'features', variant='score_replace_cluster')
            return {}, set(), 0
        with patch.object(paired,'bounded_cache',return_value=cache), \
             patch.object(paired.rb,'_build_bull_day_context',return_value=None), \
             patch.object(paired.rb,'_build_candidates_for_symbol',series), \
             patch.object(paired.rb,'build_replay_candidate_snapshot',build), \
             patch.object(paired.rb,'simulate_portfolio',
                AsyncMock(return_value=([],paired.rb.ReplayRunStats()))), \
             patch('builtins.print') as log:
            asyncio.run(paired.run_arms(cache,['BTCUSDT'],0,900000,self.root))
            self.assertIs(paired.rb._build_candidates_for_symbol,series)
        self.assertEqual(series.await_count,2)
        for call in series.call_args_list:
            self.assertEqual(call.args,('BTCUSDT','1h','data','features'))
            self.assertEqual(call.kwargs,{'variant':'score_replace_cluster'})
        rows = [json.loads(call.args[0]) for call in log.call_args_list]
        self.assertEqual([row['arm'] for row in rows
            if row['phase']=='ARM_COMPLETE'],['champion','candidate'])
        self.assertTrue(all(call.kwargs['flush'] for call in log.call_args_list))

    def test_frozen_preflight_and_receipt_no_overwrite(self):
        archive = self.root/'archive'
        archive.mkdir()
        (archive/'manifest.json').write_text(json.dumps({'start_ms':0,'end_ms':180*86400000,
            'input_hashes':{}}))
        out = self.root/'result'
        with patch.object(paired,'preflight',return_value=['disabled']):
            result = asyncio.run(paired.run(archive,self.root/'champion.json',
                                           self.root/'candidate.json',out))
        self.assertEqual(result['state'],'BLOCKED')
        self.assertFalse(result['runtime_eligible'])
        receipt = json.loads((out/'receipt.json').read_bytes())
        self.assertEqual(receipt['result.json'],paired.sha(out/'result.json'))
        with self.assertRaises(FileExistsError):
            asyncio.run(paired.run(archive,self.root/'champion.json',self.root/'candidate.json',out))

    def test_market_tamper_blocks_before_replay(self):
        archive = self.root/'bad'
        (archive/'market').mkdir(parents=True)
        (archive/'market'/'x.json').write_bytes(b'changed')
        (archive/'manifest.json').write_text(json.dumps({'input_hashes':{'x.json':'wrong'}}))
        with self.assertRaisesRegex(ValueError,'market input drift'):
            asyncio.run(paired.run(archive,self.root/'champion.json',self.root/'candidate.json',self.root/'out'))


if __name__ == '__main__':
    unittest.main()
