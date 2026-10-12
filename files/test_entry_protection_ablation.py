import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import replay_backtest as rb
import run_entry_protection_ablation as audit


class ProtectionAblationTests(unittest.TestCase):
    def test_four_isolated_arms(self):
        self.assertEqual([audit.options(a) for a in audit.ARMS],
            [(True,True),(False,True),(True,False),(False,False)])
        with self.assertRaises(ValueError): audit.options('unknown')

    def test_invalid_account_never_becomes_profit_evidence(self):
        for report in ({}, {'decision_grade':False}):
            self.assertEqual(audit.financial_state(report),'INVALID_ACCOUNT_EVIDENCE')
        self.assertEqual(audit.financial_state({'decision_grade':True}),'DIAGNOSTIC_ONLY')

    def test_output_preserves_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            audit.publish(path,{'state':'UNKNOWN'})
            with self.assertRaises(FileExistsError): audit.publish(path,{'state':'PASS'})

    def test_real_simulator_default_blocks_but_opt_in_bypasses_capacity(self):
        data=np.zeros(70,dtype=rb._KLINE_DTYPE)
        step=rb.BAR_MS['15m']
        data['t']=1774994400000+np.arange(70)*step
        for key in ('o','h','l','c'): data[key]=100.
        data['v']=10.
        feat={'atr':np.ones(70)}
        frame=int(data['t'][60])+step
        candidates=[rb.ReplayCandidate(sym=s,tf='15m',mode='breakout',ts_ms=frame,
            i=60,price=100.,trail_k=2.,max_hold_bars=6,score=100.,top_gainer_score=100.)
            for s in ('BTCUSDT','ETHUSDT')]
        packs={(s,'15m'):(data,feat) for s in ('BTCUSDT','ETHUSDT')}
        for kwargs,expected in (({},1),({'enforce_position_capacity':False},2)):
            state={}
            with patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
                asyncio.run(rb.simulate_portfolio(['BTCUSDT','ETHUSDT'],['15m'],packs,
                    {s:(data,feat) for s in ('BTCUSDT','ETHUSDT')},{},None,
                    max_open_positions=1,enable_replacement=False,replace_min_delta=8,
                    variant='baseline',candidate_snapshot=({frame:candidates},{frame},2),
                    stream_state=state,**kwargs))
            self.assertEqual(len(state['open_positions']),expected)
        state={}
        with patch.object(rb,'_load_temporal_scout_events',return_value=({},{})), \
             patch.object(rb,'_signal_cluster_cap_replay',return_value=1):
            _,stats=asyncio.run(rb.simulate_portfolio(['BTCUSDT','ETHUSDT'],['15m'],packs,
                {s:(data,feat) for s in ('BTCUSDT','ETHUSDT')},{},None,
                max_open_positions=1,enable_replacement=True,replace_min_delta=8,
                variant='score_replace_cluster',top_gainer_score_min=34.,
                candidate_snapshot=({frame:candidates},{frame},2),
                stream_state=state,enforce_position_capacity=False))
        self.assertEqual(len(state['open_positions']),1)
        self.assertEqual(stats.skipped_cluster_cap,1)

    def test_runner_regenerates_chase_population_and_preserves_capital_contract(self):
        data=np.zeros(70,dtype=rb._KLINE_DTYPE)
        data['t']=1774994400000+np.arange(70)*900000
        for key in ('o','h','l','c'): data[key]=100.
        data['v']=10.
        account={'decision_grade':False,'portfolio':{},'net_alpha_after_costs':None,
                 'coverage':{'contract_violations':['capacity_exceeded']}}
        chase_seen=[]
        async def builder(*args,**kwargs):
            chase_seen.append(audit.config.TOP_GAINER_CHASE_GUARD_ENABLED)
            return {},set(),0
        sim=AsyncMock(return_value=([],rb.ReplayRunStats()))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/'archive'; archive.mkdir()
            (archive/'manifest.json').write_text(json.dumps({'start_ms':1774994400000,
                'end_ms':1774994400000+69*900000,'archive_start_ms':1774994400000,
                'eligible_symbols':['BTCUSDT'],'input_hashes':{}}))
            original=audit.config.TOP_GAINER_CHASE_GUARD_ENABLED
            with patch.object(audit,'frozen_arm',return_value=nullcontext()), \
                 patch.object(audit.rb,'_build_market_cache_index',return_value={}), \
                 patch.object(audit.rb,'_load_cached_klines',return_value=data), \
                 patch.object(audit,'validate_series'), \
                 patch.object(audit.rb,'compute_features',return_value={}), \
                 patch.object(audit.rb,'_aggregate_1h_to_4h',return_value=data), \
                 patch.object(audit.rb,'_build_bull_day_context',return_value=None), \
                 patch.object(audit.rb,'build_replay_candidate_snapshot',builder), \
                 patch.object(audit.rb,'simulate_portfolio',sim), \
                 patch.object(audit,'evaluate_portfolio_alpha',return_value=account), \
                 patch('builtins.print'):
                result=asyncio.run(audit.run(archive,root/'output'))
            self.assertEqual(chase_seen,[original,False])
            self.assertEqual(audit.config.TOP_GAINER_CHASE_GUARD_ENABLED,original)
            self.assertEqual(sim.await_count,4)
            self.assertEqual([c.kwargs['enforce_position_capacity'] for c in sim.call_args_list],
                             [True,False,True,False])
            self.assertTrue(all(c.kwargs['max_open_positions']==10 for c in sim.call_args_list))
            self.assertFalse(result['runtime_eligible'])
            self.assertFalse(result['production_guards_changed'])
            self.assertTrue(all(a['financial_state']=='INVALID_ACCOUNT_EVIDENCE'
                                for a in result['arms'].values()))


if __name__=='__main__': unittest.main()
