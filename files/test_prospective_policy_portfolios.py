import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import prospective_policy_portfolios as producer
import replay_backtest as rb
import portfolio_alpha as alpha
import independent_portfolio_gate as gate
from validated_ranker_rollout import canonical, seal, sha
from logical_learning_authority import LogicalAuthority, CONTRACT
from test_validated_ranker_rollout import bundle, evidence, AUTHORITY, NOW, CANDIDATE, CHAMPION

STEP = producer.STEP


class ProspectivePortfolioTests(unittest.TestCase):
    def data(self, n=40):
        rows = np.zeros(n, dtype=rb._KLINE_DTYPE)
        rows['t'] = np.arange(n)*STEP
        rows['o'] = rows['h'] = rows['l'] = rows['c'] = 100
        rows['v'] = 10
        return rows

    def test_persistent_position_no_forced_sell_and_restart_exit(self):
        data = self.data()
        trade = rb.ReplayTrade(sym='BTCUSDT',tf='15m',mode='trend',entry_ts=30*STEP,
                              entry_price=100,entry_i=29,trail_k=2,max_hold_bars=10,trail_stop=0)
        state = {'open_positions':[asdict(trade)], 'cooldown_until':{'A': 44*STEP}}
        def run(ts, state):
            with patch.object(rb, '_load_temporal_scout_events', return_value=({}, {})):
                return asyncio.run(rb.simulate_portfolio(['BTCUSDT'], ['15m'],
                    {('BTCUSDT','15m'):(data,{})}, {'BTCUSDT':(data,{})}, {}, None,
                    max_open_positions=10,enable_replacement=False,replace_min_delta=8,
                    candidate_snapshot=({}, {ts}, 0), stream_state=state))
        with patch.object(rb, '_update_trade_progress', return_value=None):
            closed, _ = run(31*STEP, state)
        self.assertEqual(closed, [])
        self.assertEqual(len(state['open_positions']), 1)
        state = json.loads(canonical(state))  # real checkpoint/restart boundary
        with patch.object(rb, '_update_trade_progress', return_value='weak'):
            closed, _ = run(32*STEP, state)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].exit_reason, 'weak')
        self.assertEqual(state['open_positions'], [])
        self.assertEqual(state['cooldown_until']['A'], 44*STEP)
        self.assertIn('BTCUSDT', state['last_closed_by_symbol'])
        with self.assertRaises(ValueError):
            run(32*STEP, state)

    def test_candidate_stream_only_latest_closed_index_and_keeps_state(self):
        async def build(data, state, frame):
            return await rb._build_candidates_for_symbol('BTCUSDT', '15m', data, {}, {}, {}, None,
                                                       stream_state=state, frame_ms=frame)
        state = {}
        with patch.object(rb, '_entry_candidate', return_value=None) as entry, \
             patch.object(rb, '_discovery_catchup_replay_candidate', return_value=None):
            asyncio.run(build(self.data(), state, 39*STEP))
            self.assertEqual(entry.call_count, 1)
            self.assertEqual(entry.call_args.args[1], 38)
            asyncio.run(build(self.data(41), state, 40*STEP))
            self.assertEqual(entry.call_count, 2)
            self.assertEqual(entry.call_args.args[1], 39)
        self.assertEqual(state['last_i'], 39)

    def test_closed_merge_rejects_gap_revision_future_and_missing_latest(self):
        row = {'t':0,'o':100.,'h':100.,'l':100.,'c':100.,'v':10.}
        self.assertEqual(producer.merge_closed([], [row], STEP, STEP), [row])
        for incoming, frame in (([dict(row,c=101.)],STEP),
                                ([dict(row,t=2*STEP)],3*STEP), ([dict(row,t=STEP)],STEP),
                                ([row],2*STEP)):
            with self.assertRaises(ValueError):
                producer.merge_closed([row], incoming, STEP, frame)

    def test_clock_never_backfills_missing_forward_frames(self):
        producer.validate_clock({}, STEP, STEP+1000)
        for state, frame, now in (({},STEP,STEP-1), ({},STEP,STEP+120001),
                                 ({'last_frame':STEP},STEP,STEP),
                                 ({'last_frame':STEP},3*STEP,3*STEP)):
            with self.assertRaises(ValueError):
                producer.validate_clock(state, frame, now)

    def test_chain_corruption_blocks(self):
        body = {'frame':STEP}
        frames = [{'previous':None,'body':body,'sha256':sha(canonical(body))}]
        producer.verify_chain(frames)
        frames[0]['body']['frame'] += STEP
        with self.assertRaises(ValueError):
            producer.verify_chain(frames)

    def test_full_paired_frame_atomic_checkpoint_and_unsigned_export(self):
        frame = NOW*1000//STEP*STEP
        manifest = {'symbols':['BTCUSDT'],'registered_ms':frame-3*producer.DAY}
        incoming = {}
        for tf, step in rb.BAR_MS.items():
            end = frame//step*step
            incoming['BTCUSDT/'+tf] = [dict(t=end-(100-i)*step,
                o=100.,h=100.,l=100.,c=100.,v=10.) for i in range(100)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(producer, 'frozen_arm', return_value=nullcontext()), \
                 patch.object(producer, 'verify_registration'), \
                 patch.object(producer.time, 'time', return_value=(frame+1000)/1000), \
                 patch.object(rb, 'build_replay_candidate_snapshot', return_value=({},set(),0)), \
                 patch.object(rb, '_load_temporal_scout_events', return_value=({},{})):
                state = asyncio.run(producer.advance(root,manifest,{},incoming,frame,frame+1000))
            stored = json.loads((root/'checkpoint.json').read_bytes())
            self.assertEqual(stored['sha256'],sha(canonical(stored['body'])))
            self.assertEqual(set(state['arms']),{'candidate','champion'})
            self.assertEqual(state['last_frame'],frame)
            producer.export_unsigned(root,manifest,state)
            exported = json.loads((root/'sealed_unsigned.json').read_bytes())
            self.assertEqual(exported['prices']['BTCUSDT'],[[frame,100.]])
            self.assertEqual(exported['candidate_trades'],[])

    def test_partial_population_rejected_before_any_policy_step(self):
        with self.assertRaisesRegex(ValueError,'partial'):
            asyncio.run(producer.advance(Path('.'),{'symbols':['BTCUSDT']},{},{},STEP,STEP+1000))

    def test_open_holdings_valued_without_a_fake_sell(self):
        trade = {'sym':'A','entry_ts':STEP,'entry_price':100,'exit_ts':0,'exit_price':0,
                 'position_open':True}
        result = alpha._simulate_account([trade], price_series_by_symbol={'A':[(STEP,100),(2*STEP,110)]},
            valuation_timestamps=[STEP,2*STEP],capacity=10,initial_capital=10000,fee_bps=7.5,slippage_bps=5)
        self.assertGreater(result.ending_equity,10000)
        self.assertEqual(result.violations, [])

    def test_open_positions_do_not_make_closed_denominator_sufficient(self):
        value = bundle()
        value['candidate_trades'] = [dict(t,position_open=True,exit_ts=0,exit_price=0)
                                     for t in value['candidate_trades']]
        with self.assertRaisesRegex(ValueError, 'insufficient trades'):
            gate.compare_accounts(value,NOW)

    def test_logical_certificate_cannot_masquerade_as_os_isolation(self):
        raw, old = evidence(bundle())
        cert = dict(old['body'],contract=CONTRACT,isolation_mode='logical_same_user',
                    os_access_isolation=False,training_holdout_excluded=True)
        cert.pop('no_trainer_holdout_access')
        logical = LogicalAuthority(AUTHORITY)
        with patch.object(gate,'compare_accounts',return_value={'passed':True}):
            gate.evaluate_bundle(raw,seal(cert,AUTHORITY),logical,sha(CANDIDATE),sha(CHAMPION),NOW)
        with self.assertRaises(ValueError):
            gate.evaluate_bundle(raw,seal(dict(cert,os_access_isolation=True),AUTHORITY),
                                 logical,sha(CANDIDATE),sha(CHAMPION),NOW)
        with self.assertRaises(ValueError):
            gate.evaluate_bundle(raw,seal(cert,AUTHORITY),AUTHORITY,sha(CANDIDATE),sha(CHAMPION),NOW)


if __name__ == '__main__':
    unittest.main()
