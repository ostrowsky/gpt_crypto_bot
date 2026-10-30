"""EOS02/08/11 sealed full-scope orchestration and artifact integrity."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import _support
from binance_demo_bot.execution_order_study import (make_plan, verify_files, seal_registration,
    save_report, load_report, group_results, _summary)


class ExecutionStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime')
        self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.parent=dict(seeds=[11,23,37,53,71],splits=[['train',0,900000],['tune',900000,1800000],['test',1800000,2700000]])

    def test_EOS02_exact168_schedule_all_original_arms_seeds_segments_costs(self):
        plan=make_plan(self.parent,dict(CatBoost=.002,GRU=.002))
        self.assertEqual(len(plan),168); self.assertEqual(len({r['id'] for r in plan}),168)
        for order in ('STOP_FIRST','TARGET_FIRST'):
            for segment in ('tune','test'):
                for cost in ('base','stress'):
                    rows=[r for r in plan if (r['order'],r['segment'],r['cost'])==(order,segment,cost)]
                    self.assertEqual(len(rows),21)
                    for kind in ('PPO','MYOPIC','EXIT'):
                        self.assertEqual({r['seed'] for r in rows if r['kind']==kind},set(self.parent['seeds']))
        self.assertFalse(any(r['segment']=='train' for r in plan))

    def test_EOS02_changed_seed_population_is_rejected(self):
        parent=copy.deepcopy(self.parent); parent['seeds'].pop()
        with self.assertRaises(ValueError): make_plan(parent,dict(CatBoost=.002,GRU=.002))

    def test_EOS08_file_hash_verification_detects_tampering_unicode(self):
        p=self.root/'牛来.json'; p.write_text('native evidence',encoding='utf-8')
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
        self.assertEqual(verify_files(files)['files_n'],1)
        p.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'HASH'): verify_files(files)

    def test_EOS08_registration_immutable_and_resume_exact(self):
        binding=dict(protocol='EOS-1',models=['PPO'],promotion_allowed=False)
        first=seal_registration(self.root,binding); second=seal_registration(self.root,binding)
        self.assertEqual(first,second)
        with self.assertRaises(ValueError): seal_registration(self.root,dict(binding,models=['GRU']))

    def test_EOS11_compressed_report_resume_binding_and_integrity(self):
        task=make_plan(self.parent,dict(CatBoost=.002,GRU=.002))[0]
        detail=dict(state='COMPLETE',net_pnl_usdt=-.5,equity_curve=[{'origin_ms':900000,'equity_after':99.5}])
        self.assertIsNone(load_report(self.root,task,'a'*64))
        save_report(self.root,task,'a'*64,detail)
        self.assertEqual(load_report(self.root,task,'a'*64),detail)
        with self.assertRaises(ValueError): load_report(self.root,task,'b'*64)
        path=next((self.root/'portfolios').glob('*.gz')); path.write_bytes(path.read_bytes()+b'bad')
        with self.assertRaisesRegex(ValueError,'INTEGRITY'): load_report(self.root,task,'a'*64)

    def test_EOS10_seed_results_are_mean_not_sum_or_best_seed(self):
        rows=[]
        for seed,net in zip(self.parent['seeds'],(-1,-2,-3,-4,-5)):
            rows.append(dict(order='STOP_FIRST',segment='test',cost='base',label=f'PPO-{seed}',
                state='COMPLETE',net_pnl_usdt=net,trades_n=seed,max_drawdown_usdt=1,
                open_exposure_usdt=0,complete_days_n=10,partial_days_n=1,active_days_n=1,
                mean_daily_net_pnl_usdt=net/10,ambiguous_executions_n=0,entry_killed=True,
                risk_locked_steps_n=5,fees_usdt=.1))
        grouped=group_results(rows)
        self.assertEqual(len(grouped),1); self.assertEqual(grouped[0]['net_pnl_usdt'],-3.)
        self.assertEqual(grouped[0]['runs_n'],5)
        self.assertEqual(grouped[0]['net_min_usdt'],-5); self.assertEqual(grouped[0]['net_max_usdt'],-1)

    def test_EOS10_buy_hold_has_one_entry_day_even_with_zero_closed_cycles(self):
        task=make_plan(self.parent,dict(CatBoost=.002,GRU=.002))[3]
        report=dict(unmanaged_risk=True,state='COMPLETE',trades_n=0)
        row=_summary(task,report,dict(state='PASS'),dict(state='MATCH'))
        self.assertEqual(row['active_days_n'],1)
        self.assertEqual(row['first_fill_ms'],task['start'])


if __name__=='__main__': unittest.main()
