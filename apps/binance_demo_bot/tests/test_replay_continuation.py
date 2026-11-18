"""FM01..04 acceptance tests before continuation implementation."""
import json
from pathlib import Path
import tempfile
import unittest
import _support
from binance_demo_bot.replay_continuation import continuation_scope, remap_inputs, remaining_computations, load_completed
from binance_demo_bot.minute_replay_runner import minute_plan
from binance_demo_bot.intrabar_probe import _bytes_sha

class ContinuationTests(unittest.TestCase):
    def test_FM03_budget_is_remaining_not_reset_and_old_scope_untouched(self):
        old=dict(attempt_limit=2000,created_at_ms=1,expires_at_ms=2,symbols=['AAAUSDT'],start=0,end=900000,files={},prior_imported_attempts_n=10)
        scope=continuation_scope(old,111,now_ms=10000)
        self.assertEqual(scope['attempt_limit'],1889);self.assertEqual(scope['inherited_native_attempts_n'],111)
        self.assertEqual(scope['prior_imported_attempts_n'],121);self.assertEqual(scope['expires_at_ms'],7210000)
        self.assertEqual(old['expires_at_ms'],2);self.assertEqual(old['attempt_limit'],2000)
        for n in (2000,2001,-1):
            with self.assertRaises(ValueError):continuation_scope(old,n,now_ms=10000)

    def test_FM04_only_EOF_doc_changes_can_map_to_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            f=Path(root);doc=f/'spec.md';doc.write_bytes(b'fixed spec\n\n');snap=f/'frozen.md';snap.write_bytes(doc.read_bytes());digest=_bytes_sha(doc)
            doc.write_bytes(b'fixed spec\n');files=remap_inputs({str(doc):digest},{str(doc):dict(path=str(snap),sha256=digest)})
            self.assertEqual(files[str(snap)],digest);self.assertEqual(files[str(doc)],_bytes_sha(doc))
            doc.write_bytes(b'changed behavior\n')
            with self.assertRaises(ValueError):remap_inputs({str(doc):digest},{str(doc):dict(path=str(snap),sha256=digest)})
            code=f/'code.py';code.write_bytes(b'new code')
            with self.assertRaises(ValueError):remap_inputs({str(code):digest},{str(code):dict(path=str(snap),sha256=digest)})

    def test_FM01_all_neural_results_are_reused_and_only_ten_fixed_tasks_missing(self):
        parent=dict(seeds=[11,23,37,53,71],splits=[['train',0,900000],['tune',900000,1800000],['test',1800000,2700000]])
        plan=minute_plan(parent,dict(CatBoost=0,GRU=0));ids={t['id'] for t in plan[:87]}
        missing=remaining_computations(plan,ids)
        self.assertEqual(len(missing),10);self.assertTrue(all(t['label'] in ('CatBoost','GRU','EXIT_fixed') for t in missing))
        ids.remove('MINUTE_STOP_FIRST-tune-base-PPO-11')
        with self.assertRaises(ValueError):remaining_computations(plan,ids)

    def test_FM01_absent_report_is_not_partial_financial_state(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(load_completed(Path(root),dict(id='absent'),'a'*64))


class CompleteReportReuseTests(unittest.TestCase):
    def test_FM01_only_complete_reconciled_reports_can_be_reused(self):
        from binance_demo_bot.execution_order_study import save_report
        from binance_demo_bot.rl_direction_core import ResearchAccount
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);task=dict(id='complete');account=ResearchAccount();account.step(0,None,0,lambda *_:None)
            report=account.report(0,900000);save_report(folder,task,'a'*64,report)
            self.assertEqual(load_completed(folder,task,'a'*64)['state'],'COMPLETE')
            other=dict(id='incomplete');save_report(folder,other,'a'*64,dict(report,state='INCOMPLETE',net_pnl_usdt=None))
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):load_completed(folder,other,'a'*64)


class ContinuationBindingTests(unittest.TestCase):
    def test_FM04_top_level_scope_digest_must_identify_new_scope(self):
        from binance_demo_bot.replay_continuation import make_binding
        with tempfile.TemporaryDirectory() as root:
            scope=Path(root)/'data-scope.json';scope.write_text('{"new":true}',encoding='utf-8')
            old=dict(data_scope_sha256='a'*64,protocol='parent',files={})
            new=make_binding(old,scope,protocol='continuation')
            self.assertEqual(new['data_scope_sha256'],_bytes_sha(scope));self.assertEqual(old['data_scope_sha256'],'a'*64)



class RequestGuardTests(unittest.TestCase):
    def test_FM07_imported_packets_are_manifest_bound_not_rehashed_per_request(self):
        from binance_demo_bot.replay_continuation import request_guard_files
        from binance_demo_bot.intrabar_probe import _read_sealed
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);code=root/'controller.py';code.write_text('fixed source',encoding='utf-8')
            files={str(root/'frames'/f'{i}.json'):'a'*64 for i in range(3000)}
            files[str(code)]=_bytes_sha(code);original=dict(files);manifest=root/'imports.json'
            guard=request_guard_files(files,manifest)
            self.assertEqual(len(guard),2);self.assertEqual(guard[str(code)],_bytes_sha(code))
            self.assertEqual(guard[str(manifest)],_bytes_sha(manifest));self.assertEqual(files,original)
            self.assertEqual(_read_sealed(manifest)['files'],files)

if __name__=='__main__':unittest.main()
