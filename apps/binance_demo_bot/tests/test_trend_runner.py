"""FRI-05 research registration/archive integrity before execution."""
import hashlib,importlib,json,tempfile,unittest,zlib
from pathlib import Path
import _support
from test_trend_research import bars,STEP
from binance_demo_bot.adapter import _sha,AdapterError
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.usdt_history import HistoryArchive

class TrendRunnerTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.trend_runner')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temp.cleanup);self.paths=AppPaths(Path(temp.name).resolve())
        self.folder=self.paths.runtime_path('all-usdt');self.folder.mkdir(parents=True)
        self.cohort=['AAAUSDT','BBBUSDT'];self.cutoff=80*STEP
        self.registration=dict(symbols=self.cohort,cutoff_ms=self.cutoff,cohort_sha256=_sha(self.cohort))
        (self.folder/'history-registration.json').write_text(json.dumps(self.registration))
        self.archive=HistoryArchive(self.paths,cutoff_ms=self.cutoff);self.addCleanup(self.archive.close)
    def ready(self,symbol):
        rows=bars(80)
        self.archive.append_page(symbol,rows,requested_start_ms=0,received_at_ms=self.cutoff+1,source_sha256=_sha(rows))
        self.archive.record_reference(symbol,dict(evidence_kind='SIMULATION',alpha_qualified=False,verdict='NOT_ALPHA_EVIDENCE'))
    def test_fri05_waits_without_claiming_complete_or_running_partial(self):
        self.ready('AAAUSDT');report=self.api.run_research(self.paths)
        self.assertEqual(report['state'],'WAITING_FOR_HISTORY');self.assertEqual(report['ready_n'],1)
        self.assertEqual(report['symbols_N'],2);self.assertFalse(report['promotion_allowed'])
        self.assertTrue(self.paths.runtime_path('trend-research/registration.json').is_file())
    def test_fri05_complete_cohort_evaluates_three_time_segments_and_cost_stress(self):
        for symbol in self.cohort:self.ready(symbol)
        report=self.api.run_research(self.paths)
        self.assertEqual(report['state'],'RESEARCH_COMPLETE');self.assertEqual(report['evaluated_n'],2)
        self.assertEqual(set(report['results']),{'train_base','tune_base','test_base','train_stress','tune_stress','test_stress'})
        self.assertEqual(report['results']['test_base']['fee_bps'],10)
        self.assertEqual(report['results']['test_stress']['fee_bps'],20)
        self.assertFalse(report['promotion_allowed']);self.assertEqual(report['training_state'],'NOT_TRAINED')
    def test_fri05_partial_is_explicit_and_cannot_grant_full_cohort_pass(self):
        self.ready('AAAUSDT');report=self.api.run_research(self.paths,allow_partial=True)
        self.assertEqual(report['state'],'PROVISIONAL_RESEARCH_COMPLETE');self.assertFalse(report['cohort_complete'])
        self.assertIn('MAXIMUM_COHORT_INCOMPLETE',report['readiness']['reasons'])
    def test_fri05_registration_tamper_is_not_silently_overwritten(self):
        self.api.run_research(self.paths)
        path=self.paths.runtime_path('trend-research/registration.json');value=json.loads(path.read_text())
        value['policy']['stop']=.5;path.write_text(json.dumps(value))
        with self.assertRaises(AdapterError):self.api.run_research(self.paths)
    def test_fri05_valid_shorter_archive_cannot_satisfy_longer_registered_cutoff(self):
        for symbol in self.cohort:self.ready(symbol)
        self.registration['cutoff_ms']+=20*STEP
        (self.folder/'history-registration.json').write_text(json.dumps(self.registration))
        with self.assertRaises(AdapterError) as caught:self.api.run_research(self.paths)
        self.assertEqual(caught.exception.reason,'RESEARCH_ARCHIVE_CUTOFF_MISMATCH')
        self.assertFalse(self.paths.runtime_path('trend-research/full/report.json').exists())
    def test_fri05_compressed_source_tamper_aborts_before_results(self):
        for symbol in self.cohort:self.ready(symbol)
        self.archive._con.execute("UPDATE pages SET body=? WHERE symbol='AAAUSDT'",(b'bad',))
        with self.assertRaises(AdapterError):self.api.run_research(self.paths)
        self.assertFalse(self.paths.runtime_path('trend-research/full/report.json').exists())
    def test_fri05_cached_report_tamper_cannot_masquerade_as_completed_run(self):
        for symbol in self.cohort:self.ready(symbol)
        self.api.run_research(self.paths);path=self.paths.runtime_path('trend-research/full/report.json')
        value=json.loads(path.read_text());value['promotion_allowed']=True;path.write_text(json.dumps(value))
        with self.assertRaises(AdapterError):self.api.run_research(self.paths)
    def test_fri05_immutable_registration_pins_source_policy_and_cohort_before_result(self):
        self.api.run_research(self.paths)
        value=json.loads(self.paths.runtime_path('trend-research/registration.json').read_text())
        self.assertEqual(value['cohort_sha256'],_sha(self.cohort));self.assertEqual(value['policy']['version'],'trend-breakout-v1')
        self.assertIn('trend_research.py',value['source_sha256'])
        self.assertIn('trend_runner.py',value['source_sha256'])

if __name__=='__main__':unittest.main()
