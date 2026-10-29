"""FSA08/09 archive, frozen registration and financial verdict contracts."""
import hashlib, importlib, json, sqlite3, tempfile, unittest, zlib
from pathlib import Path
from contextlib import closing
import _support

class FeatureAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.feature_ablation_runner')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def archive(self):
        path=self.root/'history.sqlite'
        rows=[[0,'10','11','9','10','2',899999,'20',3,'1','10','0']]
        raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode(); blob=zlib.compress(raw)
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            db.execute('CREATE TABLE pages(symbol,start_ms,source_sha,blob_sha,body,received_ms)')
            db.execute('INSERT INTO pages VALUES(?,?,?,?,?,?)',('BTCUSDT',0,hashlib.sha256(raw).hexdigest(),hashlib.sha256(blob).hexdigest(),blob,1800000))
        return path,rows
    def test_fsa08_verified_read_preserves_taker_fields_and_receipts(self):
        path,rows=self.archive();series,manifest=self.api.read_archive(path,['BTCUSDT'],900000)
        self.assertEqual(series['BTCUSDT'],rows)
        self.assertEqual(manifest['BTCUSDT']['raw_rows_n'],1)
        self.assertEqual(manifest['BTCUSDT']['received_min_ms'],1800000)
        self.assertFalse(manifest['BTCUSDT']['historical_receipts_certified'])
    def test_fsa08_bad_compressed_hash_rejected(self):
        path,_=self.archive()
        with closing(sqlite3.connect(path, isolation_level=None)) as db:db.execute("UPDATE pages SET blob_sha='wrong'")
        with self.assertRaisesRegex(ValueError,'ARCHIVE_HASH'):self.api.read_archive(path,['BTCUSDT'],900000)
    def test_fsa08_valid_compression_with_changed_content_rejected(self):
        path,_=self.archive();blob=zlib.compress(b'[]')
        with closing(sqlite3.connect(path, isolation_level=None)) as db:db.execute('UPDATE pages SET body=?,blob_sha=?',(blob,hashlib.sha256(blob).hexdigest()))
        with self.assertRaisesRegex(ValueError,'ARCHIVE_HASH'):self.api.read_archive(path,['BTCUSDT'],900000)
    def test_fsa08_missing_symbol_never_becomes_empty_complete_cohort(self):
        path,_=self.archive()
        with self.assertRaisesRegex(ValueError,'MISSING_SYMBOL'):self.api.read_archive(path,['BTCUSDT','ETHUSDT'],900000)
    def test_fsa08_registration_reopen_identical_but_changed_rejected(self):
        folder=self.root/'study';self.api.freeze_registration(folder,{'seed':42})
        self.api.freeze_registration(folder,{'seed':42})
        with self.assertRaisesRegex(ValueError,'REGISTRATION_CHANGED'):self.api.freeze_registration(folder,{'seed':43})
        self.assertEqual(json.loads((folder/'registration.json').read_text()),{'seed':42})
    def test_fsa09_no_support_from_positive_mean_with_negative_bound(self):
        result=self.api.verdict({'paired_days_N':400,'lower':-.01,'mean':.1},absolute_lower=.02,drawdown=1,reference_drawdown=1,stress_pnl=1)
        self.assertEqual(result,'NOT_SUPPORTED')
    def test_fsa09_insufficient_sample_and_risk_fail_cannot_support(self):
        self.assertEqual(self.api.verdict({'paired_days_N':2,'lower':1,'mean':1},absolute_lower=1,drawdown=0,reference_drawdown=0,stress_pnl=1),'INCONCLUSIVE')
        self.assertEqual(self.api.verdict({'paired_days_N':400,'lower':1,'mean':1},absolute_lower=1,drawdown=2,reference_drawdown=0,stress_pnl=1),'NOT_SUPPORTED')
    def test_fsa09_bootstrap_schema_is_normalized_for_verdict(self):
        raw={'paired_days_N':400,'mean_delta_usdt':.04,'ci_low_usdt':.02,'ci_high_usdt':.06,'state':'COMPLETE'}
        normalized=self.api.normalize_interval(raw)
        self.assertEqual(normalized['lower'],.02)
        self.assertEqual(normalized['mean'],.04)
        self.assertEqual(self.api.verdict(normalized,absolute_lower=.01,drawdown=1,reference_drawdown=1,stress_pnl=1),'POSITIVE_PATTERN_POWER_UNCERTIFIED')
    def test_fsa09_frozen_training_parameters(self):
        p=self.api.MODEL_PARAMS
        self.assertEqual((p['iterations'],p['depth'],p['random_seed']),(300,5,42))
        self.assertTrue(p['has_time']);self.assertFalse(p['allow_writing_files'])
        self.assertEqual(p['bootstrap_type'],'No')

if __name__=='__main__':unittest.main()
