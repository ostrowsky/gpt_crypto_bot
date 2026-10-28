"""USD-05: complete native pagination and immutable resumable compressed pages."""
import importlib,tempfile,unittest
from pathlib import Path
import _support
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.adapter import AdapterError,_sha


def bar(t,close='100'):
    return [t,'100','101','99',close,'10',t+899999,'1000',10,'5','500','0']


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.usdt_history')
        root=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.paths=AppPaths(Path(tmp.name).resolve())
        self.archive=self.api.HistoryArchive(self.paths,cutoff_ms=2700000);self.addCleanup(self.archive.close)
    def put(self,rows,start=0):
        return self.archive.append_page('ETHUSDT',rows,requested_start_ms=start,received_at_ms=3000000,source_sha256=_sha(rows))
    def test_usd05_empty_archive_starts_at_zero(self):
        self.assertEqual(self.archive.next_start('ETHUSDT'),0);self.assertEqual(self.archive.ready_symbols(),[])
    def test_usd05_short_page_is_not_false_completion(self):
        self.put([bar(0)]);self.assertEqual(self.archive.summary('ETHUSDT')['state'],'COLLECTING')
        self.assertEqual(self.archive.next_start('ETHUSDT'),900000)
    def test_usd05_empty_terminal_page_certifies_available_end(self):
        self.put([bar(0)]);self.put([],900000)
        self.assertEqual(self.archive.summary('ETHUSDT')['state'],'MAXIMUM_AVAILABLE_FETCH_COMPLETE')
        self.assertEqual(self.archive.summary('ETHUSDT')['rows_n'],1)
    def test_usd05_resume_keeps_original_cutoff_and_rows(self):
        self.put([bar(0)]);self.archive.close()
        self.archive=self.api.HistoryArchive(self.paths,cutoff_ms=2700000);self.addCleanup(self.archive.close)
        self.assertEqual(self.archive.next_start('ETHUSDT'),900000);self.assertEqual(self.archive.bars('ETHUSDT'),[bar(0)])
    def test_usd05_cutoff_cannot_move_after_restart(self):
        self.put([bar(0)]);self.archive.close()
        with self.assertRaises(AdapterError):self.api.HistoryArchive(self.paths,cutoff_ms=3600000)
    def test_usd05_open_bar_rejected(self):
        with self.assertRaises(AdapterError):self.put([bar(2700000)])
    def test_usd05_wrong_cursor_does_not_skip_history(self):
        with self.assertRaises(AdapterError):self.put([bar(900000)],start=900000)
    def test_usd05_duplicate_page_does_not_duplicate_rows(self):
        self.put([bar(0)]);self.put([bar(0)])
        self.assertEqual(self.archive.summary('ETHUSDT')['rows_n'],1)
    def test_usd05_conflicting_duplicate_is_quarantined(self):
        self.put([bar(0)])
        with self.assertRaises(AdapterError):self.put([bar(0,'102')])
    def test_usd05_nonprogress_page_rejected(self):
        with self.assertRaises(AdapterError):self.put([bar(900000),bar(0)])
    def test_usd05_source_hash_mismatch_rejected(self):
        with self.assertRaises(AdapterError):self.archive.append_page('ETHUSDT',[bar(0)],requested_start_ms=0,
            received_at_ms=3000000,source_sha256='f'*64)

    def test_usd05_shortened_native_candle_preserved_for_quarantine(self):
        row=bar(0);row[6]=653799;self.put([row])
        self.assertEqual(self.archive.bars('ETHUSDT'),[row])
    def test_usd05_reference_required_in_addition_to_complete_history(self):
        self.put([bar(0),bar(900000),bar(1800000)])
        self.assertEqual(self.archive.ready_symbols(),[])
        self.archive.record_reference('ETHUSDT',dict(evidence_kind='SIMULATION',alpha_qualified=False,verdict='NOT_ALPHA_EVIDENCE',trades_n=0))
        self.assertEqual(self.archive.ready_symbols(),['ETHUSDT'])
    def test_usd05_incomplete_history_cannot_be_validated(self):
        self.put([bar(0)])
        with self.assertRaises(AdapterError):self.archive.record_reference('ETHUSDT',dict(evidence_kind='SIMULATION'))
    def test_usd05_empty_symbol_is_not_execution_ready(self):
        self.put([]);self.assertEqual(self.archive.ready_symbols(),[])

    def test_usd05_selected_history_verified_before_execution(self):
        self.put([bar(0),bar(900000),bar(1800000)])
        self.archive.record_reference('ETHUSDT',dict(evidence_kind='SIMULATION',alpha_qualified=False,verdict='NOT_ALPHA_EVIDENCE',trades_n=0))
        result=self.api.verify_reference(self.paths,'ETHUSDT')
        self.assertEqual(result['rows_n'],3);self.assertEqual(result['manifest_sha256'],self.archive.summary('ETHUSDT')['manifest_sha256'])

    def test_usd05_tampered_compressed_page_cannot_authorize_trade(self):
        self.put([bar(0),bar(900000),bar(1800000)])
        self.archive.record_reference('ETHUSDT',dict(evidence_kind='SIMULATION',alpha_qualified=False,verdict='NOT_ALPHA_EVIDENCE',trades_n=0))
        self.archive._con.execute("UPDATE pages SET body=x'00' WHERE symbol='ETHUSDT'")
        with self.assertRaises(AdapterError):self.api.verify_reference(self.paths,'ETHUSDT')


if __name__=='__main__':unittest.main()
