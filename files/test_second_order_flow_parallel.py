import tempfile,unittest
from pathlib import Path
import numpy as np
from prepare_second_order_flow_parallel import completed_prefix,assemble


class ParallelPrepTests(unittest.TestCase):
    def test_published_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=root/'coverage.json';receipt.write_text('published',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'already published'):assemble(root,root,{})
            self.assertEqual(receipt.read_text(encoding='utf-8'),'published')

    def test_reused_asset_requires_all_source_rows_parts_and_sealed_clocks(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';books=root/'books'
            (raw/'BTCUSDT').mkdir(parents=True);(books/'BTCUSDT').mkdir(parents=True)
            for i in range(2):
                pq.write_table(pa.table({'E':[1,2,3]}),raw/'BTCUSDT'/f'{i}.parquet')
                np.savez(books/'BTCUSDT'/f'{i:03d}.npz',time=np.array([1000+i*2000,2000+i*2000]),segment=np.ones(2))
            log=root/'prep.log'
            log.write_text("One-second OFI BTCUSDT 0.parquet audit {'raw_rows': 3}\nOne-second OFI BTCUSDT 1.parquet audit {'raw_rows': 6}\n",encoding='utf-8')
            report=completed_prefix(raw,books,log);self.assertEqual(len(report['parts']),2);self.assertEqual(report['audit']['raw_rows'],6)
            np.savez(books/'BTCUSDT'/'001.npz',time=np.array([4000,5000]),segment=np.ones(2))
            with self.assertRaisesRegex(ValueError,'clock broken'):completed_prefix(raw,books,log)
            log.write_text("One-second OFI BTCUSDT 0.parquet audit {'raw_rows': 3}\n",encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'prefix incomplete'):completed_prefix(raw,books,log)

    def test_reused_log_cannot_invent_denominator(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'BTCUSDT').mkdir();pq.write_table(pa.table({'E':[1,2]}),root/'BTCUSDT'/'0.parquet')
            log=root/'log';log.write_text("One-second OFI BTCUSDT 0.parquet audit {'raw_rows': 200}\n",encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'denominator mismatch'):completed_prefix(root,root,log)


if __name__=='__main__':unittest.main()
