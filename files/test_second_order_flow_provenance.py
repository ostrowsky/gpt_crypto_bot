import json,tempfile,unittest
from pathlib import Path
import numpy as np
from minute_direction_data import SYMBOLS,sha
from evaluate_second_order_flow import validate_source,load_asset


class OFIProvenanceTests(unittest.TestCase):
    def test_maximum_manifest_hashes_files_and_prepared_parts_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);rows=[];assets={}
            for s in SYMBOLS:
                (raw/s).mkdir();parts=[]
                for i in range(30):
                    p=raw/s/f'{i}.parquet';p.write_bytes(bytes([i]))
                    rows.append(dict(path=p.relative_to(raw).as_posix(),lfs=dict(oid=sha(p))));parts.append(dict(file=str(i)))
                assets[s]=dict(parts=parts)
            manifest=dict(dataset='predict-quant/binance-future-orderbook',revision='b8590b83452d7a32fbb274ff7741b6db000b3984',files=rows)
            (raw/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            cov=dict(assets=assets,source_manifest_sha256=sha(raw/'manifest.json'),source_hash=sha(Path(__file__).with_name('second_order_flow_data.py')))
            validate_source(raw,raw,cov)
            extra=raw/SYMBOLS[0]/'extra.parquet';extra.write_bytes(b'1')
            with self.assertRaisesRegex(ValueError,'do not match'):validate_source(raw,raw,cov)
            extra.unlink();assets[SYMBOLS[0]]['parts'].pop()
            with self.assertRaisesRegex(ValueError,'coverage incomplete'):validate_source(raw,raw,cov)
            assets[SYMBOLS[0]]['parts'].append(dict(file='29'))
            (raw/rows[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash drift'):validate_source(raw,raw,cov)

    def test_prepared_source_drift_and_clock_holes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=SYMBOLS[0];(root/s).mkdir();p=root/s/'0.npz'
            np.savez(p,time=np.array([1000,2000]),book=np.ones((2,20)))
            cov=dict(assets={s:dict(parts=[dict(file=p.name,sha256=sha(p))])})
            self.assertEqual(len(load_asset(root,s,cov)['time']),2)
            np.savez(p,time=np.array([1000,3000]),book=np.ones((2,20)))
            with self.assertRaisesRegex(ValueError,'source drift'):load_asset(root,s,cov)
            cov['assets'][s]['parts'][0]['sha256']=sha(p)
            with self.assertRaisesRegex(ValueError,'clock broken'):load_asset(root,s,cov)


if __name__=='__main__':unittest.main()
