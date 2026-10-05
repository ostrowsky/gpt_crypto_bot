import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from historical_signal_evaluation import sha
from prepare_capacity_baseline_cache import prepare
import replay_backtest as rb


class CacheRepairTests(unittest.TestCase):
    def test_verified_archive_replaces_gapped_local_without_imputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'archive';(a/'market').mkdir(parents=True);local=root/'local';local.mkdir()
            bindings={}
            for tf,step in [('15m',900000),('1h',3600000)]:
                rows=[{'t':t,'o':100,'h':101,'l':99,'c':100,'v':1} for t in range(0,7200000,step)]
                name=f'BTCUSDT_{tf}_0_7200000.json';p=a/'market'/name;p.write_text(json.dumps(rows));bindings[name]=sha(p)
            (a/'manifest.json').write_text(json.dumps({'archive_start_ms':0,'archive_end_ms':7200000,'input_hashes':bindings}))
            r=prepare(a,local,root/'out',['BTCUSDT','UNKNOWN'],3600000,7200000)
            self.assertEqual([x['missing'] for x in r['series'][:2]],[0,0])
            self.assertGreater(r['series'][2]['missing'],0)
            self.assertEqual(r['series'][0]['origin']['kind'],'verified_archive')
            self.assertEqual(json.loads((root/'out'/'UNKNOWN_15m_2700000_7200000.json').read_text()),[])
            (a/'market'/'BTCUSDT_15m_0_7200000.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                prepare(a,local,root/'bad',['BTCUSDT'],3600000,7200000)


if __name__=='__main__':unittest.main()
