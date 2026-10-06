import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from run_leader_mission_reaudit import load_market,verify_parents
from historical_signal_evaluation import sha


class RunnerTests(unittest.TestCase):
    def test_raw_grid_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'market').mkdir();p=d/'market'/'A_15m.json'
            rows=[dict(t=i*900000,o=1,h=2,l=.5,c=1) for i in range(20)];p.write_text(json.dumps(rows))
            manifest=dict(eligible_symbols=['A'],archive_start_ms=0,end_ms=20*900000,input_hashes={p.name:sha(p)})
            self.assertEqual(len(load_market(d,manifest)['A']['time']),20)
            rows.pop(5);p.write_text(json.dumps(rows));manifest['input_hashes'][p.name]=sha(p)
            with self.assertRaises(ValueError):load_market(d,manifest)

    def test_parent_receipt_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            for name,value in [('result.json',{}),('registration.json',{'sources':{}}),('trades_control.json',[])]:
                (d/name).write_text(json.dumps(value))
            (d/'receipt.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()}))
            (d/'independent_verification.json').write_text(json.dumps({'status':'PASS'}))
            parents={name:str(d) for name in ('turnover','impulse','joint','exit','capacity','execution')}
            with patch('run_leader_mission_reaudit.PARENTS',parents):
                self.assertEqual(len(verify_parents()[0]),6)
                (d/'trades_control.json').write_text('[{}]')
                with self.assertRaises(ValueError):verify_parents()


if __name__=='__main__':unittest.main()
