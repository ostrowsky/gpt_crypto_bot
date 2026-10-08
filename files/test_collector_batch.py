import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import critic_dataset as cd

class BatchTests(unittest.TestCase):
    def test_duplicate_snapshot_preserves_actual_buy_and_future_labels(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cd,'CRITIC_FILE',Path(d)/'data.jsonl'),patch.object(cd,'_logged_candidates',cd.OrderedDict()):
            old=dict(id='one',decision=dict(stage='monitor',action='take'),labels=dict(ret_5=12))
            cd.CRITIC_FILE.write_text(json.dumps(old)+'\n',encoding='utf-8')
            new=lambda k:dict(id=k,decision=dict(stage='collector',action='candidate'))
            stats=cd.append_collector_batch([new('one'),new('two'),new('two')])
            rows=[json.loads(x) for x in cd.CRITIC_FILE.read_text().splitlines()]
            self.assertEqual(rows[0],old);self.assertEqual(len(rows),2);self.assertEqual(stats,dict(new_ids=1,existing_ids=1))
            self.assertEqual(cd.append_collector_batch([new('two')])['new_ids'],0)

    def test_malformed_history_prevents_all_append_and_id_acknowledgment(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cd,'CRITIC_FILE',Path(d)/'data.jsonl'),patch.object(cd,'_logged_candidates',cd.OrderedDict()):
            cd.CRITIC_FILE.write_text('{bad\n',encoding='utf-8');before=cd.CRITIC_FILE.read_bytes()
            with self.assertRaises(cd.DatasetIntegrityError):cd.append_collector_batch([dict(id='new',decision=dict(stage='collector',action='candidate'))])
            self.assertEqual(cd.CRITIC_FILE.read_bytes(),before);self.assertFalse(cd._is_logged('new'))

if __name__=='__main__':unittest.main()
