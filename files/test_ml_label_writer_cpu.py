import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ml_dataset as ml


class ScopedWriterTests(unittest.TestCase):
    def rewrite(self, raw, mutate, scope, *, fail=False):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rows.jsonl'; path.write_text(raw,encoding='utf-8')
            with patch.object(ml,'ML_FILE',path):
                if fail:
                    with patch.object(ml,'_atomic_replace_with_retry',side_effect=OSError('failed')):
                        with self.assertRaises(OSError):
                            ml._rewrite_records(mutate,mutation_scope=scope)
                    self.assertEqual(path.read_text(encoding='utf-8'),raw)
                    self.assertFalse(list(path.parent.glob('*.tmp')))
                    return
                result=ml._rewrite_records(mutate,mutation_scope=scope)
            return result,path.read_text(encoding='utf-8')

    def test_only_matching_record_serialized_and_all_records_validated(self):
        raw='{"id":"a","v":0}\n{"id": "b", "seq": [1,2,3]}\n'
        def mutate(row):
            if row['id']=='a': row['v']=1; return True
            self.fail('out-of-scope mutator called')
        original=json.dumps
        with patch.object(ml.json,'dumps',wraps=original) as encode:
            counts,out=self.rewrite(raw,mutate,lambda r:r.get('id')=='a')
        self.assertEqual(encode.call_count,1)
        self.assertEqual(counts,(1,0))
        self.assertIn('{"id": "b", "seq": [1,2,3]}',out)

    def test_compact_and_escaped_duplicate_ids_are_updated(self):
        raw='{"id": "a", "v":0}\n{"id":"\\u0061","v":0}\n'
        def mutate(row): row['v']=1; return True
        counts,out=self.rewrite(raw,mutate,lambda r:r.get('id')=='a')
        self.assertEqual(counts,(2,0))
        self.assertTrue(all(json.loads(l)['v']==1 for l in out.splitlines()))

    def test_in_scope_false_return_side_effect_is_preserved(self):
        def mutate(row):
            row['provenance']='updated'
            return row['id']=='a'
        counts,out=self.rewrite('{"id":"a"}\n{"id":"b"}\n',mutate,lambda r:True)
        self.assertEqual(counts,(1,0))
        self.assertTrue(all(json.loads(l)['provenance']=='updated' for l in out.splitlines()))

    def test_malformed_cleanup_and_missing_final_newline(self):
        def mutate(row): row['v']=1; return True
        counts,out=self.rewrite('{"id":"a"}\nbad\n[]\n{"id":"b"}',mutate,
                                lambda r:r.get('id')=='a')
        self.assertEqual(counts,(1,2))
        self.assertTrue(out.endswith('\n'))
        self.assertEqual(len(out.splitlines()),2)

    def test_no_change_preserves_file_and_atomic_failure_cleans_temp(self):
        raw='{"id":"a"}\nbad\n'
        self.assertEqual(self.rewrite(raw,lambda r:False,lambda r:True),((0,0),raw))
        def mutate(row): row['v']=1; return True
        self.rewrite(raw,mutate,lambda r:True,fail=True)

    def test_scoped_and_legacy_semantic_outputs_match(self):
        raw='{"id":"a", "labels": {}}\n {"id":"b", "seq":[1.2,3.4]}\n'
        def mutate(row):
            if row['id']=='a': row['labels']['ret_3']=1.2; return True
            return False
        old=self.rewrite(raw,mutate,None)
        new=self.rewrite(raw,mutate,lambda r:r.get('id')=='a')
        self.assertEqual(old[0],new[0])
        self.assertEqual([json.loads(l) for l in old[1].splitlines()],
                         [json.loads(l) for l in new[1].splitlines()])


if __name__=='__main__': unittest.main()
