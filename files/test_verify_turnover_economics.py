import json
from pathlib import Path
import tempfile
import unittest
from historical_signal_evaluation import sha
from verify_turnover_economics import verify


class ReceiptTests(unittest.TestCase):
    def test_result_mutation_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'result.json';p.write_text('{}')
            (root/'receipt.json').write_text(json.dumps({'result.json':sha(p)}))
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'receipt drift'):verify(root,root,root)
            (root/'receipt.json').write_text(json.dumps({'../result.json':'0'*64}))
            with self.assertRaisesRegex(ValueError,'receipt drift'):verify(root,root,root)


if __name__=='__main__':unittest.main()
