import json
from pathlib import Path
import tempfile
import unittest

from verify_capacity_catboost import checksum, verify


class VerificationTests(unittest.TestCase):
    def test_changed_prediction_or_report_cannot_be_verified(self):
        for name in ('test_predictions.npz','result.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); p=root/name; p.write_bytes(b'frozen')
                (root/'receipt.json').write_text(json.dumps({name:checksum(p)}))
                p.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed result artifact'):
                    verify(root,root)

    def test_unsafe_receipt_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'receipt.json').write_text(json.dumps({'../outside':'0'*64}))
            with self.assertRaisesRegex(ValueError,'changed result artifact'):
                verify(root,root)


if __name__ == '__main__': unittest.main()
