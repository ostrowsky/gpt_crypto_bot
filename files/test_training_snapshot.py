from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import training_snapshot as snapshots
import forward_evidence_service as service
from test_independent_signal_evaluator import NOW, row


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.output = self.root/'training.jsonl'
        self.cutoff = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def publish(self, data=b'{"id":"a"}\n'):
        tmp = self.root/'new.tmp'; tmp.write_bytes(data)
        return snapshots.publish(tmp, self.output, 1, self.cutoff)

    def test_reader_pins_old_file_while_new_snapshot_publishes(self):
        self.output.write_bytes(b'legacy')
        first = self.publish()
        old, _ = snapshots.resolve(self.output)
        with old.open('rb') as reader, self.output.open('rb'):
            second = self.publish(b'{"id":"b"}\n')
            self.assertEqual(reader.read(), b'{"id":"a"}\n')
        new, body = snapshots.resolve(self.output)
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertEqual(new.read_bytes(), b'{"id":"b"}\n')
        self.assertEqual(self.output.read_bytes(), b'legacy')
        self.assertEqual(body['cutoff'], self.cutoff.isoformat())

    def test_same_content_idempotent(self):
        first = self.publish(); self.assertEqual(first, self.publish())
        self.assertEqual(len(list((self.root/'training_snapshots').glob('*.jsonl'))), 1)

    def test_failed_pointer_publication_preserves_prior(self):
        self.publish(); before = snapshots.pointer(self.output).read_bytes()
        with patch.object(Path, 'replace', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError): self.publish(b'new')
        self.assertEqual(snapshots.pointer(self.output).read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_corrupt_snapshot_rejected_and_not_repaired_silently(self):
        self.publish(); path, _ = snapshots.resolve(self.output); path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError): snapshots.resolve(self.output)
        with self.assertRaises(ValueError): self.publish()

    def test_unsafe_pointer_and_descriptor_fields(self):
        for field, value in (('path','../raw.jsonl'), ('sha256','x'*64), ('bytes',1),
                             ('rows',-1), ('rows',True), ('contract','legacy')):
            self.publish(); p = snapshots.pointer(self.output)
            body = json.loads(p.read_bytes()); body[field] = value
            p.write_text(json.dumps(body))
            with self.assertRaises(ValueError): snapshots.resolve(self.output)

    def test_missing_pointer_does_not_use_legacy(self):
        self.output.write_bytes(b'legacy')
        with self.assertRaises(FileNotFoundError): snapshots.resolve(self.output)

    def test_immutable_export_respects_original_provenance_cutoff(self):
        cutoff = NOW+timedelta(days=2)
        dataset = self.root/'dataset.jsonl'
        eligible, future = row(0,-5), row(1)
        dataset.write_text('\n'.join(json.dumps(r) for r in (eligible,future,{'id':'legacy'})))
        self.output.write_bytes(b'legacy-reader-held')
        with self.output.open('rb'):
            exported = service.export_training(dataset,self.output,cutoff,immutable=True)
        path, body = snapshots.resolve(self.output)
        self.assertEqual(exported, body)
        self.assertEqual(body['rows'], 1)
        self.assertEqual(json.loads(path.read_bytes())['id'], eligible['id'])
        self.assertEqual(self.output.read_bytes(), b'legacy-reader-held')

    def test_empty_snapshot_count_is_not_training_approval(self):
        tmp = self.root/'empty.tmp';tmp.write_bytes(b'')
        body = snapshots.publish(tmp,self.output,0,self.cutoff)
        self.assertEqual(body['rows'],0)
        self.assertIn('not-learning-approval',body['scope'])

    def test_duplicate_training_identity_dedup_and_conflict_block(self):
        cutoff = NOW+timedelta(days=2)
        eligible = row(0,-5)
        dataset = self.root/'dataset.jsonl'
        dataset.write_text('\n'.join(json.dumps(eligible) for _ in range(2)))
        body = service.export_training(dataset,self.output,cutoff,immutable=True)
        self.assertEqual(body['rows'],1)
        before = snapshots.pointer(self.output).read_bytes()
        other = dict(eligible, f={'changed':1})
        dataset.write_text('\n'.join(json.dumps(r) for r in (eligible,other)))
        with self.assertRaisesRegex(ValueError,'conflicting'):
            service.export_training(dataset,self.output,cutoff,immutable=True)
        self.assertEqual(snapshots.pointer(self.output).read_bytes(),before)


if __name__ == '__main__': unittest.main()
