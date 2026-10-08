import os,tempfile,unittest
from pathlib import Path
from evidence_snapshot_io import open_snapshot,publish_snapshot

class SnapshotTests(unittest.TestCase):
    def test_reader_keeps_old_bytes_without_blocking_atomic_replace(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'snapshot';new=Path(d)/'new';target.write_bytes(b'old');new.write_bytes(b'new')
            with open_snapshot(target,'rb') as reader:
                publish_snapshot(new,target)
                self.assertEqual(reader.read(),b'old')
            with open_snapshot(target) as reader:self.assertEqual(reader.read(),'new')

    def test_absent_file_and_write_request_do_not_fabricate_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'absent'
            with self.assertRaises(FileNotFoundError):
                with open_snapshot(p):pass
            with self.assertRaises(ValueError):
                with open_snapshot(p,'w'):pass
            self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
