from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from process_lock import process_lock


class ProcessLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'collector.lock'

    def child(self):
        code = (
            "import sys,time; from pathlib import Path; "
            "sys.path.insert(0,sys.argv[1]); from process_lock import process_lock; "
            "ctx=process_lock(Path(sys.argv[2])); ctx.__enter__(); "
            "print('LOCKED',flush=True); time.sleep(60)"
        )
        proc = subprocess.Popen([sys.executable, '-c', code,
            str(Path(__file__).parent), str(self.path)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.assertEqual(proc.stdout.readline().strip(), 'LOCKED')
        return proc

    def test_legacy_orphan_file_is_not_ownership(self):
        self.path.touch()
        with process_lock(self.path):
            self.assertTrue(self.path.exists())
        with process_lock(self.path):
            pass

    def test_live_owner_cannot_be_stolen(self):
        child = self.child()
        with self.assertRaises(BlockingIOError):
            with process_lock(self.path):
                self.fail('concurrent owner')
        child.kill()
        child.communicate(timeout=10)

    def test_abrupt_death_releases_lock_without_deleting_file(self):
        child = self.child()
        child.kill()
        child.communicate(timeout=10)
        self.assertTrue(self.path.exists())
        with process_lock(self.path):
            pass

    def test_exception_releases_ownership(self):
        with self.assertRaisesRegex(ValueError, 'test'):
            with process_lock(self.path):
                raise ValueError('test')
        with process_lock(self.path):
            pass


if __name__ == '__main__':
    unittest.main()
