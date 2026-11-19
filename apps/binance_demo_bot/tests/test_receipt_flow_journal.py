"""JR01/02: real file partial-write recovery, not silent log loss."""
import errno
import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from binance_demo_bot import receipt_flow_journal as journal


class DurableJournalContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime')
        self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'events.jsonl.gz.partial'

    def test_JR01_exact_text_order_and_multiple_gzip_members(self):
        text='{"n":1}\n{"n":2}\n'
        with journal.DurableGzipText(self.path,chunk_bytes=32) as out:
            for _ in range(20):out.write(text)
        expected=(text*20).replace('\n',os.linesep).encode()
        self.assertEqual(gzip.decompress(self.path.read_bytes()),expected)
        self.assertEqual(out.proof()['uncompressed_sha256'],hashlib.sha256(expected).hexdigest())
        self.assertGreater(out.proof()['members_n'],1)
        self.assertEqual(out.proof()['records_n'],40)

    def test_JR02_partial_append_retry_rolls_back_only_own_member(self):
        calls=[]
        def append(path,member):
            calls.append(len(member))
            if len(calls)==1:
                with path.open('ab') as stream:stream.write(member[:len(member)//2])
                raise OSError(errno.EINVAL,'injected partial write')
            journal.append_bytes(path,member)
        with journal.DurableGzipText(self.path,append=append,chunk_bytes=32) as out:
            out.write('first\n');out.write('second\n')
        self.assertEqual(gzip.decompress(self.path.read_bytes()),'first\nsecond\n'.replace('\n',os.linesep).encode())
        self.assertEqual(len(calls),2);self.assertEqual(out.proof()['recoveries_n'],1)

    def test_JR02_persistent_io_error_is_not_complete(self):
        calls=[]
        def fail(path,data):calls.append(1);raise OSError(errno.EINVAL,'persistent')
        with self.assertRaises(OSError):
            with journal.DurableGzipText(self.path,append=fail) as out:out.write('one\n')
        self.assertEqual(len(calls),3)
        self.assertFalse(out.proof()['complete'])
        self.assertTrue(self.path.exists())

    def test_JR02_changed_committed_prefix_cannot_be_truncated_as_recovery(self):
        calls=[]
        def append(path,member):
            calls.append(1)
            if len(calls)==1:journal.append_bytes(path,member)
            else:
                with path.open('r+b') as stream:stream.write(b'x')
                raise OSError(errno.EINVAL,'tamper')
        with self.assertRaisesRegex(ValueError,'PREFIX'):
            with journal.DurableGzipText(self.path,append=append,chunk_bytes=32) as out:
                out.write('a'*20+'\n');out.write('b'*20+'\n')
        self.assertEqual(self.path.read_bytes()[:1],b'x')
        self.assertFalse(out.proof()['complete'])

    def test_JR01_existing_path_is_never_overwritten(self):
        self.path.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):journal.DurableGzipText(self.path)
        self.assertEqual(self.path.read_bytes(),b'keep')

    def test_JR05_original_gzip_text_bytes_match_including_newlines(self):
        old=self.path.parent/'original.gz';text='one\n\u0442\u0435\u0441\u0442\r\ntwo\n'
        with gzip.open(old,'wt',encoding='utf-8') as stream:stream.write(text)
        with journal.DurableGzipText(self.path) as out:out.write(text)
        self.assertEqual(gzip.decompress(old.read_bytes()),gzip.decompress(self.path.read_bytes()))

    def test_JR02_short_successful_write_is_detected_and_retried(self):
        calls=[]
        def append(path,member):
            calls.append(1)
            if len(calls)==1:
                with path.open('ab') as stream:stream.write(member[:3])
            else:journal.append_bytes(path,member)
        with journal.DurableGzipText(self.path,append=append) as out:out.write('one\n')
        self.assertEqual(gzip.decompress(self.path.read_bytes()),'one\n'.replace('\n',os.linesep).encode())
        self.assertEqual(out.proof()['recoveries_n'],1)


if __name__=='__main__':unittest.main()
