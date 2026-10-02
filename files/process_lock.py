"""Nonblocking OS-owned lock; process death releases ownership, not evidence."""
from contextlib import contextmanager
import os


@contextmanager
def process_lock(path):
    # Never unlink this inode: unlinking lets a second writer lock another file.
    # Existing empty legacy sentinel files are harmless; ownership is in the OS.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b', buffering=0) as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BlockingIOError('another process owns '+str(path)) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
