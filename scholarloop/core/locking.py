"""Advisory process lock, released by the OS even after a crash."""

from contextlib import contextmanager
import errno
import time


@contextmanager
def exclusive_lock(path, wait=0.0):
    """Hold the lock on `path`. With `wait` 0 a competing holder raises BlockingIOError at once
    (one session per directory); with `wait` > 0 the lock is retried for that many seconds, for
    short critical sections several processes legitimately share (budget and spend files)."""
    deadline = time.monotonic() + wait
    while True:
        try:
            with _lock_once(path):
                yield
            return
        except _Busy as busy:
            if time.monotonic() >= deadline:
                raise busy.error from None
            time.sleep(0.01)


class _Busy(Exception):
    """Raised only while acquiring, so an error from the caller's own block is never retried."""
    def __init__(self, error):
        self.error = error


@contextmanager
def _lock_once(path):
    with path.open("a+b") as stream:
        try:
            import fcntl
        except ImportError:  # Windows
            import msvcrt
            if stream.seek(0, 2) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                # Windows reports a competing lock as EACCES, not EAGAIN.
                # Keep real file-access failures from open/write distinct.
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise _Busy(BlockingIOError(errno.EAGAIN, "Another process holds this lock", str(path))) from exc
                raise
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise _Busy(exc) from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
