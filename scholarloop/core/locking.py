"""Nonblocking advisory process lock, released by the OS even after a crash."""

from contextlib import contextmanager
import errno


@contextmanager
def exclusive_lock(path):
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
                    raise BlockingIOError(errno.EAGAIN, "Another process holds this lock", str(path)) from exc
                raise
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
