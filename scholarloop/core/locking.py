"""Nonblocking advisory process lock, released by the OS even after a crash."""

from contextlib import contextmanager


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
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
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
