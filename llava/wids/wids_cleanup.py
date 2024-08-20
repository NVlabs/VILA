"""
This module provides utilities for managing files in a directory.

It includes a function `keep_most_recent_files` that keeps the most recent
files in a directory, deleting the rest based on the maximum size of the directory
in bytes and the maximum number of files to keep.

The cleanup job can be run in the background using `create_cleanup_background_process`.
"""

import errno
import fcntl
import glob
import os
import time

import numpy as np


def keep_most_recent_files(pattern, maxsize=int(1e12), maxfiles=1000, debug=False):
    """Keep the most recent files in a directory, deleting the rest.

    The maxsize is the maximum size of the directory in bytes. The maxfiles is
    the maximum number of files to keep. The files are sorted by modification
    time, and the most recent files are kept. If the directory is already
    smaller than maxsize, then no files are deleted. If there are fewer than
    maxfiles, then no files are deleted."""

    # get the list of files in the directory
    fnames = glob.glob(pattern)
    # compute a list of (mtime, fname, size) triples
    files = []
    for fname in fnames:
        try:
            s = os.stat(fname)
        except FileNotFoundError:
            continue
        files.append((s.st_mtime, fname, s.st_size))
    # sort the list by mtime, most recent first
    files.sort(reverse=True)
    # compute an accumulated total of the file sizes in order using np.cumsum
    sizes = np.cumsum([size for mtime, fname, size in files])
    # compute a cutoff index based on maxsize
    cutoff = np.searchsorted(sizes, maxsize)
    # compute a cutoff index based on maxfiles
    cutoff = min(cutoff, maxfiles)
    # delete the files above the cutoff in reverse order
    for mtime, fname, size in files[cutoff:][::-1]:
        try:
            os.unlink(fname)
        except FileNotFoundError:
            pass


class ExclusiveLock:
    """A simple non-blocking exclusive lock using fcntl."""

    def __init__(self, lockfile):
        self.lockfile = lockfile

    def try_lock(self):
        try:
            self.lock = open(self.lockfile, "w")
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            else:
                raise

    def release_lock(self):
        self.lock.close()
        os.unlink(self.lockfile)


def create_cleanup_background_process(pattern, maxsize=int(1e12), maxfiles=1000, every=60):
    """Create a background process that keeps a directory below a certain size."""

    def cleanup_worker(every):
        # use a lock file to ensure that only one cleanup worker is running
        lockfile = os.path.join(os.path.dirname(pattern), ".cleanup.lock")
        lock = ExclusiveLock(lockfile)
        if not lock.try_lock():
            return
        while True:
            keep_most_recent_files(pattern, maxsize=maxsize, maxfiles=maxfiles)
            time.sleep(every)

    import multiprocessing

    p = multiprocessing.Process(target=cleanup_worker, args=(every,))
    p.start()
    return p
