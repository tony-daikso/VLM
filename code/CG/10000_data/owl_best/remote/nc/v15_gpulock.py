"""Run the unchanged vert15_full.py extract, holding the cg10k GPU flock only around the GPU stages
(worker.seg_nnunet, pilot_vert15.embed_units) - same wrapper as rib15_gpulock.py.
usage: v15_gpulock.py <lockfile> <vert15_full.py> extract args...   (run from the src dir)"""
import sys, os, fcntl, functools
LOCK = sys.argv.pop(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
sys.argv.pop(1)
import worker, pilot_vert15


def locked(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        fd = os.open(LOCK, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fn(*a, **k)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    return w


worker.seg_nnunet = locked(worker.seg_nnunet)
pilot_vert15.embed_units = locked(pilot_vert15.embed_units)
import vert15_full
sys.argv[0] = "vert15_full.py"
vert15_full.main()
