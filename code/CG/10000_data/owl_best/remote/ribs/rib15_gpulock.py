"""Run the unchanged rib15_full.py extract, but hold the cg10k GPU flock ONLY around the GPU stages
(worker.seg_nnunet and pilot_vert15.embed_units), so CPU work (NIfTI load, ROI, 1.5 mm resampling) of several rib
workers overlaps and the nc workers (--gpu-lock, same file) still never share the 10 GB card with a rib GPU stage.
usage: rib15_gpulock.py <lockfile> <rib15_full.py extract args...>   (run from the src dir)"""
import sys, os, fcntl, functools
LOCK = sys.argv.pop(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))   # src dir of rib15_full.py
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
import rib15_full
sys.argv[0] = "rib15_full.py"  # remaining argv = extract args
rib15_full.main()
