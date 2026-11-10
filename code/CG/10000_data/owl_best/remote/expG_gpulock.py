"""Run the unchanged extract_bones.py (Exp G single-rib units), holding the cg10k GPU flock only around the per-case GPU
stage (extract_bones.embed_bones), so the nc workers (--gpu-lock, same file) keep priority between cases on the 3080.
Peak ~3 GB (one process).  usage: expG_gpulock.py <lockfile> <extract_bones.py args...>   (run from the src dir)"""
import sys, os, fcntl, functools
LOCK = sys.argv.pop(1)
sys.path.insert(0, os.getcwd())
import extract_bones as X
_f = X.embed_bones
@functools.wraps(_f)
def locked(*a, **k):
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX); return _f(*a, **k)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
X.embed_bones = locked
sys.argv[0] = "extract_bones.py"
X.main()
