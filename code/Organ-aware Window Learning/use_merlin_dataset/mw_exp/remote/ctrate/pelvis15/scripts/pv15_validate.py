"""load-test every feature npz in f_<enc>: keys X/meta/nvox, consistent rows, finite X. usage: pv15_validate.py f_292 f_295"""
import os, sys, numpy as np, zipfile
from concurrent.futures import ProcessPoolExecutor
def chk(p):
    try:
        z = np.load(p); X, m, n = z["X"], z["meta"], z["nvox"]
        if not (len(X) == len(m) == len(n)) or len(X) == 0: return "shape", X.shape
        if not np.isfinite(X.astype(np.float32)).all(): return "nonfinite", X.shape
        return "ok", X.shape[1]
    except Exception as e: return "err:" + repr(e)[:80], None
roi = {f[:-4] for f in os.listdir("roi") if f.endswith(".npz") and ".tmp" not in f}
for d in sys.argv[1:]:
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f)
    tmp = [f for f in os.listdir(d) if ".tmp" in f]
    with ProcessPoolExecutor(6) as ex: res = list(ex.map(chk, [os.path.join(d, f) for f in fs], chunksize=64))
    bad = [(f, r) for f, r in zip(fs, res) if r[0] != "ok"]
    dims = {r[1] for r in res if r[0] == "ok"}
    have = {f[:-4] for f in fs}
    print(d, "files", len(fs), "tmp", len(tmp), "bad", len(bad), "dims", dims, "missing_vs_roi", len(roi - have), flush=True)
    with open(f"logs/validate_{d}.txt", "w") as fh:
        for f, r in bad: fh.write(f"{f}\t{r}\n")
        for c in sorted(roi - have): fh.write(f"{c}\tmissing\n")
