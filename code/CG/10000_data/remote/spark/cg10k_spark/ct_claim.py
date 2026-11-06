"""Runs ON CT-rate (piped via ssh stdin) for the Spark helper of the cg10k OWL run.

  claim   SET N ORDER TAG   -> O_EXCL-claim up to N not-done, not-failed, stable cases of SET in CT-rate's own
                               claims dir (content 'TAG:1'; TAG != CT-rate hostname, so cg10k_run.py never takes
                               these claims over), print claimed ids. ORDER = asc|desc.
  release SET TAG id...     -> remove our claims (only those whose content starts with 'TAG:')
  done    SET id...         -> print the ids that are done on CT-rate (5 emb + 3 bone npz)
  status  TAG               -> per set: n_nifti n_done n_spark_claims
"""
import json
import os
import sys
import time
from pathlib import Path

R = Path("/media/daikso/CTDATA/cg10k")
STATE = Path("/home/daikso/OWL/cg10k_infer/state")
IN = {"main": R / "nifti_3mm", "delay": R / "nifti_3mm_delay", "nc": R / "nifti_3mm_nc"}
WINS = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
BWINS = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]


def done(s, cid):
    o = R / "owl" / s
    return all((o / "emb" / w / f"{cid}.npz").exists() for w in WINS) and \
        all((o / "emb_bones" / w / f"{cid}.npz").exists() for w in BWINS)


def ids(s):
    out = []
    now = time.time()
    if not IN[s].is_dir():
        return []
    for n in os.listdir(IN[s]):
        if n.startswith(".") or not n.endswith(".nii.gz") or ".tmp" in n:
            continue
        st = os.stat(IN[s] / n)
        if st.st_size == 0 or now - st.st_mtime < 300:
            continue
        out.append(n[:-7])
    return sorted(out)


def failed(s):
    f = STATE / s / "failed.jsonl"
    return {json.loads(l)["id"] for l in open(f)} if f.exists() else set()


def main():
    cmd = sys.argv[1]
    if cmd == "claim":
        s, n, order, tag = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
        cdir = STATE / s / "claims"
        cdir.mkdir(parents=True, exist_ok=True)
        fl = failed(s)
        got = []
        lst = ids(s)
        if order == "desc":
            lst = lst[::-1]
        for cid in lst:
            if len(got) >= n:
                break
            if cid in fl or (cdir / cid).exists() or done(s, cid):
                continue
            try:
                fd = os.open(cdir / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{tag}:1".encode())
            os.close(fd)
            if done(s, cid):
                (cdir / cid).unlink()
                continue
            got.append(cid)
        print(" ".join(got))
    elif cmd == "release":
        s, tag = sys.argv[2], sys.argv[3]
        for cid in sys.argv[4:]:
            c = STATE / s / "claims" / cid
            try:
                if c.read_text().startswith(f"{tag}:"):
                    c.unlink()
            except FileNotFoundError:
                pass
    elif cmd == "done":
        s = sys.argv[2]
        print(" ".join(c for c in sys.argv[3:] if done(s, c)))
    elif cmd == "status":
        tag = sys.argv[2]
        for s in IN:
            ii = ids(s)
            nd = sum(done(s, c) for c in ii)
            cl = 0
            cdir = STATE / s / "claims"
            if cdir.exists():
                for c in cdir.iterdir():
                    try:
                        cl += c.read_text().startswith(f"{tag}:")
                    except FileNotFoundError:
                        pass
            print(f"{s} nifti={len(ii)} done={nd} spark_claims={cl}")


if __name__ == "__main__":
    main()
