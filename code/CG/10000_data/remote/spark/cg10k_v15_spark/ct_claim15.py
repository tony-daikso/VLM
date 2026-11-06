"""Runs ON CT-rate (piped via ssh stdin) for the Spark helper of the cg10k 1.5 mm vertebra run (2026-11-06).

vert15_full.py on CT-rate claims with a plain O_EXCL create of owl_v15/<set>/claims/<id> and NEVER takes a claim
over (no stale logic): any existing claim file is skipped. So a Spark claim (same O_EXCL, content 'TAG:1') is never
processed by CT-rate, and Spark never claims an id CT-rate holds or has done (emb15/<id>.npz).

  claim   SET N ORDER TAG -> claim up to N ids (NIfTI with a 3 mm mask, no emb15, no claim); print them. ORDER asc|desc
  release SET TAG id...   -> remove our claims (content starts with 'TAG:')
  done    SET id...       -> print ids with emb15/<id>.npz on CT-rate
  mine    SET TAG         -> print ids whose claim is ours
  status  TAG             -> per set: n_ids n_emb15 n_claims n_spark_claims
"""
import os
import sys
from pathlib import Path

R = Path("/media/daikso/CTDATA/cg10k")
O = R / "owl_v15"
IN = {"main": R / "nifti_3mm", "delay": R / "nifti_3mm_delay"}


def ids(s):
    m = R / "owl" / s / "masks"
    out = [n[:-7] for n in os.listdir(IN[s]) if n.endswith(".nii.gz") and "tmp" not in n and not n.startswith(".")]
    return sorted(c for c in out if (m / f"{c}.nii.gz").exists())


def ours(c, tag):
    try:
        return c.read_text().startswith(f"{tag}:")
    except FileNotFoundError:
        return False


def main():
    cmd = sys.argv[1]
    if cmd == "claim":
        s, n, order, tag = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
        cdir, edir = O / s / "claims", O / s / "emb15"
        cdir.mkdir(parents=True, exist_ok=True)
        lst = ids(s)
        if order == "desc":
            lst = lst[::-1]
        got = []
        for cid in lst:
            if len(got) >= n:
                break
            if (cdir / cid).exists() or (edir / f"{cid}.npz").exists():
                continue
            try:
                fd = os.open(cdir / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{tag}:1".encode())
            os.close(fd)
            if (edir / f"{cid}.npz").exists():
                (cdir / cid).unlink()
                continue
            got.append(cid)
        print(" ".join(got))
    elif cmd == "release":
        s, tag = sys.argv[2], sys.argv[3]
        for cid in sys.argv[4:]:
            c = O / s / "claims" / cid
            if ours(c, tag):
                c.unlink(missing_ok=True)
    elif cmd == "done":
        s = sys.argv[2]
        print(" ".join(c for c in sys.argv[3:] if (O / s / "emb15" / f"{c}.npz").exists()))
    elif cmd == "mine":
        s, tag = sys.argv[2], sys.argv[3]
        cdir = O / s / "claims"
        print(" ".join(sorted(c.name for c in cdir.iterdir() if c.stat().st_size and ours(c, tag))) if cdir.exists() else "")
    elif cmd == "status":
        tag = sys.argv[1 + 1]
        for s in IN:
            ii = ids(s)
            e = O / s / "emb15"
            nd = sum((e / f"{c}.npz").exists() for c in ii)
            cdir = O / s / "claims"
            cl = list(cdir.iterdir()) if cdir.exists() else []
            sp = sum(1 for c in cl if c.stat().st_size and ours(c, tag))
            print(f"{s} ids={len(ii)} emb15={nd} claims={len(cl)} spark_claims={sp}")


if __name__ == "__main__":
    main()
