"""Runs ON CT-rate (piped via `ssh CT-rate python - ...` stdin) for the rented-GPU helper (l40s26747, 2026-11-06).

Claims use the SAME files as CT-rate / Spark, content 'TAG:1' (TAG = l40s26747, never CT-rate's hostname):
  v15 (1.5 mm vertebra, vert15_full.py): owl_v15/<set>/claims/<id>   (CT-rate never takes a claim over)
  nc  (3 mm OWL, cg10k_run.py):           ~/OWL/cg10k_infer/state/nc/claims/<id> (taken over only if host==CT-rate)

  v15claim SET N TAG FRAC   -> claim up to N ids (have 3 mm mask, no emb15, no claim); order = from FRAC of the
                               sorted list upward, then from FRAC downward (avoids CT-rate asc head / Spark desc tail)
  v15install SET id...      -> move owl_v15/SET/.l40s_stage/{seg15,emb15}/<id>.npz into place (seg15 first,
                               emb15 = done marker last; same fs -> rename), print installed ids
  v15done SET id...         -> ids with emb15/<id>.npz
  ncclaim N TAG             -> claim up to N nc ids (stable >= 300 s, not done, not failed, no claim), ascending
  ncinstall id...           -> move owl/nc/.l40s_stage/{masks,prep,patches,emb/*,emb_bones/*} into place (bones last)
  ncdone id...              -> ids done (5 emb + 3 bone npz)
  ncsyncdone                -> prints 1 if nifti_3mm_nc/.SYNC_DONE exists
  release KIND SET TAG id...-> remove our claims (KIND v15|nc; content starts with 'TAG:')
  mine KIND SET TAG         -> our claims
  status TAG                -> counts
"""
import json
import os
import sys
import time
from pathlib import Path

R = Path("/media/daikso/CTDATA/cg10k")
O15 = R / "owl_v15"
NCSTATE = Path("/home/daikso/OWL/cg10k_infer/state/nc")
IN15 = {"main": R / "nifti_3mm", "delay": R / "nifti_3mm_delay"}
NCIN = R / "nifti_3mm_nc"
NCOUT = R / "owl" / "nc"
WINS = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
BWINS = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
STAGE = ".l40s_stage"


def cdir(kind, s):
    return O15 / s / "claims" if kind == "v15" else NCSTATE / "claims"


def ours(c, tag):
    try:
        return c.read_text().startswith(f"{tag}:")
    except FileNotFoundError:
        return False


def excl(p, tag):
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{tag}:1".encode())
    os.close(fd)
    return True


def ids15(s):
    m = R / "owl" / s / "masks"
    out = [n[:-7] for n in os.listdir(IN15[s]) if n.endswith(".nii.gz") and "tmp" not in n and not n.startswith(".")]
    return sorted(c for c in out if (m / f"{c}.nii.gz").exists())


def nc_done(cid):
    return all((NCOUT / "emb" / w / f"{cid}.npz").exists() for w in WINS) and \
        all((NCOUT / "emb_bones" / w / f"{cid}.npz").exists() for w in BWINS)


def nc_ids():
    if not NCIN.is_dir():
        return []
    now, out = time.time(), []
    for n in os.listdir(NCIN):
        if n.startswith(".") or not n.endswith(".nii.gz") or ".tmp" in n:
            continue
        st = os.stat(NCIN / n)
        if st.st_size == 0 or now - st.st_mtime < 300:
            continue
        out.append(n[:-7])
    return sorted(out)


def nc_failed():
    f = NCSTATE / "failed.jsonl"
    return {json.loads(l)["id"] for l in open(f)} if f.exists() else set()


def main():
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "v15claim":
        s, n, tag, frac = a[1], int(a[2]), a[3], float(a[4])
        cd, ed = cdir("v15", s), O15 / s / "emb15"
        cd.mkdir(parents=True, exist_ok=True)
        lst = ids15(s)
        i0 = int(len(lst) * frac)
        lst = lst[i0:] + lst[:i0][::-1]
        got = []
        for cid in lst:
            if len(got) >= n:
                break
            if (cd / cid).exists() or (ed / f"{cid}.npz").exists():
                continue
            if not excl(cd / cid, tag):
                continue
            if (ed / f"{cid}.npz").exists():
                (cd / cid).unlink()
                continue
            got.append(cid)
        print(" ".join(got))
    elif cmd == "v15install":
        s = a[1]
        base = O15 / s
        ok = []
        for cid in a[2:]:
            st = base / STAGE
            e = st / "emb15" / f"{cid}.npz"
            if not e.exists():
                continue
            g = st / "seg15" / f"{cid}.npz"
            if g.exists():
                os.replace(g, base / "seg15" / f"{cid}.npz")
            os.replace(e, base / "emb15" / f"{cid}.npz")
            ok.append(cid)
        print(" ".join(ok))
    elif cmd == "v15done":
        s = a[1]
        print(" ".join(c for c in a[2:] if (O15 / s / "emb15" / f"{c}.npz").exists()))
    elif cmd == "ncclaim":
        n, tag = int(a[1]), a[2]
        cd = cdir("nc", "nc")
        cd.mkdir(parents=True, exist_ok=True)
        fl = nc_failed()
        got = []
        for cid in nc_ids():
            if len(got) >= n:
                break
            if cid in fl or (cd / cid).exists() or nc_done(cid):
                continue
            if not excl(cd / cid, tag):
                continue
            if nc_done(cid):
                (cd / cid).unlink()
                continue
            got.append(cid)
        print(" ".join(got))
    elif cmd == "ncinstall":
        st = NCOUT / STAGE
        ok = []
        for cid in a[1:]:
            have = all((st / "emb" / w / f"{cid}.npz").exists() for w in WINS) and \
                all((st / "emb_bones" / w / f"{cid}.npz").exists() for w in BWINS)
            if not have:
                continue
            for rel in (f"masks/{cid}.nii.gz", f"prep/{cid}.npz", f"patches/{cid}.json"):
                if (st / rel).exists():
                    (NCOUT / rel).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(st / rel, NCOUT / rel)
            for sub, ws in (("emb", WINS), ("emb_bones", BWINS)):
                for w in ws:
                    (NCOUT / sub / w).mkdir(parents=True, exist_ok=True)
                    os.replace(st / sub / w / f"{cid}.npz", NCOUT / sub / w / f"{cid}.npz")
            ok.append(cid)
        print(" ".join(ok))
    elif cmd == "ncdone":
        print(" ".join(c for c in a[1:] if nc_done(c)))
    elif cmd == "ncsyncdone":
        print(1 if (NCIN / ".SYNC_DONE").exists() else 0)
    elif cmd == "release":
        kind, s, tag = a[1], a[2], a[3]
        for cid in a[4:]:
            c = cdir(kind, s) / cid
            if ours(c, tag):
                c.unlink(missing_ok=True)
    elif cmd == "mine":
        kind, s, tag = a[1], a[2], a[3]
        cd = cdir(kind, s)
        print(" ".join(sorted(c.name for c in cd.iterdir() if c.is_file() and ours(c, tag))) if cd.exists() else "")
    elif cmd == "status":
        tag = a[1]
        for s in IN15:
            ii = ids15(s)
            nd = sum((O15 / s / "emb15" / f"{c}.npz").exists() for c in ii)
            cd = cdir("v15", s)
            cl = list(cd.iterdir()) if cd.exists() else []
            print(f"v15 {s} ids={len(ii)} emb15={nd} claims={len(cl)} ours={sum(ours(c, tag) for c in cl)}")
        ii = nc_ids()
        cd = cdir("nc", "nc")
        cl = list(cd.iterdir()) if cd.exists() else []
        print(f"nc nifti={len(ii)} done={sum(nc_done(c) for c in ii)} claims={len(cl)} ours={sum(ours(c, tag) for c in cl)}"
              f" sync_done={(NCIN / '.SYNC_DONE').exists()}")
    else:
        raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main()
