"""Spark helper for the CT-rate cg10k 1.5 mm vertebra run (src_v15/vert15_full.py, 2026-11-06).

Rounds (pipelined):
  * claim ROUND ids ON CT-rate (ct_claim15.py: O_EXCL in owl_v15/<set>/claims, content 'spark-6736:1'; CT-rate's
    vert15_full.py skips any existing claim and never takes one over). Order: delay ascending, then main descending
    (CT-rate works main ascending, then delay).
  * pull NIfTI + 3 mm mask into in/<set>/{ct,masks}; while round k runs, round k+1 is claimed + pulled.
  * N_W workers of the unchanged src_v15/vert15_full.py extract (--ct-local) on the round's id list -> out/<set>.
  * sync: rsync seg15 then emb15 (temp + rename; emb15 = CT-rate's done marker) -> verify on CT-rate -> release
    claims. Ids without local emb15 (Spark failure) are released -> CT-rate redoes them (logged in state/spark_failed.txt).
  * STOP file (~/OWL/cg10k_v15_spark/STOP): no new rounds; the running round finishes, syncs, releases, exit.
  * SIGTERM/exit: workers terminated, finished cases synced, all other claims released.
  * start: leftover Spark claims on CT-rate (crash) -> synced if done locally, else released.
"""
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path.home() / "OWL/cg10k_v15_spark"
CT = "daikso@192.168.0.189"
R = "/media/daikso/CTDATA/cg10k"
O = f"{R}/owl_v15"
CTPY = "/home/daikso/miniconda3/envs/radar/bin/python"
PY = str(Path.home() / "radar_env/bin/python")
TAG = "spark-6736"
SETS = [("delay", f"{R}/nifti_3mm_delay", "asc"), ("main", f"{R}/nifti_3mm", "desc")]
N_W = int(os.environ.get("N_W", "3"))
THREADS = int(os.environ.get("THREADS", "4"))
ROUND = int(os.environ.get("ROUND", "60"))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-c", "aes128-gcm@openssh.com"]
RSH = " ".join(SSH)

claimed = {}       # (set, id) -> round no, claimed on CT-rate and not yet synced/released
procs = []
exhausted = set()
n_synced = 0
n_failed = 0


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def ct(*args, tries=4):
    for k in range(tries):
        p = subprocess.run(SSH + [CT, CTPY, "-", *args], stdin=open(D / "ct_claim15.py"), capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
        log("ct_claim error", args[:3], p.stderr[-300:])
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"ct_claim {args[:2]} failed")


def rsync(src, dst, files):
    lst = D / "state" / f"files.{os.getpid()}.txt"
    lst.write_text("\n".join(files) + "\n")
    for k in range(4):
        p = subprocess.run(["rsync", "-a", "--ignore-missing-args", "-e", RSH, "--files-from", str(lst), src, dst],
                           capture_output=True, text=True)
        if p.returncode == 0:
            return
        log("rsync error", src, dst, p.stderr[-300:])
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"rsync {src} -> {dst} failed")


def release(s, cs):
    for i in range(0, len(cs), 200):
        ct("release", s, TAG, *cs[i:i + 200])


def sync(s, cs):
    """sync finished cs of set s, release all cs; returns (n_ok, failed list)."""
    global n_synced, n_failed
    o = D / "out" / s
    ready = [c for c in cs if (o / "emb15" / f"{c}.npz").exists()]
    if ready:
        rsync(f"{o}/", f"{CT}:{O}/{s}/", [f"seg15/{c}.npz" for c in ready])
        rsync(f"{o}/", f"{CT}:{O}/{s}/", [f"emb15/{c}.npz" for c in ready])
        ok = set(ct("done", s, *ready).split())
    else:
        ok = set()
    bad = [c for c in ready if c not in ok]
    if bad:
        log("WARNING not on CT-rate after sync (claims released anyway):", s, bad)
    fail = [c for c in cs if c not in ok]
    release(s, cs)
    for c in cs:
        claimed.pop((s, c), None)
        for f in (D / "in" / s / "ct" / f"{c}.nii.gz", D / "in" / s / "masks" / f"{c}.nii.gz"):
            f.unlink(missing_ok=True)
    if fail:
        with open(D / "state" / "spark_failed.txt", "a") as fh:
            fh.writelines(f"{time.strftime('%F %T')}\t{s}\t{c}\n" for c in fail)
        n_failed += len(fail)
    n_synced += len(ok)
    return len(ok), fail


def claim_round(k):
    """claim up to ROUND ids from the first non-exhausted set, pull inputs. returns (set, ids) or None."""
    for s, src, order in SETS:
        if s in exhausted:
            continue
        ids = ct("claim", s, str(ROUND), order, TAG).split()
        if not ids:
            exhausted.add(s)
            log(f"{s}: nothing left to claim")
            continue
        for c in ids:
            claimed[(s, c)] = k
        d = D / "in" / s
        (d / "ct").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        rsync(f"{CT}:{src}/", f"{d}/ct/", [f"{c}.nii.gz" for c in ids])
        rsync(f"{CT}:{R}/owl/{s}/masks/", f"{d}/masks/", [f"{c}.nii.gz" for c in ids])
        (D / "state" / f"round_{k}.txt").write_text("\n".join(ids) + "\n")
        log(f"round {k}: claimed+pulled {len(ids)} {s}: {ids[0]}..{ids[-1]}")
        return s, ids
    return None


def run_round(k, s, ids):
    o = D / "out" / s
    for c in ids:  # local leftovers of an interrupted run
        if not (o / "emb15" / f"{c}.npz").exists():
            (o / "claims" / c).unlink(missing_ok=True)
    for p in list((o / "emb15").glob("*.tmp*")) + list((o / "seg15").glob("*.tmp*")) if o.exists() else []:
        p.unlink(missing_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    ps = []
    for i in range(N_W):
        lf = open(D / "logs" / f"w{i}.log", "a")
        ps.append(subprocess.Popen([PY, "-u", "vert15_full.py", "extract", "--ids", str(D / "state" / f"round_{k}.txt"),
                                    "--out", str(o), "--mask-dirs", str(D / "in" / s / "masks"),
                                    "--ct-local", str(D / "in" / s / "ct"), "--device", "cuda", "--wid", str(i),
                                    "--threads", str(THREADS)], cwd=D / "src_v15", env=env, stdout=lf,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        time.sleep(3)
    return ps


def cleanup():
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(60)
        except Exception:  # noqa: BLE001
            p.kill()
    by = {}
    for (s, c) in list(claimed):
        by.setdefault(s, []).append(c)
    for s, cs in by.items():
        try:
            n, fail = sync(s, cs)
            log(f"exit: synced {n} {s}, released {len(fail)} unfinished claims")
        except Exception as e:  # noqa: BLE001
            log("EXIT SYNC/RELEASE FAILED", s, len(cs), e)


def on_signal(sig, frm):
    log(f"signal {sig} -> exit")
    sys.exit(1)


def main():
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    atexit.register(cleanup)
    (D / "state").mkdir(exist_ok=True)
    log(f"spark_coord15 start pid {os.getpid()} N_W={N_W} THREADS={THREADS} ROUND={ROUND}")
    for s, _, _ in SETS:  # leftovers from a crashed run
        left = ct("mine", s, TAG).split()
        if left:
            n, fail = sync(s, left)
            log(f"startup: {len(left)} leftover {s} claims -> synced {n}, released {len(fail)}")
    k = int((D / "state" / "round_no").read_text()) if (D / "state" / "round_no").exists() else 0
    nxt = None if (D / "STOP").exists() else claim_round(k)
    while nxt is not None:
        s, ids = nxt
        t0 = time.time()
        procs[:] = run_round(k, s, ids)
        (D / "state" / "round_no").write_text(str(k + 1))
        nxt = None if (D / "STOP").exists() else claim_round(k + 1)
        for p in procs:
            p.wait()
        procs.clear()
        n, fail = sync(s, ids)
        dt = time.time() - t0
        log(f"round {k} {s}: {n}/{len(ids)} ok in {dt:.0f}s ({60 * n / max(dt, 1):.1f}/min), failed->released "
            f"{fail}; total synced {n_synced} failed {n_failed}")
        k += 1
        if nxt is None and (D / "STOP").exists():
            log("STOP file -> no further rounds")
    log(f"done: synced {n_synced}, failed {n_failed}; status: " + ct("status", TAG).replace("\n", " | "))


if __name__ == "__main__":
    main()
