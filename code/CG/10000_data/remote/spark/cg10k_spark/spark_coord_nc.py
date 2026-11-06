"""Spark helper for the CT-rate cg10k OWL 3 mm run (2026-11-06).

Loop:
  * keep <= QUEUE cases in flight: claim them ON CT-rate (ct_claim.py, O_EXCL in CT-rate's own claims dir, content
    'spark-6736:1' -> CT-rate workers skip them and never take them over), pull NIfTIs into in/<set>/
    (priority: delay ascending, then main descending = from the tail; CT-rate works main ascending)
  * Spark workers (src/cg10k_run.py, unchanged, --watch) process in/<set> -> out/<set>/
  * a case is synced back when its 8 npz exist locally: rsync masks/prep/patches first, then emb, then emb_bones
    (rsync writes dot-temp files + rename -> no half files; CT-rate 'done' = the npz files, so they go last),
    verify done on CT-rate, then release the claim
  * Spark failures (state/<set>/failed.jsonl) -> claim released, CT-rate does them
  * STOP file (~/OWL/cg10k_spark/STOP): no new claims; finish in-flight, sync, exit
  * on exit (any reason) all claims of not-yet-synced cases are released
At the end: touches in/<set>/.SYNC_DONE (Spark workers exit), then waits for run_all on CT-rate and re-runs
infer_head on CT-rate if the probs csv is older than our last sync while run_all is no longer running.
"""
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path.home() / "OWL/cg10k_spark"
CT = "daikso@192.168.0.189"
R = "/media/daikso/CTDATA/cg10k"
CTD = "/home/daikso/OWL/cg10k_infer"
CTPY = "/home/daikso/miniconda3/envs/radar/bin/python"
TAG = "spark-6736"
SETS = [("nc", f"{R}/nifti_3mm_nc", "desc")]   # nc copy (2026-11-06): CT-rate run_nc.sh does infer_head
WINS = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
BWINS = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
QUEUE = int(os.environ.get("QUEUE", "12"))
BATCH = int(os.environ.get("BATCH", "6"))
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-c", "aes128-gcm@openssh.com"]
RSH = " ".join(SSH)

inflight = {}      # (set, id) -> t_claimed
exhausted = set()
last_sync = 0.0
n_synced = 0


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def ct(*args, tries=3):
    for k in range(tries):
        p = subprocess.run(SSH + [CT, CTPY, "-", *args], stdin=open(D / "ct_claim.py"), capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
        log("ct_claim error", args[:3], p.stderr[-300:])
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"ct_claim {args[:2]} failed")


def rsync(src, dst, files=None):
    cmd = ["rsync", "-a", "-e", RSH]
    if files is not None:
        lst = D / "state" / f"files.{os.getpid()}.txt"
        lst.write_text("\n".join(files) + "\n")
        cmd += ["--files-from", str(lst)]
    for k in range(3):
        p = subprocess.run(cmd + [src, dst], capture_output=True, text=True)
        if p.returncode == 0:
            return
        log("rsync error", src, dst, p.stderr[-300:])
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"rsync {src} -> {dst} failed")


def local_done(s, cid):
    o = D / "out" / s
    return all((o / "emb" / w / f"{cid}.npz").exists() for w in WINS) and \
        all((o / "emb_bones" / w / f"{cid}.npz").exists() for w in BWINS)


def local_failed(s):
    f = D / "state" / s / "failed.jsonl"
    return {json.loads(l)["id"] for l in open(f)} if f.exists() else set()


def release_all():
    by = {}
    for (s, c) in list(inflight):
        by.setdefault(s, []).append(c)
    for s, cs in by.items():
        try:
            ct("release", s, TAG, *cs)
            log(f"released {len(cs)} {s} claims on exit")
        except Exception as e:  # noqa: BLE001
            log("RELEASE FAILED", s, cs, e)
    inflight.clear()


def on_signal(sig, frm):
    log(f"signal {sig} -> exit")
    sys.exit(1)


def sync_back():
    global last_sync, n_synced
    for s, _, _ in SETS:
        ready = [c for (ss, c) in inflight if ss == s and local_done(s, c)]
        if not ready:
            continue
        o = f"{D}/out/{s}/"
        dst = f"{CT}:{R}/owl/{s}/"
        rsync(o, dst, [f"masks/{c}.nii.gz" for c in ready] + [f"prep/{c}.npz" for c in ready] +
              [f"patches/{c}.json" for c in ready])
        rsync(o, dst, [f"emb/{w}/{c}.npz" for c in ready for w in WINS])
        rsync(o, dst, [f"emb_bones/{w}/{c}.npz" for c in ready for w in BWINS])
        ok = set(ct("done", s, *ready).split())
        bad = [c for c in ready if c not in ok]
        if bad:
            log("WARNING not done on CT-rate after sync:", s, bad)
        good = [c for c in ready if c in ok]
        if good:
            ct("release", s, TAG, *good)
            for c in good:
                inflight.pop((s, c), None)
                try:
                    (D / "in" / s / f"{c}.nii.gz").unlink()
                except FileNotFoundError:
                    pass
            n_synced += len(good)
            last_sync = time.time()
            (D / "state" / "last_sync").write_text(str(last_sync))
            log(f"synced {len(good)} {s} -> CT-rate (total {n_synced}, in flight {len(inflight)})")


def handle_failed():
    for s, _, _ in SETS:
        fl = local_failed(s)
        bad = [c for (ss, c) in inflight if ss == s and c in fl]
        if bad:
            ct("release", s, TAG, *bad)
            for c in bad:
                inflight.pop((s, c), None)
            log(f"Spark failed {s} {bad} -> claims released (CT-rate will do them)")


def refill(stop):
    if stop:
        return
    while len(inflight) < QUEUE:
        got = False
        for s, src, order in SETS:
            if s in exhausted:
                continue
            ids = ct("claim", s, str(min(BATCH, QUEUE - len(inflight))), order, TAG).split()
            if not ids:
                # nc: the Mac is still converting -> only exhausted once CT-rate has .SYNC_DONE for this set
                p = subprocess.run(SSH + [CT, f"test -f {src}/.SYNC_DONE"], capture_output=True)
                if p.returncode != 0:
                    continue
                exhausted.add(s)
                log(f"{s}: nothing left to claim")
                continue
            dst = D / "in" / s
            dst.mkdir(parents=True, exist_ok=True)
            for c in ids:
                inflight[(s, c)] = time.time()
            rsync(f"{CT}:{src}/", f"{dst}/", [f"{c}.nii.gz" for c in ids])
            log(f"claimed+pulled {len(ids)} {s}: {ids[0]}..{ids[-1]}")
            got = True
            break
        if not got:
            return


def finish_infer():
    """after our last sync: make sure CT-rate's probs csvs include our cases."""
    if n_synced == 0:
        return
    log("waiting for CT-rate run_all to finish (to verify probs include Spark cases)")
    t_end = time.time() + 12 * 3600
    while time.time() < t_end:
        # only act when run_all really completed (last log line 'ALL DONE', no run_all / worker running)
        p = subprocess.run(SSH + [CT, f"pgrep -f cg10k_infer/run_all.sh >/dev/null && echo RUN; "
                                      f"pgrep -f src/cg10k_run.py >/dev/null && echo RUN; "
                                      f"tail -1 {CTD}/logs/run_all.log | grep -q 'ALL DONE' || echo NOTDONE; "
                                      f"stat -c %Y {R}/owl/probs_main.csv {R}/owl/probs_delay.csv 2>/dev/null"],
                           capture_output=True, text=True)
        out = p.stdout.split()
        if p.returncode in (0, 1) and "RUN" not in out and "NOTDONE" not in out:
            mt = [int(x) for x in out if x.isdigit()]
            if len(mt) == 2 and min(mt) > last_sync:
                log("probs_main/delay.csv are newer than the last Spark sync -> nothing to do")
                return
            log("run_all not running and probs older than last Spark sync / missing -> running infer_head on CT-rate")
            for s in ("main", "delay"):
                cmd = (f"cd {CTD} && {CTPY} -u src/infer_head.py --model {CTD}/model/E3_late.pt --emb-dir {R}/owl/{s}/emb "
                       f"--bone-emb-dir {R}/owl/{s}/emb_bones --per-seed --out {R}/owl/probs_{s}.csv "
                       f"> logs/infer_{s}_spark.log 2>&1; echo exit=$?; tail -1 logs/infer_{s}_spark.log")
                p2 = subprocess.run(SSH + [CT, cmd], capture_output=True, text=True)
                log(f"infer_head {s}:", p2.stdout.strip().replace("\n", " | "))
            return
        time.sleep(300)
    log("gave up waiting for run_all (12 h)")


def main():
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    atexit.register(release_all)
    (D / "state").mkdir(exist_ok=True)
    log(f"spark_coord_nc start pid {os.getpid()} QUEUE={QUEUE} BATCH={BATCH}")
    while True:
        stop = (D / "STOP").exists()
        handle_failed()
        sync_back()
        refill(stop)
        if not inflight and (stop or len(exhausted) == len(SETS)):
            break
        time.sleep(15)
    log(f"all claimed cases synced ({n_synced}); stop={stop}")
    for s, _, _ in SETS:
        (D / "in" / s).mkdir(parents=True, exist_ok=True)
        (D / "in" / s / ".SYNC_DONE").touch()
    # finish_infer: not for nc (run_nc.sh on CT-rate waits for our claims, then runs infer_head)
    log("spark_coord exit")


if __name__ == "__main__":
    main()
