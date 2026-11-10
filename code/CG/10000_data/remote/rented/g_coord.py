"""Mac relay coordinator for the rented L40S (root@tw-07.access.glows.ai:26747, claim tag l40s26747), 2026-11-06.

The glows box cannot reach CT-rate (LAN 192.168.0.189) and CT-rate has no ssh key of its own, so the Mac relays:
data is streamed `ssh CT-rate tar c | ssh L40S tar x` (and back), nothing is staged on the Mac disk. Uploads to
glows are ~4 MB/s per ssh stream, so every transfer is split into PAR parallel streams.

Phase 1 - v15 (1.5 mm vertebra, src_v15/vert15_full.py unchanged), rounds of ROUND ids:
  claim on CT-rate (ct_claim_r.py v15claim: main from FRAC of the sorted id list upward, then downward; delay after)
  -> push NIfTI + 3 mm mask -> L40S run_round15.sh (N_W workers) -> pull emb15/seg15 into owl_v15/<set>/.l40s_stage
  -> v15install (rename, emb15 last) -> verify -> release claims -> delete the case files on the L40S.
  Round k+1 is claimed + pushed while round k computes; round k is synced while k+1 computes.
  Ids without emb15 on the L40S (failure) are released (CT-rate redoes them), logged in state/failed_v15.txt.
Phase 2 - nc (3 mm OWL, src/cg10k_run.py unchanged): L40S run_nc_workers.sh (persistent --watch workers);
  keep <= QUEUE cases in flight: ncclaim -> push into in/nc (stage dir + mv) -> poll ncdone.py -> pull into
  owl/nc/.l40s_stage -> ncinstall -> verify -> release -> delete on L40S. Ends when CT-rate has
  nifti_3mm_nc/.SYNC_DONE and nothing is claimable; then touches in/nc/.SYNC_DONE on the L40S, waits for CT-rate
  run_nc.sh to finish and re-runs infer_head nc on CT-rate if probs_nc.csv is older than our last nc sync
  (run_nc.sh only waits for 'spark' claims, not ours).
Control: STOP file (rented/STOP): no new claims; in-flight work finishes, syncs, exits.
         SIGTERM: L40S workers killed, finished cases synced, the other claims released.
Resume: rerun; leftover claims of our tag are synced (if done on the L40S) or released at start.
At normal completion all CG10k data on the L40S (in/, out/, smoke/) is deleted.
"""
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
ST = D / "state"
CT = "CT-rate"
CTPY = "/home/daikso/miniconda3/envs/radar/bin/python"
L = ["-p", "26747", "root@tw-07.access.glows.ai"]
G = "/root/g4090"
R = "/media/daikso/CTDATA/cg10k"
TAG = "l40s26747"
SSHO = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", "-c", "aes128-gcm@openssh.com"]
V15SETS = [("main", f"{R}/nifti_3mm", float(os.environ.get("FRAC", "0.5"))), ("delay", f"{R}/nifti_3mm_delay", 0.0)]
N_W = int(os.environ.get("N_W", "4"))
THREADS = int(os.environ.get("THREADS", "3"))
ROUND = int(os.environ.get("ROUND", "80"))
PAR = int(os.environ.get("PAR", "4"))
NC_W = int(os.environ.get("NC_W", "3"))
NC_TH = int(os.environ.get("NC_TH", "4"))
QUEUE = int(os.environ.get("QUEUE", "24"))
DO_V15 = os.environ.get("DO_V15", "1") == "1"
DO_NC = os.environ.get("DO_NC", "1") == "1"

claimed15 = {}   # (set, id) -> round
inflight_nc = {}  # id -> t
stats = {"v15": 0, "v15_fail": 0, "nc": 0, "nc_fail": 0}
last_nc_sync = 0.0
remote_running = False


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def stop():
    return (D / "STOP").exists()


def run(cmd, tries=4, inp=None, check=True):
    for k in range(tries):
        p = subprocess.run(cmd, capture_output=True, text=True, input=inp)
        if p.returncode == 0:
            return p.stdout.strip()
        log("cmd error", " ".join(cmd)[:160], p.returncode, p.stderr[-300:])
        time.sleep(10 * (k + 1))
    if check:
        raise RuntimeError(f"failed: {' '.join(cmd)[:200]}")
    return None


def ct(*args):
    return run(["ssh", *SSHO, CT, CTPY, "-", *args], inp=(D / "ct_claim_r.py").read_text())


def ctb(*args):  # chunked (long id lists)
    head, ids = list(args[:-1]), list(args[-1])
    out = []
    for i in range(0, len(ids), 300):
        out.append(ct(*head, *ids[i:i + 300]))
    return " ".join(o for o in out if o).split()


def lsh(cmd, tries=4, check=True):
    return run(["ssh", *SSHO, *L, cmd], tries=tries, check=check)


def pipe(src_host, src_dir, files, dst_host, dst_dir):
    """stream files (relative to src_dir) from src_host to dst_host:dst_dir via PAR parallel tar pipes."""
    files = list(files)
    if not files:
        return
    chunks = [files[i::PAR] for i in range(PAR) if files[i::PAR]]

    def side(host):
        return ["ssh", *SSHO, *(L if host == "L" else [CT])]

    # create all target dirs ONCE before the parallel tars (CTDATA is NTFS/fuseblk: concurrent mkdir by several
    # tar x processes raced -> 'Cannot open: No such file or directory')
    dirs = sorted({str(Path(dst_dir) / Path(f).parent) for f in files})
    run(side(dst_host) + ["mkdir -p " + " ".join(dirs)])
    for k in range(4):
        ps = []
        for ch in chunks:
            src = side(src_host) + [f"cd {src_dir} && tar cf - --ignore-failed-read {' '.join(ch)} 2>/dev/null; true"]
            dst = side(dst_host) + [f"mkdir -p {dst_dir} && tar xf - -C {dst_dir}"]
            p1 = subprocess.Popen(src, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p2 = subprocess.Popen(dst, stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            p1.stdout.close()
            ps.append((p1, p2))
        bad = False
        for p1, p2 in ps:
            err = p2.communicate()[1]
            p1.wait()
            if p2.returncode != 0:
                bad = True
                log("pipe error", src_host, "->", dst_host, err[-300:].decode(errors="replace"))
        if not bad:
            return
        time.sleep(15 * (k + 1))
    raise RuntimeError("pipe failed")


# ------------------------------------------------------------------------------------------------- v15
def v15_claim(k):
    for s, src, frac in V15SETS:
        if (ST / f"v15_exhausted_{s}").exists():
            continue
        ids = ct("v15claim", s, str(ROUND), TAG, str(frac)).split()
        if not ids:
            (ST / f"v15_exhausted_{s}").touch()
            log(f"v15 {s}: nothing left to claim")
            continue
        for c in ids:
            claimed15[(s, c)] = k
        t = time.time()
        pipe("CT", src, [f"{c}.nii.gz" for c in ids], "L", f"{G}/in/v15/{s}/ct")
        pipe("CT", f"{R}/owl/{s}/masks", [f"{c}.nii.gz" for c in ids], "L", f"{G}/in/v15/{s}/masks")
        lsh(f"mkdir -p {G}/state && printf '%s\\n' {' '.join(ids)} > {G}/state/v15_round_{k}.txt")
        log(f"v15 round {k}: claimed+pushed {len(ids)} {s} {ids[0]}..{ids[-1]} in {time.time() - t:.0f}s")
        return s, ids
    return None


def v15_start(k, s):
    global remote_running
    lsh(f"rm -f {G}/state/v15_round_{k}.done; nohup setsid bash {G}/bin/run_round15.sh {k} {s} {N_W} {THREADS} "
        f">> {G}/logs/round15.log 2>&1 < /dev/null & echo started")
    remote_running = True


def v15_wait(k):
    global remote_running
    while True:
        o = lsh(f"test -f {G}/state/v15_round_{k}.done && echo DONE || echo RUN", tries=20)
        if o == "DONE":
            remote_running = False
            return
        time.sleep(15)


def v15_sync(s, ids):
    have = lsh(f"cd {G}/out/v15/{s}/emb15 && ls {' '.join(c + '.npz' for c in ids)} 2>/dev/null; true").split()
    ready = [h[:-4] for h in have]
    ok = set()
    if ready:
        pipe("L", f"{G}/out/v15/{s}", [f"seg15/{c}.npz" for c in ready] + [f"emb15/{c}.npz" for c in ready],
             "CT", f"{R}/owl_v15/{s}/.l40s_stage")
        ctb("v15install", s, ready)
        ok = set(ctb("v15done", s, ready))
    fail = [c for c in ids if c not in ok]
    ctb("release", "v15", s, TAG, ids)
    if fail:
        with open(ST / "failed_v15.txt", "a") as fh:
            fh.writelines(f"{time.strftime('%F %T')}\t{s}\t{c}\n" for c in fail)
        stats["v15_fail"] += len(fail)
    stats["v15"] += len(ok)
    for c in ids:
        claimed15.pop((s, c), None)
    rm = " ".join(f"in/v15/{s}/ct/{c}.nii.gz in/v15/{s}/masks/{c}.nii.gz out/v15/{s}/emb15/{c}.npz "
                  f"out/v15/{s}/seg15/{c}.npz out/v15/{s}/claims/{c}" for c in ids)
    lsh(f"cd {G} && rm -f {rm}")
    return len(ok), fail


def v15_leftovers():
    for s, _, _ in V15SETS:
        left = ct("mine", "v15", s, TAG).split()
        if left:
            n, fail = v15_sync(s, left)
            log(f"startup v15: {len(left)} leftover {s} claims -> synced {n}, released {len(fail)}")


def phase_v15():
    if lsh("pgrep -f [v]ert15_full.py >/dev/null && echo RUN || echo IDLE") == "RUN":
        log("v15 workers from a previous run still active on the L40S -> waiting")
        while lsh("pgrep -f [v]ert15_full.py >/dev/null && echo RUN || echo IDLE", tries=20) == "RUN":
            time.sleep(20)
    v15_leftovers()
    k = int((ST / "v15_round_no").read_text()) if (ST / "v15_round_no").exists() else 0
    cur = None if stop() else v15_claim(k)
    if cur is not None:
        v15_start(k, cur[0])
    while cur is not None:
        s, ids = cur
        (ST / "v15_round_no").write_text(str(k + 1))
        t0 = time.time()
        nxt = None if stop() else v15_claim(k + 1)   # claim + push k+1 while k computes
        v15_wait(k)
        dt = time.time() - t0
        if nxt is not None:
            v15_start(k + 1, nxt[0])                  # k+1 computes while k syncs
        n, fail = v15_sync(s, ids)
        log(f"v15 round {k} {s}: {n}/{len(ids)} ok, round wall {dt:.0f}s ({60 * n / max(dt, 1):.1f}/min); "
            f"failed->released {fail}; total {stats['v15']} failed {stats['v15_fail']}")
        with open(ST / "speed.tsv", "a") as fh:
            fh.write(f"{time.strftime('%F %T')}\tv15\t{s}\t{k}\t{n}\t{dt:.0f}\n")
        k += 1
        cur = nxt
        if cur is None and stop():
            log("STOP -> no further v15 rounds")
    log(f"v15 phase end: synced {stats['v15']}, failed {stats['v15_fail']}")


# ------------------------------------------------------------------------------------------------- nc
NC_REL = ("masks/{c}.nii.gz prep/{c}.npz patches/{c}.json "
          + " ".join(f"emb/{w}/{{c}}.npz" for w in
                     ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"])
          + " " + " ".join(f"emb_bones/{w}/{{c}}.npz" for w in ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]))


def nc_sync():
    global last_nc_sync
    ids = sorted(inflight_nc)
    if not ids:
        return
    o = lsh(f"/root/radar_env/bin/python {G}/bin/ncdone.py {' '.join(ids)}")
    ln = {x[:2]: x[2:].split() for x in o.splitlines()}
    done, failed = ln.get("D:", []), ln.get("F:", [])
    if done:
        files = [f for c in done for f in NC_REL.format(c=c).split()]
        pipe("L", f"{G}/out/nc", files, "CT", f"{R}/owl/nc/.l40s_stage")
        ctb("ncinstall", done)
        ok = set(ctb("ncdone", done))
        bad = [c for c in done if c not in ok]
        if bad:
            log("WARNING nc not done on CT-rate after install (released):", bad)
            failed += bad
        good = [c for c in done if c in ok]
        if good:
            ctb("release", "nc", "nc", TAG, good)
            for c in good:
                inflight_nc.pop(c, None)
            rm = " ".join(f"in/nc/{c}.nii.gz " + " ".join("out/nc/" + f for f in NC_REL.format(c=c).split())
                          for c in good)
            lsh(f"cd {G} && rm -f {rm}")
            stats["nc"] += len(good)
            last_nc_sync = time.time()
            (ST / "last_nc_sync").write_text(str(last_nc_sync))
            log(f"nc synced {len(good)} (total {stats['nc']}, in flight {len(inflight_nc)})")
            with open(ST / "speed.tsv", "a") as fh:
                fh.write(f"{time.strftime('%F %T')}\tnc\tnc\t-\t{len(good)}\t-\n")
    if failed:
        ctb("release", "nc", "nc", TAG, failed)
        for c in failed:
            inflight_nc.pop(c, None)
        lsh(f"cd {G} && rm -f " + " ".join(f"in/nc/{c}.nii.gz" for c in failed))
        with open(ST / "failed_nc.txt", "a") as fh:
            fh.writelines(f"{time.strftime('%F %T')}\tnc\t{c}\n" for c in failed)
        stats["nc_fail"] += len(failed)
        log(f"nc failed on L40S {failed} -> released (CT-rate does them)")


def nc_refill():
    if stop():
        return False
    exhausted = False
    while len(inflight_nc) < QUEUE:
        ids = ct("ncclaim", str(min(8, QUEUE - len(inflight_nc))), TAG).split()
        if not ids:
            exhausted = ct("ncsyncdone") == "1"
            break
        for c in ids:
            inflight_nc[c] = time.time()
        pipe("CT", f"{R}/nifti_3mm_nc", [f"{c}.nii.gz" for c in ids], "L", f"{G}/in/nc/.stage")
        lsh(f"cd {G}/in/nc && mv .stage/*.nii.gz . ")
        log(f"nc claimed+pushed {len(ids)}: {ids[0]}..{ids[-1]} (in flight {len(inflight_nc)})")
    return exhausted


def nc_finish_infer():
    if stats["nc"] == 0 and not (ST / "last_nc_sync").exists():
        return
    ls_ = float((ST / "last_nc_sync").read_text()) if (ST / "last_nc_sync").exists() else last_nc_sync
    log("waiting for CT-rate run_nc.sh to end (to make sure probs_nc.csv includes the L40S cases)")
    t_end = time.time() + 12 * 3600
    while time.time() < t_end:
        o = run(["ssh", *SSHO, CT, f"pgrep -f [c]g10k_infer/run_nc.sh >/dev/null && echo RUN; "
                 f"pgrep -f '[s]rc/cg10k_run.py --set nc' >/dev/null && echo RUN; "
                 f"grep -l '^\\(spark\\|{TAG}\\)' ~/OWL/cg10k_infer/state/nc/claims/* 2>/dev/null | head -1; "
                 f"stat -c %Y {R}/owl/probs_nc.csv 2>/dev/null; true"], tries=6).split()
        if "RUN" not in o and not any("/claims/" in x for x in o):
            mt = [int(x) for x in o if x.isdigit()]
            if mt and mt[0] > ls_:
                log("probs_nc.csv is newer than our last nc sync -> ok")
                return
            log("run_nc.sh ended, probs_nc.csv missing/older than our last sync -> infer_head nc on CT-rate")
            D2 = "/home/daikso/OWL/cg10k_infer"
            out = run(["ssh", *SSHO, CT, f"cd {D2} && {CTPY} -u src/infer_head.py --model {D2}/model/E3_late.pt "
                       f"--emb-dir {R}/owl/nc/emb --bone-emb-dir {R}/owl/nc/emb_bones --per-seed "
                       f"--out {R}/owl/probs_nc.csv > logs/infer_nc_l40s.log 2>&1; echo exit=$?; "
                       f"tail -1 logs/infer_nc_l40s.log"], tries=1, check=False)
            log("infer_head nc:", (out or "").replace("\n", " | "))
            return
        time.sleep(300)
    log("gave up waiting for run_nc.sh (12 h)")


def phase_nc():
    left = ct("mine", "nc", "nc", TAG).split()
    for c in left:
        inflight_nc[c] = time.time()
    if left:
        log(f"startup nc: {len(left)} leftover claims -> in flight (pushed again if missing on L40S)")
        have = set(x[:-7] for x in lsh(f"ls {G}/in/nc 2>/dev/null; true").split())
        miss = [c for c in left if c not in have]
        if miss:
            pipe("CT", f"{R}/nifti_3mm_nc", [f"{c}.nii.gz" for c in miss], "L", f"{G}/in/nc/.stage")
            lsh(f"cd {G}/in/nc && mv .stage/*.nii.gz . ")
    if lsh("pgrep -f '[c]g10k_run.py --set nc' >/dev/null && echo RUN || echo IDLE") == "IDLE":
        lsh(f"rm -f {G}/in/nc/.SYNC_DONE; N_W={NC_W} TH={NC_TH} nohup setsid bash {G}/bin/run_nc_workers.sh "
            f">> {G}/logs/nc_workers.log 2>&1 < /dev/null & echo started")
        log(f"nc workers started on L40S (N_W={NC_W} TH={NC_TH})")
    last_status = 0
    while True:
        nc_sync()
        exhausted = nc_refill()
        if not inflight_nc and (stop() or exhausted):
            break
        if time.time() - last_status > 1800:
            last_status = time.time()
            log("status: " + ct("status", TAG).replace("\n", " | "))
        time.sleep(20)
    lsh(f"touch {G}/in/nc/.SYNC_DONE")
    log(f"nc phase end: synced {stats['nc']}, failed {stats['nc_fail']}; stop={stop()}")
    return not stop()


def cleanup():
    if remote_running or claimed15:
        try:
            lsh("pkill -f [v]ert15_full.py; true", tries=2, check=False)
            time.sleep(5)
            by = {}
            for (s, c) in list(claimed15):
                by.setdefault(s, []).append(c)
            for s, cs in by.items():
                n, fail = v15_sync(s, cs)
                log(f"exit: v15 synced {n} {s}, released {len(fail)}")
        except Exception as e:  # noqa: BLE001
            log("EXIT v15 SYNC/RELEASE FAILED", e)
    if inflight_nc:
        try:
            lsh(f"touch {G}/state_nc/STOP; pkill -f '[c]g10k_run.py --set nc'; true", tries=2, check=False)
            time.sleep(5)
            nc_sync()
            if inflight_nc:
                ctb("release", "nc", "nc", TAG, sorted(inflight_nc))
                log(f"exit: released {len(inflight_nc)} nc claims")
                inflight_nc.clear()
        except Exception as e:  # noqa: BLE001
            log("EXIT nc SYNC/RELEASE FAILED", e)


def on_signal(sig, frm):
    log(f"signal {sig} -> exit")
    sys.exit(1)


def main():
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    atexit.register(cleanup)
    ST.mkdir(exist_ok=True)
    log(f"g_coord start pid {os.getpid()} N_W={N_W} THREADS={THREADS} ROUND={ROUND} PAR={PAR} NC_W={NC_W} "
        f"QUEUE={QUEUE} DO_V15={DO_V15} DO_NC={DO_NC}")
    if stop():
        log("STOP exists - remove it to run")
        return
    if DO_V15:
        phase_v15()
    complete = False
    if DO_NC and not stop():
        complete = phase_nc()
    log("status: " + ct("status", TAG).replace("\n", " | "))
    if complete:
        lsh(f"rm -rf {G}/in {G}/out {G}/smoke; ls {G}", tries=3, check=False)
        log("all CG10k data on the L40S deleted (in/ out/ smoke/)")
        nc_finish_infer()
    log("g_coord exit")


if __name__ == "__main__":
    main()
