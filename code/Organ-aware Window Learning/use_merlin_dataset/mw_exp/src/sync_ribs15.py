"""Exp L relay (runs on the Mac; the 4090 has no route to the A6000): every 2 min unions the claims of the 4090
(/root/OWL/out_ribs15/claims) and the A6000 (outputs_ribs15/claims) (add-only, ids only), and every 10 min pulls the
4090 emb15/seg15 (+ timing/failed) to a Mac staging dir and pushes them to A6000 outputs_ribs15/ (--ignore-existing).
No raw CT moves through here.  stop: touch <stage>/STOP_SYNC
"""
import subprocess, sys, time
from pathlib import Path

G = ["ssh", "-p", "25966", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "root@tw-07.access.glows.ai"]
A6 = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "A6000"]
G_OUT, A_OUT = "/root/OWL/out_ribs15", "Desktop/Tony/VLM/OWL/mw_exp/outputs_ribs15"
ST = Path(sys.argv[1])


def sh(cmd, inp=None, timeout=900):
    return subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=timeout, check=True).stdout


def ls(host, d):
    return set(sh(host + [f"ls -f {d} 2>/dev/null | grep -v '^\\.' || true"]).split())


def claims():
    gc, ac = ls(G, f"{G_OUT}/claims"), ls(A6, f"{A_OUT}/claims")
    ta, tg = sorted(gc - ac), sorted(ac - gc)
    if ta:
        sh(A6 + [f"cd {A_OUT}/claims && xargs touch"], inp="\n".join(ta) + "\n")
    if tg:
        sh(G + [f"cd {G_OUT}/claims && xargs touch"], inp="\n".join(tg) + "\n")
    return len(ta), len(tg)


def outputs():
    for d in ("emb15", "seg15"):
        (ST / d).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--exclude", "*.tmp*", "--exclude", ".*", "-e", "ssh -p 25966",
                        f"root@tw-07.access.glows.ai:{G_OUT}/{d}/", f"{ST}/{d}/"], check=True, timeout=3600)
        subprocess.run(["rsync", "-a", "--ignore-existing", f"{ST}/{d}/", f"A6000:{A_OUT}/{d}/"], check=True, timeout=3600)
    (ST / "meta").mkdir(exist_ok=True)
    subprocess.run(f"rsync -a -e 'ssh -p 25966' 'root@tw-07.access.glows.ai:{G_OUT}/timing_w*.csv' "
                   f"'root@tw-07.access.glows.ai:{G_OUT}/failed.txt' '{ST}/meta/' ; "
                   f"rsync -a '{ST}/meta/' A6000:{A_OUT}/from_4090/", shell=True, timeout=600)


ST.mkdir(parents=True, exist_ok=True)
log = open(ST / "sync.log", "a")
last_out = 0.0
while not (ST / "STOP_SYNC").exists():
    try:
        a, g = claims()
        msg = f"claims +A6000 {a} +4090 {g}"
        if time.time() - last_out > 600:
            outputs(); last_out = time.time()
            msg += f"; outputs synced, stage emb15 {len(list((ST / 'emb15').glob('*.npz')))}"
        print(time.strftime("%F %T"), msg, file=log, flush=True)
    except Exception as e:  # noqa: BLE001
        print(time.strftime("%F %T"), f"error {e!r}"[:300], file=log, flush=True)
    time.sleep(120)
