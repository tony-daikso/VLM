"""Mac-side feeder for the streamed full run (python3, stdlib only).

Sources of CTs:
  - local   : the Mac's external drive (rsync to the host)
  - relay   : a remote host that stores the CTs (ssh cat | ssh host), e.g. CT-rate box
Hosts:
  - mode=copy : CTs are copied into the host's inbox (deleted by the worker after processing)
  - mode=ref  : the host already stores the CTs; the feeder only writes {id}.ref tickets
A case is done when {out}/patches/{id}.json exists on any host. Restart-safe: state is rebuilt from the hosts.
When the queue is empty and nothing is in flight, STOP is written to every inbox.

Usage: python3 feeder.py feeder_config.json
"""
from __future__ import annotations

import json
import random
import shlex
import subprocess
import sys
import threading
import time

SSH_OPTS = "-o ControlMaster=auto -o ControlPath=/tmp/owl_cm_%r@%h:%p -o ControlPersist=15m -o ServerAliveInterval=30"
LOCK = threading.Lock()


def sh(cmd: str, timeout: int = 600) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[:120]}... -> {r.returncode}: {r.stderr[-300:]}")
    return r.stdout


def ssh(host: str, cmd: str, timeout: int = 600) -> str:
    if host == "local":  # the Mac itself acts as a host
        return sh(cmd, timeout)
    return sh(f"ssh {SSH_OPTS} {host} {json.dumps(cmd)}", timeout)


class State:
    def __init__(self, cfg):
        self.cfg = cfg
        self.all_ids = [l.strip() for l in open(cfg["ids_file"]) if l.strip()]
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.inbox: dict[str, set[str]] = {h: set() for h in cfg["hosts"]}
        self.copying: dict[str, set[str]] = {h: set() for h in cfg["hosts"]}
        self.queue: list[str] = []
        self.t0 = time.time()
        self.n_done0 = None

    def refresh(self):
        done, failed = set(), set()
        for h, hc in self.cfg["hosts"].items():
            out = ssh(h, f"ls {hc['out']}/patches 2>/dev/null | sed 's/.json$//'; echo ===; "
                         f"find {hc['inbox']} -name 'AC*' -not -path '*/failed/*' | sed 's#.*/##'; echo ===; ls {hc['inbox']}/failed 2>/dev/null")
            d, ib, fl = out.split("===\n")
            done |= {x for x in d.split() if x.startswith("AC")}
            self.inbox[h] = {x.split(".")[0] for x in ib.split() if x.startswith("AC")}
            failed |= {x.split(".")[0] for x in fl.split() if x.startswith("AC")}
        with LOCK:
            self.done, self.failed = done, failed
            busy = set().union(*self.inbox.values(), *self.copying.values())
            self.queue = [i for i in self.all_ids if i not in done and i not in failed and i not in busy]
            if self.n_done0 is None:
                self.n_done0 = len(done)

    def pending(self, h):
        return len(self.inbox[h]) + len(self.copying[h])

    def pop(self, h):
        with LOCK:
            if not self.queue:
                return None
            cid = self.queue.pop(0)
            self.copying[h].add(cid)
            return cid

    def release(self, h, cid, ok):
        with LOCK:
            self.copying[h].discard(cid)
            if ok:
                self.inbox[h].add(cid)
            else:
                self.queue.insert(0, cid)


def pick_copy_host(st: State, src: dict):
    cands = []
    for h, hc in st.cfg["hosts"].items():
        if hc["mode"] != "copy" or ("hosts" in src and h not in src["hosts"]):
            continue
        p = st.pending(h)
        if p < hc["max_inflight"]:
            cands.append((p / hc["workers"], h))
    return min(cands)[1] if cands else None


def copy_loop(st: State, src: dict, stop: threading.Event):
    while not stop.is_set():
        h = pick_copy_host(st, src)
        if h is None:
            time.sleep(3)
            continue
        cid = st.pop(h)
        if cid is None:
            time.sleep(10)
            continue
        hc = st.cfg["hosts"][h]
        part, final = f"{hc['inbox']}/.{cid}.part", f"{hc['inbox']}/{cid}.nii.gz"
        try:
            if src["kind"] == "local":
                sh(f"rsync -a -e {json.dumps('ssh ' + SSH_OPTS)} '{src['dir']}/{cid}.nii.gz' {h}:{part} && "
                   f"ssh {SSH_OPTS} {h} 'mv {part} {final}'")
            elif src["kind"] == "pull" and h == src.get("storage_host"):  # storage host: HDD -> its own SSD inbox
                ssh(h, f"cp {src['dir']}/{cid}.nii.gz {part} && mv {part} {final}")
            elif src["kind"] == "pull":  # destination host pulls straight from the storage host (no Mac hop)
                inner = (f"ssh -c aes128-gcm@openssh.com -o BatchMode=yes {src['remote']} "
                         f"\"cat {src['dir']}/{cid}.nii.gz\" > {part} && mv {part} {final}")
                ssh(h, inner)
            else:  # relay through the Mac
                sh(f"ssh {SSH_OPTS} {src['host']} 'cat {src['dir']}/{cid}.nii.gz' | "
                   f"ssh {SSH_OPTS} {h} 'cat > {part} && mv {part} {final}'")
            st.release(h, cid, True)
        except Exception as e:  # noqa: BLE001
            print(f"[copy {src['name']}->{h}] {cid} failed: {e}", flush=True)
            st.release(h, cid, False)
            time.sleep(5)


def azure_loop(st: State, src: dict, stop: threading.Event):
    """Destination host downloads a batch of CTs straight from Azure, `batch` in parallel (single Azure streams are
    slow, ~0.5 MB/s, but they add up). The SAS lives in a 600 file on the host and never appears in this process."""
    batch = src.get("batch", 8)
    while not stop.is_set():
        cands = [(st.pending(h) / st.cfg["hosts"][h]["workers"], h) for h in src["hosts"]
                 if st.pending(h) + batch <= st.cfg["hosts"][h]["max_inflight"]]
        if not cands:
            time.sleep(5)
            continue
        h = min(cands)[1]
        ids = [c for c in (st.pop(h) for _ in range(batch)) if c]
        if not ids:
            time.sleep(10)
            continue
        ib = st.cfg["hosts"][h]["inbox"]
        cmd = (f"cd {ib} && export SAS=$(cat {src['sas_file']}) && printf '%s\\n' {' '.join(ids)} | "
               f"xargs -P {batch} -I@ sh -c 'curl -sf --retry 3 --retry-delay 10 --speed-limit 20000 --speed-time 120 "
               f"--max-time 2400 -o .@.part \"{src['url']}/@.nii.gz?$SAS\" && mv .@.part @.nii.gz || rm -f .@.part'; "
               f"ls {' '.join(i + '.nii.gz' for i in ids)} 2>/dev/null; true")
        try:
            # own (non-multiplexed) connection: long-running batches must not eat the shared master's session slots
            out = sh(f"ssh -o ControlMaster=no -o ControlPath=none -o ServerAliveInterval=30 {h} {shlex.quote(cmd)}", timeout=3000)
            got = set(x.split(".")[0] for x in out.split())
        except Exception as e:  # noqa: BLE001
            got = set()
            print(f"[azure->{h}] batch failed: {str(e)[:200]}", flush=True)
        for c in ids:
            st.release(h, c, c in got)
        if len(got) < len(ids):
            time.sleep(30)


def ref_loop(st: State, h: str, stop: threading.Event):
    hc = st.cfg["hosts"][h]
    while not stop.is_set():
        n = hc["max_inflight"] - st.pending(h)
        if n >= 2:
            ids = [c for c in (st.pop(h) for _ in range(n)) if c]
            if ids:
                cmd = "; ".join(f"echo {hc['ct_dir']}/{c}.nii.gz > {hc['inbox']}/.{c}.t && mv {hc['inbox']}/.{c}.t {hc['inbox']}/{c}.ref"
                                for c in ids)
                try:
                    ssh(h, cmd)
                    for c in ids:
                        st.release(h, c, True)
                except Exception as e:  # noqa: BLE001
                    print(f"[ref {h}] failed: {e}", flush=True)
                    for c in ids:
                        st.release(h, c, False)
        time.sleep(3)


def main():
    cfg = json.load(open(sys.argv[1]))
    st = State(cfg)
    for h, hc in cfg["hosts"].items():
        ssh(h, f"mkdir -p {hc['inbox']}/claimed {hc['inbox']}/failed; rm -f {hc['inbox']}/STOP {hc['inbox']}/.*.part")
    st.refresh()
    print(f"{len(st.all_ids)} ids, {len(st.done)} already done, {len(st.queue)} queued", flush=True)
    stop = threading.Event()
    threads = [threading.Thread(target=azure_loop if s["kind"] == "azure" else copy_loop, args=(st, s, stop), daemon=True)
               for s in cfg["sources"] for _ in range(s.get("threads", 1))]
    threads += [threading.Thread(target=ref_loop, args=(st, h, stop), daemon=True)
                for h, hc in cfg["hosts"].items() if hc["mode"] == "ref"]
    for t in threads:
        t.start()
    last = 0
    while True:
        time.sleep(cfg.get("refresh_s", 30))
        try:
            st.refresh()
        except Exception as e:  # noqa: BLE001
            print("refresh failed:", e, flush=True)
            continue
        n_new = len(st.done) - st.n_done0
        rate = n_new / max(time.time() - st.t0, 1)
        left = len(st.all_ids) - len(st.done) - len(st.failed)
        if time.time() - last > 300:
            last = time.time()
            eta = left / rate / 3600 if rate > 0 else float("nan")
            print(f"{time.strftime('%H:%M')} done {len(st.done)}/{len(st.all_ids)} failed {len(st.failed)} "
                  f"queue {len(st.queue)} inflight " + " ".join(f"{h}:{st.pending(h)}" for h in cfg["hosts"]) +
                  f" | {rate*3600:.0f}/h ETA {eta:.1f}h", flush=True)
        if not st.queue and all(st.pending(h) == 0 for h in cfg["hosts"]):
            stop.set()
            for h, hc in cfg["hosts"].items():
                ssh(h, f"touch {hc['inbox']}/STOP {hc['out']}/STOP_EXTRACT")
            print(f"ALL DONE: {len(st.done)} done, {len(st.failed)} failed", flush=True)
            break


if __name__ == "__main__":
    main()
