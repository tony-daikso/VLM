# Rented GPU status (updated 2026-11-06 12:10)

## Available: 1x L40S 46 GB (glows) — busy with CG10k v15, then CG10k C- (nc) 3 mm

| item | value |
|---|---|
| ssh | `ssh -p 26747 root@tw-07.access.glows.ai` (root; Mac key installed; no `~/.ssh/config` alias) |
| GPU | NVIDIA L40S 46068 MiB, driver 580.178.04 (CUDA 13.0 driver) |
| CPU / RAM / disk | 12 cores (load ~4 with 4 v15 workers), 2 TB RAM, `/` overlay 200 GB (work dir `/root/g4090`), `/datadrive` = user's own data (SSL_shortcut, VLM, copy.py; 35 GB free) — do not touch the user's files |
| OS / python | Ubuntu 20.04, system python 3.8 (unused) |
| env | `/root/radar_env` (uv venv, python 3.12.16) — torch 2.7.1+cu126, torchvision 0.22.1, nnunetv2 2.8.1, TotalSegmentator 2.18.0, numpy 2.5.4, nibabel 5.4.2, scipy 1.18.1, scikit-image 0.26.0, SimpleITK 2.5.7, timm 1.0.22 (= CT-rate `radar` env); uv in `/root/.local/bin` |
| TS weights | `/root/.totalsegmentator/nnunet/results/` Dataset292 (vertebrae 1.5 mm) + Dataset297 (total 3 mm fast), copied from CT-rate, md5 identical per file. **Dataset295/others (ribs) are NOT there** — copy from CT-rate `~/.totalsegmentator` the same way if needed |
| code | `/root/g4090/code/{src,src_v15,*.yaml}` = copy of CT-rate `~/OWL/cg10k_infer` (config_cg10k.yaml ts_model_dir -> /root/...) ; runners `/root/g4090/bin/` |
| reachability | the glows box can NOT reach CT-rate (LAN). CT-rate CAN reach glows outbound but has no ssh key of its own -> the coordinator runs on the Mac and streams `ssh CT-rate tar c \| ssh L40S tar x` (no Mac disk staging). Upload Mac->glows ~4 MB/s per ssh stream, ~15-60 MB/s with 6 parallel streams; download is fast |

Smoke (2 v15 + 1 3 mm cases already done on CT-rate, recomputed on the L40S): emb/emb_max cosine 1.000000 for every
unit/window/stage (max |diff| 2.4e-4 = fp16 rounding); seg15 voxel agreement 0.99999, 3 mm mask voxel agreement
0.99997; bone emb cosine >= 0.99999. Per case: v15 0.8-3.9 s (peak 2.2-4.1 GB), 3 mm 4.1 s (single worker).

## How work is dispatched (scripts in `remote/rented/`)

* `g_coord.py` (Mac, started by `run_g.sh`: nohup + `caffeinate -w`), claim tag **`l40s26747:1`**, CT-rate helper
  `ct_claim_r.py` (piped over ssh stdin).
  * Phase 1 v15: rounds of 100 ids, main from 50 % of the sorted id list upward (then downward), then delay;
    claim (O_EXCL in `owl_v15/<set>/claims`) -> push NIfTI + 3 mm mask -> `/root/g4090/bin/run_round15.sh`
    (4 workers x 3 threads of the unchanged `vert15_full.py`) -> pull into `owl_v15/<set>/.l40s_stage` -> rename into
    seg15 then emb15 -> verify -> release -> delete on L40S. Round k+1 is pushed while k computes; k syncs while k+1 computes.
  * Phase 2 nc (C- 3 mm, automatic after v15): `run_nc_workers.sh` (3 persistent `cg10k_run.py --watch --gpu-lock`
    workers, 4 threads) ; Mac keeps <= 24 nc cases in flight (claims in CT-rate `~/OWL/cg10k_infer/state/nc/claims`),
    syncs masks/prep/patches/emb/emb_bones via `owl/nc/.l40s_stage` + rename (bones last). At the end it waits for
    CT-rate `run_nc.sh` and re-runs `infer_head` nc if `probs_nc.csv` is older than the last L40S sync
    (run_nc.sh only waits for `spark` claims).
* Log `remote/rented/logs/coord.log`, speeds `remote/rented/state/speed.tsv`, failures `state/failed_{v15,nc}.txt`;
  L40S logs `/root/g4090/logs/`.
* STOP: `touch remote/rented/STOP` (current round / in-flight finish, sync, release, exit). Fast: `kill -TERM $(cat remote/rented/state/coord.pid)`.
  Resume: `rm STOP; bash remote/rented/run_g.sh` (leftover `l40s26747` claims synced or released).
* When the nc phase completes normally the coordinator deletes `/root/g4090/{in,out,smoke}` (all CG10k data) on the L40S.

## For a rib-fracture extraction job on this host
GPU is in use until CG10k nc is finished (nc NIfTIs start arriving from the Mac ~13:20). Free VRAM is ~35 GB while v15 runs.
Reuse the pattern: copy weights from CT-rate via tar pipe, write a CT-rate-side claim helper, and use a Mac relay
like `g_coord.py` (`pipe()` = parallel tar streams; pre-create target dirs because CTDATA is NTFS/fuseblk and
parallel `tar x` mkdir races fail).

## Not usable
* RTX 4090 `ssh -p 25595 glows@tw-07.access.glows.ai`: no NVIDIA driver (nouveau loaded, no /dev/nvidia*), sudo needs a
  password -> abandoned. Only uv (`~/.local/bin`) and an empty partial setup in `~`; no CG10k data was ever copied there.
* RTX 4090 port 25966 / 2x L40S port 25185: closed since the morning.
