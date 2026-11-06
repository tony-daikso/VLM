# CG10k OWL pipeline — remote-host copies

Snapshot (2026-11-06) of the scripts that run on the GPU hosts, kept here so the pipeline is reproducible from git.
These are copies; the running versions live on the hosts.

- `ctrate/` — CT-rate (RTX 3080) `~/OWL/cg10k_infer/`: 3 mm extraction driver `src/cg10k_run.py` (gpu-lock, done-cache, recycle), `run_all.sh`, 1.5 mm vertebra extraction `src_v15/` + `run_v15.sh`, C- set `run_nc.sh`, configs/organ maps.
- `spark/` — Spark (GB10) helpers that share work with CT-rate via pre-created claims: `cg10k_spark/` (3 mm, `spark-6736:` claims) and `cg10k_v15_spark/` (1.5 mm).
- `a6000/` — A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/`: `probe.py` with `--save-model/--hp`, `infer_head.py`, `verify_saved_head.py`, `run_save_expB.sh` (E3_late expB weights, val reproduce 0.8846).

Mac-side scripts (series selection / DICOM→NIfTI, sync, DICOM header index, evaluation) are in the parent folder.
Data and outputs are not in git (see VLM/.gitignore).
