"""bd15 progress / label-subset helper (train/val only).
  bd15_status.py ROOT status            -> one-line progress
  bd15_status.py ROOT tier0_done        -> exit 0 when tier0 is finished (every tier0 id embedded or terminal-failed, or
                                           >= 99.5 % embedded and no claimed tier0 id still pending for > 30 min)
  bd15_status.py ROOT labels OUT [0|all]-> labels csv restricted to ids with an embedding (tier0 ids or all)"""
import sys, os, json, glob, time
import pandas as pd
R = os.path.expanduser(sys.argv[1]); cmd = sys.argv[2]
ids = pd.read_csv(f"{R}/inputs/ids_prio.csv")
emb = {os.path.basename(f)[:-4] for f in glob.glob(f"{R}/emb/*.npz") if ".tmp" not in f}
crops = {os.path.basename(f)[:-4] for f in glob.glob(f"{R}/crops/*.npz") if ".tmp" not in f}
term = set()
for f in glob.glob(f"{R}/logs/crop_info_w*.jsonl"):
    for l in open(f):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("status") in ("nomask", "outside", "err"):
            term.add(d["study_id"])
t0 = ids[ids.tier == 0].study_id
if cmd == "status":
    print(f"{time.strftime('%m-%d %H:%M')} crops {len(crops)} emb {len(emb)} failed {len(term)} | tier0 emb {t0.isin(emb).sum()}/{len(t0)}"
          f" | all emb {ids.study_id.isin(emb).sum()}/{len(ids)}", flush=True)
elif cmd == "tier0_done":
    left = [s for s in t0 if s not in emb and s not in term]
    if not left:
        sys.exit(0)
    if len(left) <= 0.005 * len(t0):
        old = [s for s in left if os.path.exists(f"{R}/claims/{s}") and time.time() - os.path.getmtime(f"{R}/claims/{s}") > 1800]
        unclaimed = [s for s in left if not os.path.exists(f"{R}/claims/{s}")]
        if len(old) == len(left) and not unclaimed:
            sys.exit(0)
    sys.exit(1)
elif cmd == "labels":
    lab = pd.read_csv(f"{R}/inputs/labels_bd.csv"); assert set(lab.split) <= {"train", "val"}
    lab = lab[lab.split.isin(["train", "val"])]
    keep = set(t0) if sys.argv[4] == "0" else set(ids.study_id)
    lab = lab[lab.study_id.isin(keep) & lab.study_id.isin(emb)]
    lab.to_csv(sys.argv[3], index=False)
    print(f"labels {sys.argv[3]}: train {(lab.split == 'train').sum()} (pos {int(lab[lab.split == 'train'].y.sum())}) val {(lab.split == 'val').sum()}", flush=True)
