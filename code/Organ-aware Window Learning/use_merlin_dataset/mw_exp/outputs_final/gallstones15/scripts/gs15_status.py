"""gs15 progress / label-subset helper (train/val only).
  gs15_status.py ROOT status            -> one-line progress
  gs15_status.py ROOT tier0_done        -> exit 0 when every tier0 id is embedded or terminal (or >= 99.5 % and the rest
                                           claimed > 30 min ago)
  gs15_status.py ROOT all_cropped       -> exit 0 when every id has a crop or is terminal (same tolerance)
  gs15_status.py ROOT labels OUT [0|all]-> labels csv (labels_gs.csv rows, train/val only) restricted to ids with an embedding"""
import sys, os, json, glob, time
import pandas as pd
R = os.path.expanduser(sys.argv[1]); cmd = sys.argv[2]
ids = pd.read_csv(f"{R}/inputs/ids_prio.csv")
emb = {os.path.basename(f)[:-4] for f in glob.glob(f"{R}/emb/*.npz") if ".tmp" not in f}
crops = {os.path.basename(f)[:-4] for f in glob.glob(f"{R}/crops/*.npz")}
term = set()
for f in glob.glob(f"{R}/logs/crop_info_*.jsonl") + glob.glob(f"{R}/logs/mac/crop_info_*.jsonl"):
    for l in open(f):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("status") in ("nomask", "outside", "err"):
            term.add(d["study_id"])


def done(sub, have):
    left = [s for s in sub if s not in have and s not in term]
    if not left:
        return True
    if len(left) <= 0.005 * len(sub):
        old = [s for s in left if os.path.exists(f"{R}/claims/{s}") and time.time() - os.path.getmtime(f"{R}/claims/{s}") > 1800]
        return len(old) == len(left)
    return False


t0 = ids[ids.tier == 0].study_id
if cmd == "status":
    print(f"{time.strftime('%m-%d %H:%M')} crops {len(crops)} emb {len(emb)} failed {len(term)} | tier0 emb {t0.isin(emb).sum()}/{len(t0)}"
          f" | all emb {ids.study_id.isin(emb).sum()}/{len(ids)}", flush=True)
elif cmd == "tier0_done":
    sys.exit(0 if done(t0, emb) else 1)
elif cmd == "all_cropped":
    sys.exit(0 if done(ids.study_id, crops | emb) else 1)
elif cmd == "all_embedded":
    sys.exit(0 if done(ids.study_id, emb) else 1)
elif cmd == "labels":
    lab = pd.read_csv(f"{R}/inputs/labels_gs.csv"); assert set(lab.split) <= {"train", "val"}
    lab = lab[lab.split.isin(["train", "val"])]
    keep = set(t0) if sys.argv[4] == "0" else set(ids.study_id)
    lab = lab[lab.study_id.isin(keep) & lab.study_id.isin(emb)]
    lab.to_csv(sys.argv[3], index=False)
    print(f"labels {sys.argv[3]}: train {(lab.split == 'train').sum()} (pos {int(lab[lab.split == 'train'].y.sum())}) val {(lab.split == 'val').sum()}", flush=True)
