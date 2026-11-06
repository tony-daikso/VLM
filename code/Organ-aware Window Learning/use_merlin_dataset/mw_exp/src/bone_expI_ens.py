"""Exp I: seed-mean groups + Caruana greedy ensemble chosen on TRAIN OOF only (bone_ens.py recipe).
Selection score = train AUROC of the mean of fold-z-scored group logits (z within each CV fold, removes cross-fold
calibration offsets). Saved val/train logits: each group's raw seed-mean logit z-scored with its train mean/std
(as bone_ens.py), averaged over the selected multiset. Val is never used.
usage: python bone_expI_ens.py --g ref='.../oof_bone_E_ref_s*.npz' --g lv1='...' --out .../expI/groups/ens_X.npz
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from probe import auroc  # noqa: E402
from bone_expI_eval import fold_of  # noqa: E402

B = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones"
ap = argparse.ArgumentParser()
ap.add_argument("--g", action="append", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-gain", type=float, default=1e-4); ap.add_argument("--max-steps", type=int, default=20)
a = ap.parse_args()
meta = pd.read_csv(B / "expI/meta.csv", index_col=0, keep_default_na=False, na_values=[""])
fo = fold_of(meta).values
ids0 = None; RAW, FZ = {}, {}
for spec in a.g:
    n, p = spec.split("=", 1)
    fs = sorted(glob.glob(p)); assert fs, p
    r, z_ = [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}
        if ids0 is None:
            ids0, split, y = [str(i) for i in z["ids"]], z["split"], z["y"]
            assert ids0 == list(meta.index)
        assert [str(i) for i in z["ids"]] == ids0
        L = np.asarray(z["logit"], float); r.append(L)
        Lz = L.copy()
        for k in range(5):
            m = fo == k; Lz[m] = (L[m] - L[m].mean()) / L[m].std()
        z_.append(Lz)
    RAW[n], FZ[n] = np.mean(r, 0), np.mean(z_, 0)
tr = split == "train"
ZS = {n: (v - v[tr].mean()) / v[tr].std() for n, v in RAW.items()}
FZn = {n: (v - v[tr].mean()) / v[tr].std() for n, v in FZ.items()}
for n in RAW:
    print(f"{n:30s} train OOF (fold-z) {auroc(y[tr], FZ[n][tr]):.4f}")
sel, best = [], -1
for _ in range(a.max_steps):
    cand = {g: auroc(y[tr], (np.sum([FZn[s] for s in sel], 0) + FZn[g])[tr]) for g in FZn}
    g, s = max(cand.items(), key=lambda kv: kv[1])
    if s < best + a.min_gain:
        break
    sel.append(g); best = s
    print(f"greedy +{g}: train OOF (fold-z) {s:.4f}")
ens = np.mean([ZS[s] for s in sel], 0)
ens_fz = np.mean([FZn[s] for s in sel], 0)
ens = np.where(tr, ens_fz, ens)            # train rows: fold-z version (what was selected on); val: train-z raw logits
np.savez(a.out, ids=np.array(ids0), split=split, logit=ens.astype(np.float32), y=y, exp="greedy:" + "+".join(sel))
Path(a.out).with_suffix(".json").write_text(json.dumps({"greedy": sel, "train_oof_foldz": best,
    "groups": {n: auroc(y[tr], FZ[n][tr]) for n in RAW}}, indent=1, ensure_ascii=False))
print("ens =", sel, f"train {best:.4f} ->", a.out)
