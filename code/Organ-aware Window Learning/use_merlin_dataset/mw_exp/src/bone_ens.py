"""Experiment F: seed-average bone OOF members per config and choose the ensemble composition on TRAIN OOF only.

Input: final_F1/members/oof_bone_E_<variant>[__<model>__<pool>__<win>][_s<k>].npz (bone_label_exp.py, 10 seeds).
Step 1  per group: mean logit over seeds -> groups/<group>.npz (oof format: ids, split, logit, y), train OOF AUROC
        per seed and of the seed mean (+ val AUROC printed for information only, never used to select).
Step 2  greedy forward selection (with replacement, Caruana) of groups on TRAIN OOF AUROC of the mean of
        train-z-scored group logits; stops when the gain < --min-gain. -> groups/bone_ens.npz + selection.json.
usage: python bone_ens.py [--dir .../final_F1]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe import auroc  # noqa: E402

D = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/final_F1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(D))
    ap.add_argument("--min-gain", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=20)
    a = ap.parse_args()
    d = Path(a.dir); (d / "groups").mkdir(exist_ok=True)
    grp = {}
    for f in sorted(glob.glob(str(d / "members/oof_bone_E_*.npz"))):
        g = re.sub(r"_s\d+$", "", Path(f).stem.replace("oof_bone_E_", ""))
        g = g if "__" in g else g + "__site_max__mean__all3"
        grp.setdefault(g, []).append(f)
    info, L = {}, {}
    ids = split = y = None
    for g, fs in grp.items():
        Ls, cvs = [], []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            assert set(np.unique(z["split"])) <= {"train", "val"}
            if ids is None:
                ids, split, y = z["ids"], z["split"], z["y"]
            assert (z["ids"] == ids).all()
            Ls.append(z["logit"]); cvs.append(auroc(y[split == "train"], z["logit"][split == "train"]))
        m = np.mean(Ls, 0); L[g] = m
        tr, va = split == "train", split == "val"
        info[g] = {"n_seeds": len(fs), "cv_seeds": cvs, "cv_mean_seed": float(np.mean(cvs)), "cv_sd_seed": float(np.std(cvs)),
                   "cv_ens": auroc(y[tr], m[tr]), "val_ens_info_only": auroc(y[va], m[va])}
        np.savez(d / "groups" / f"{g}.npz", ids=ids, split=split, logit=m.astype(np.float32), y=y, exp=g, files=json.dumps(fs))
        print(f"{g:40s} seeds={len(fs):2d} train OOF: seed {np.mean(cvs):.4f}±{np.std(cvs):.4f} -> ens {info[g]['cv_ens']:.4f}"
              f" | val (info) {info[g]['val_ens_info_only']:.4f}")
    tr = split == "train"
    Z = {g: (v - v[tr].mean()) / v[tr].std() for g, v in L.items() if info[g]["n_seeds"] >= 5}
    sel, cur, best = [], None, -1
    for _ in range(a.max_steps):
        cand = {g: auroc(y[tr], (np.sum([Z[s] for s in sel], 0) + Z[g])[tr] / (len(sel) + 1)) for g in Z}
        g, s = max(cand.items(), key=lambda kv: kv[1])
        if s < best + a.min_gain:
            break
        sel.append(g); best = s
        print(f"greedy +{g}: train OOF AUROC {s:.4f}")
    ens = np.mean([Z[s] for s in sel], 0)
    va = split == "val"
    print(f"bone_ens = {sel}: train OOF {best:.4f} | val (info) {auroc(y[va], ens[va]):.4f}")
    np.savez(d / "groups" / "bone_ens.npz", ids=ids, split=split, logit=ens.astype(np.float32), y=y, exp="greedy:" + "+".join(sel))
    (d / "groups" / "selection.json").write_text(json.dumps({"groups": info, "greedy": sel, "greedy_train_oof": best}, indent=1))


if __name__ == "__main__":
    main()
