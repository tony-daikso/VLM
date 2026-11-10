"""CG10k rows for the fracture member matrices, built exactly like bone_probe.build / rib_probe.build
(E [N,U,3 windows (baseline, soft_tissue, bone),2 (emb, emb_max),448] f16, P [N,U], ids, units).
usage: cg_frac_rows.py <emb_dir root (contains baseline/ soft_tissue_L40_W400/ bone_L400_W1800/)> <id prefix> <out.npz>"""
import sys, os, numpy as np
root, pre, out = sys.argv[1:4]
WINS = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
sets = [{f[:-4] for f in os.listdir(f"{root}/{w}") if f.endswith(".npz") and ".tmp" not in f} for w in WINS]
ids = sorted(set.intersection(*sets))
z0 = np.load(f"{root}/{WINS[0]}/{ids[0]}.npz"); U = len(z0["organ_names"]); units = list(z0["organ_names"])
E = np.zeros((len(ids), U, 3, 2, 448), np.float16); P = np.zeros((len(ids), U), bool)
for i, cid in enumerate(ids):
    for w, wn in enumerate(WINS):
        z = np.load(f"{root}/{wn}/{cid}.npz")
        E[i, :, w, 0], E[i, :, w, 1] = z["emb"], z["emb_max"]
        if w == 0: P[i] = z["organ_present"]; assert list(z["organ_names"]) == units
np.savez(out, E=E, P=P, ids=np.array([pre + i for i in ids]), units=np.array(units)); print(out, E.shape, P.mean(0).round(3))
