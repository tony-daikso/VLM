"""Osteopenia x Exp J: slice the 1.5 mm per-vertebra embeddings (outputs_vert15/merged_final/emb15) to L1-L4,
train/val only. The id list comes from analysis_osteopenia/labels_trval.csv, whose split column is read first and
filtered to train/val; labels are not used here. Test never loaded.
Output: emb15_L1L4_trval.npz  E [N,4,3,448] f16 (mean pool), Emax [N,4,3,448] f16 (max pool), P [N,4] bool,
        nvox [N,4], ids [N]
usage: python ov_build.py OUT.npz"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/")
EMB = ROOT + "outputs_vert15/merged_final/emb15/"
sp = pd.read_csv(ROOT + "outputs_final/analysis_osteopenia/labels_trval.csv", usecols=["study_id", "split"])
sp = sp[sp.split.isin(["train", "val"])]
ids = [i for i in sp.study_id if os.path.exists(EMB + f"{i}.npz")]
LV = ["L1", "L2", "L3", "L4"]
N = len(ids); E = np.zeros((N, 4, 3, 448), np.float16); Em = None
P = np.zeros((N, 4), bool); NV = np.zeros((N, 4), np.int64)
for n, cid in enumerate(ids):
    z = np.load(EMB + f"{cid}.npz")
    li = [list(z["levels"]).index(l) for l in LV]
    E[n] = z["emb"][li]; P[n] = z["present"][li]; NV[n] = z["nvox"][li]
    if n % 2000 == 0: print(n, flush=True)
np.savez(sys.argv[1], E=E, P=P, nvox=NV, ids=np.array(ids))
print("DONE", N, "present frac", P.mean(0))
