"""Exp O probe inputs (train/val only; test ids are dropped from the split file before any label is read).
main_<code>.npz = subset of the *_notest main embedding matrix (organ units for U_org); labels_<code>.csv = study_id, split, y, fold.
codes: bd (biliary y_bdd, biliary folds), fa2/asc2 (copies of Exp M fa/asc inputs), frp (any fracture) / frpp (pelvis/sacrum/hip
site positives vs all negatives; other-site positives y = NaN, kept for prediction only), go (bowel obstruction y_sbo, its folds).
Run in outputs_final/region_units."""
import sys, os, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src"))
P = "probe_inputs/"
MAT = "../results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
z = np.load(MAT, allow_pickle=True); ids = [str(i) for i in z["ids"]]; units = [str(u) for u in z["units"]]
E = None
def main(code, us):
    global E
    if E is None: E = z["E"]
    ix = [units.index(u) for u in us]
    np.savez(P + f"main_{code}.npz", ids=np.array(ids), units=np.array(us), E=E[:, ix], P=z["P"][:, ix])
for a, b in (("fa", "fa2"), ("asc", "asc2")):
    shutil.copy(P + f"main_{a}.npz", P + f"main_{b}.npz"); shutil.copy(P + f"labels_{a}.csv", P + f"labels_{b}.csv")
# biliary
W = "../biliary_improve/work/"
lab = pd.read_csv(W + "labels_trval.csv")[["study_id", "split", "y_bdd"]].rename(columns={"y_bdd": "y"})
lab = lab[lab.split.isin(["train", "val"])]
fo = pd.read_csv(W + "oof_train.csv")[["study_id", "fold"]]
lab = lab.merge(fo, on="study_id", how="left"); lab = lab[lab.y.notna()]
assert lab[lab.split == "train"].fold.notna().all(); lab.to_csv(P + "labels_bd.csv", index=False); main("bd", ["pancreas", "gallbladder", "liver"])
# bowel obstruction
A = "../analysis_bowel_obstruction/"
lab = pd.read_csv(A + "labels_trval.csv")[["study_id", "split", "y_sbo"]].rename(columns={"y_sbo": "y"})
lab = lab[lab.split.isin(["train", "val"])].merge(pd.read_csv(A + "oof_train.csv")[["study_id", "fold"]], on="study_id", how="left")
lab = lab[lab.y.notna()]; assert lab[lab.split == "train"].fold.notna().all()
lab.to_csv(P + "labels_go.csv", index=False); main("go", ["stomach", "small_bowel"])
# fracture
from final_stack import split_trval, load_labels_trval
spl = split_trval(); L = load_labels_trval(set(spl.index)); L["split"] = spl.reindex(L.index).values
L = L[L.y.notna()].copy(); L.index.name = "study_id"; L = L.reset_index()
pel = L.sites.apply(lambda s: any(x in s for x in ("pelvis", "sacrum", "hip_femur")))
rng = np.random.default_rng(0); L["fold"] = np.nan
for yv in (0, 1):
    m = np.where((L.split == "train") & (L.y == yv))[0]; m = rng.permutation(m); L.loc[m, "fold"] = np.arange(len(m)) % 5
L[["study_id", "split", "y", "fold"]].to_csv(P + "labels_frp.csv", index=False)
Lp = L.copy(); Lp["y"] = np.where(L.y == 0, 0.0, np.where(pel, 1.0, np.nan))
Lp[["study_id", "split", "y", "fold"]].to_csv(P + "labels_frpp.csv", index=False)
L.assign(pelvis=pel.astype(int))[["study_id", "split", "y", "fold", "pelvis"]].to_csv(P + "frac_sites_trval.csv", index=False)
main("frp", ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right"]); shutil.copy(P + "main_frp.npz", P + "main_frpp.npz")
print("train/val pos:", {c: pd.read_csv(P + f"labels_{c}.csv").groupby("split").y.agg(["count", "sum"]).values.tolist() for c in ["bd", "fa2", "asc2", "frp", "frpp", "go"]})
