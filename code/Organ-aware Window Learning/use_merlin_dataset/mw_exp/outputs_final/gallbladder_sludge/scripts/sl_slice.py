"""Slice inputs for the 膽泥 / gallstone-suppression work to train/val study_ids only (test never read) for CPU work on the Mac.
Tables: same list as gallstones_final/scripts/gf_slice.py + sludge labels + subgroup flags + other-finding OOF.
Embeddings: main notest matrix -> gallbladder+liver units of train/val ids; GS5 GB-crop emb and Exp N unit-U emb packed to npz."""
import pandas as pd, numpy as np, os, sys, glob
I = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final"; O = sys.argv[1]
spl = pd.read_csv("/home/advantech/Desktop/Tony/VLM/OWL/data/split.csv")
ids = set(spl[spl.split.isin(["train", "val"])].study_id)
F = ["analysis_gallstones/labels_trval.csv", "analysis_gallstones/hu_a6.csv", "analysis_gallstones/hu_ct.csv", "analysis_gallstones/oof_train.csv",
     "analysis_gallstones/scores_val_descriptive.csv", "analysis_absent_gb/labels_trval.csv", "analysis_absent_gb/feat_a6.csv", "analysis_absent_gb/feat_ct.csv",
     "analysis_absent_gb/oof_train.csv", "analysis_absent_gb/scores_val_descriptive.csv", "gallstones_improve/gs1_feat3mm.csv",
     "gallstones_improve/gs3b/gs3b_oof_train.csv", "gallstones_improve/gs3b/gs3b_val.csv", "gallstones_improve/gs5/gbx_oof_train.csv",
     "gallstones_improve/gs5/gbx_val.csv", "gallstones_improve/gs4/oof13_train.csv", "gallstones_improve/gs4/oof13_val.csv",
     "gallstones_improve/gs5/stack/oof_train.csv",
     "region_units/gallstones/stack/gbn_oof_train.csv", "region_units/gallstones/stack/gbn_val.csv",
     "region_units/gallstones/stack/oof_train.csv", "region_units/gallstones/stack/val_scores.csv",
     "analysis_gallstones/subgroups/case_flags_trval.csv", "gallbladder_sludge/labels_trval_sludge.csv"]
for f in F:
    d = pd.read_csv(f"{I}/{f}"); n = len(d); d = d[d.study_id.isin(ids)]
    if "split" in d: d = d[d.split.isin(["train", "val"])]
    os.makedirs(os.path.dirname(f"{O}/{f}"), exist_ok=True); d.to_csv(f"{O}/{f}", index=False); print(f, n, "->", len(d), flush=True)
if os.path.exists(f"{O}/emb_gb_liver_trval.npz"): z = None
else: z = np.load(f"{I}/results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
if z is not None:
  mids = list(z["ids"]); units = list(z["units"]); keep = np.array([s in ids for s in mids]); ui = [units.index("gallbladder"), units.index("liver")]
  E = z["E"][keep][:, ui]; P = z["P"][keep][:, ui]
  np.savez(f"{O}/emb_gb_liver_trval.npz", ids=np.array(mids)[keep], E=E.astype(np.float16), P=P.astype(np.float32), units=np.array(["gallbladder", "liver"]))
  print("main", E.shape, flush=True); del z, E
for tag, d in (("gs5", f"{I}/gallstones_improve/gs5/emb"), ("unitU", f"{I}/region_units/gallstones/emb")):
    fs = sorted(f for f in glob.glob(f"{d}/*.npz") if os.path.basename(f)[:-4] in ids)
    q0 = np.load(fs[0]); EM = np.zeros((len(fs),) + q0["emb"].shape, np.float16)
    PR = np.zeros((len(fs),) + np.asarray(q0["present"]).shape, bool); RV = np.zeros((len(fs),) + q0["region_vox"].shape, np.float32)
    for i, f in enumerate(fs):
        q = np.load(f); PR[i] = q["present"]
        if "emb" in q.files: EM[i] = q["emb"]; RV[i] = q["region_vox"]
        else: PR[i] = False
    np.savez(f"{O}/emb_{tag}_trval.npz", ids=np.array([os.path.basename(f)[:-4] for f in fs]), emb=EM, present=PR, region_vox=RV)
    print(tag, EM.shape, flush=True)
