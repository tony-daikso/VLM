"""Convert Exp O pelvis-unit head scores (train OOF + val full-train refit) and the Exp L final stack scores to the
final_stack npz member format (ids, split, logit). Train/val only (asserted)."""
import numpy as np, pandas as pd
RU = "outputs_final/region_units/fracture_pelvis_r2/"; OUT = "outputs_fracture/final_combo/members/"
for sub, col, name in [("out_pel", "U_org", "U_org_pel"), ("out_pel", "U", "U_pel"), ("out", "U_org", "U_org_out")]:
    o = pd.read_csv(RU + sub + "/oof_train.csv"); v = pd.read_csv(RU + sub + "/val_scores.csv")
    ids = np.r_[o.study_id.values, v.study_id.values].astype(str); sp = np.r_[["train"] * len(o), ["val"] * len(v)]
    np.savez(OUT + name + ".npz", ids=ids, split=sp, logit=np.r_[o[col].values, v[col].values].astype(float))
    print(name, len(o), len(v))
z = np.load("outputs_fracture/expL/final/SL_L_repl-rl03-lrc/scores.npz", allow_pickle=True)
assert set(np.unique(z["split"])) <= {"train", "val"}
np.savez(OUT + "expL_final_stack.npz", ids=z["ids"], split=z["split"], logit=z["stack"].astype(float)); print("expL stack", z["stack_kind"], len(z["ids"]))
