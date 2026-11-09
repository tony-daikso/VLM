"""Re-create the per-case val scores of the appendicitis_rlq_unit stack "REF＋U_rlq_cs" (y_ap and y_ap_def)
exactly as rlq_eval.py does (deterministic LR C=0.1 on all train OOF rows -> val). train/val only; test never read.
Writes integrated_best/scores_v2/appendicitis_val.csv and split_v2/ap_def_val.csv (study_id, y, stack)."""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; O = R + "appendicitis_rlq_unit/out/"; IB = R + "integrated_best/"
lab = pd.read_csv(R + "appendicitis_split/labels_trval_split.csv"); assert set(lab.split) <= {"train", "val"}
oof = pd.read_csv(O + "oof_train.csv"); val = pd.read_csv(O + "val_scores.csv")
hu = pd.concat([pd.read_csv(R + "analysis_appendicitis/hu_a6.csv"), pd.read_csv(R + "analysis_appendicitis/hu_ct.csv")]).drop_duplicates("study_id")
LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
        [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
hu = hu[["study_id"] + LOCAL]
T = oof.merge(lab, on="study_id", how="left").merge(hu, on="study_id", how="left")
V = val.merge(lab, on="study_id", how="left").merge(hu, on="study_id", how="left")
assert (T.split == "train").all() and (V.split == "val").all()
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
os.makedirs(IB + "scores_v2", exist_ok=True); os.makedirs(IB + "split_v2", exist_ok=True)
for tg, out, ref in (("y_ap", IB + "scores_v2/appendicitis_val.csv", .920365), ("y_ap_def", IB + "split_v2/ap_def_val.csv", .936213)):
    Tm = T[T[tg].notna() & T[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    Vm = V[V[tg].notna() & V[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    cols = [f"{tg}__E3_late", f"{tg}__E3cs", f"{tg}__U_rlq_cs"] + LOCAL
    X = Tm[cols]; med = X.median(); X = X.fillna(med); XV = Vm[cols].fillna(med)
    s = lr().fit(X, Tm[tg].values.astype(int)).decision_function(XV)
    a = roc_auc_score(Vm[tg].astype(int), s); print(tg, len(Vm), int(Vm[tg].sum()), round(a, 6), "expected", ref)
    assert abs(a - ref) < 5e-4
    pd.DataFrame(dict(study_id=Vm.study_id, y=Vm[tg].astype(int), stack=s)).to_csv(out, index=False)
