"""Extra checks: is the HU stacking gain an age/body-size shortcut? age-matched AUROC for stacks; RLQ-local-only HU set."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score as A
D = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_appendicitis/"
lab = pd.read_csv(D + "labels_trval.csv"); hu = pd.concat([pd.read_csv(D + "hu_a6.csv"), pd.read_csv(D + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender"]]
T = lab[lab.split == "train"].merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left").merge(pd.read_csv(D + "oof_train.csv"), on="study_id")
T = T[T.y_ap.notna()].reset_index(drop=True); y = T.y_ap.values.astype(int); pos, neg = y == 1, y == 0
def oof_lr(X, C=0.1):
    X = X.fillna(X.median()); p = np.zeros(len(T))
    for k in range(5):
        tr, te = T.fold != k, T.fold == k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(X[tr], y[tr]); p[te] = m.decision_function(X[te])
    return p
def matched(s, reps=20, k=5):
    v = []; Ag = T.Age.values; G = T.Gender.values; ni = np.where(neg & ~np.isnan(Ag))[0]
    for r in range(reps):
        rr = np.random.default_rng(r); sel = []
        for i in np.where(pos & ~np.isnan(Ag))[0]:
            c = ni[(np.abs(Ag[ni] - Ag[i]) <= 2) & (G[ni] == G[i])]
            if len(c): sel += list(rr.choice(c, min(k, len(c)), replace=False)) + [i]
        sel = np.array(sel); v.append(A(y[sel], s[sel]))
    return np.mean(v)
LOCAL = [c for c in T.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
        [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
SIZE = ["colon_vox", "sb_vox", "rlq_r15_n", "rlq_r30_n", "rlq_seed_vox", "glob_fat_med", "glob_fluid_frac", "rlq_z_rel_spine_mm"]
L = T[LOCAL]; Sz = np.log1p(T[SIZE].clip(lower=0)).assign(glob_fat_med=T.glob_fat_med, rlq_z_rel_spine_mm=T.rlq_z_rel_spine_mm)
age = pd.DataFrame({"age": T.Age.fillna(T.Age.median())})
rows = []
sc = {"E0": T.E0.values, "E3_late": T.E3_late.values, "E3_late_colon_sb": T.E3_late_colon_sb.values,
      "HU_local_only": oof_lr(L), "HU_size_only": oof_lr(Sz), "age_only": -T.Age.fillna(55).values,
      "E3+HU_local": oof_lr(pd.concat([L, T[["E3_late"]]], axis=1)),
      "E3+HU_size": oof_lr(pd.concat([Sz, T[["E3_late"]]], axis=1)),
      "E3+HU_local+size": oof_lr(pd.concat([L, Sz, T[["E3_late"]]], axis=1)),
      "E3cs+HU_local": oof_lr(pd.concat([L, T[["E3_late_colon_sb"]]], axis=1)),
      "E3+E3cs+HU_local": oof_lr(pd.concat([L, T[["E3_late", "E3_late_colon_sb"]]], axis=1)),
      "E3+E3cs+HU_local+size": oof_lr(pd.concat([L, Sz, T[["E3_late", "E3_late_colon_sb"]]], axis=1))}
for k, s in sc.items():
    rows.append(dict(model=k, auroc=round(A(y, s), 4), age_matched=round(matched(s), 4),
                     spearman_age_neg=round(pd.Series(s[neg]).corr(T.Age[neg].reset_index(drop=True), method="spearman"), 3)))
r = pd.DataFrame(rows); print(r.to_string()); r.to_csv(D + "extra_age_matched.csv", index=False)
print("age median: neg explicit-normal-appendix", T.Age[neg & (T.kw_negphrase == 1)].median(), "neg not_mentioned", T.Age[neg & (T.ap_cat == "not_mentioned")].median(), "pos", T.Age[pos].median())
nn = neg & (T.kw_negphrase == 1); nm = neg & (T.ap_cat == "not_mentioned")
print("matched AUROC E3 pos vs explicit-normal neg only: ", round(A(y[pos | nn], T.E3_late[pos | nn]), 4))
for nm_, s in [("E3_late", T.E3_late.values), ("E3+HU_local", sc["E3+HU_local"])]:
    a = matched(s); print(nm_, "matched", round(a, 4))
# within explicit-normal negatives restricted & age-matched
