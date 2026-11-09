"""Extra checks: negatives with large measured diameter, metal shortcut, repaired subgroup geometry, val subgroup counts."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
T = pd.read_csv(D + "per_case_train_oof.csv"); y = T.y_aaa.values.astype(int)
lab = pd.read_csv(D + "labels_trval.csv").set_index("study_id")
geo = pd.concat([pd.read_csv(D + "geom_a6.csv"), pd.read_csv(D + "geom_ct.csv")]).drop_duplicates("study_id").set_index("study_id")
T = T.join(geo[["abd_eq_rmax_x" if False else "abd_feret_rmax", "infra_nsl", "z_ren", "a_zmin", "infra_multicomp_frac", "inV_n800", "v_n1200"]], on="study_id")
T["kw_normal_caliber"] = lab.loc[T.study_id, "kw_normal_caliber"].values
T["txt_max_cm"] = lab.loc[T.study_id, "txt_max_cm"].values
pos, ng = y == 1, y == 0
big = ng & (T.infra_eq_rmax >= 30)
print("neg >=30mm:", big.sum(), T[big].aaa_cat.value_counts().to_dict(), "kw_ectasia", int(T[big].kw_ectasia.sum()), "kw_normal_caliber", int(T[big].kw_normal_caliber.sum()),
      "q_diam stated", int(T[big].q_diam_cm.notna().sum()), "median age", T[big].Age.median(), "multicomp>0", int((T[big].infra_multicomp_frac > 0).sum()),
      "infra_nsl median", T[big].infra_nsl.median(), "vs all neg", T[ng].infra_nsl.median())
print(T[big][["aaa_cat", "q_diam_cm", "kw_ectasia", "kw_normal_caliber", "infra_eq_rmax", "infra_feret_rmax", "abd_eq_rmax", "infra_multicomp_frac", "E3_late", "Age"]].round(2).sort_values("infra_eq_rmax").to_string(index=False))
for nm, m in (("repaired pos", pos & (T.repaired == 1)), ("unrepaired pos", pos & (T.repaired == 0)), ("neg", ng)):
    print(nm, "n", m.sum(), "metal v_n1600>0 %.3f" % (T.v_n1600[m] > 0).mean(), "v_n2000>0 %.3f" % (T.v_n2000[m] > 0).mean(),
          "median infra_eq %.1f" % T.infra_eq_rmax[m].median(), "measured<30 %.3f" % (T.infra_eq_rmax[m] < 30).mean(), "q_diam stated %.3f" % T.q_diam_cm[m].notna().mean())
s = T.E3_late.values
mn = ng & (T.v_n1600 > 0)
print("E3 among negatives: metal vs no metal median score", np.median(s[mn]), np.median(s[ng & ~mn]), "AUROC metal->score in negs %.3f" % A(mn[ng], s[ng]))
print("AUROC v_n1600 for y (all) %.3f; repaired pos vs neg %.3f" % (A(y, T.v_n1600.fillna(0)), A(y[(pos & (T.repaired == 1)) | ng], T.v_n1600.fillna(0)[(pos & (T.repaired == 1)) | ng])))
# size not stated positives: what are they
ns = pos & T.q_diam_cm.isna()
print("size-not-stated pos:", ns.sum(), "repaired", int(T.repaired[ns].sum()), "median measured", T.infra_eq_rmax[ns].median(), "kw_iliac", int(T.kw_iliac_aneur[ns].sum()), "kw_thoracic", int(T.kw_thoracic[ns].sum()))
# FN@95
f95 = T.fn95_cat.fillna("") != ""
print(T[f95][["q_diam_cm", "q_repaired", "kw_evar", "kw_iliac_aneur", "kw_thoracic", "infra_eq_rmax", "infra_feret_rmax", "v_n1600", "Age", "rank_E3", "fn95_cat"]].round(2).to_string(index=False))
# measured <30 positives composition
m30 = pos & (T.infra_eq_rmax < 30)
print("pos measured<30:", m30.sum(), "repaired", int(T.repaired[m30].sum()), "size not stated", int(T.q_diam_cm[m30].isna().sum()), "reported 3.0-3.4", int(((T.q_diam_cm >= 3) & (T.q_diam_cm < 3.5))[m30].sum()), "feret>=30", int((T.infra_feret_rmax[m30] >= 30).sum()))
V = pd.read_csv(D + "scores_val_descriptive_stack.csv").join(lab[["q_repaired", "kw_evar", "q_diam_cm"]], on="study_id")
vp = V[V.y_aaa == 1]
print("val pos", len(vp), "repaired", int(((vp.q_repaired == "yes") | (vp.kw_evar == 1)).sum()), "size stated", int(vp.q_diam_cm.notna().sum()), "measured>=30", int((vp.infra_eq_rmax >= 30).sum()))
