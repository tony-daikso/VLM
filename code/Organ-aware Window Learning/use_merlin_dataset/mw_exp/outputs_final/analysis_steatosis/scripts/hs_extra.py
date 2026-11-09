import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score as A
from sklearn.ensemble import HistGradientBoostingClassifier
S = sys.argv[1]
T = pd.read_csv(S + "/out/per_case_train_oof.csv")
for f in ["oof_train_nofocal.csv", "oof_train_concat_diff.csv", "oof_train_concat.csv"]:
    T = T.merge(pd.read_csv(S + "/" + f).drop(columns="fold"), on="study_id")
y = T.y_hs.values.astype(int)
lab = pd.read_csv(S + "/data/labels_trval.csv").set_index("study_id")
cat = lab.loc[T.study_id, "hs_cat"].values
diff = (y == 1) & (T.subtype == "瀰漫/steatosis").values
foc = (y == 1) & (T.subtype == "局部脂肪浸潤(無瀰漫字眼)").values
out = []
for c in ["E3_late", "E3_late_nofocal", "E3_late_concat_diff", "E3_late+HU_GBM"]:
    out.append({"score": c, "all_labels": A(y, T[c]), "diffuse_pos_vs_all_neg": A(y[diff | (y == 0)], T[c][diff | (y == 0)]),
                "focal_pos_vs_all_neg": A(y[foc | (y == 0)], T[c][foc | (y == 0)]),
                "diffuse_pos_vs_no_neg": A(y[diff | (cat == "no")], T[c][diff | (cat == "no")])})
# stack concat_diff logit + HU GBM (same folds); reuse analysis design via import
sys.argv = ["x", S]
