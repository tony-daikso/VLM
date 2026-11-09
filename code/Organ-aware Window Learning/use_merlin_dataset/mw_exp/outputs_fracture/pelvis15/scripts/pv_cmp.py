import sys, glob, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
r = pd.read_csv("ref_train.csv", index_col=0)
for run in sys.argv[1:]:
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"runs/{run}/score_fold*.csv"))]); s = s[s.split == "train"].set_index("study_id")
    d = r.join(s[[c for c in ("ft", "ft_side", "ft_mix") if c in s]], how="inner"); d = d[d.t_pel.notna()]
    out = {}
    for c in [c for c in ("ft", "ft_side", "ft_mix") if c in d]:
        X = d[["U_org_pel", "U_pel", c]].values; X = (X - X.mean(0)) / X.std(0)
        p0 = cross_val_predict(LogisticRegression(), X[:, :2], d.t_pel, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
        p1 = cross_val_predict(LogisticRegression(), X, d.t_pel, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
        out[c] = (round(auc(d.t_pel, d[c]), 4), "ref2", round(auc(d.t_pel, p0), 4), "+probe", round(auc(d.t_pel, p1), 4))
    print(run, len(d), out)
