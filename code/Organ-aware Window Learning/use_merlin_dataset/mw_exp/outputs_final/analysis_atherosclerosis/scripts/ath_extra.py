import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
D = os.path.dirname(os.path.abspath(__file__)) + "/"
T = pd.read_csv(D + "per_case_train_oof.csv"); y = T.y_ath.values.astype(int)
S = T.E3_late.values; ST = T["p_E3+hu_gbm"].values
q50 = np.percentile(T.calc[y == 1], 50)
txt = (y == 0) & ((T.kw_vasc_calc == 1) | (T.kw_athero == 1) | (T.kw_calc_plaque == 1)).values & (T.ath_cat != "no").values
hi = (y == 0) & (T.calc.values >= q50); bad = txt | hi
print("suspect-neg frac", bad[y == 0].mean().round(4))
for nm, s in (("E0", T.E0.values), ("E3", S), ("E3+HU", ST), ("hu", T.p_hu_gbm.values)):
    print(nm, "all %.4f clean %.4f" % (A(y, s), A(y[~bad], s[~bad])))
# per-positive rank decomposition: AUROC = mean_i P(s_i > neg)
neg = np.sort(S[y == 0])
r = np.searchsorted(neg, S[y == 1]) / len(neg)
P = T[y == 1].assign(r=r)
print("overall mean r", r.mean().round(4))
for k, m in (("grade mild", P.grade == "mild"), ("text mild", P.g_mild == 1), ("age<60", P.Age < 60), ("aortoiliac kw", P.kw_aortoiliac == 1)):
    print(k, "frac %.3f r %.3f rest %.3f cost %.4f" % (m.mean(), P.r[m].mean(), P.r[~m].mean(), m.mean() * (P.r[~m].mean() - P.r[m].mean())))
# cost of suspect negatives: per-negative
pos = np.sort(S[y == 1]); rn = 1 - np.searchsorted(pos, S[y == 0]) / len(pos)  # frac of pos above each neg
bn = bad[y == 0]
print("neg suspect: frac %.3f mean(pos above) %.3f rest %.3f cost %.4f" % (bn.mean(), rn[bn].mean(), rn[~bn].mean(), bn.mean() * (rn[~bn].mean() - rn[bn].mean())))
for lo, hi_ in ((60, 200), (70, 200)):
    m = (T.Age >= lo).values
    print(f"age>={lo}: E3 %.4f E3+HU %.4f hu %.4f age %.4f" % (A(y[m], S[m]), A(y[m], ST[m]), A(y[m], T.p_hu_gbm.values[m]), A(y[m], T.Age.values[m])))
print("aorta vox spearman age", T[["a_vox", "Age"]].corr("spearman").iloc[0, 1].round(3))
print("FN grade dist", T[T.fn_cat != ""].grade.value_counts(normalize=True).round(3).to_dict() if "fn_cat" in T else "")
