"""Univariate AUROC of low-density features (train/val only). Comparisons per split:
 overall pos vs neg | FN-E vs neg | FN-E vs neg (GB present >=100 vox) | FN-E vs TP | FN-F vs neg ; plus partial (residual) AUROC:
 within positives+negatives, does the feature separate after N_U?  -> AUROC among cases with N_U below the train-OOF 90%-spec thr."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
MW = "/Users/tony.tu/Desktop/戴承智慧/VLM/code/Organ-aware Window Learning/use_merlin_dataset/mw_exp/outputs_final/"
F, OUT = sys.argv[1], sys.argv[2]
D = pd.read_csv(MW + "analysis_gallstones_v2/per_case_trval.csv"); assert set(D.split) <= {"train", "val"}
X = pd.read_csv(F).drop(columns=["split"], errors="ignore"); cols = [c for c in X.columns if c.startswith(("ld_", "ld15_")) and X[c].dtype != object]
D = D.merge(X, on="study_id", how="inner")
D["grp"] = np.where(D.err == "TP", "TP", np.where(D.err == "FN", "FN_" + D.cat.fillna("?").str[0], "neg"))
thr = np.quantile(D.query("split=='train' and y_gs==0").N_U, 0.9)
def A(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5: return np.nan
    return auc(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
rows = []
for sp in ("val", "train"):
    d = D[D.split == sp]; neg = d[d.y_gs == 0]; negp = neg[neg.gb_vox >= 100]; pos = d[d.y_gs == 1]
    E = d[d.grp == "FN_E"]; Fh = d[d.grp == "FN_F"]; TP = d[d.grp == "TP"]
    low = d[d.N_U <= thr]  # below operating threshold: FN vs TN
    for c in cols:
        rows.append(dict(split=sp, feat=c, n_pos=int(pos[c].notna().sum()), n_E=int(E[c].notna().sum()),
                         auc_pos_vs_neg=A(pos[c], neg[c]), auc_E_vs_neg=A(E[c], neg[c]), auc_E_vs_negGB=A(E[c], negp[c]),
                         auc_E_vs_TP=A(E[c], TP[c]), auc_F_vs_neg=A(Fh[c], neg[c]),
                         auc_FN_vs_TN_belowthr=A(low[low.y_gs == 1][c], low[low.y_gs == 0][c]),
                         med_E=float(E[c].median()), med_negGB=float(negp[c].median()), med_TP=float(TP[c].median())))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
v = R[R.split == "val"].set_index("feat"); t = R[R.split == "train"].set_index("feat")
S = v[["n_pos", "n_E", "auc_pos_vs_neg", "auc_E_vs_neg", "auc_E_vs_negGB", "auc_E_vs_TP", "auc_F_vs_neg", "auc_FN_vs_TN_belowthr", "med_E", "med_negGB", "med_TP"]].join(
    t[["auc_E_vs_negGB", "auc_FN_vs_TN_belowthr"]].add_prefix("tr_"))
print(S.round(3).to_string())
