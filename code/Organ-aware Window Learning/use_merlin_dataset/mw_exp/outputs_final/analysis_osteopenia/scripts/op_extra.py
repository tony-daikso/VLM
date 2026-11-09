"""Does the frozen lumbar_spine embedding linearly encode vertebral HU? Ridge (5-fold, train only) from each window's
standardized embedding -> L1 body HU (fracture_features L1_hu). Reports R2, MAE, AUROC of decoded HU for osteopenia."""
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
import threadpoolctl; threadpoolctl.threadpool_limits(4)
import os
A = os.environ.get("OP_DIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_osteopenia/")
z = np.load(A + "emb_lumbar_L1_trval.npz")
ids = [str(i) for i in z["ids"]]; units = [str(u) for u in z["units"]]
lab = pd.read_csv(A + "labels_trval.csv"); lab = lab[(lab.split == "train") & lab.y_op.notna()].set_index("study_id")
fr = pd.read_csv(A + "bundle_trval.csv").set_index("study_id")["L1_hu"].dropna()
fr = fr[fr.index.isin(set(lab.index))]
keep = [i for i, s in enumerate(ids) if s in fr.index]
rows = []
for un in ["lumbar_spine", "L1"]:
    g = units.index(un)
    E = z["E"][keep, g].astype(np.float32); P = z["P"][keep, g]
    sid = [ids[i] for i in keep]; ok = P > 0
    E, sid = E[ok], np.array(sid)[ok]
    yhu = fr.loc[sid].values; yop = lab.loc[sid, "y_op"].values
    for w, wn in enumerate(["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]) :
        X = E[:, w]; X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        pr = np.zeros(len(X))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X):
            m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(X[a], yhu[a]); pr[b] = m.predict(X[b])
        rows.append(dict(unit=un, window=wn, n=len(X), R2_L1HU=r2_score(yhu, pr), MAE_HU=np.abs(yhu - pr).mean(),
                         auc_decodedHU=roc_auc_score(yop, -pr), auc_trueHU=roc_auc_score(yop, -yhu)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(A + "embedding_decodes_L1HU.csv", index=False)
