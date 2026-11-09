"""push90 B tier check (train only): does the PE tier score add to the current stack OOF on tier rows?
inputs: PE run dirs (score_fold*.csv, train rows), rsP_mix_repl scores.npz (train = nested-CV OOF of the selected stack),
spine15_ft oof. Metrics on tier train rows: AUROC for t_pe and for official y; LR stack CV (sitewise folds) [stack] vs
[stack + pe] for y, paired bootstrap of the gain. usage: pe_tier_eval.py LABELS RUN_DIR [RUN_DIR ...]"""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from pathlib import Path
FR = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture"
lab = pd.read_csv(sys.argv[1]); lab = lab[lab.split == "train"].set_index("study_id")
z = np.load(FR / "final_ft/rsP_mix_repl/scores.npz", allow_pickle=True); t = z["split"] == "train"
stk = pd.Series(z["selected"][t], index=z["ids"][t].astype(str))
sp = pd.read_csv(FR / "spine15_ft/oof_train.csv").set_index("study_id")["ft"]


def cv(X, y, f):
    p = np.zeros(len(y))
    for k in np.unique(f):
        a, b = f != k, f == k; mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
        p[b] = LogisticRegression(max_iter=2000).fit((X[a] - mu) / sd, y[a]).decision_function((X[b] - mu) / sd)
    return p


for run in sys.argv[2:]:
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(run + "/score_fold*.csv"))]); P = P[P.split == "train"].set_index("study_id")
    for col in ("ft_head", "ft_side"):
        D = lab.join(P[col].rename("pe"), how="inner").join(stk.rename("stk")).join(sp.rename("sp")).dropna(subset=["pe", "stk", "sp"])
        m = D.t_pe.notna(); y = D.y.values.astype(int); f = D.fold.values
        s0 = cv(D[["stk"]].values, y, f); s1 = cv(D[["stk", "pe"]].values, y, f)
        r = np.random.default_rng(0); g = []
        for _ in range(1000):
            i = r.integers(0, len(y), len(y)); g.append(auc(y[i], s1[i]) - auc(y[i], s0[i]))
        print(f"{run} {col} n={len(D)} t_pe: pe {auc(D.t_pe[m], D.pe[m]):.3f} stack {auc(D.t_pe[m], D.stk[m]):.3f} sp15 {auc(D.t_pe[m], D.sp[m]):.3f} | "
              f"y: pe {auc(y, D.pe):.3f} stack {auc(y, s0):.4f} stack+pe {auc(y, s1):.4f} gain {np.mean(g):+.4f} [{np.percentile(g, 2.5):+.4f}, {np.percentile(g, 97.5):+.4f}]", flush=True)
