"""Paired bootstrap CI for MIL-max gain (train OOF and val descriptive). Usage: hn_boot.py <dir>"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = sys.argv[1].rstrip("/") + "/"
lab = pd.read_csv(D + "labels_trval.csv")[["study_id", "split", "y_hn", "hn_side"]]
rows = []
for nm, f in [("train_oof", "oof_train.csv"), ("val_descriptive", "scores_val_descriptive.csv")]:
    d = lab.merge(pd.read_csv(D + f), on="study_id"); d = d[d.y_hn.notna()].reset_index(drop=True)
    y = d.y_hn.values.astype(int)
    for a, b in [("E0", "E0_max"), ("E3_late", "E3_late_max")]:
        for sub, m in [("all", np.ones(len(d), bool)), ("unilateral pos + all neg", (y == 0) | d.hn_side.isin(["left", "right"]).values),
                       ("bilateral pos + all neg", (y == 0) | (d.hn_side == "bilateral").values)]:
            yy, sa, sb = y[m], d[a].values[m], d[b].values[m]
            base = roc_auc_score(yy, sb) - roc_auc_score(yy, sa)
            rng = np.random.default_rng(0); P, N = np.where(yy == 1)[0], np.where(yy == 0)[0]; g = []
            for _ in range(2000):
                ix = np.concatenate([rng.choice(P, len(P)), rng.choice(N, len(N))])
                g.append(roc_auc_score(yy[ix], sb[ix]) - roc_auc_score(yy[ix], sa[ix]))
            g = np.array(g)
            rows.append(dict(set=nm, compare=f"{b} - {a}", subset=sub, n_pos=len(P), auc_a=round(roc_auc_score(yy, sa), 4), auc_b=round(roc_auc_score(yy, sb), 4),
                             gain=round(base, 4), ci_lo=round(np.quantile(g, .025), 4), ci_hi=round(np.quantile(g, .975), 4), p_le0=round((g <= 0).mean(), 4)))
r = pd.DataFrame(rows); r.to_csv(D + "bootstrap_mil_max_gain.csv", index=False); print(r.to_string())
