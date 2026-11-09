"""rib_straight tier comparison (train only; val never read). For each tier run (runs/<name>/oof_train.csv, in_pool rows):
FT tier OOF (t_rib) vs (i) rib_v3 lev28 OOF on the same rows (trained on ALL train), (ii) the tier-trained rib_v3 probe
(runs/probe_tier_sl, ft_side), (iii) LR stack CV on tier rows (sitewise folds): [RV3] vs [RV3 + FT] (and + probe), paired
bootstrap of the stack gain. usage: rs_cmp.py RUN [RUN ...] -> runs/tier_cmp.json + printed table"""
import sys, os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
B = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture"); RS = B + "/rib_straight"
lab = pd.read_csv(RS + "/in/labels.csv"); lab = lab[lab.split == "train"].set_index("study_id")
ids = [l.strip() for l in open(RS + "/in/ids_tier.txt") if l.strip()]
T = lab.loc[ids, ["t_rib", "fold"]].copy()
T["rv3"] = pd.read_csv(B + "/rib_v3/cand/lev28/oof_train.csv").set_index("study_id")["ft"].reindex(T.index)
pf = sorted(glob.glob(RS + "/runs/probe_tier_sl/score_fold*.csv"))
if len(pf) == 5:
    P = pd.concat([pd.read_csv(f) for f in pf]); P = P[P.split == "train"].set_index("study_id")
    T["probe"] = P["ft_side"].reindex(T.index)


def stack_cv(X, y, fold):
    p = np.zeros(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr]); p[te] = m.decision_function((X[te] - mu) / sd)
    return p


def boot(y, a, b, n=2000, seed=0):
    r = np.random.default_rng(seed); d = []; ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        s = np.r_[r.choice(ip, len(ip)), r.choice(ineg, len(ineg))]; d.append(auc(y[s], a[s]) - auc(y[s], b[s]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


out = {}
for run in sys.argv[1:]:
    f = RS + f"/runs/{run}/oof_train.csv"
    if not os.path.exists(f): print(run, "missing"); continue
    d = pd.read_csv(f).set_index("study_id")["ft"]; X = T.join(d.rename("ft")).dropna(subset=["ft", "rv3", "t_rib"])
    y = X.t_rib.values.astype(int); fo = X.fold.values
    r = {"n": len(X), "pos": int(y.sum()), "ft": auc(y, X.ft), "rv3_fulltrain": auc(y, X.rv3)}
    if "probe" in X and X.probe.notna().all(): r["probe_tier"] = auc(y, X.probe)
    s0 = stack_cv(X[["rv3"]].values, y, fo); s1 = stack_cv(X[["rv3", "ft"]].values, y, fo)
    r["stack_rv3"] = auc(y, s0); r["stack_rv3_ft"] = auc(y, s1); r["stack_gain_ci"] = boot(y, s1, s0)
    r["ft_minus_probe_ci"] = boot(y, X.ft.values, X.probe.values) if "probe_tier" in r else None
    r["spearman_ft_rv3"] = float(X[["ft", "rv3"]].corr("spearman").iloc[0, 1])
    s = json.load(open(RS + f"/runs/{run}/summary.json")) if os.path.exists(RS + f"/runs/{run}/summary.json") else {}
    r["folds_done"] = s.get("folds_done"); out[run] = r
    print(run, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(RS + "/runs/tier_cmp.json", "w"), indent=1)
