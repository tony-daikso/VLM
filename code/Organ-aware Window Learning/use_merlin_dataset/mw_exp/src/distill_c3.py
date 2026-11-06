"""Distillation C3 (gallstones): teacher = GS1-V4 stack (E3_late + gallbladder HU + absent-GB + cholecystectomy head +
3 mm texture; train-OOF logits from gallstones_improve/gs1/gs1_oof_train.csv), student = embedding-only E3_late
gallbladder-unit probe (analysis_gallstones/scripts/gs_oof.py recipe: probe.py Probe 'late', joint gallstones +
surgically-absent-GB heads, lr 1e-3 / wd 1e-4, inner 15 % early stop on mean hard AUROC, StratifiedKFold(5, rs=0)).
Student gallstone target on train = alpha * y + (1 - alpha) * q_V4 (Platt on train OOF, temperature T).
alpha = 1 reproduces the hard-label E3_late. Train OOF = selection; val = full-train fit (same as gs_oof.py), seeds
42/43 averaged; val never used to fit/stop/select; test never read (all inputs are train/val only).
usage: python distill_c3.py --dir C3DIR --alpha 0.5 --T 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from probe import Probe, auroc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(a.threads)
D = Path(a.dir)
out = D / f"c3_a{a.alpha:g}_T{a.T:g}.csv"
z = np.load(D / "gb_unit.npz", allow_pickle=True)
ids = [str(i) for i in z["ids"]]
E = torch.from_numpy(z["E"].astype(np.float32)); P = torch.from_numpy(z["P"].astype(np.float32))
lab = pd.read_csv(D / "labels_trval.csv").set_index("study_id")
assert set(lab.split) <= {"train", "val"}
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_gs", "y_chole"]].values.astype(np.float32)
tr = S == "train"
pres = P[torch.from_numpy(tr), 0] > 0
rows = E[torch.from_numpy(tr)][pres][:, 0]
mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
W = E.shape[2]
# teacher (train OOF only)
t = pd.read_csv(D / "gs1_oof_train.csv").set_index("study_id")
zt = pd.Series(t["V4"]).reindex(ids).values.astype(float)
okt = tr & ~np.isnan(zt) & ~np.isnan(Y[:, 0])
lr = LogisticRegression(C=1e6, max_iter=1000).fit(zt[okt, None], Y[okt, 0].astype(int))
q = 1 / (1 + np.exp(-(lr.coef_[0, 0] * zt + lr.intercept_[0]) / a.T))
TGT = Y.copy()
soft = okt
TGT[soft, 0] = a.alpha * Y[soft, 0] + (1 - a.alpha) * q[soft]
print(f"alpha={a.alpha} T={a.T}: teacher rows {int(okt.sum())}/{int(tr.sum())}", flush=True)


def fit(idx_tr, idx_es, idx_pred, seed, lr_=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, W, E.shape[3], [[0], [0]], None)
    opt = torch.optim.Adam(m.parameters(), lr=lr_, weight_decay=wd)
    Et, Pt, Yt, Tt = E[idx_tr], P[idx_tr], torch.from_numpy(Y[idx_tr]), torch.from_numpy(TGT[idx_tr])
    valid = ~torch.isnan(Yt); npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = valid.sum(0) / (2 * npos.clamp(min=1)); wneg = valid.sum(0) / (2 * nneg.clamp(min=1))
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y, v, tt = Yt[j], valid[j], Tt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg) * v
            loss = (F.binary_cross_entropy_with_logits(lg, tt.nan_to_num(0), reduction="none") * w).sum() / v.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(E[idx_es], P[idx_es]).numpy()
        sc = np.nanmean([auroc(Y[idx_es][:, f], pv[:, f]) for f in range(2)])
        if sc > best + 1e-5:
            best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(E[idx_pred], P[idx_pred])[:, 0].numpy()


tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
strat = np.nan_to_num(Y[tri, 0], nan=2).astype(int) * 2 + np.nan_to_num(Y[tri, 1], nan=0).astype(int)
rng = np.random.default_rng(0)
col = np.zeros(len(tri))
for k, (a_, b_) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(tri, strat)):
    a_ = rng.permutation(a_); nes = int(0.15 * len(a_)); es, a2 = a_[:nes], a_[nes:]
    col[b_] = np.mean([fit(tri[a2], tri[es], tri[b_], sd) for sd in (42, 43)], 0)
p = rng.permutation(len(tri)); nes = int(0.15 * len(p))
vv = np.mean([fit(tri[p[nes:]], tri[p[:nes]], vai, sd) for sd in (42, 43)], 0)
yt = Y[tri, 0]; ok = ~np.isnan(yt)
print(f"train OOF AUROC {auroc(yt[ok], col[ok]):.4f}", flush=True)
pd.concat([pd.DataFrame({"study_id": [ids[i] for i in tri], "split": "train", "y": Y[tri, 0], "score": col}),
           pd.DataFrame({"study_id": [ids[i] for i in vai], "split": "val", "y": Y[vai, 0], "score": vv})]).to_csv(out, index=False)
print("->", out)
