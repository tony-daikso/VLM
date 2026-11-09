"""MIL probe on frozen per-instance (rib x sector) features (rib_feat2.py) -> rib-fracture logit. Train/val only (asserted); target t_rib.
Folds = labels.fold (sitewise outer folds); best epoch by inner 12 % holdout of the training folds; fold model scores its
held-out fold + val (val AUROC never computed). Output OUT/score_fold{k}.csv (combine.py format) + summary.json (train OOF).
usage: rib_probe.py LABELS FEATS OUT [--hid 256] [--wd 1e-2] [--drop 0.3] [--epochs 40] [--seeds 3] [--pool attmax|lse] [--blocks all]"""
import os, sys, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("feats"); ap.add_argument("out")
ap.add_argument("--hid", type=int, default=256); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--pool", default="attmax"); ap.add_argument("--blocks", default="all"); ap.add_argument("--ribemb", action="store_true")
ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--half", action="store_true")
ap.add_argument("--extra", default="", help="npz dir of another feature set to concat (same format)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
have = {f[:-4] for f in os.listdir(a.feats) if f.endswith(".npz") and ".tmp" not in f}
lab = lab[lab.study_id.isin(have)].reset_index(drop=True)
def _ld(p):
    with np.load(p) as z: return {k: z[k] for k in ("X", "inst")}
Z = [_ld(os.path.join(a.feats, c + ".npz")) for c in lab.study_id]
IM = max(len(z["inst"]) for z in Z); D0 = Z[0]["X"].shape[1]
B = {"s0": (0, 64), "s1": (64, 192), "s2": (192, 448), "s3": (448, 960)}
cols = np.arange(D0) if a.blocks == "all" else np.r_[tuple(np.arange(*B[b]) for b in a.blocks.split(","))]
X = np.zeros((len(Z), IM, len(cols)), np.float32); P = np.zeros((len(Z), IM), bool); II = np.zeros((len(Z), IM), np.int64)
for n, z in enumerate(Z):
    k = len(z["inst"]); X[n, :k] = z["X"][:, cols]; P[n, :k] = True; II[n, :k] = z["inst"]
del Z
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int); yt = lab.t_rib.values
Xs = X[tr][P[tr]]; mu = Xs.mean(0); sd = Xs.std(0) + 1e-4; del Xs; X = np.where(P[..., None], (X - mu) / sd, 0).astype(np.float32)
D = X.shape[2]; Xt = (torch.from_numpy(X).half() if a.half else torch.from_numpy(X)).to(dev); del X; Pt = torch.from_numpy(P).to(dev); It = torch.from_numpy(II).to(dev); NSEC = 12
print(f"N {len(lab)} train {tr.sum()} rib+ {(yt == 1).sum()} D {D}", flush=True)


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Dropout(a.drop), nn.Linear(D, a.hid), nn.GELU(), nn.Dropout(a.drop))
        s.re = nn.Embedding(int(II.max()) + 1, a.hid) if a.ribemb else None
        s.att = nn.Linear(a.hid, 1); s.inst = nn.Linear(a.hid, 1); s.head = nn.Linear(2 * a.hid, 1)
    def forward(s, x, p, ii):
        h = s.f(x.float())
        if s.re is not None: h = h + s.re(ii)
        if a.pool == "lse":
            sc = s.inst(h)[..., 0].masked_fill(~p, -1e4); return torch.logsumexp(sc, 1)
        at = s.att(h)[..., 0].masked_fill(~p, -1e4); w = torch.softmax(at, 1)[..., None]
        z = torch.cat([(w * h).sum(1), h.masked_fill(~p[..., None], -1e4).amax(1)], 1)
        return s.head(z)[:, 0]


def pred(m, idx):
    m.eval()
    with torch.no_grad(): return torch.cat([m(Xt[idx[i:i + 2048]], Pt[idx[i:i + 2048]], It[idx[i:i + 2048]]) for i in range(0, len(idx), 2048)]).cpu().numpy()


def fit(itr, ies, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = M().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    yy = torch.from_numpy(np.nan_to_num(yt).astype(np.float32)).to(dev)
    pw = torch.tensor((yt[itr] == 0).sum() / (yt[itr] == 1).sum(), device=dev); best, bst = -1, None
    for ep in range(a.epochs):
        m.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // 256)):
            b = torch.from_numpy(b).to(dev)
            loss = F.binary_cross_entropy_with_logits(m(Xt[b], Pt[b], It[b]), yy[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sc = auc(yt[ies], pred(m, torch.from_numpy(ies).to(dev)))
        if sc > best: best, bst = sc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst); return m, best


trainable = tr & ~np.isnan(yt); res = {}; oof = np.full(len(lab), np.nan)
for k in range(5):
    a_ = np.where(trainable & (fold != k))[0]; idx = np.where((tr & (fold == k)) | ~tr)[0]; ps = []
    for sd_ in range(a.seed0, a.seed0 + a.seeds):
        r = np.random.default_rng(100 * sd_ + k).permutation(a_); ne = int(0.12 * len(r))
        m, b = fit(r[ne:], r[:ne], 100 * sd_ + k); ps.append(pred(m, torch.from_numpy(idx).to(dev)))
    p = np.mean(ps, 0); oof[idx[tr[idx]]] = p[tr[idx]]
    pd.DataFrame({"study_id": lab.study_id.values[idx], "split": lab.split.values[idx], "fold": k, "ft": p}).to_csv(f"{a.out}/score_fold{k}.csv", index=False)
m_ = trainable & ~np.isnan(oof); t0 = m_ & (lab.tier.values == 0)
res = {"oof_t_rib": float(auc(yt[m_], oof[m_])), "oof_t_rib_tier0": float(auc(yt[t0], oof[t0])) if t0.sum() else None,
       "n_train_scored": int(m_.sum()), "args": vars(a)}
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("PROBE", a.out, json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)
