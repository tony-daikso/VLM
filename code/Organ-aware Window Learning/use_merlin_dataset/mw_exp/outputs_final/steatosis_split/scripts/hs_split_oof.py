"""5-fold train OOF + full-train -> val scores for the split steatosis findings (y_diffuse / y_focal),
probe.py recipe (E3_late, liver-unit inputs, pooled 12-unit train standardization, class-balanced BCE,
Adam lr 1e-3 / wd 1e-4, early stopping on an inner 15 % TRAIN holdout, 2 seeds) — same as
analysis_steatosis/scripts/hs_oof_spleen.py, only the label column changes.
Variants: liver | concat (liver||spleen) | concat_diff (liver||(liver-spleen)).
Folds: StratifiedKFold(5, shuffle, random_state=0) on the target label (train rows with a defined label).
Usage: hs_split_oof.py <src dir with probe.py> <data dir> <target col> <variant> <out dir> [threads]"""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 2)
sys.path.insert(0, sys.argv[1])
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
D, TGT, VAR, O = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
z = np.load(D + "/liver_emb_trval.npz"); st = np.load(D + "/std_stats_main12.npz")
zs = np.load(D + "/spleen_emb_trval.npz"); assert list(zs["ids"]) == list(z["ids"])
ids = list(z["ids"]); E = torch.from_numpy(np.stack([z["E"], zs["E"]], 1)); P = torch.from_numpy(np.stack([z["P"], zs["P"]], 1))
lab = pd.read_csv(D + "/labels_trval_split.csv").set_index("study_id")
lab = lab[lab.split.isin(["train", "val"])]
keep = [i for i, s in enumerate(ids) if s in lab.index and not np.isnan(lab.at[s, TGT])]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
mu, sd = torch.from_numpy(st["mu"]), torch.from_numpy(st["sd"]).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
if VAR == "liver":
    E, P = E[:, :1], P[:, :1]
elif VAR == "concat":
    E = torch.cat([E[:, 0], E[:, 1]], -1)[:, None]; P = P[:, :1]
elif VAR == "concat_diff":
    E = torch.cat([E[:, 0], E[:, 0] - E[:, 1]], -1)[:, None]; P = P[:, :1]
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, [TGT]].values.astype(np.float32)
W = E.shape[2]


def fit(idx_tr, idx_es, idx_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, W, E.shape[3], [[0]], None)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[idx_tr], P[idx_tr], torch.from_numpy(Y[idx_tr])
    npos = Yt.sum(0); nneg = len(Yt) - npos
    wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(m(Et[j], Pt[j]), y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            sc = auroc(Y[idx_es][:, 0], m(E[idx_es], P[idx_es])[:, 0].numpy())
        if sc > best + 1e-5:
            best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(E[idx_pred], P[idx_pred])[:, 0].numpy()


tri = np.where(S == "train")[0]; vai = np.where(S == "val")[0]
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tri, Y[tri, 0]))
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1, "y": Y[tri, 0]})
for k, (a, b) in enumerate(folds):
    oof.loc[b, "fold"] = k
val = pd.DataFrame({"study_id": [ids[i] for i in vai], "y": Y[vai, 0]})
rng = np.random.default_rng(0); seeds = [42, 43]; col = np.zeros(len(tri)); name = f"E3_late_{VAR}"
for k, (a, b) in enumerate(folds):
    a = rng.permutation(a); nes = int(0.15 * len(a))
    col[b] = np.mean([fit(tri[a[nes:]], tri[a[:nes]], tri[b], s) for s in seeds], 0)
    print(TGT, VAR, "fold", k, flush=True)
oof[name] = col
a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
val[name] = np.mean([fit(tri[a[nes:]], tri[a[:nes]], vai, s) for s in seeds], 0)
oof.to_csv(f"{O}/oof_{TGT}_{VAR}.csv", index=False); val.to_csv(f"{O}/val_{TGT}_{VAR}.csv", index=False)
print(f"{TGT} {VAR}: train OOF {auroc(Y[tri,0], col):.4f} | val {auroc(Y[vai,0], val[name].values):.4f}", flush=True)
