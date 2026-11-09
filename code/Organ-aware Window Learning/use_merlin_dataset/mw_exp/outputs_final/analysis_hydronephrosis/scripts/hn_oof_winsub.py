"""Window-subset E3_late OOF (same recipe as hn_oof.py): baseline+one window pairs and leave-one-window-out.
Usage: hn_oof_winsub.py <dir> <src dir>. Writes oof_winsub.csv (train OOF) + val_winsub.csv (descriptive)."""
import sys, runpy, numpy as np, pandas as pd
D, SRC = sys.argv[1].rstrip("/") + "/", sys.argv[2]
WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
subs = {f"pair_baseline+{w}": [0, i] for i, w in enumerate(WINS) if i}
subs.update({f"LOWO_-{w}": [j for j in range(5) if j != i] for i, w in enumerate(WINS)})
subs["soft+narrow+liver(no baseline,bone)"] = [1, 2, 3]
g = runpy.run_path(SRC + "/../scripts/hn_oof.py", run_name="lib") if False else None
import torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, SRC)
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
z = np.load(D + "kidney_emb_trval.npz")
ids = list(z["ids"]); E0 = torch.from_numpy(z["E"].astype(np.float32)); P = torch.from_numpy(z["P"].astype(np.float32))
lab = pd.read_csv(D + "labels_trval.csv").set_index("study_id")
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_hn"]].values.astype(np.float32)
tr = S == "train"
rows = E0[torch.from_numpy(tr)][P[torch.from_numpy(tr)] > 0]
E0 = ((E0 - rows.mean(0)) / rows.std(0).clamp(min=1e-6)) * (P[..., None, None] > 0)
def fit(E, idx_tr, idx_es, idx_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 2, E.shape[2], E.shape[3], [[0, 1]])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[idx_tr], P[idx_tr], torch.from_numpy(Y[idx_tr])
    npos = Yt.sum(0); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): sc = auroc(Y[idx_es][:, 0], m(E[idx_es], P[idx_es])[:, 0].numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): return m(E[idx_pred], P[idx_pred])[:, 0].numpy()
tri = np.where(tr)[0]; tri = tri[~np.isnan(Y[tri, 0])]; vai = np.where(S == "val")[0]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri]}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
for name, ws in subs.items():
    E = E0[:, :, ws]; rng = np.random.default_rng(0); col = np.zeros(len(tri))
    for k, (a, b) in enumerate(skf.split(tri, Y[tri, 0].astype(int))):
        a = rng.permutation(a); nes = int(0.15 * len(a))
        col[b] = np.mean([fit(E, tri[a[nes:]], tri[a[:nes]], tri[b], s) for s in (42, 43)], 0)
    a = rng.permutation(tri); nes = int(0.15 * len(a))
    vv = np.mean([fit(E, a[nes:], a[:nes], vai, s) for s in (42, 43)], 0)
    oof[name] = col; val[name] = vv
    print(f"{name}: train OOF AUROC {auroc(Y[tri,0], col):.4f} | val (descriptive) {auroc(Y[vai,0], vv):.4f}", flush=True)
    oof.to_csv(D + "oof_winsub.csv", index=False); val.to_csv(D + "val_winsub.csv", index=False)
print("WINSUB_DONE")
