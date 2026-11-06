"""BD5 stage C: probe on porta-hepatis crop-unit embeddings (bd5_extract.py), same recipe/folds/seeds as bd_oof.py.
Variants (E3_late-style 'late' head: one linear head per window, gated):
  P5_porta      : 4 porta regions (mean|max, 896 each) concatenated per window
  P5_cat4_porta : liver+pancreas+gallbladder+small_bowel (main emb) + 4 porta regions concatenated per window
Usage: bd5_probe.py EMB_NPZ PORTA_DIR LABELS_CSV OOF_REF_CSV OUT_DIR [variant ...]
Writes OUT_DIR/bd5_oof_train.csv and bd5_val.csv (train OOF + full-train->val)."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
EMB, PD, LAB, REF, OUT = sys.argv[1:6]
z = np.load(EMB); ids = list(z["ids"]); units = list(z["units"])
sel = [units.index(u) for u in ("liver", "pancreas", "gallbladder", "small_bowel")]
E = z["E"][:, sel].astype(np.float32); P = z["P"][:, sel].astype(np.float32); del z
lab = pd.read_csv(LAB).set_index("study_id"); lab = lab[lab.split.isin(["train", "val"])]
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
N, W = len(ids), E.shape[2]
EP = np.zeros((N, 4, W, 896), np.float32); PP = np.zeros((N, 4), np.float32)
for i, s in enumerate(ids):
    f = f"{PD}/{s}.npz"
    if os.path.exists(f):
        q = np.load(f)
        if bool(q["present"]):
            EP[i] = q["emb"].astype(np.float32); PP[i] = (q["region_vox"] > 0).astype(np.float32)
print("porta coverage", PP[:, 0].mean().round(4), flush=True)
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_bdd"]].values.astype(np.float32); tr = S == "train"


def standardize(X, Pm):
    X = torch.from_numpy(X); Pm = torch.from_numpy(Pm); t = torch.from_numpy(tr)
    out = torch.zeros_like(X)
    for u in range(X.shape[1]):                      # per-unit stats over present train rows (porta dims differ by unit role)
        r = X[t][:, u][Pm[t][:, u] > 0]
        mu, sd = r.mean(0), r.std(0).clamp(min=1e-6)
        out[:, u] = ((X[:, u] - mu) / sd) * (Pm[:, u, None, None] > 0)
    return out, Pm


Em, Pm_ = standardize(E, P); Ep, Pp_ = standardize(EP, PP)


def cat_view(parts):
    X = torch.cat([x.permute(0, 2, 1, 3).reshape(N, W, -1) for x, _ in parts], 2)[:, None]
    Pm = torch.cat([p for _, p in parts], 1).max(1, keepdim=True).values
    return X, Pm


views = {"P5_porta": cat_view([(Ep, Pp_)]), "P5_cat4_porta": cat_view([(Em, Pm_), (Ep, Pp_)])}


def fit(X, Pm, idx_tr, idx_es, idx_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, X.shape[2], X.shape[3], [[0]], None)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[idx_tr], Pm[idx_tr], torch.from_numpy(Y[idx_tr])
    npos = Yt.sum(); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            sc = auroc(Y[idx_es][:, 0], m(X[idx_es], Pm[idx_es])[:, 0].numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(X[idx_pred], Pm[idx_pred])[:, 0].numpy()


tri = np.where(tr)[0]; vai = np.where(S == "val")[0]; tri = tri[~np.isnan(Y[tri, 0])]; vai = vai[~np.isnan(Y[vai, 0])]
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tri, Y[tri, 0].astype(int)))
ref = pd.read_csv(REF).set_index("study_id"); fid = np.zeros(len(tri), int)
for k, (a, b) in enumerate(folds): fid[b] = k
assert (ref.loc[[ids[i] for i in tri], "fold"].values == fid).all()
fo, fv = OUT + "/bd5_oof_train.csv", OUT + "/bd5_val.csv"
oof = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": fid})
val = pd.read_csv(fv) if os.path.exists(fv) else pd.DataFrame({"study_id": [ids[i] for i in vai]})
for name in (sys.argv[6:] or list(views)):
    X, Pm = views[name]; rng = np.random.default_rng(0); col = np.zeros(len(tri))
    for k, (a, b) in enumerate(folds):
        a = rng.permutation(a); nes = int(0.15 * len(a))
        col[b] = np.mean([fit(X, Pm, tri[a[nes:]], tri[a[:nes]], tri[b], s) for s in (42, 43)], 0)
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(X, Pm, tri[a[nes:]], tri[a[:nes]], vai, s) for s in (42, 43)], 0)
    oof[name] = col; val[name] = vv
    print(f"{name}: train OOF {auroc(Y[tri, 0], col):.4f} | val {auroc(Y[vai, 0], vv):.4f}", flush=True)
    oof.to_csv(fo, index=False); val.to_csv(fv, index=False)
print("BD5_PROBE_DONE")
