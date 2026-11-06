"""BD2: probe head with an auxiliary CBD-diameter target. Same embeddings, standardisation, folds, seeds and
optimiser as analysis_biliary/scripts/bd_oof.py (probe.py recipe); the head has 2 outputs on the same unit view:
  out0 = biliary dilation logit (class-balanced BCE), out1 = standardized log(CBD mm) (masked MSE, only where the
  train report states a CBD diameter). loss = BCE + lam * MSE. Early stopping on inner 15% train holdout AUROC.
lam = 0 is the plain control (identical to bd_oof). TRAIN 5-fold OOF -> selection; full-train -> val scores (report).
Usage: bd2_aux.py EMB_NPZ LABELS_CSV CBD_CSV OOF_REF_CSV OUT_DIR  [variant ...]"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
EMB, LAB, CBD, REF, OUT = sys.argv[1:6]
z = np.load(EMB)
ids = list(z["ids"]); units = list(z["units"]); MAIN = units[:12]
E = z["E"][:, :12].astype(np.float32); P = z["P"][:, :12].astype(np.float32)
del z
lab = pd.read_csv(LAB).set_index("study_id")
lab = lab[lab.split.isin(["train", "val"])]
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = torch.from_numpy(E[keep]), torch.from_numpy(P[keep])
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_bdd"]].values.astype(np.float32)
cbd = pd.read_csv(CBD).set_index("study_id").reindex(ids)["cbd_mm"].values
tr = S == "train"
lc = np.log(cbd); mu_c, sd_c = np.nanmean(lc[tr]), np.nanstd(lc[tr])
Z = ((lc - mu_c) / sd_c).astype(np.float32)          # standardized log CBD (nan = not stated)
Z[~tr] = np.nan                                       # never train on val diameters
trt = torch.from_numpy(tr)
rows = E[trt][P[trt] > 0]
mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
del rows
W = E.shape[2]; ui = {u: MAIN.index(u) for u in MAIN}


def view(kind, us):
    idx = [ui[u] for u in us]
    if kind == "mean":
        return E[:, idx], P[:, idx]
    return E[:, idx].permute(0, 2, 1, 3).reshape(E.shape[0], 1, W, -1), P[:, idx].max(1, keepdim=True).values


class MTL(torch.nn.Module):
    """shared low-rank projection per window (rank r) -> 2 task heads per window -> window gate (late fusion).
    The only deviation from the linear probe: the auxiliary CBD target can now shape a SHARED representation
    (in the linear 'late' probe the two heads share only the gate, so the aux loss cannot help)."""
    def __init__(self, W, D, r=32, nt=2):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(W, D, r) * (1.0 / D ** 0.5))
        self.a = torch.nn.Parameter(torch.randn(nt, W, r) * 0.01); self.b = torch.nn.Parameter(torch.zeros(nt, W))
        self.gate = torch.nn.Parameter(torch.zeros(W))
    def forward(self, X, P):
        h = torch.einsum("nwd,wdr->nwr", X[:, 0], self.U)
        o = torch.einsum("nwr,twr->ntw", h, self.a) + self.b
        return (o * torch.softmax(self.gate, 0)).sum(-1)


def fit(X, Pm, fu, idx_tr, idx_es, idx_pred, spec, seed, lam, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MTL(X.shape[2], X.shape[3]) if spec["mode"] == "mtl" else Probe(spec["mode"], X.shape[1], X.shape[2], X.shape[3], fu, spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[idx_tr], Pm[idx_tr], torch.from_numpy(Y[idx_tr])
    Zt = torch.from_numpy(Z[idx_tr]); zv = ~torch.isnan(Zt)
    valid = ~torch.isnan(Yt); npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = valid.sum(0) / (2 * npos.clamp(min=1)); wneg = valid.sum(0) / (2 * nneg.clamp(min=1))
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y, v = Yt[j], valid[j]
            out = m(Et[j], Pt[j]); lg = out[:, :1]
            w = torch.where(y == 1, wpos, wneg) * v
            loss = (F.binary_cross_entropy_with_logits(lg, y.nan_to_num(0), reduction="none") * w).sum() / v.sum().clamp(min=1)
            if lam > 0:
                zm = zv[j]
                if zm.any():
                    loss = loss + lam * ((out[:, 1][zm] - Zt[j][zm]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(X[idx_es], Pm[idx_es])[:, 0].numpy()
        sc = auroc(Y[idx_es][:, 0], pv)
        if sc > best + 1e-5:
            best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        o = m(X[idx_pred], Pm[idx_pred]).numpy()
    return o[:, 0], o[:, 1]


LP = ["liver", "pancreas"]; LPGD = LP + ["gallbladder", "small_bowel"]
specs = {}
for lam in (0.0, 0.3, 1.0):
    specs[f"E3_late_aux{lam:g}"] = ("mean", LP, {"mode": "late"}, lam)
    specs[f"E3late_cat4_aux{lam:g}"] = ("cat", LPGD, {"mode": "late"}, lam)
for lam in (0.0, 1.0):
    specs[f"E3late_cat4_mtl{lam:g}"] = ("cat", LPGD, {"mode": "mtl"}, lam)
only = sys.argv[6:] or [k for k in specs if "mtl" not in k]
tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
tri = tri[~np.isnan(Y[tri, 0])]
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tri, Y[tri, 0].astype(int)))
ref = pd.read_csv(REF).set_index("study_id")
fold_id = np.zeros(len(tri), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k
assert (ref.loc[[ids[i] for i in tri], "fold"].values == fold_id).all(), "folds differ from analysis_biliary oof_train.csv"
fo, fv = OUT + "/bd2_oof_train.csv", OUT + "/bd2_val.csv"
oof = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": fold_id})
val = pd.read_csv(fv) if os.path.exists(fv) else pd.DataFrame({"study_id": [ids[i] for i in vai]})
SEEDS = [42, 43]
for name in only:
    kind, us, spec, lam = specs[name]
    X, Pm = view(kind, us)
    fu = [[0] if kind == "cat" else list(range(len(us)))] * 2
    rng = np.random.default_rng(0)
    col = np.zeros(len(tri)); colr = np.zeros(len(tri))
    for k, (a, b) in enumerate(folds):
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        r = [fit(X, Pm, fu, tri[a2], tri[es], tri[b], spec, s, lam) for s in SEEDS]
        col[b] = np.mean([x[0] for x in r], 0); colr[b] = np.mean([x[1] for x in r], 0)
    oof[name] = col; oof[name + "_reg"] = colr * sd_c + mu_c
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    r = [fit(X, Pm, fu, tri[a[nes:]], tri[a[:nes]], vai, spec, s, lam) for s in SEEDS]
    val[name] = np.mean([x[0] for x in r], 0); val[name + "_reg"] = np.mean([x[1] for x in r], 0) * sd_c + mu_c
    print(f"{name}: train OOF AUROC {auroc(Y[tri, 0], col):.4f} | val {auroc(Y[vai, 0], val[name].values):.4f}", flush=True)
    oof.to_csv(fo, index=False); val.to_csv(fv, index=False)
print("BD2_DONE")
