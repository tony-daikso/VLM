"""5-fold OOF on TRAIN for biliary_ductal_dilation with the probe.py recipe (standardize with pooled train statistics of
the 12 main units, class-balanced BCE, Adam lr 1e-3 / wd 1e-4, mean over present units + present-fraction feature,
early stopping on an inner 15% train holdout instead of val; 2 seeds averaged).
Units as in finding_organ_map.yaml: [liver, pancreas]. Variants test window choice (E1_*), unit choice and pooling of
units (mean vs concat, extra units). Also full-train -> val scores (descriptive only, nothing tuned on val)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_biliary/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"])
MAIN = units[:12]
E = z["E"][:, :12].astype(np.float32); P = z["P"][:, :12].astype(np.float32)
del z
lab = pd.read_csv(A + "labels_trval.csv").set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = torch.from_numpy(E[keep]), torch.from_numpy(P[keep])
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_bdd"]].values.astype(np.float32)
tr = S == "train"
trt = torch.from_numpy(tr)
rows = E[trt][P[trt] > 0]                                # pooled stats of present main units (probe.py --standardize)
mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
del rows
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
ui = {u: MAIN.index(u) for u in MAIN}


def view(kind, us):
    """mean: [N,U',W,D] the listed units (probe averages present ones); cat: one pseudo-unit = concat of units."""
    idx = [ui[u] for u in us]
    if kind == "mean":
        return E[:, idx], P[:, idx], [list(range(len(idx)))]
    Ec = E[:, idx].permute(0, 2, 1, 3).reshape(E.shape[0], 1, W, -1) if kind == "cat" else None
    if kind == "catwin":                                   # concat units AND windows -> single window
        Ec = E[:, idx].reshape(E.shape[0], 1, 1, -1)
    return Ec, (P[:, idx].max(1, keepdim=True).values), [[0]]


def fit(X, Pm, fu, idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe(spec["mode"], X.shape[1], X.shape[2], X.shape[3], fu, spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[idx_tr], Pm[idx_tr], torch.from_numpy(Y[idx_tr])
    valid = ~torch.isnan(Yt); npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = valid.sum(0) / (2 * npos.clamp(min=1)); wneg = valid.sum(0) / (2 * nneg.clamp(min=1))
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y, v = Yt[j], valid[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg) * v
            loss = (F.binary_cross_entropy_with_logits(lg, y.nan_to_num(0), reduction="none") * w).sum() / v.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(X[idx_es], Pm[idx_es]).numpy()
        sc = auroc(Y[idx_es][:, 0], pv[:, 0])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(X[idx_pred], Pm[idx_pred])[:, 0].numpy()

LP = ["liver", "pancreas"]
specs = {"E0": ("mean", LP, {"mode": "single", "win_idx": 0}),
         "E3_late": ("mean", LP, {"mode": "late"})}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = ("mean", LP, {"mode": "single", "win_idx": i})
specs.update({
    "E0_liver_only": ("mean", ["liver"], {"mode": "single", "win_idx": 0}),
    "E0_pancreas_only": ("mean", ["pancreas"], {"mode": "single", "win_idx": 0}),
    "E0_cat_liver_panc": ("cat", LP, {"mode": "single", "win_idx": 0}),
    "E3late_cat_liver_panc": ("cat", LP, {"mode": "late"}),
    "E0_cat_+gb_duod": ("cat", LP + ["gallbladder", "small_bowel"], {"mode": "single", "win_idx": 0}),
    "E3late_cat_+gb_duod": ("cat", LP + ["gallbladder", "small_bowel"], {"mode": "late"}),
    "E0_cat_all12": ("cat", MAIN, {"mode": "single", "win_idx": 0}),
    "E2_catwin_liver_panc": ("catwin", LP, {"mode": "single", "win_idx": 0}),
})
only = sys.argv[1:] or list(specs)
tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
ok = ~np.isnan(Y[tri, 0]); tri = tri[ok]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
folds = list(skf.split(tri, Y[tri, 0].astype(int)))
try:
    oof = pd.read_csv(A + "oof_train.csv"); val = pd.read_csv(A + "scores_val_descriptive.csv")
except FileNotFoundError:
    oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
for k, (a, b) in enumerate(folds):
    oof.loc[b, "fold"] = k
SEEDS = [42, 43]
for name in only:
    kind, us, spec = specs[name]
    X, Pm, fu = view(kind, us)
    rng = np.random.default_rng(0)
    col = np.zeros(len(tri))
    for k, (a, b) in enumerate(folds):
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        col[b] = np.mean([fit(X, Pm, fu, tri[a2], tri[es], tri[b], spec, s) for s in SEEDS], 0)
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(X, Pm, fu, tri[a[nes:]], tri[a[:nes]], vai, spec, s) for s in SEEDS], 0)
    val[name] = vv
    print(f"{name}: train OOF AUROC {auroc(Y[tri, 0], col):.4f} | val (descriptive) {auroc(Y[vai, 0], vv):.4f}", flush=True)
    oof.to_csv(A + "oof_train.csv", index=False); val.to_csv(A + "scores_val_descriptive.csv", index=False)
print("OOF_DONE")
