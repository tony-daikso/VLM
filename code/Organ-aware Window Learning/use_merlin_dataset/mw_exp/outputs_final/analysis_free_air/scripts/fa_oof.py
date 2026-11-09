"""5-fold OOF on TRAIN for free_air with the probe.py recipe (units small_bowel+colon+stomach = finding_organ_map,
standardize per unit on train-present rows, class-balanced BCE, lr 1e-3 / wd 1e-4, early stopping on an inner 15% TRAIN holdout).
Joint findings restricted to those whose units are inside {small_bowel, colon, stomach}: free_air, bowel_obstruction, appendicitis
(they share the E3_late gate in expB). ES criterion = mean AUROC of the joint findings (as in the gallstones template).
Also: full-train -> val scores with the same recipe (descriptive only). Test ids are never loaded (notest matrix + label filter)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_free_air/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"]); UI = [units.index(u) for u in ("small_bowel", "colon", "stomach")]
E = torch.from_numpy(z["E"][:, UI].astype(np.float32)); P = torch.from_numpy(z["P"][:, UI].astype(np.float32))
del z
lab = pd.read_csv(A + "labels_trval.csv").set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_fa", "y_sbo", "y_app"]].values.astype(np.float32)
FU = [[0, 1, 2], [0, 1], [1]]
tr = S == "train"; trt = torch.from_numpy(tr)
for u in range(3):
    pres = P[trt, u] > 0
    rows = E[trt][pres][:, u]
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    E[:, u] = ((E[:, u] - mu) / sd) * (P[:, u, None, None] > 0)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
print("N", len(ids), "train", tr.sum(), "missing-unit frac", (P[trt] <= 0).float().mean(0).tolist(), flush=True)

def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe(spec["mode"], 3, W, E.shape[3], FU, spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[idx_tr], P[idx_tr], torch.from_numpy(Y[idx_tr])
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
            pv = m(E[idx_es], P[idx_es]).numpy()
        sc = np.nanmean([auroc(Y[idx_es][:, f], pv[:, f]) for f in range(3)])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(E[idx_pred], P[idx_pred])[:, 0].numpy(), ep

specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"}}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}
seeds = {"E0": [42, 43], "E3_late": [42, 43]}
tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
strat = np.nan_to_num(Y[tri, 0], nan=2).astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
rng = np.random.default_rng(0)
for name, spec in specs.items():
    col = np.zeros(len(tri))
    for k, (a, b) in enumerate(skf.split(tri, strat)):
        oof.loc[b, "fold"] = k
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        ps = [fit(tri[a2], tri[es], tri[b], spec, sd)[0] for sd in seeds.get(name, [42])]
        col[b] = np.mean(ps, 0)
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(tri[a[nes:]], tri[a[:nes]], vai, spec, sd)[0] for sd in seeds.get(name, [42])], 0)
    val[name] = vv
    print(f"{name}: train OOF AUROC {auroc(Y[tri,0], col):.4f} | val (descriptive) {auroc(Y[vai,0], vv):.4f}", flush=True)
    oof.to_csv(A + "oof_train.csv", index=False); val.to_csv(A + "scores_val_descriptive.csv", index=False)
print("OOF_DONE")
