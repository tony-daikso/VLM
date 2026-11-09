"""5-fold OOF on TRAIN for osteopenia with the probe.py recipe (lumbar_spine unit, standardize on train-present rows,
class-balanced BCE, lr 1e-3 / wd 1e-4 = expB chosen HP for both E0 and E3_late, early stopping on an inner 15% train
holdout instead of val). Single finding (after exp B only osteopenia maps to lumbar_spine).
Extra: E1 single windows, and E0/E3_late on the L1 bone unit (emb_bones). Full-train -> val scores (descriptive only)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
import os
SRC = os.environ.get("OWL_SRC", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src"); sys.path.insert(0, SRC)
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
A = os.environ.get("OP_DIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_osteopenia/")
z = np.load(A + "emb_lumbar_L1_trval.npz")   # slice of the *_notest emb matrix (scripts/op_dump.py)
ids = list(z["ids"]); units = [str(u) for u in z["units"]]
UN = ["lumbar_spine", "L1"]; gi = [units.index(u) for u in UN]
E = torch.from_numpy(z["E"][:, gi].astype(np.float32)); P = torch.from_numpy(z["P"][:, gi].astype(np.float32))
del z
lab = pd.read_csv(A + "labels_trval.csv").set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_op"]].values.astype(np.float32)
tr = S == "train"; trt = torch.from_numpy(tr)
for u in range(len(UN)):
    pres = P[trt, u] > 0
    rows = E[trt][pres][:, u]
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    E[:, u] = ((E[:, u] - mu) / sd) * (P[:, u, None, None] > 0)
print("present frac (train):", [(P[trt, u] > 0).float().mean().item() for u in range(len(UN))], flush=True)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]

def fit(u, idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    Eu, Pu = E[:, u:u + 1], P[:, u:u + 1]
    m = Probe(spec["mode"], 1, W, E.shape[3], [[0]], spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = Eu[idx_tr], Pu[idx_tr], torch.from_numpy(Y[idx_tr])
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
            pv = m(Eu[idx_es], Pu[idx_es]).numpy()
        sc = auroc(Y[idx_es][:, 0], pv[:, 0])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(Eu[idx_pred], Pu[idx_pred])[:, 0].numpy(), ep

specs = {"E0": (0, {"mode": "single", "win_idx": 0}), "E3_late": (0, {"mode": "late"})}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = (0, {"mode": "single", "win_idx": i})
specs["L1unit_E0"] = (1, {"mode": "single", "win_idx": 0}); specs["L1unit_E3_late"] = (1, {"mode": "late"})
seeds = {"E0": [42, 43], "E3_late": [42, 43]}
tri = np.where(tr & ~np.isnan(Y[:, 0]))[0]; vai = np.where((S == "val") & ~np.isnan(Y[:, 0]))[0]
strat = Y[tri, 0].astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
rng = np.random.default_rng(0)
for name, (u, spec) in specs.items():
    col = np.zeros(len(tri))
    for k, (a, b) in enumerate(skf.split(tri, strat)):
        oof.loc[b, "fold"] = k
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        ps = [fit(u, tri[a2], tri[es], tri[b], spec, sd)[0] for sd in seeds.get(name, [42])]
        col[b] = np.mean(ps, 0)
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(u, tri[a[nes:]], tri[a[:nes]], vai, spec, sd)[0] for sd in seeds.get(name, [42])], 0)
    val[name] = vv
    print(f"{name}: train OOF AUROC {auroc(Y[tri,0], col):.4f} | val (descriptive) {auroc(Y[vai,0], vv):.4f}", flush=True)
    oof.to_csv(A + "oof_train.csv", index=False); val.to_csv(A + "scores_val_descriptive.csv", index=False)
print("OOF_DONE")
