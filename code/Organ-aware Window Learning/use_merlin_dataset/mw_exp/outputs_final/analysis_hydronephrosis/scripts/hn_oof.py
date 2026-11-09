"""5-fold OOF on TRAIN for hydronephrosis with the probe.py recipe (kidney_left+kidney_right units, mean over present units
+ present-fraction feature, standardize on train kidney rows, class-balanced BCE, Adam lr 1e-3 / wd 1e-4 = expB HP,
early stopping on an inner 15% train holdout instead of val, 2 seeds). Only hydronephrosis is trained (only finding on kidneys).
Extra specs: *_max = MIL max over the two kidneys with a shared head (+ per-unit bias) -> also per-side logits.
Also: full-train -> val scores with the same recipe (descriptive only). Usage: hn_oof.py <dir with kidney_emb_trval.npz, labels_trval.csv> <src dir>"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
D, SRC = sys.argv[1].rstrip("/") + "/", sys.argv[2]
sys.path.insert(0, SRC)
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
z = np.load(D + "kidney_emb_trval.npz")
ids = list(z["ids"]); E = torch.from_numpy(z["E"].astype(np.float32)); P = torch.from_numpy(z["P"].astype(np.float32))
lab = pd.read_csv(D + "labels_trval.csv").set_index("study_id")
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_hn"]].values.astype(np.float32)
tr = S == "train"
rows = E[torch.from_numpy(tr)][P[torch.from_numpy(tr)] > 0]          # [m,W,D] present kidney rows
mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]

def make(spec, seed):
    torch.manual_seed(seed)
    return Probe(spec["mode"], 2, W, E.shape[3], [[0, 1]], spec.get("win_idx"),
                 finding_sites=[[[0, 1]]] if spec.get("agg") == "max" else None)

def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    np.random.seed(seed); m = make(spec, seed)
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
        sc = auroc(Y[idx_es][:, 0], pv[:, 0])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        Ep, Pp = E[idx_pred], P[idx_pred]
        out = {"s": m(Ep, Pp)[:, 0].numpy()}
        if spec.get("agg") == "max" or spec["mode"] in ("single", "late"):
            # per-side scores: only one kidney 'present' (the other zeroed) -> side logit
            for k, nm in ((0, "L"), (1, "R")):
                Pk = Pp.clone(); Pk[:, 1 - k] = 0
                out[nm] = np.where(Pp[:, k].numpy() > 0, m(Ep, Pk)[:, 0].numpy(), np.nan)
        if spec["mode"] == "late":
            out["gate"] = torch.softmax(m.gate, -1).detach().numpy().mean(0)
    return out

specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"},
         "E0_max": {"mode": "single", "win_idx": 0, "agg": "max"}, "E3_late_max": {"mode": "late", "agg": "max"}}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}
only = sys.argv[3:] or list(specs)
seeds = [42, 43]
tri = np.where(tr)[0]; tri = tri[~np.isnan(Y[tri, 0])]; vai = np.where(S == "val")[0]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
gates = {}
for name in only:
    spec = specs[name]; rng = np.random.default_rng(0)
    col = {k: np.zeros(len(tri)) for k in ("s", "L", "R")}
    for k, (a, b) in enumerate(skf.split(tri, Y[tri, 0].astype(int))):
        oof.loc[b, "fold"] = k
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        outs = [fit(tri[a2], tri[es], tri[b], spec, sd_) for sd_ in seeds]
        for c in col:
            if c in outs[0]: col[c][b] = np.mean([o[c] for o in outs], 0)
        if "gate" in outs[0]: gates.setdefault(name, []).append(np.mean([o["gate"] for o in outs], 0).tolist())
    oof[name] = col["s"]
    if spec.get("agg") == "max" or name in ("E0", "E3_late"):
        oof[name + "_L"] = col["L"]; oof[name + "_R"] = col["R"]
    a = rng.permutation(tri); nes = int(0.15 * len(a))
    outs = [fit(a[nes:], a[:nes], vai, spec, sd_) for sd_ in seeds]
    val[name] = np.mean([o["s"] for o in outs], 0)
    print(f"{name}: train OOF AUROC {auroc(Y[tri,0], col['s']):.4f} | val (descriptive) {auroc(Y[vai,0], val[name].values):.4f}", flush=True)
    oof.to_csv(D + "oof_train.csv", index=False); val.to_csv(D + "scores_val_descriptive.csv", index=False)
json.dump(gates, open(D + "gates.json", "w"))
print("OOF_DONE")
