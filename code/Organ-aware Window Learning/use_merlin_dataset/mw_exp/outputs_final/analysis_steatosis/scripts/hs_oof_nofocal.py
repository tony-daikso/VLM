"""5-fold OOF on TRAIN for hepatic steatosis, probe.py recipe on the liver unit:
pooled 12-unit train standardization (as probe.py --standardize), class-balanced BCE, Adam lr 1e-3 / wd 1e-4
(expB chosen HP for E0 and E3_late), early stopping on an inner 15% train holdout (never val), 2 seeds.
Modes: E0, E3_late, E1_<window>. Also full-train -> val scores (descriptive only).
Deviation: steatosis trained alone (probe.py trains 14 findings jointly; the liver gate is shared with ascites/BDD)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(2)
sys.path.insert(0, sys.argv[1])          # mw_exp/src (probe.Probe)
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
D = sys.argv[2]
z = np.load(D + "/data/liver_emb_trval.npz"); st = np.load(D + "/data/std_stats_main12.npz")
ids = list(z["ids"]); E = torch.from_numpy(z["E"])[:, None]; P = torch.from_numpy(z["P"])[:, None]
lab = pd.read_csv(D + "/data/labels_trval.csv").set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index and not np.isnan(lab.at[s, "y_hs"])]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
mu, sd = torch.from_numpy(st["mu"]), torch.from_numpy(st["sd"]).clamp(min=1e-6)
E = ((E - mu) / sd) * (P[..., None, None] > 0)
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_hs"]].values.astype(np.float32)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]


def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe(spec["mode"], 1, W, E.shape[3], [[0]], spec.get("win_idx"))
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
        return m(E[idx_pred], P[idx_pred])[:, 0].numpy(), (torch.softmax(m.gate, -1).numpy() if m.gate is not None else None)


specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"}}
specs = {"E3_late": {"mode": "late"}}
pf = lab.loc[ids, "pure_focal"].values.astype(bool) & (Y[:, 0] == 1)   # focal-fat-only positives
tri = np.where(S == "train")[0]; vai = np.where(S == "val")[0]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
folds = list(skf.split(tri, Y[tri, 0]))
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
for k, (a, b) in enumerate(folds):
    oof.loc[b, "fold"] = k
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
log = open(D + "/oof_nofocal.log", "a")
for name, spec in specs.items():
    rng = np.random.default_rng(0)
    seeds = [42, 43] if name in ("E0", "E3_late") else [42]
    col = np.zeros(len(tri)); gates = []
    for k, (a, b) in enumerate(folds):
        a = rng.permutation(a); nes = int(0.15 * len(a))
        a2, es = a[nes:], a[:nes]
        a2 = a2[~pf[tri[a2]]]; es = es[~pf[tri[es]]]
        outs = [fit(tri[a2], tri[es], tri[b], spec, s) for s in seeds]
        col[b] = np.mean([o[0] for o in outs], 0)
        gates += [o[1] for o in outs if o[1] is not None]
    oof[name + "_nofocal"] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    a2, es = a[nes:], a[:nes]; a2 = a2[~pf[tri[a2]]]; es = es[~pf[tri[es]]]
    val[name + "_nofocal"] = np.mean([fit(tri[a2], tri[es], vai, spec, s)[0] for s in seeds], 0)
    msg = f"{name}: train OOF AUROC {auroc(Y[tri,0], col):.4f} | val (descriptive) {auroc(Y[vai,0], val[name + '_nofocal'].values):.4f}"
    if gates:
        msg += " | liver gate " + str(np.round(np.mean(gates, 0)[0], 3).tolist())
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    oof.to_csv(D + "/oof_train_nofocal.csv", index=False); val.to_csv(D + "/scores_val_nofocal.csv", index=False)
print("OOF_DONE", flush=True)
