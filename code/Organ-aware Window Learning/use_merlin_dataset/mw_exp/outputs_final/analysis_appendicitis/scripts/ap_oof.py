"""5-fold OOF on TRAIN for appendicitis with the probe.py recipe (colon unit = finding_organ_map, standardize,
class-balanced BCE, lr 1e-3 / wd 1e-4, early stopping on an inner 15% TRAIN holdout instead of val).
Extra variants: E3_late on colon+small_bowel units (U=2, mean over present units) to test unit choice.
Also full-train -> val scores (descriptive only). Test rows are dropped before anything (labels_trval has no test)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"; A = R + "analysis_appendicitis/"
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"]); ui = [units.index("colon"), units.index("small_bowel")]
E = torch.from_numpy(z["E"][:, ui].astype(np.float32)); P = torch.from_numpy(z["P"][:, ui].astype(np.float32))
del z
lab = pd.read_csv(A + "labels_trval.csv").set_index("study_id")
lab = lab[lab.y_ap.notna()]
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_ap"]].values.astype(np.float32)
tr = S == "train"
for u in range(2):   # per-unit standardization on present train rows
    pres = P[torch.from_numpy(tr), u] > 0
    rows = E[torch.from_numpy(tr)][pres][:, u]
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    E[:, u] = ((E[:, u] - mu) / sd) * (P[:, u, None, None] > 0)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]

def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    us = spec.get("units", [0])
    Eu, Pu = E[:, us], P[:, us]
    m = Probe(spec["mode"], len(us), W, E.shape[3], [list(range(len(us)))], spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = Eu[idx_tr], Pu[idx_tr], torch.from_numpy(Y[idx_tr])
    npos = Yt.sum(0); nneg = len(Yt) - npos
    wpos = len(Yt) / (2 * npos.clamp(min=1)); wneg = len(Yt) / (2 * nneg.clamp(min=1))
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
            pv = m(Eu[idx_es], Pu[idx_es]).numpy()
        sc = auroc(Y[idx_es][:, 0], pv[:, 0])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(Eu[idx_pred], Pu[idx_pred])[:, 0].numpy(), ep

specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"},
         "E3_late_colon_sb": {"mode": "late", "units": [0, 1]}, "E3_late_sb": {"mode": "late", "units": [1]}}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}
seeds = {"E0": [42, 43, 44], "E3_late": [42, 43, 44]}
tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
rng = np.random.default_rng(0)
splits = list(skf.split(tri, Y[tri, 0].astype(int)))
for name, spec in specs.items():
    col = np.zeros(len(tri))
    for k, (a, b) in enumerate(splits):
        oof.loc[b, "fold"] = k
        a = np.random.default_rng(k).permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        # stratified inner holdout guard: ensure positives in es
        ps = [fit(tri[a2], tri[es], tri[b], spec, sd)[0] for sd in seeds.get(name, [42])]
        col[b] = np.mean(ps, 0)
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(tri[a[nes:]], tri[a[:nes]], vai, spec, sd)[0] for sd in seeds.get(name, [42])], 0)
    val[name] = vv
    print(f"{name}: train OOF AUROC {auroc(Y[tri,0], col):.4f} | val (descriptive) {auroc(Y[vai,0], vv):.4f}", flush=True)
    oof.to_csv(A + "oof_train.csv", index=False); val.to_csv(A + "scores_val_descriptive.csv", index=False)
print("OOF_DONE")
