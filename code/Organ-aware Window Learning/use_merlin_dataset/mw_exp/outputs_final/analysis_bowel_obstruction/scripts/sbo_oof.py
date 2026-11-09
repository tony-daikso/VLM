"""5-fold OOF on TRAIN for bowel_obstruction with the probe.py recipe (finding_organ_map units small_bowel+colon,
standardize per unit on train-present rows, class-balanced BCE, lr 1e-3 / wd 1e-4, early stopping on an inner 15% TRAIN
holdout). Joint findings restricted to those whose units are inside the unit set: bowel_obstruction + appendicitis (colon)
(they share the E3_late gate in expB). ES criterion = mean AUROC of the joint findings (as in the gallstones/free-air
templates). Variant "_st": bowel_obstruction mapped to small_bowel+colon+stomach.
Also: full-train -> val scores with the same recipe (descriptive only). Input is a train/val-only slice (no test ids).
Usage: sbo_oof.py emb_npz labels_csv src_dir out_dir [only_specs] [label_variant]
label_variant (training labels only; OOF AUROC is always computed on the original labels):
  dropamb : train negatives whose report affirms ileus / obstruction / dilated small bowel, or category probably_no, are ignored
  ambpos  : those negatives are trained as positives (label-boundary probe)"""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(4)
emb, labf, src, O = sys.argv[1:5]
sys.path.insert(0, src)
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
z = np.load(emb)
ids = list(z["ids"]); E0 = torch.from_numpy(z["E"].astype(np.float32)); P0 = torch.from_numpy(z["P"].astype(np.float32))
lab = pd.read_csv(labf).set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index and lab.loc[s, "split"] in ("train", "val")]
ids = [ids[i] for i in keep]; E0, P0 = E0[keep], P0[keep]
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_sbo", "y_app"]].values.astype(np.float32)
tr = S == "train"; trt = torch.from_numpy(tr)
Yeval = Y.copy()
LV = sys.argv[6] if len(sys.argv) > 6 else ""
if LV:
    L2 = lab.loc[ids]
    amb = ((L2.y_sbo == 0) & ((L2.sbo_cat == "probably_no") | (L2.rep_aff_ileus == 1) | (L2.rep_aff_obstruct == 1) | (L2.rep_aff_sb_dilated == 1))).values
    Y[amb, 0] = np.nan if LV == "dropamb" else 1.0
    print("label variant", LV, "ambiguous negatives", int(amb.sum()), "of which train", int((amb & tr).sum()), flush=True)
for u in range(E0.shape[1]):
    pres = P0[trt, u] > 0
    rows = E0[trt][pres][:, u]
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    E0[:, u] = ((E0[:, u] - mu) / sd) * (P0[:, u, None, None] > 0)
W = E0.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
print("N", len(ids), "train", tr.sum(), "val", (S == "val").sum(), "missing-unit frac", (P0[trt] <= 0).float().mean(0).tolist(), flush=True)
UNITSETS = {"": ([0, 1], [[0, 1], [1]]), "_st": ([0, 1, 2], [[0, 1, 2], [1]])}


def fit(E, P, FU, idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe(spec["mode"], E.shape[1], W, E.shape[3], FU, spec.get("win_idx"))
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
        sc = np.nanmean([auroc(Y[idx_es][:, f][~np.isnan(Y[idx_es][:, f])], pv[:, f][~np.isnan(Y[idx_es][:, f])]) for f in range(Y.shape[1])])
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
specs["E0_st"] = {"mode": "single", "win_idx": 0, "us": "_st"}
specs["E3_late_st"] = {"mode": "late", "us": "_st"}
seeds = {"E0": [42, 43], "E3_late": [42, 43], "E0_st": [42, 43], "E3_late_st": [42, 43]}
tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
strat = np.nan_to_num(Yeval[tri, 0], nan=2).astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
rng = np.random.default_rng(0)
only = sys.argv[5].split(",") if len(sys.argv) > 5 else None
for name, spec in specs.items():
    if only and name not in only: continue
    ui, FU = UNITSETS[spec.get("us", "")]
    E, P = E0[:, ui], P0[:, ui]
    col = np.zeros(len(tri))
    for k, (a, b) in enumerate(skf.split(tri, strat)):
        oof.loc[b, "fold"] = k
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        ps = [fit(E, P, FU, tri[a2], tri[es], tri[b], spec, sd)[0] for sd in seeds.get(name, [42])]
        col[b] = np.mean(ps, 0)
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(E, P, FU, tri[a[nes:]], tri[a[:nes]], vai, spec, sd)[0] for sd in seeds.get(name, [42])], 0)
    val[name] = vv
    ev = lambda yy, ss: auroc(yy[~np.isnan(yy)], ss[~np.isnan(yy)])
    print(f"{name}{('_' + LV) if LV else ''}: train OOF AUROC {ev(Yeval[tri,0], col):.4f} | val (descriptive) {ev(Yeval[vai,0], vv):.4f}", flush=True)
    sfx = ("_" + LV) if LV else ""
    oof.to_csv(O + f"/oof_train{sfx}.csv", index=False); val.to_csv(O + f"/scores_val_descriptive{sfx}.csv", index=False)
print("OOF_DONE")
