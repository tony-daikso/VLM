"""5-fold OOF on TRAIN for ascites with the probe.py recipe (units liver+spleen = finding_organ_map, mean over present
units + present-fraction feature, standardize, class-balanced BCE, lr 1e-3 / wd 1e-4 = expB HP, early stopping on an
inner 15% TRAIN holdout instead of val). Single finding (ascites) -> the E3_late gate is ascites-only.
Extra spec E3_late_all12: ascites mapped to all 12 main units (dilution test). Val scores = descriptive only."""
import sys, os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "2")))
A = os.environ.get("ADIR", "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_ascites/")
sys.path.insert(0, A + "scripts")   # probe.py copied read-only from A6000 mw_exp/src
from probe import Probe, auroc
from sklearn.model_selection import StratifiedKFold
z = np.load(A + "emb12_trval.npz")   # 12 main units, train+val rows only (scripts/asc_dump_emb.py)
ids = list(z["ids"]); units = list(z["units"])[:12]
E = torch.from_numpy(z["E"][:, :12].astype(np.float32)); P = torch.from_numpy(z["P"][:, :12].astype(np.float32))
del z
lab = pd.read_csv(A + "labels_trval.csv").set_index("study_id")
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
S = lab.loc[ids, "split"].values
Y = lab.loc[ids, ["y_asc"]].values.astype(np.float32)
amb = (lab.loc[ids, "asc_cat"].values == "probably_no") & np.isin(lab.loc[ids, "amount"].values, ["trace", "small"])
# alternative TRAINING labels (evaluation always on the original label):
#  dropamb: ambiguous negatives (probably_no with trace/small fluid) get NaN (= ignored) in training
#  anyfluid: they are relabelled positive
Y_alt = {"dropamb": np.where(amb[:, None], np.nan, Y).astype(np.float32), "anyfluid": np.where(amb[:, None], 1.0, Y).astype(np.float32)}
tr = S == "train"; trt = torch.from_numpy(tr)
for u in range(E.shape[1]):     # per-unit standardisation on train present rows (probe --standardize uses pooled main units)
    pass
pres = P[trt] > 0                                   # [m,U]
rows = E[trt][pres]                                 # [k,W,D] pooled over the 12 main units (as probe.py)
mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6); del rows
E = ((E - mu) / sd) * (P[..., None, None] > 0)
W = E.shape[2]; WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
LS = [units.index("liver"), units.index("spleen")]

def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    fu = [list(range(12))] if spec.get("all") else [LS]
    m = Probe(spec["mode"], 12, W, E.shape[3], fu, spec.get("win_idx"))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Ytrain = Y_alt[spec["ylab"]] if spec.get("ylab") else Y
    Et, Pt, Yt = E[idx_tr], P[idx_tr], torch.from_numpy(Ytrain[idx_tr])
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
        out = m(E[idx_pred], P[idx_pred])[:, 0].numpy()
        g = torch.softmax(m.gate, -1)[LS].mean(0).numpy() if m.gate is not None else None
    return out, ep, g

specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"}}
for i, w in enumerate(WINS[1:], 1):
    specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}
specs["E3_late_all12"] = {"mode": "late", "all": True}
specs["E3_late_dropamb"] = {"mode": "late", "ylab": "dropamb"}
specs["E3_late_anyfluid"] = {"mode": "late", "ylab": "anyfluid"}
seeds = {"E0": [42, 43], "E3_late": [42, 43]}
only = sys.argv[1:] or list(specs)
tri = np.where(tr & ~np.isnan(Y[:, 0]))[0]; vai = np.where((S == "val") & ~np.isnan(Y[:, 0]))[0]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": -1})
val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
if os.path.exists(A + "oof_train.csv"):
    o = pd.read_csv(A + "oof_train.csv"); oof = oof.merge(o.drop(columns=["fold"]), on="study_id", how="left")
    v = pd.read_csv(A + "scores_val_descriptive.csv"); val = val.merge(v, on="study_id", how="left")
yt = Y[tri, 0]
for name in only:
    spec = specs[name]; t0 = time.time(); rng = np.random.default_rng(0)
    col = np.zeros(len(tri)); gates = []
    for k, (a, b) in enumerate(skf.split(tri, yt.astype(int))):
        oof.loc[b, "fold"] = k
        a = rng.permutation(a); nes = int(0.15 * len(a)); es, a2 = a[:nes], a[nes:]
        rs = [fit(tri[a2], tri[es], tri[b], spec, sd) for sd in seeds.get(name, [42])]
        col[b] = np.mean([r[0] for r in rs], 0); gates += [r[2] for r in rs if r[2] is not None]
    oof[name] = col
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(tri[a[nes:]], tri[a[:nes]], vai, spec, sd)[0] for sd in seeds.get(name, [42])], 0)
    val[name] = vv
    gs = f" gate(liver/spleen mean)={np.round(np.mean(gates, 0), 3).tolist()}" if gates else ""
    print(f"{name}: train OOF AUROC {auroc(yt, col):.4f} | val (descriptive) {auroc(Y[vai,0], vv):.4f} | {time.time()-t0:.0f}s{gs}", flush=True)
    oof.to_csv(A + "oof_train.csv", index=False); val.to_csv(A + "scores_val_descriptive.csv", index=False)
print("OOF_DONE")
