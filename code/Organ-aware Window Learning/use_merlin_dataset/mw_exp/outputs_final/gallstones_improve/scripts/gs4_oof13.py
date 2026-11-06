"""GS4: train 5-fold OOF logits of the 13 OTHER findings (not gallstones) with the probe.py E3_late recipe
(multi-label late head over all findings' units, finding_organ_map.yaml incl. fracture MIL max, --standardize stats,
class-balanced BCE, lr 1e-3 / wd 1e-4 = expB chosen HP, seeds 42+43, early stopping on an inner 15% train holdout
(mean AUROC over the 13 findings) instead of val). Folds = the GS1 gallstone folds (analysis_gallstones/oof_train.csv).
The gallstone label is never read (usecols excludes it). Val: full-train fit -> val scores. Test is never loaded
(emb matrix *_notest; split filtered before labels are read).
usage: gs4_oof13.py <out_dir> [--smoke]"""
import sys, json, time, numpy as np, pandas as pd, torch, torch.nn.functional as F, yaml
torch.set_num_threads(4)
SRC = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src"; sys.path.insert(0, SRC)
from probe import Probe, auroc
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/"; O = sys.argv[1]; SMOKE = "--smoke" in sys.argv
import os; os.makedirs(O, exist_ok=True)
fmap = yaml.safe_load(open(R + "finding_organ_map.yaml"))
# ---- split first, then labels (train/val rows only; gallstones column never read) ----
spl = pd.read_csv(R + "../data/split.csv").set_index("study_id")["split"]
trval = set(spl[spl.isin(["train", "val"])].index)
hdr = pd.read_csv(R + "../data/labels_disease14.csv", nrows=0).columns
FIND = [f for f in fmap if f != "gallstones" and f in hdr]
assert "gallstones" not in FIND and len(FIND) == 13, FIND
lab = pd.concat([c[c.study_id.isin(trval)] for c in pd.read_csv(R + "../data/labels_disease14.csv", usecols=["study_id"] + FIND, chunksize=5000)]).set_index("study_id")
folds = pd.read_csv(R + "outputs_final/analysis_gallstones/oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id")["fold"]
z = np.load(R + "outputs_final/results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"])
keep = np.array([(s in lab.index) and spl.get(s) in ("train", "val") for s in ids])
need = sorted({units.index(u) for f in FIND for u in fmap[f]["organs"]})
print("loading E", flush=True)
E = torch.from_numpy(z["E"][keep][:, need].astype(np.float32)); P = torch.from_numpy(z["P"][keep][:, need].astype(np.float32)); del z
ids = [s for s, k in zip(ids, keep) if k]; units = [units[i] for i in need]
S = spl.loc[ids].values; Y = lab.loc[ids, FIND].values.astype(np.float32)
tr = S == "train"
# ---- standardize as probe.py --standardize: main units pooled stats, bone units per-unit stats (train, present only) ----
BONES = {"T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "ribs_left", "ribs_right", "sacrum", "hip_left", "hip_right", "femur_left", "femur_right"}
mi = [i for i, u in enumerate(units) if u not in BONES]; bi = [i for i, u in enumerate(units) if u in BONES]
trt = torch.from_numpy(tr)
rows = E[trt][:, mi][P[trt][:, mi] > 0]; mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6); del rows
E[:, mi] = ((E[:, mi] - mu) / sd) * (P[:, mi][..., None, None] > 0)
for i in bi:
    r = E[trt, i][P[trt, i] > 0]; E[:, i] = ((E[:, i] - r.mean(0)) / r.std(0).clamp(min=1e-6)) * (P[:, i][:, None, None] > 0)
fu = [[units.index(u) for u in fmap[f]["organs"]] for f in FIND]
def sites_of(f):
    fm = fmap[f]
    if fm.get("agg", "mean") != "max": return None
    return [[units.index(u) for u in us] for us in (fm.get("sites") or {"all": fm["organs"]}).values()]
fsites = [sites_of(f) for f in FIND]
W, D = E.shape[2], E.shape[3]
print(f"N={len(ids)} train={int(tr.sum())} val={int((S=='val').sum())} U={len(units)} findings={FIND}", flush=True)

def fit(i_tr, i_es, i_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", len(units), W, D, fu, finding_sites=fsites)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[i_tr], P[i_tr], torch.from_numpy(Y[i_tr])
    valid = ~torch.isnan(Yt); npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = valid.sum(0) / (2 * npos.clamp(min=1)); wneg = valid.sum(0) / (2 * nneg.clamp(min=1))
    best, bst, bad = -1, None, 0
    for ep in range(2 if SMOKE else max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y, v = Yt[j], valid[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg) * v
            loss = (F.binary_cross_entropy_with_logits(lg, y.nan_to_num(0), reduction="none") * w).sum() / v.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = m(E[i_es], P[i_es]).numpy()
        sc = np.nanmean([auroc(Y[i_es][:, f], pv[:, f]) for f in range(len(FIND))])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return torch.cat([m(E[i_pred[k:k + 2048]], P[i_pred[k:k + 2048]]) for k in range(0, len(i_pred), 2048)]).numpy(), ep, best

tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
fold = folds.reindex([ids[i] for i in tri]).values
assert not np.isnan(fold.astype(float)).any(), "train case without a GS1 fold"
SEEDS = [42] if SMOKE else [42, 43]
rng = np.random.default_rng(0)
oof = np.zeros((len(tri), len(FIND)), np.float32); t0 = time.time()
for k in range(5):
    b = np.where(fold == k)[0]; a = rng.permutation(np.where(fold != k)[0]); nes = int(0.15 * len(a))
    if SMOKE: a = a[:1500]; nes = 300
    outs = []
    for sd_ in SEEDS:
        p, ep, bs_ = fit(tri[a[nes:]], tri[a[:nes]], tri[b], sd_); outs.append(p)
        print(f"fold {k} seed {sd_}: epochs {ep+1} inner-ES mean AUROC {bs_:.4f}  {time.time()-t0:.0f}s", flush=True)
    oof[b] = np.mean(outs, 0)
    if SMOKE and k == 0: break
a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
if SMOKE: a = a[:1500]; nes = 300
val = np.mean([fit(tri[a[nes:]], tri[a[:nes]], vai, sd_)[0] for sd_ in SEEDS], 0)
po = pd.DataFrame(oof, columns=["o13_" + f for f in FIND]); po.insert(0, "fold", fold.astype(int)); po.insert(0, "study_id", [ids[i] for i in tri])
pv = pd.DataFrame(val, columns=["o13_" + f for f in FIND]); pv.insert(0, "study_id", [ids[i] for i in vai])
res = {f: dict(oof=auroc(Y[tri, j], oof[:, j]), val=auroc(Y[vai, j], val[:, j])) for j, f in enumerate(FIND)}
for f, r in res.items(): print(f"{f}: train OOF {r['oof']:.4f} | val {r['val']:.4f}", flush=True)
if not SMOKE:
    po.to_csv(f"{O}/oof13_train.csv", index=False); pv.to_csv(f"{O}/oof13_val.csv", index=False)
    json.dump(dict(findings=FIND, seeds=SEEDS, hp=dict(lr=1e-3, wd=1e-4), auroc=res), open(f"{O}/oof13_summary.json", "w"), indent=1)
print("OOF13_DONE", f"{time.time()-t0:.0f}s", flush=True)
