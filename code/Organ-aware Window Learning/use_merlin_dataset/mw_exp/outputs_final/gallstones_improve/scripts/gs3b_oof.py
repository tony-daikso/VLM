"""GS3b: gallstone probe when the GB mask is missing (GB unit embedding = 0 -> constant logit). Same recipe as
analysis_gallstones/scripts/gs_oof.py (probe.py 'late' head, joint gallstones + surgically_absent_gallbladder, class-balanced
BCE, lr 1e-3 / wd 1e-4, seeds 42+43, early stopping on an inner 15% train holdout), GS1 folds, val = full-train head.
Variants (finding_organ_map.yaml untouched; implemented here):
  g3b_base    : gallbladder only (= E3_late reproduction, same ES split sequence as the variants)
  g3b_livfb   : (a) GB missing (present=0 or 3 mm GB < 100 vox) -> liver embedding (liver stats) in the GB slot,
                    + indicator gb_missing_used_liver appended as an extra input dim (all windows)
  g3b_gbliv   : (b) units [gallbladder, liver], mean over present units + present fraction (probe.py multi-unit)
  g3b_gbliv_max: (b') sites {gb:[gallbladder], liver:[liver]}, MIL (per-site shared head, logsumexp over sites)
usage: gs3b_oof.py OUT_DIR [--smoke]"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
OUT = sys.argv[1]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
lab = pd.read_csv(R + "analysis_gallstones/labels_trval.csv"); lab = lab[lab.split.isin(["train", "val"])].set_index("study_id")
folds = pd.read_csv(R + "analysis_gallstones/oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id")["fold"]
z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"]); gi, li = units.index("gallbladder"), units.index("liver")
keep = np.array([s in lab.index for s in ids])
E = torch.from_numpy(z["E"][keep][:, [gi, li]].astype(np.float32)); P = torch.from_numpy(z["P"][keep][:, [gi, li]].astype(np.float32)); del z
ids = [s for s, k in zip(ids, keep) if k]
g1 = pd.read_csv(R + "gallstones_improve/gs1_feat3mm.csv").drop_duplicates("study_id").set_index("study_id")["g1_vox"]
gbv = g1.reindex(ids).fillna(0).values
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_gs", "y_chole"]].values.astype(np.float32); tr = S == "train"
t = torch.from_numpy(tr)
for u in range(2):  # per-unit standardisation (present train rows), as gs_oof.py does for the GB unit
    r = E[t][:, u][P[t][:, u] > 0]; mu, sd = r.mean(0), r.std(0).clamp(min=1e-6)
    E[:, u] = ((E[:, u] - mu) / sd) * (P[:, u, None, None] > 0)
miss = torch.from_numpy((P[:, 0].numpy() == 0) | (gbv < 100))
print("N", len(ids), "train", int(tr.sum()), "GB missing", int(miss.sum()), "liver present among missing", float(P[miss, 1].mean()), flush=True)
# views: (X [N,U,W,D], P [N,U], finding_units, finding_sites)
Xa = E[:, :1].clone(); Pa = P[:, :1].clone()
use_l = miss & (P[:, 1] > 0)
Xa[use_l, 0] = E[use_l, 1]; Pa[use_l, 0] = 1.0
ind = use_l.float()[:, None, None, None].expand(-1, 1, Xa.shape[2], 1)
Xa = torch.cat([Xa, ind], 3)
VIEWS = {"g3b_base": (E[:, :1], P[:, :1], [[0], [0]], None),
         "g3b_livfb": (Xa, Pa, [[0], [0]], None),
         "g3b_gbliv": (E, P, [[0, 1], [0]], None),
         "g3b_gbliv_max": (E, P, [[0, 1], [0]], [[[0], [1]], None])}


def fit(V, i_tr, i_es, i_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    X, Pm, fu, fs = V
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", X.shape[1], X.shape[2], X.shape[3], fu, None, finding_sites=fs)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[i_tr], Pm[i_tr], torch.from_numpy(Y[i_tr])
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
        with torch.no_grad(): pv = m(X[i_es], Pm[i_es]).numpy()
        sc = np.nanmean([auroc(Y[i_es][:, f], pv[:, f]) for f in range(2)])
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): return m(X[i_pred], Pm[i_pred])[:, 0].numpy()


tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
fold = folds.reindex([ids[i] for i in tri]).values.astype(int)
SEEDS = [42] if SMOKE else [42, 43]
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": fold}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
for name, V in VIEWS.items():
    rng = np.random.default_rng(0); col = np.zeros(len(tri))
    for k in range(5):
        b = np.where(fold == k)[0]; a = rng.permutation(np.where(fold != k)[0]); nes = int(0.15 * len(a))
        col[b] = np.mean([fit(V, tri[a[nes:]], tri[a[:nes]], tri[b], sd) for sd in SEEDS], 0)
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(V, tri[a[nes:]], tri[a[:nes]], vai, sd) for sd in SEEDS], 0)
    oof[name] = col; val[name] = vv
    print(f"{name}: train OOF {auroc(Y[tri, 0], col):.4f} | val {auroc(Y[vai, 0], vv):.4f}", flush=True)
    oof.to_csv(f"{OUT}/gs3b_oof_train.csv", index=False); val.to_csv(f"{OUT}/gs3b_val.csv", index=False)
print("GS3B_DONE", flush=True)
