"""RLQ crop-unit probe (E3_late-style 'late' head: one linear head per window, gated), same recipe as ap_oof.py /
bd5_probe.py: per-unit standardisation on present TRAIN rows, class-balanced BCE, Adam lr 1e-3 / wd 1e-4, early stopping
on an inner 15% TRAIN holdout, 2 seeds (42, 43). Folds = analysis_appendicitis/oof_train.csv folds (stratified on the
combined label; the split targets are subsets of those rows, as in appendicitis_split/ap_split_probe.py).
Targets: y_ap (combined), y_ap_def (definite acute), y_ap_unc (uncertain/treated/follow-up/post-op);
rows with NaN target are dropped from training AND evaluation of that target. Test is never loaded
(emb matrix = *_notest cache, labels_trval_split.csv has train/val only, crop embeddings were made for train+val ids only).
Variants (all per-window concatenation of unit vectors, one late head):
  E3_late    : colon (main emb, mean 448)             -- reference, recomputed here
  E3cs       : colon + small_bowel (main emb)          -- reference
  U_rlq      : 4 RLQ regions (mean|max 896 each)
  U_rlq_cs   : colon + small_bowel + 4 RLQ regions
Usage: rlq_probe.py RLQ_EMB_DIR OUT_DIR [--device cuda] [--targets ...] [--variants ...]
Writes OUT_DIR/oof_train.csv (study_id, fold, {target}__{variant}) and OUT_DIR/val_scores.csv."""
import sys, os, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/"
sys.path.insert(0, R + "src")
from probe import Probe, auroc
ap = argparse.ArgumentParser()
ap.add_argument("emb_dir"); ap.add_argument("out")
ap.add_argument("--device", default="cpu")
ap.add_argument("--targets", default="y_ap,y_ap_def,y_ap_unc")
ap.add_argument("--variants", default="E3_late,E3cs,U_rlq,U_rlq_cs")
a = ap.parse_args(); dev = torch.device(a.device)
os.makedirs(a.out, exist_ok=True)
z = np.load(R + "outputs_final/results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"]); ui = [units.index("colon"), units.index("small_bowel")]
E = z["E"][:, ui].astype(np.float32); P = z["P"][:, ui].astype(np.float32); del z
lab = pd.read_csv(R + "outputs_final/appendicitis_split/labels_trval_split.csv").set_index("study_id")
assert set(lab.split) <= {"train", "val"}
lab = lab[lab.y_ap.notna()]
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]; lab = lab.loc[ids]
N, W = len(ids), E.shape[2]
EP = np.zeros((N, 4, W, 896), np.float32); PP = np.zeros((N, 4), np.float32)
for i, s in enumerate(ids):
    f = f"{a.emb_dir}/{s}.npz"
    if os.path.exists(f):
        q = np.load(f)
        if bool(q["present"]):
            EP[i] = q["emb"].astype(np.float32); PP[i] = (q["region_vox"] > 0).astype(np.float32)
print("RLQ coverage per region", PP.mean(0).round(4), flush=True)
S = lab.split.values; tr = S == "train"


def standardize(X, Pm):
    X = torch.from_numpy(X); Pm = torch.from_numpy(Pm); t = torch.from_numpy(tr)
    out = torch.zeros_like(X)
    for u in range(X.shape[1]):
        r = X[t][:, u][Pm[t][:, u] > 0]
        mu, sd = r.mean(0), r.std(0).clamp(min=1e-6)
        out[:, u] = ((X[:, u] - mu) / sd) * (Pm[:, u, None, None] > 0)
    return out, Pm


Em, Pm_ = standardize(E, P); Ep, Pp_ = standardize(EP, PP); del E, EP


def cat_view(parts):
    X = torch.cat([x.permute(0, 2, 1, 3).reshape(N, W, -1) for x, _ in parts], 2)[:, None]
    Pm = torch.cat([p for _, p in parts], 1).max(1, keepdim=True).values
    return X.to(dev), Pm.to(dev)


VIEWS = {"E3_late": lambda: cat_view([(Em[:, 1:2], Pm_[:, 1:2])]),
         "E3cs": lambda: cat_view([(Em, Pm_)]),
         "U_rlq": lambda: cat_view([(Ep, Pp_)]),
         "U_rlq_cs": lambda: cat_view([(Em, Pm_), (Ep, Pp_)])}
ref = pd.read_csv(R + "outputs_final/analysis_appendicitis/oof_train.csv").set_index("study_id")["fold"]


def fit(X, Pm, Y, idx_tr, idx_es, idx_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, X.shape[2], X.shape[3], [[0]], None).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = X[idx_tr], Pm[idx_tr], torch.from_numpy(Y[idx_tr]).to(dev)
    npos = Yt.sum(); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et), device=dev)
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            sc = auroc(Y[idx_es][:, 0], m(X[idx_es], Pm[idx_es])[:, 0].float().cpu().numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(X[idx_pred], Pm[idx_pred])[:, 0].float().cpu().numpy()


fo, fv = a.out + "/oof_train.csv", a.out + "/val_scores.csv"
tri_all = np.where(tr)[0]; vai_all = np.where(S == "val")[0]
oof = pd.read_csv(fo) if os.path.exists(fo) else pd.DataFrame({"study_id": [ids[i] for i in tri_all], "fold": ref.loc[[ids[i] for i in tri_all]].values})
val = pd.read_csv(fv) if os.path.exists(fv) else pd.DataFrame({"study_id": [ids[i] for i in vai_all]})
fold_all = oof.fold.values
for vname in a.variants.split(","):
    X, Pm = VIEWS[vname]()
    for tg in a.targets.split(","):
        col_name = f"{tg}__{vname}"
        if col_name in oof.columns and col_name in val.columns:
            continue
        Y = lab[[tg]].values.astype(np.float32)
        okt = ~np.isnan(Y[tri_all, 0]); okv = ~np.isnan(Y[vai_all, 0])
        tri = tri_all[okt]; vai = vai_all[okv]; fid = fold_all[okt]
        col = np.full(len(tri_all), np.nan); cc = np.zeros(len(tri))
        for k in range(5):
            a_ = np.where(fid != k)[0]; b_ = np.where(fid == k)[0]
            a_ = np.random.default_rng(k).permutation(a_); nes = int(0.15 * len(a_))
            cc[b_] = np.mean([fit(X, Pm, Y, tri[a_[nes:]], tri[a_[:nes]], tri[b_], s) for s in (42, 43)], 0)
        col[okt] = cc
        rr = np.random.default_rng(0).permutation(len(tri)); nes = int(0.15 * len(rr))
        vv = np.full(len(vai_all), np.nan)
        vv[okv] = np.mean([fit(X, Pm, Y, tri[rr[nes:]], tri[rr[:nes]], vai, s) for s in (42, 43)], 0)
        oof[col_name] = col; val[col_name] = vv
        print(f"{col_name}: train OOF {auroc(Y[tri, 0], cc):.4f} | val {auroc(Y[vai, 0], vv[okv]):.4f}", flush=True)
        oof.to_csv(fo, index=False); val.to_csv(fv, index=False)
    del X, Pm
    if dev.type == "cuda": torch.cuda.empty_cache()
print("RLQ_PROBE_DONE", flush=True)
