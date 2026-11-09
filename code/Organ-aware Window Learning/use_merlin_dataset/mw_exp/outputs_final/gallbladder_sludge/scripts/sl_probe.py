"""膽泥 heads: the same image heads as the gallstones best (Exp N G5c+U) retrained on sludge targets.
Recipe identical to gs3b_oof.py / gs5_probe.py / nu_probe.py: probe.py 'late' head (one linear head per window mixed by
the window gate), per-unit/region standardisation on present TRAIN rows, class-balanced BCE, Adam lr 1e-3 / wd 1e-4,
early stopping on an inner 15% train holdout (patience 10, max 100 ep), seeds 42+43, 5 folds = GS1 folds (= absent-GB folds,
identical on all 15,309 train ids), val = refit on all train -> val once. Single target per head (gs3b used a joint
gallstones+cholecystectomy head; here single target for simplicity).
Views:
  base    : main TS gallbladder unit (= E3_late GB head)
  livfb   : GB missing (present=0 or 3 mm GB < 100 vox) -> liver embedding in the GB slot + indicator dim (= g3b_livfb)
  gbx_r0 / gbx_r1 / gbx_cat : GS5 GB crop unit (dilated GB / fossa / r0|r1|r2)
  U       : Exp N GB/GB-fossa unit (3 regions x 7 windows concatenated)
Targets (labels_trval_sludge.csv): y_sludge (膽泥, all), y_sludge_only (sludge & no stones vs rest; gallstone-known rows).
Input = train/val slice only (test never present). usage: sl_probe.py DATA OUT --views a,b --targets t1,t2"""
import sys, os, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, os.environ["MW_SRC"])
from probe import Probe, auroc
ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--views", default="base,livfb,gbx_r0,gbx_r1,gbx_cat,U"); ap.add_argument("--targets", default="y_sludge,y_sludge_only")
ap.add_argument("--tag", default="a")
A = ap.parse_args(); D = A.data; os.makedirs(A.out, exist_ok=True)
lab = pd.read_csv(f"{D}/gallbladder_sludge/labels_trval_sludge.csv")
assert set(lab.split) <= {"train", "val"}
fold = pd.read_csv(f"{D}/analysis_absent_gb/oof_train.csv")[["study_id", "fold"]]
lab = lab.merge(fold, on="study_id", how="left").reset_index(drop=True)
lab = lab[lab.y_sludge.notna()].reset_index(drop=True)
ids = list(lab.study_id); N = len(ids); tr = (lab.split == "train").values; t = torch.from_numpy(tr.copy())
VIEWS = A.views.split(",")


def stdz(X, Pr):  # X [N,U,W,D], Pr [N,U]
    for u in range(X.shape[1]):
        m = t & (Pr[:, u] > 0)
        r = X[m][:, u]; mu, sd = r.mean(0), r.std(0).clamp(min=1e-6)
        X[:, u] = ((X[:, u] - mu) / sd) * (Pr[:, u, None, None] > 0)
    return X


V = {}
if {"base", "livfb"} & set(VIEWS):
    z = np.load(f"{D}/emb_gb_liver_trval.npz"); pos = {s: i for i, s in enumerate(z["ids"])}; k = [pos[s] for s in ids]
    E = torch.from_numpy(z["E"][k].astype(np.float32)); P = torch.from_numpy(z["P"][k].astype(np.float32)); del z
    E = stdz(E, P)
    g1 = pd.read_csv(f"{D}/gallstones_improve/gs1_feat3mm.csv").drop_duplicates("study_id").set_index("study_id")["g1_vox"]
    gbv = g1.reindex(ids).fillna(0).values
    miss = torch.from_numpy((P[:, 0].numpy() == 0) | (gbv < 100))
    V["base"] = (E[:, :1].contiguous(), P[:, :1].contiguous())
    Xa = E[:, :1].clone(); Pa = P[:, :1].clone(); use_l = miss & (P[:, 1] > 0)
    Xa[use_l, 0] = E[use_l, 1]; Pa[use_l, 0] = 1.0
    Xa = torch.cat([Xa, use_l.float()[:, None, None, None].expand(-1, 1, Xa.shape[2], 1)], 3)
    V["livfb"] = (Xa, Pa); del E
if any(v.startswith("gbx") for v in VIEWS):
    z = dict(np.load(f"{D}/emb_gs5_trval.npz")); pos = {s: i for i, s in enumerate(z["ids"])}
    X = np.zeros((N, 3, 5, 896), np.float32); Pm = np.zeros((N, 3), np.float32)
    for i, s in enumerate(ids):
        j = pos.get(s)
        if j is not None and bool(z["present"][j]): X[i] = z["emb"][j]; Pm[i] = (z["region_vox"][j] > 0)
    del z; X = stdz(torch.from_numpy(X), torch.from_numpy(Pm)); Pm = torch.from_numpy(Pm)
    def view(us): return torch.cat([X[:, u] for u in us], 2)[:, None].contiguous(), Pm[:, us].max(1, keepdim=True).values
    for nm, us in (("gbx_r0", [0]), ("gbx_r1", [1]), ("gbx_cat", [0, 1, 2])):
        if nm in VIEWS: V[nm] = view(us)
    del X
if "U" in VIEWS:
    z = dict(np.load(f"{D}/emb_unitU_trval.npz")); pos = {s: i for i, s in enumerate(z["ids"])}
    K, Rn, W, Dd = z["emb"].shape[1:]
    X = np.zeros((N, Rn, W, Dd), np.float32); RP = np.zeros((N, Rn), np.float32)
    for i, s in enumerate(ids):
        j = pos.get(s)
        if j is None: continue
        X[i] = z["emb"][j, 0]; RP[i] = (z["region_vox"][j, 0] > 0) & bool(np.asarray(z["present"][j]).reshape(-1)[0])
    del z
    X = stdz(torch.from_numpy(X), torch.from_numpy(RP))                       # per region (unit axis = region)
    V["U"] = (X.permute(0, 2, 1, 3).reshape(N, W, Rn * Dd)[:, None].contiguous(), torch.from_numpy(RP).max(1, keepdim=True).values)
    del X
print({k: tuple(v[0].shape) for k, v in V.items()}, flush=True)


def fit(Xv, Pv, Y, i_tr, i_es, i_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, Xv.shape[2], Xv.shape[3], [[0]], None)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = Xv[i_tr], Pv[i_tr], torch.from_numpy(Y[i_tr])
    npos = Yt.sum(); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j]); w = torch.where(y == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): sc = auroc(Y[i_es, 0], m(Xv[i_es], Pv[i_es])[:, 0].numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): return m(Xv[i_pred], Pv[i_pred])[:, 0].numpy()


for tg in A.targets.split(","):
    fo, fv = f"{A.out}/oof_{tg}_{A.tag}.csv", f"{A.out}/val_{tg}_{A.tag}.csv"
    ok = lab[tg].notna().values
    tri = np.where(tr & ok)[0]; vai = np.where(~tr & ok)[0]; fd = lab.fold.values[tri].astype(int)
    Y = lab[[tg]].values.astype(np.float32)
    oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": fd}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
    print(tg, "train", len(tri), int(Y[tri].sum()), "val", len(vai), int(Y[vai].sum()), "pos per fold", np.bincount(fd, weights=Y[tri, 0]).tolist(), flush=True)
    for nm, (Xv, Pv) in V.items():
        rng = np.random.default_rng(0); col = np.zeros(len(tri))
        for k in range(5):
            b = np.where(fd == k)[0]; a = rng.permutation(np.where(fd != k)[0]); nes = int(0.15 * len(a))
            col[b] = np.mean([fit(Xv, Pv, Y, tri[a[nes:]], tri[a[:nes]], tri[b], sd) for sd in (42, 43)], 0)
        a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
        vv = np.mean([fit(Xv, Pv, Y, tri[a[nes:]], tri[a[:nes]], vai, sd) for sd in (42, 43)], 0)
        oof["h_" + nm] = col; val["h_" + nm] = vv
        print(f"{tg} {nm}: train OOF {auroc(Y[tri, 0], col):.4f} | val {auroc(Y[vai, 0], vv):.4f}", flush=True)
        oof.to_csv(fo, index=False); val.to_csv(fv, index=False)
print("SL_PROBE_DONE", flush=True)
