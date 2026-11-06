"""GS5 stage C: gallstone probe on the GB crop-unit embeddings (gs5_extract.py: [3 regions, 5 windows, 896 = mean|max]).
Recipe as analysis_gallstones/scripts/gs_oof.py (probe.py 'late' head = per-window linear heads mixed by a window gate,
per-region standardisation on present train rows, class-balanced BCE, lr 1e-3 / wd 1e-4, seeds 42+43, early stopping
on an inner 15% train holdout). Folds = the GS1 folds. Val = full-train head.
Views: gbx_r0 (GB dilated 6 mm), gbx_r1 (GB dil6 U fossa30), gbx_cat (r0|r1|r2 concatenated per window).
usage: gs5_probe.py CROP_EMB_DIR OUT_DIR [--smoke]"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
ED, OUT = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv; os.makedirs(OUT, exist_ok=True)
lab = pd.read_csv(R + "analysis_gallstones/labels_trval.csv"); lab = lab[lab.split.isin(["train", "val"])]
lab = lab[lab.y_gs.notna()].set_index("study_id")
folds = pd.read_csv(R + "analysis_gallstones/oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id")["fold"]
ids = [s for s in lab.index if os.path.exists(f"{ED}/{s}.npz")]
print("cases with crop emb", len(ids), "of", len(lab), flush=True)
N = len(ids); EP = np.zeros((N, 3, 5, 896), np.float32); PP = np.zeros((N, 3), np.float32)
for i, s in enumerate(ids):
    q = np.load(f"{ED}/{s}.npz")
    if bool(q["present"]):
        EP[i] = q["emb"].astype(np.float32); PP[i] = (q["region_vox"] > 0).astype(np.float32)
print("region present", PP.mean(0).round(4).tolist(), flush=True)
S = lab.loc[ids, "split"].values; Y = lab.loc[ids, ["y_gs"]].values.astype(np.float32); tr = S == "train"
X = torch.from_numpy(EP); Pm = torch.from_numpy(PP); t = torch.from_numpy(tr)
for u in range(3):
    r = X[t][:, u][Pm[t][:, u] > 0]; mu, sd = r.mean(0), r.std(0).clamp(min=1e-6)
    X[:, u] = ((X[:, u] - mu) / sd) * (Pm[:, u, None, None] > 0)
def view(us):
    V = torch.cat([X[:, u] for u in us], 2)[:, None]; P_ = Pm[:, us].max(1, keepdim=True).values; return V, P_
views = {"gbx_r0": view([0]), "gbx_r1": view([1]), "gbx_cat": view([0, 1, 2])}

def fit(Xv, Pv, i_tr, i_es, i_pred, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Probe("late", 1, Xv.shape[2], Xv.shape[3], [[0]], None)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = Xv[i_tr], Pv[i_tr], torch.from_numpy(Y[i_tr])
    npos = Yt.sum(); nneg = len(Yt) - npos; wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * nneg)
    best, bst, bad = -1, None, 0
    for ep in range(2 if SMOKE else max_ep):
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

tri = np.where(tr)[0]; vai = np.where(S == "val")[0]
fold = folds.reindex([ids[i] for i in tri]).values.astype(int)
SEEDS = [42] if SMOKE else [42, 43]
oof = pd.DataFrame({"study_id": [ids[i] for i in tri], "fold": fold}); val = pd.DataFrame({"study_id": [ids[i] for i in vai]})
for name, (Xv, Pv) in views.items():
    rng = np.random.default_rng(0); col = np.zeros(len(tri))
    for k in range(5):
        b = np.where(fold == k)[0]; a = rng.permutation(np.where(fold != k)[0]); nes = int(0.15 * len(a))
        col[b] = np.mean([fit(Xv, Pv, tri[a[nes:]], tri[a[:nes]], tri[b], sd) for sd in SEEDS], 0)
    a = rng.permutation(len(tri)); nes = int(0.15 * len(a))
    vv = np.mean([fit(Xv, Pv, tri[a[nes:]], tri[a[:nes]], vai, sd) for sd in SEEDS], 0)
    oof[name] = col; val[name] = vv
    print(f"{name}: train OOF {auroc(Y[tri, 0], col):.4f} | val {auroc(Y[vai, 0], vv):.4f}", flush=True)
oof.to_csv(f"{OUT}/gbx_oof_train.csv", index=False); val.to_csv(f"{OUT}/gbx_val.csv", index=False)
print("GS5_PROBE_DONE", flush=True)
