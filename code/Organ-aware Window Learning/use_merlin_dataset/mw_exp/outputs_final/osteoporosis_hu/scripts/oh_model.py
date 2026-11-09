"""影像骨質疏鬆: embeddings-only model (NO HU features) predicting the HU-defined label.
Inputs: 3 mm lumbar_spine unit, 5 window embeddings (emb_lumbar_L1_trval.npz, unit 0) -> 5 window heads ("5 window
logits"); 1.5 mm L1-L4 per-vertebra embeddings (ol_emb15_L1L4_trval.npz, mean pool, 4 x 3 x 448, missing level = 0)
-> 1 head. Stack of the 6 component scores.
Targets: hu (continuous, Ridge), y110 / y160 (class-balanced LR). Binary candidates also include -(stacked HU
regression) as a score. Hyper-parameters (Ridge alpha / LR C for components, stack type) chosen ONLY by train
5-fold OOF (folds = analysis_osteopenia folds); chosen config refit on all labelled train and applied once to val.
Rows: labelled train / val only (labels file already restricted to train/val; embeddings filtered by id). Test never read.
usage: python oh_model.py WORK_DIR LABELS_CSV OUT_DIR"""
import sys, os, json, time, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
W, LAB, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3].rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, W)
from ib_core import auc
LOG = open(OUT + "model.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

lab = pd.read_csv(LAB).set_index("study_id")
lab = lab[lab.split.isin(["train", "val"]) & lab.hu.notna()]
fold = pd.read_csv(W + "oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id").fold
z = np.load(W + "emb_lumbar_L1_trval.npz"); i3 = {s: i for i, s in enumerate(z["ids"])}
z15 = np.load(W + "ol_emb15_L1L4_trval.npz"); i15 = {s: i for i, s in enumerate(z15["ids"])}
ids = [s for s in lab.index if s in i3 and s in i15 and (lab.split[s] == "val" or s in fold.index)]
n0 = len(lab); lab = lab.loc[ids]; isT = (lab.split == "train").values
log(f"labelled rows with embeddings + fold: train {isT.sum()}, val {(~isT).sum()} (dropped {n0 - len(ids)}: no fold / embedding)")
r3 = np.array([i3[s] for s in ids]); r15 = np.array([i15[s] for s in ids])
E3 = z["E"][r3, 0].astype(np.float32); P3 = z["P"][r3, 0] > 0                       # [N,5,448]
E15 = z15["E"][r15].astype(np.float32); P15 = z15["P"][r15]                           # [N,4,3,448]
E15 = (E15 * P15[:, :, None, None]).reshape(len(ids), -1)
del z, z15
F = fold.reindex(ids).values
comps = {f"w{k}": (E3[:, k], P3) for k in range(E3.shape[1])}
comps["v15"] = (E15, P15.any(1))

def stdz(X, P):
    mu = X[isT & P].mean(0); sd = X[isT & P].std(0) + 1e-6
    return ((X - mu) / sd) * P[:, None]

GRID = {"reg": [1, 10, 30, 1e2, 3e2, 1e3, 1e4], "cls": [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}
def model(kind, h):
    return Ridge(alpha=h) if kind == "reg" else LogisticRegression(C=h, class_weight="balanced", max_iter=3000)
def pred(m, X):
    return m.predict(X) if isinstance(m, Ridge) else m.decision_function(X)
def metric(kind, y, s):
    return -np.mean(np.abs(y - s)) if kind == "reg" else auc(y, s)

def comp_scores(kind, y):
    """per component: OOF over HP grid -> best HP by OOF -> refit all train -> val."""
    oof, val, sel = {}, {}, {}
    Tidx = np.where(isT)[0]; Vidx = np.where(~isT)[0]
    for c, (X, P) in comps.items():
        Xs = stdz(X, P); best = None
        for h in GRID[kind]:
            o = np.zeros(len(Tidx))
            for f in range(5):
                tr, te = Tidx[F[Tidx] != f], Tidx[F[Tidx] == f]
                o[F[Tidx] == f] = pred(model(kind, h).fit(Xs[tr], y[tr]), Xs[te])
            m = metric(kind, y[Tidx], o)
            if best is None or m > best[0]: best = (m, h, o)
        sel[c] = dict(hp=best[1], oof=best[0]); oof[c] = best[2]
        val[c] = pred(model(kind, best[1]).fit(Xs[Tidx], y[Tidx]), Xs[Vidx])
        log(f"  {kind} {c}: hp {best[1]} OOF {best[0]:.4f}")
    return pd.DataFrame(oof), pd.DataFrame(val), sel

def stack(kind, y, O, V):
    Tidx = np.where(isT)[0]; yT = y[Tidx]
    mk = (lambda: Ridge(alpha=1.0)) if kind == "reg" else (lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    o = np.zeros(len(Tidx))
    for f in range(5):
        tr, te = F[Tidx] != f, F[Tidx] == f
        sc = StandardScaler().fit(O.values[tr]); o[te] = pred(mk().fit(sc.transform(O.values[tr]), yT[tr]), sc.transform(O.values[te]))
    sc = StandardScaler().fit(O.values); m = mk().fit(sc.transform(O.values), yT)
    return o, pred(m, sc.transform(V.values)), dict(zip(O.columns, np.round(m.coef_.ravel(), 4).tolist()))

res = {}; info = {}
t0 = time.time()
yhu = lab.hu.values.astype(float)
O, V, sel = comp_scores("reg", yhu); o, v, coef = stack("reg", yhu, O, V)
res["hu_reg"] = (o, v); info["hu_reg"] = dict(components=sel, stack_coef=coef, oof_mae=float(np.mean(np.abs(yhu[isT] - o))))
log(f"hu regression stack OOF MAE {info['hu_reg']['oof_mae']:.2f}  ({time.time() - t0:.0f}s)")
O.assign(study_id=np.array(ids)[isT], stack=o).to_csv(OUT + "oof_hu_reg.csv", index=False)
V.assign(study_id=np.array(ids)[~isT], stack=v).to_csv(OUT + "val_hu_reg.csv", index=False)
for tgt in ("y110", "y160"):
    y = lab[tgt].values.astype(float)
    O, V, sel = comp_scores("cls", y); o, v, coef = stack("cls", y, O, V)
    oauc = auc(y[isT], o); rauc = auc(y[isT], -res["hu_reg"][0])
    info[tgt] = dict(components=sel, stack_coef=coef, oof_auc_cls_stack=oauc, oof_auc_neg_hu_reg=rauc,
                     chosen="cls_stack" if oauc >= rauc else "neg_hu_reg")
    log(f"{tgt}: OOF cls stack {oauc:.4f} | -HU regression {rauc:.4f} -> {info[tgt]['chosen']}  ({time.time() - t0:.0f}s)")
    O.assign(study_id=np.array(ids)[isT], stack=o).to_csv(OUT + f"oof_{tgt}.csv", index=False)
    V.assign(study_id=np.array(ids)[~isT], stack=v).to_csv(OUT + f"val_{tgt}.csv", index=False)
json.dump(info, open(OUT + "model_info.json", "w"), indent=1)
log("MODEL_DONE")
