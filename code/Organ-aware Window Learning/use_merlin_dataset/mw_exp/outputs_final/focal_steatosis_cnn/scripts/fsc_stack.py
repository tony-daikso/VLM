"""Focal-steatosis CNN: stack the CNN logit with the current best (Exp N 'REF＋全部單元頭', val .676).
CNN config chosen by seed-averaged train 5-fold OOF AUROC only. Stack candidates (all selected on train OOF):
  CUR       = REF (E3_late concat_diff + HU + FL) + all Exp N unit heads, 6 candidates (rebuilt, = Exp N)
  CUR＋CNN   = same design + CNN logit, 6 candidates
  REF＋CNN   = REF + CNN logit
  blend     = LR(C=1) on [CUR score, CNN logit] (5-fold CV on train OOF scores; full fit -> val)
val once; paired bootstrap vs CUR; age-matched AUROC; sens@90%spec. Age/sex never model inputs. Mac CPU only.
usage: fsc_stack.py STACKIN RUNS OUT [--nthreads 4]"""
import sys, os, json, glob, numpy as np, pandas as pd
IN, RUNS, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/", sys.argv[3].rstrip("/") + "/"
sys.path.insert(0, IN + "scripts")
import hs_split_stack as hs
from ib_core import boot_delta, spearman_neg
os.makedirs(OUT, exist_ok=True)
T, V = hs.load(IN + "stackdata", IN + "oof", "y_focal", "concat_diff"); logit = "E3_late_concat_diff"
uo = pd.read_csv(IN + "unit_out/oof_train.csv"); uv = pd.read_csv(IN + "unit_out/val_scores.csv")
UH = [c for c in uo.columns if c not in ("study_id", "fold")]
T = T.merge(uo[["study_id"] + UH].rename(columns={c: "fn_" + c for c in UH}), on="study_id", how="left")
V = V.merge(uv[["study_id"] + UH].rename(columns={c: "fn_" + c for c in UH}), on="study_id", how="left")
UH = ["fn_" + c for c in UH]
y, yv = T.y.values, V.y.values
log = open(OUT + "stack.log", "w")
def P(*s):
    print(*s, flush=True); print(*s, file=log, flush=True)
P(f"train {len(T)} ({y.sum()} pos) val {len(V)} ({yv.sum()} pos)")

# ---- CNN configs: seed-averaged OOF (train) and full-train val
cfg = {}
for d in sorted(glob.glob(RUNS + "*/")):
    c = os.path.basename(d.rstrip("/")); seeds = sorted({f.split("_s")[-1][:-4] for f in glob.glob(d + "pred_*_s*.csv")})
    oo, vf, vk = [], [], []
    for s in seeds:
        fs = [d + f"pred_{k}_s{s}.csv" for k in range(5)]
        if not all(os.path.exists(f) for f in fs + [d + f"pred_full_s{s}.csv"]): continue
        A = pd.concat([pd.read_csv(f) for f in fs])
        oo.append(A[A.split == "train"].set_index("study_id").logit.rename(f"s{s}"))
        vk.append(A[A.split == "val"].groupby("study_id").logit.mean().rename(f"s{s}"))
        vf.append(pd.read_csv(d + f"pred_full_s{s}.csv").set_index("study_id").logit.rename(f"s{s}"))
    if not oo: continue
    O = pd.concat(oo, axis=1); VF = pd.concat(vf, axis=1); VK = pd.concat(vk, axis=1)
    # per-seed standardisation (scale differs between seeds) then mean
    mu, sd = O.mean(), O.std()
    cfg[c] = dict(n=len(oo), oof=((O - mu) / sd).mean(1), val_full=((VF - mu) / sd).mean(1), val_folds=((VK - mu) / sd).mean(1),
                  seed_oof=[hs.auc(T.set_index("study_id").loc[O.index].y.values, O[s].values) for s in O.columns])
for c, r in cfg.items():
    yy = T.set_index("study_id").y
    a = hs.auc(yy.loc[r["oof"].index].values, r["oof"].values)
    r["oof_auc"] = a; P(f"CNN cfg {c}: seeds {r['n']} per-seed OOF {np.round(r['seed_oof'], 4)} seed-avg OOF {a:.4f} (n={len(r['oof'])})")
best = max(cfg, key=lambda c: cfg[c]["oof_auc"]); P(f"CNN SELECTED (train OOF): {best}")
for c, r in cfg.items():
    T[f"cnn_{c}"] = T.study_id.map(r["oof"]); V[f"cnn_{c}"] = V.study_id.map(r["val_full"]); V[f"cnnK_{c}"] = V.study_id.map(r["val_folds"])
P(f"CNN coverage train {T[f'cnn_{best}'].notna().mean():.4f} val {V[f'cnn_{best}'].notna().mean():.4f}")
CN = f"cnn_{best}"


def design(d, cols, med=None):
    X, m0 = hs.design(d, "E3+HU+FL", logit, None if med is None else med[0])
    E = d[cols].astype(float).reset_index(drop=True) if cols else pd.DataFrame(index=range(len(d)))
    m1 = E.median() if med is None else med[1]
    return pd.concat([X.reset_index(drop=True), E.fillna(m1)], axis=1), (m0, m1)


def run(cols, cn):
    s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, md = design(T[tr], cols); Xte, _ = design(T[te], cols, md)
        s[te] = hs.score(hs.CANDS[cn]().fit(Xtr, y[tr]), Xte)
    X, md = design(T, cols); m = hs.CANDS[cn]().fit(X, y); Xv, _ = design(V, cols, md)
    return s, hs.score(m, Xv)


def best_of(cols, nm):
    b = None
    for cn in hs.CANDS:
        o, v = run(cols, cn); a = hs.auc(y, o); P(f"  {nm:<12s} {cn:<24s} OOF {a:.4f}")
        if b is None or a > b[0]: b = (a, cn, o, v)
    return b


S = {}
S["CUR"] = best_of(UH, "CUR")
S["CUR＋CNN"] = best_of(UH + [CN], "CUR＋CNN")
S["REF＋CNN"] = best_of([CN], "REF＋CNN")
# blend: LR on [CUR score, CNN]
from sklearn.linear_model import LogisticRegression
co, cv = S["CUR"][2], S["CUR"][3]
cn_o = T[CN].fillna(T[CN].median()).values; cn_v = V[CN].fillna(T[CN].median()).values
Zo = np.c_[(co - co.mean()) / co.std(), cn_o]; Zv = np.c_[(cv - co.mean()) / co.std(), cn_v]
bo = np.zeros(len(T))
for k in sorted(T.fold.unique()):
    tr, te = T.fold.values != k, T.fold.values == k
    bo[te] = LogisticRegression(C=1.0).fit(Zo[tr], y[tr]).decision_function(Zo[te])
lr = LogisticRegression(C=1.0).fit(Zo, y); P(f"blend coef {lr.coef_.round(3)}")
S["blend(CUR,CNN)"] = (hs.auc(y, bo), "LR_C1", bo, lr.decision_function(Zv))
S["CNN 單獨"] = (hs.auc(y, cn_o), "cnn", cn_o, cn_v)
S["CNN 單獨（5 fold 模型平均→val）"] = (hs.auc(y, cn_o), "cnn", cn_o, V[f"cnnK_{best}"].fillna(T[CN].median()).values)
for c in cfg:
    if c != best:
        S[f"CNN {c} 單獨"] = (hs.auc(y, T[f'cnn_{c}'].fillna(T[f'cnn_{c}'].median()).values), "cnn", T[f'cnn_{c}'].fillna(T[f'cnn_{c}'].median()).values, V[f'cnn_{c}'].fillna(T[f'cnn_{c}'].median()).values)
bo_, bv_ = S["CUR"][2], S["CUR"][3]
rows = []
for nm, (a, cn, o, v) in S.items():
    r = dict(model=nm, cand=cn, **hs.metrics(y, o, T.Age, yv, v, V.Age, None if nm == "CUR" else bo_, None if nm == "CUR" else bv_))
    r["spearman_neg_val"] = spearman_neg(yv, v, V.Age.values)
    if nm != "CUR":
        r["d_val"] = boot_delta(yv, bv_, v); r["d_oof"] = boot_delta(y, bo_, o, n=1000); r["d_val_age"] = boot_delta(yv, bv_, v, n=500, age=V.Age.values)
    rows.append(r)
    P(f"{nm:<28s} [{cn}] OOF {r['oof']:.4f} val {r['val']:.4f} {np.round(r['val_ci'], 3)} age {r['oof_age']:.4f}/{r['val_age']:.4f} s90 {r['val_sens90']:.3f} (trthr {r['val_sens90_trthr']:.3f}) sp {r['spearman_neg_val']:+.3f}"
      + (f" | dVal {r['d_val']['mean']:+.4f} {np.round(r['d_val']['ci'], 4)} dOOF {r['d_oof']['mean']:+.4f} {np.round(r['d_oof']['ci'], 4)} dAge {r['d_val_age']['mean']:+.4f} {np.round(r['d_val_age']['ci'], 4)}" if nm != "CUR" else ""))
cands = [r for r in rows if r["model"] in ("CUR", "CUR＋CNN", "blend(CUR,CNN)")]
sel = max(cands, key=lambda r: r["oof"]); P(f"SELECTED (train OOF among CUR / CUR＋CNN / blend): {sel['model']} OOF {sel['oof']:.4f} val {sel['val']:.4f}")
json.dump(dict(cnn_cfg={c: dict(n=r["n"], oof=r["oof_auc"], seed_oof=r["seed_oof"]) for c, r in cfg.items()}, cnn_selected=best,
               results=rows, selected=sel["model"]), open(OUT + "results.json", "w"), indent=1, ensure_ascii=False, default=float)
pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, **{k: v[2] for k, v in S.items()}}).to_csv(OUT + "scores_oof.csv", index=False)
pd.DataFrame({"study_id": V.study_id, "y": yv, **{k: v[3] for k, v in S.items()}}).to_csv(OUT + "scores_val.csv", index=False)
P("FSC_STACK_DONE")
