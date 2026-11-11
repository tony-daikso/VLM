"""Option A (A6000, radar env, CPU only): does Pillar-0 add to OWL on Merlin?  train = 5-fold OOF, val = report only,
test never read (no test features exist; every input is filtered to train/val first).

  cls14   per class: (i) Pillar global, (ii) Pillar organ units (vec64 composites, UNITS below) -> standardised logistic
          regression, C by inner CV on train only (nested: inner leave-one-fold-out over the 4 other outer folds; val refit
          C by 5-fold CV on all train); (iii) stacks of logit(OWL prob) + Pillar OOF scores (LR, C=1). Folds = the folds of
          the formal OWL stack of that class (prep_inputs.py), so every train score is OOF w.r.t. the same partition.
  kidney  per-kidney hydronephrosis and
  vert    per-vertebra (T8-L5) fracture: the peer's unit labels / folds / probe types (report_gen/m4 improve/unit_probe.py
          run_task: torch logistic vs small MLP chosen by train OOF AUROC, fixed hyper-parameters) on Pillar vec64 / vec16 /
          vec64+vec16 of that kidney / vertebra and on Pillar global (weak baseline); OWL unit = peer probe_scores.csv;
          stacks (unit-level LR on logits, OOF with the same folds) incl. M4 deterministic measurements.
  cheap   attributes among report positives: steatosis grade, ascites amount, atherosclerosis grade (mild vs moderate+;
          AUROC) and AAA diameter / spleen length regression (ridge; Spearman, MAE) vs the segmentation measurement.
  report  aggregate tables (bootstrap 2,000; stratified for case AUROC, case-cluster for unit metrics) -> out/*.md
usage: optA.py <cmd> [--dev] [--workers 6]   (W = ~/Desktop/Tony/VLM/OWL/pillar_optA; packed Pillar shards in W/packed)
--dev: no Pillar val features yet -> train fold 4 of each task plays 'val' (pipeline check only, numbers not reported).
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
W = os.path.expanduser("~/Desktop/Tony/VLM/OWL/pillar_optA/")
PEER = os.path.expanduser("~/Desktop/Tony/VLM/OWL/report_gen/m4/code_f/")
D14 = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis", "abdominal_aortic_aneurysm",
       "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis", "ascites", "free_air", "gallstones",
       "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊已切除", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
# TS 'total' ids: 1 spleen 2 kidney_R 3 kidney_L 4 gallbladder 5 liver 6 stomach 7 pancreas 18 small_bowel 19 duodenum
# 20 colon 21 bladder 25 sacrum 26 S1 27..36 L5..T8 52 aorta 64 portal/splenic vein 65/66 iliac artery L/R 77/78 hip L/R
# 89 iliopsoas_R 92-115 ribs.  One group = exact w-weighted composite of its labels; groups are concatenated.
VERT = {"T8": 36, "T9": 35, "T10": 34, "T11": 33, "T12": 32, "L1": 31, "L2": 30, "L3": 29, "L4": 28, "L5": 27}
LEVELS = list(VERT)
UNITS = {"hepatic_steatosis": [[5], [1]], "surgically_absent_gallbladder": [[4], [5], [19]], "splenomegaly": [[1], [5]],
         "hydronephrosis": [[2], [3], [21]], "abdominal_aortic_aneurysm": [[52], [65, 66]],
         "atherosclerosis": [[52], [65, 66]], "osteopenia": [[31], [30], [29], [28]],
         "bowel_obstruction": [[18], [20], [6], [19]], "appendicitis": [[20], [18], [89]],
         "ascites": [[5], [1], [18], [20], [21]], "free_air": [[5], [6], [20], [18]], "gallstones": [[4], [5]],
         "biliary_ductal_dilation": [[5], [4], [7], [19], [64]],
         "fracture": ["VERT_MEANMAX", list(range(92, 116)), [25, 26, 77, 78]]}
CS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
B = 2000


# ------------------------------------------------------------------ Pillar features
class PF:
    def __init__(self):
        sh = sorted(glob.glob(W + "packed/shard_*.npz"))
        assert sh, "no packed shards"
        zs = [np.load(s) for s in sh]
        self.ids = np.concatenate([z["ids"] for z in zs])
        _, first = np.unique(self.ids, return_index=True)
        keep = np.sort(first)
        self.ids = self.ids[keep]
        self.idx = {c: i for i, c in enumerate(self.ids)}
        for k in ("global", "vec64", "vec16", "w64", "w16"):
            setattr(self, k, np.concatenate([z[k] for z in zs])[keep])
        print(f"Pillar: {len(self.ids)} cases from {len(sh)} shards", flush=True)

    def rows(self, ids):
        r = np.array([self.idx.get(c, -1) for c in ids])
        return r, r >= 0

    def comp(self, r, labels, res="64"):
        """[len(r), 384] w-weighted composite of `labels` (zeros where absent / r<0)."""
        V, Wt = getattr(self, "vec" + res), getattr(self, "w" + res)
        rr = np.where(r >= 0, r, 0)
        num = np.zeros((len(r), 384), np.float32)
        den = np.zeros(len(r), np.float32)
        for lb in labels:
            w = Wt[rr, lb].astype(np.float32)
            num += w[:, None] * V[rr, lb].astype(np.float32)
            den += w
        out = np.where(den[:, None] > 0, num / np.maximum(den, 1e-9)[:, None], 0)
        out[r < 0] = 0
        return out.astype(np.float32), den

    def unit_X(self, r, d):
        parts = []
        for g in UNITS[d]:
            if g == "VERT_MEANMAX":
                V = np.stack([self.comp(r, [VERT[lv]])[0] for lv in LEVELS], 1)          # [n,10,384]
                pres = np.stack([self.comp(r, [VERT[lv]])[1] > 0 for lv in LEVELS], 1)
                cnt = np.maximum(pres.sum(1), 1)[:, None]
                parts.append((V * pres[..., None]).sum(1) / cnt)
                parts.append(np.where(pres[..., None], V, -1e4).max(1) * (pres.any(1)[:, None]))
            else:
                parts.append(self.comp(r, g)[0])
        return np.concatenate(parts, 1)

    def global_X(self, r):
        X = getattr(self, "global")[np.where(r >= 0, r, 0)].astype(np.float32)
        X[r < 0] = 0
        return X


# ------------------------------------------------------------------ metrics / bootstrap
def auc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1 = y.sum()
    if n1 == 0 or n1 == len(y):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def boot_case(y, S, pairs=(), seed=0, b=B):
    """stratified bootstrap of case AUROC. S: {name: scores}; pairs: [(a, b)] -> a - b paired."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pt = {k: auc(y, v) for k, v in S.items()}
    bs = {k: [] for k in S}
    bd = {p: [] for p in pairs}
    for _ in range(b):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a = {k: auc(y[i], v[i]) for k, v in S.items()}
        for k in S:
            bs[k].append(a[k])
        for p in pairs:
            bd[p].append(a[p[0]] - a[p[1]])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {k: {"auroc": pt[k], "ci": ci(bs[k])} for k in S}
    for p in pairs:
        out[f"{p[0]}-{p[1]}"] = {"delta": pt[p[0]] - pt[p[1]], "ci": ci(bd[p]),
                                 "p_le0": float(np.mean(np.array(bd[p]) <= 0))}
    return out


def boot_cluster(n_cases, metric_fns, pairs=(), seed=0, b=B):
    """case-cluster bootstrap; metric_fns: {name: f(case_index_array) -> float}."""
    rng = np.random.default_rng(seed)
    allc = np.arange(n_cases)
    pt = {k: f(allc) for k, f in metric_fns.items()}
    bs = {k: [] for k in metric_fns}
    bd = {p: [] for p in pairs}
    for _ in range(b):
        i = rng.choice(n_cases, n_cases)
        a = {k: f(i) for k, f in metric_fns.items()}
        for k in a:
            bs[k].append(a[k])
        for p in pairs:
            bd[p].append(a[p[0]] - a[p[1]])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {k: {"value": pt[k], "ci": ci(bs[k])} for k in metric_fns}
    for p in pairs:
        out[f"{p[0]}-{p[1]}"] = {"delta": pt[p[0]] - pt[p[1]], "ci": ci(bd[p]),
                                 "p_le0": float(np.mean(np.array(bd[p]) <= 0))}
    return out


# ------------------------------------------------------------------ nested logistic / ridge
def _fit_path(X, y, Cs, kind="lr"):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    ms = []
    if kind == "lr":
        m = LogisticRegression(max_iter=400, tol=1e-4, warm_start=True)
        for C in Cs:                                            # ascending C, warm start along the path
            m.set_params(C=C)
            m.fit(Xs, y)
            ms.append((m.coef_.copy(), m.intercept_.copy()))
    else:
        for a in Cs:
            m = Ridge(alpha=a).fit(Xs, y)
            ms.append((m.coef_.copy()[None], np.atleast_1d(m.intercept_).copy()))
    return sc, ms


def _pred(sc, cm, X):
    return (sc.transform(X) @ cm[0].T).ravel() + cm[1][0]


def nested(X, y, fold, is_tr, is_va, Cs=CS, kind="lr"):
    """X all rows; y (nan = unlabelled); fold (train rows 0..4). Train rows: OOF with C chosen by inner CV over the other
    4 folds; val rows: full-train refit with C chosen by 5-fold CV on train. Score = decision function."""
    from scipy.stats import spearmanr
    crit = (lambda yy, ss: auc(yy, ss)) if kind == "lr" else (lambda yy, ss: -np.mean(np.abs(yy - ss)))
    lab = ~np.isnan(y)
    tr = is_tr & lab
    out = np.full(len(y), np.nan)
    K = sorted(set(fold[tr]))
    chosen = []

    def cv_pick(rows_fold_set):
        pred = {c: np.full(len(y), np.nan) for c in range(len(Cs))}
        for j in rows_fold_set:
            a = tr & np.isin(fold, [f for f in rows_fold_set if f != j])
            b = tr & (fold == j)
            if a.sum() < 5 or b.sum() == 0:
                continue
            sc, ms = _fit_path(X[a], y[a], Cs, kind)
            for c, cm in enumerate(ms):
                pred[c][b] = _pred(sc, cm, X[b])
        m = tr & np.isin(fold, rows_fold_set)
        perf = [crit(y[m], pred[c][m]) for c in range(len(Cs))]
        perf = [p if np.isfinite(p) else np.nan for p in perf]
        return (int(np.nanargmax(perf)) if not np.all(np.isnan(perf)) else len(Cs) // 2), perf, pred
    for k in K:
        inner = [f for f in K if f != k]
        c, _, _ = cv_pick(inner)
        chosen.append(Cs[c])
        a = tr & (fold != k)
        sc, ms = _fit_path(X[a], y[a], [Cs[c]] if kind != "lr" else Cs[:c + 1], kind)
        b = is_tr & (fold == k)
        out[b] = _pred(sc, ms[-1], X[b])
    c, perf, _ = cv_pick(K)
    sc, ms = _fit_path(X[tr], y[tr], [Cs[c]] if kind != "lr" else Cs[:c + 1], kind)
    out[is_va] = _pred(sc, ms[-1], X[is_va])
    return out, {"C_outer": chosen, "C_refit": Cs[c], "cv_perf": [float(p) for p in perf]}


def stack_lr(Z, y, fold, is_tr, is_va, C=1.0):
    """low-dim stack: LR on Z (n, p) (nan -> train median); train OOF with folds, val from all-train fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = Z.copy()
    tr = is_tr & ~np.isnan(y)
    med = np.nanmedian(Z[tr], 0)
    Z = np.where(np.isnan(Z), med, Z)
    out = np.full(len(y), np.nan)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))  # noqa: E731
    for k in sorted(set(fold[tr])):
        a = tr & (fold != k)
        out[is_tr & (fold == k)] = mk().fit(Z[a], y[a]).decision_function(Z[is_tr & (fold == k)])
    m = mk().fit(Z[tr], y[tr])
    out[is_va] = m.decision_function(Z[is_va])
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ cls14
def _cls_task(arg):
    d, fs, dev = arg
    import threadpoolctl
    with threadpoolctl.threadpool_limits(1):
        C = pd.read_csv(W + "in/cases.csv")
        r, have = G_PF.rows(C.study_id.values)
        y = C[f"y_{d}"].values.astype(float)
        fold = C[f"fold_{d}"].values
        is_tr = (C.split == "train").values & have
        is_va = (C.split == "val").values & have
        if dev:
            is_va = is_tr & (fold == 4)
            is_tr = is_tr & (fold != 4)
            fold = np.where(is_tr, fold, -1)
        X = G_PF.global_X(r) if fs == "pg" else G_PF.unit_X(r, d)
        t0 = time.time()
        s, info = nested(X, y, fold, is_tr, is_va)
        info.update(dim=int(X.shape[1]), sec=round(time.time() - t0, 1),
                    train_oof_auroc=auc(y[is_tr], s[is_tr]))
        return d, fs, s, info


def cmd_cls14(a):
    global G_PF
    G_PF = PF()
    C = pd.read_csv(W + "in/cases.csv")
    jobs = [(d, fs, a.dev) for d in (a.only.split(",") if a.only else D14) for fs in ("pg", "pu")]
    jobs.sort(key=lambda j: -(len(UNITS[j[0]]) if j[1] == "pu" else 3))
    import multiprocessing as mp
    res = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for d, fs, s, info in pool.imap_unordered(_cls_task, jobs):
            res[(d, fs)] = (s, info)
            print(f"[{time.strftime('%H:%M')}] {d} {fs} dim {info['dim']} {info['sec']}s C_refit {info['C_refit']} "
                  f"train OOF {info['train_oof_auroc']:.4f}", flush=True)
    tag = "_dev" if a.dev else ""
    S = C[["study_id", "split"]].copy()
    meta = {}
    for d in D14:
        if (d, "pg") not in res:
            continue
        S[f"{d}__pg"], S[f"{d}__pu"] = res[(d, "pg")][0], res[(d, "pu")][0]
        meta[d] = {"pg": res[(d, "pg")][1], "pu": res[(d, "pu")][1]}
    S.to_csv(W + f"out/cls14_pillar_scores{tag}.csv", index=False)
    json.dump(meta, open(W + f"out/cls14_pillar_info{tag}.json", "w"), indent=1)


def cls14_eval(dev=False):
    tag = "_dev" if dev else ""
    C = pd.read_csv(W + "in/cases.csv")
    S = pd.read_csv(W + f"out/cls14_pillar_scores{tag}.csv")
    assert (S.study_id.values == C.study_id.values).all()
    out = {}
    for d in D14:
        if f"{d}__pg" not in S:
            continue
        y = C[f"y_{d}"].values.astype(float)
        fold = C[f"fold_{d}"].values
        pg, pu = S[f"{d}__pg"].values, S[f"{d}__pu"].values
        have = ~np.isnan(pg) & ~np.isnan(pu)
        is_tr = (C.split == "train").values & have
        is_va = (C.split == "val").values & have
        if dev:
            is_va = is_tr & (fold == 4)
            is_tr = is_tr & (fold != 4)
            fold = np.where(is_tr, fold, -1)
        ow = logit(C[f"owl_{d}"].values)
        st = stack_lr(np.c_[ow, pg, pu], y, fold, is_tr, is_va)
        stg = stack_lr(np.c_[ow, pg], y, fold, is_tr, is_va)
        stu = stack_lr(np.c_[ow, pu], y, fold, is_tr, is_va)
        sc = {"owl": ow, "pg": pg, "pu": pu, "st": st, "st_g": stg, "st_u": stu}
        ev = is_va & ~np.isnan(y) & ~np.isnan(ow)
        tr = is_tr & ~np.isnan(y) & ~np.isnan(ow)
        o = {"n_val": int(ev.sum()), "pos_val": int(y[ev].sum()), "n_train": int(tr.sum()), "pos_train": int(y[tr].sum()),
             "n_val_no_pillar": int(((C.split == "val").values & ~have & ~np.isnan(y)).sum()),
             "train_oof": {k: auc(y[tr], v[tr]) for k, v in sc.items()}}
        o["val"] = boot_case(y[ev], {k: v[ev] for k, v in sc.items()},
                             pairs=[("st", "owl"), ("st_g", "owl"), ("st_u", "owl"), ("pu", "pg")], seed=zlib.crc32(d.encode()))
        out[d] = o
        print(d, {k: round(v["auroc"], 4) for k, v in o["val"].items() if "auroc" in v},
              "st-owl", round(o["val"]["st-owl"]["delta"], 4), np.round(o["val"]["st-owl"]["ci"], 4), flush=True)
    json.dump(out, open(W + f"out/cls14_metrics{tag}.json", "w"), indent=1)
    return out


# ------------------------------------------------------------------ unit attributes (peer definitions)
def _peer():
    sys.path.insert(0, PEER + "improve")
    sys.path.insert(1, PEER)
    import unit_probe as up
    return up


def unit_inputs(task, dev):
    U = np.load(W + "in/units.npz", allow_pickle=True)
    ids, split = U["ids"], U["split"].astype(str)
    fold = (U["fh"] if task == "kidney" else U["ff"]).copy()
    Y = U["UK"] if task == "kidney" else U["UV"]
    S_owl = U["kid_owl"] if task == "kidney" else U["vert_owl"]
    mask = U["kid_mask"] if task == "kidney" else U["vert_mask"]
    ycase = U["ycase"][:, 0 if task == "kidney" else 1]
    if dev:
        tr = split == "train"
        split = np.where(tr & (fold == 4), "val", np.where(tr, "train", "excluded"))
        dev_f = np.array([(zlib.crc32(c.encode()) // 7) % 5 for c in ids])   # dev only: 5 new folds over the rest
        fold = np.where(split == "train", dev_f, -1)
    return ids, split, fold, Y, S_owl, mask, ycase


def unit_feats(pf, r, task, fs):
    labs = [[2], [3]] if task == "kidney" else [[VERT[lv]] for lv in LEVELS]   # kidney: (right, left) = TS 2, 3
    if fs == "pglob":
        g = pf.global_X(r).astype(np.float16)
        return np.repeat(g[:, None, None, :], len(labs), 1)
    toks = []
    for res in ({"p64": ["64"], "p16": ["16"], "p64_16": ["64", "16"]}[fs]):
        toks.append(np.stack([pf.comp(r, lb, res)[0] for lb in labs], 1))      # [N, nU, 384]
    return np.stack(toks, 2).astype(np.float16)                                  # [N, nU, T, 384]


def m4_unit_feats(ids, task):
    A = pd.read_csv(W + "in/attrs.csv", low_memory=False).set_index("study_id")
    A = A.reindex(ids)
    if task == "kidney":
        cols = ["flcc_ml", "sinus_fl_ml", "lowhu_frac", "lowhu_ml", "flcc_ap_mm", "flcc_tr_mm", "shell_fl", "cyst_ml", "ml"]
        Z = np.stack([np.stack([A[f"kidney_{s}_{c}"].values.astype(float) for c in cols], -1) for s in ("r", "l")], 1)
        prim = Z[..., 0]                                                         # flcc_ml (fluid proxy)
    else:
        cols = ["g15_loss", "g15_loss_a", "g15_loss_m", "g15_loss_p", "g15_wedge", "g15_crush", "g15_bicon", "ml_nb",
                "zext_nb", "g15_hu_body"]
        Z = np.stack([np.stack([A[f"{lv}_{c}"].values.astype(float) for c in cols], -1) for lv in LEVELS], 1)
        prim = Z[..., 0]                                                         # height loss vs neighbours
    signed = np.sign(Z) * np.log1p(np.abs(Z))
    return signed, prim, cols


def unit_stack(Zu, Y, fold, split, present, C=1.0):
    """unit-level LR on Zu [N, nU, p] (+ unit one-hot), rows = present units; OOF on train folds, val refit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    N, nU, p = Zu.shape
    rn, ru = np.where(present)
    Z = np.c_[Zu[rn, ru], np.eye(nU)[ru]]
    tr = split[rn] == "train"
    va = split[rn] == "val"
    y = Y[rn, ru]
    med = np.nanmedian(Z[tr], 0)
    miss = np.isnan(Z[:, :p]).astype(float)
    Z = np.c_[np.where(np.isnan(Z), med, Z), miss[:, miss[tr].std(0) > 0]]
    lab = ~np.isnan(y)
    f = fold[rn]
    s = np.full(len(rn), np.nan)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))  # noqa: E731
    for k in range(5):
        a = tr & lab & (f != k)
        s[tr & (f == k)] = mk().fit(Z[a], y[a]).predict_proba(Z[tr & (f == k)])[:, 1]
    s[va] = mk().fit(Z[tr & lab], y[tr & lab]).predict_proba(Z[va])[:, 1]
    out = np.full((N, nU), np.nan)
    out[rn, ru] = s
    return out


def cmd_units(a, task):
    import torch
    torch.set_num_threads(a.workers)
    up = _peer()
    pf = PF()
    ids, split, fold, Y, S_owl, mask, ycase = unit_inputs(task, a.dev)
    r, have = pf.rows(ids)
    present = mask & have[:, None] & ~np.isnan(S_owl)          # same unit rows for every method
    tag = "_dev" if a.dev else ""
    logf = open(W + f"out/{task}_fit{tag}.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"[{time.strftime('%m-%d %H:%M')}] {task}: {len(ids)} cases, Pillar present {int(have.sum())}, "
        f"units used train {int(present[split == 'train'].sum())} val {int(present[split == 'val'].sum())}")
    scores = {"owl": np.where(present, S_owl, np.nan)}
    info = {}
    for fs in a.fsets.split(","):
        F = unit_feats(pf, r, task, fs)
        MK = np.repeat(present[..., None], F.shape[2], 2)
        sc, kind, oof_auc, _ = up.run_task(f"{task}/{fs}", F, MK, Y, fold, split, ["logistic", "mlp"], log)
        scores[fs] = np.where(present, sc, np.nan)
        info[fs] = {"model": kind, "train_oof_auroc": oof_auc}
        del F
    Zm, prim, cols = m4_unit_feats(ids, task)
    scores["m4_raw"] = np.where(present, prim, np.nan)
    scores["m4"] = unit_stack(Zm, Y, fold, split, present)
    L = lambda k: logit(scores[k])[..., None]                  # noqa: E731
    best_p = "p64_16" if "p64_16" in scores else [k for k in scores if k.startswith("p")][0]
    scores["st_owl_p"] = unit_stack(np.concatenate([L("owl"), L(best_p)], -1), Y, fold, split, present)
    scores["st_owl_m4"] = unit_stack(np.concatenate([L("owl"), Zm], -1), Y, fold, split, present)
    scores["st_owl_p_m4"] = unit_stack(np.concatenate([L("owl"), L(best_p), Zm], -1), Y, fold, split, present)
    scores["st_p_m4"] = unit_stack(np.concatenate([L(best_p), Zm], -1), Y, fold, split, present)
    info["_stack_pillar_input"] = best_p
    info["_m4_cols"] = cols
    np.savez(W + f"out/{task}_unit_scores{tag}.npz", ids=ids, split=split, Y=Y, ycase=ycase, present=present,
             **{f"s_{k}": v for k, v in scores.items()})
    json.dump(info, open(W + f"out/{task}_unit_info{tag}.json", "w"), indent=1)
    unit_eval(task, a.dev)


def f1_thr(y, s):
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    best = (0, 0.5)
    for t in np.unique(np.round(s, 2)):
        p = s >= t
        f1 = 2 * (p & (y == 1)).sum() / max(p.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def unit_eval(task, dev=False):
    tag = "_dev" if dev else ""
    z = np.load(W + f"out/{task}_unit_scores{tag}.npz", allow_pickle=True)
    split, Y, ycase, present = z["split"], z["Y"], z["ycase"], z["present"]
    S = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
    tr, va = split == "train", split == "val"
    res = {"n_val_cases": int(va.sum()), "methods": list(S)}
    # rank-only scores (m4_raw) need a train threshold too -> f1 on train like the others
    thr = {k: f1_thr(Y[tr].ravel(), v[tr].ravel()) for k, v in S.items() if k != "m4_raw"}
    res["t_f1_train"] = thr
    vi = np.where(va & present.any(1))[0]                       # val cases with at least one evaluable unit
    res["n_val_cases"] = int(len(vi))
    Yv, Sv, Pv = Y[vi], {k: v[vi] for k, v in S.items()}, present[vi]
    yc = ycase[vi]

    def unit_auc(k):
        return lambda i: auc(Yv[i][Pv[i]], Sv[k][i][Pv[i]])
    fns = {f"auc_{k}": unit_auc(k) for k in S}
    if task == "kidney":
        # side: hydronephrosis cases with both kidney labels and both scores (peer definition)
        side_c = np.where((yc == 1) & ~np.isnan(Yv).any(1) & Pv.all(1))[0]
        uni = side_c[(Yv[side_c] == 1).sum(1) == 1]

        def exact(k):
            def f(i):
                ii = i[np.isin(i, side_c)]
                if len(ii) == 0:
                    return np.nan
                s = Sv[k][ii]
                t = Yv[ii] == 1
                pr = s >= thr[k]
                arg = np.stack([s[:, 0] >= s[:, 1], s[:, 1] > s[:, 0]], 1)
                pr = np.where(pr.any(1, keepdims=True), pr, arg)
                return float((pr == t).all(1).mean())
            return f

        def uni_acc(k):
            def f(i):
                ii = i[np.isin(i, uni)]
                if len(ii) == 0:
                    return np.nan
                return float((np.argmax(Sv[k][ii], 1) == np.argmax(Yv[ii], 1)).mean())
            return f
        for k in S:
            if k != "m4_raw":
                fns[f"side_exact_{k}"] = exact(k)
            fns[f"uni_argmax_{k}"] = uni_acc(k)
        res["n_side_cases"], res["n_unilateral"] = int(len(side_c)), int(len(uni))
        res["bilateral_share"] = float(np.mean((Yv[side_c] == 1).all(1))) if len(side_c) else float("nan")
    else:
        lev_c = np.where((yc == 1) & (np.nan_to_num(Yv, nan=0) == 1).any(1))[0]

        def top1(k):
            def f(i):
                ii = i[np.isin(i, lev_c)]
                ii = ii[~np.isnan(Sv[k][ii]).all(1)]
                if len(ii) == 0:
                    return np.nan
                t = np.nanargmax(np.where(np.isnan(Sv[k][ii]), -np.inf, Sv[k][ii]), 1)
                return float(np.mean(Yv[ii, t] == 1))
            return f
        for k in S:
            fns[f"top1_{k}"] = top1(k)
        res["n_level_cases"] = int(len(lev_c))
        res["chance_top1"] = float(np.mean([np.nansum(Yv[n] == 1) / 10 for n in lev_c])) if len(lev_c) else float("nan")
    # case-level AUROC of the max unit score vs case label
    for k in S:
        fns[f"caseauc_{k}"] = (lambda kk: lambda i: auc(yc[i], np.nanmax(np.where(np.isnan(Sv[kk][i]), -1e9, Sv[kk][i]),
                                                                          1)))(k)
    best_p = json.load(open(W + f"out/{task}_unit_info{tag}.json"))["_stack_pillar_input"]
    pairs = []
    for m in ("auc", "side_exact", "uni_argmax", "top1"):
        for a_, b_ in ((best_p, "owl"), ("st_owl_p", "owl"), ("st_owl_p_m4", "st_owl_m4")):
            if f"{m}_{a_}" in fns and f"{m}_{b_}" in fns:
                pairs.append((f"{m}_{a_}", f"{m}_{b_}"))
    res["val"] = boot_cluster(len(vi), fns, pairs=pairs, seed=7)
    res["train_oof_auc"] = {k: auc(Y[tr][present[tr]], S[k][tr][present[tr]]) for k in S}
    json.dump(res, open(W + f"out/{task}_unit_metrics{tag}.json", "w"), indent=1)
    for k, v in res["val"].items():
        print(task, k, {kk: (np.round(vv, 4).tolist() if not isinstance(vv, str) else vv) for kk, vv in v.items()})
    return res


# ------------------------------------------------------------------ cheap case-level attributes
CHEAP = {"hs_grade": ("hepatic_steatosis", "cls", None), "asc_amount": ("ascites", "cls", None),
         "ath_grade": ("atherosclerosis", "cls", None),
         "aaa_diam_mm": ("abdominal_aortic_aneurysm", "reg", "abd_major_rmax"),
         "sp_len_mm": ("splenomegaly", "reg", "sp_cc_main_cm")}


def cmd_cheap(a):
    from scipy.stats import spearmanr
    pf = PF()
    C = pd.read_csv(W + "in/cases.csv")
    A = pd.read_csv(W + "in/attrs.csv", low_memory=False).set_index("study_id").reindex(C.study_id).reset_index()
    r, have = pf.rows(C.study_id.values)
    out = {}
    rng = np.random.default_rng(3)
    for t, (d, kind, meas) in CHEAP.items():
        y = A[t].values.astype(float)
        fold = C[f"fold_{d}"].values
        is_tr = (C.split == "train").values & have & ~np.isnan(y)
        is_va = (C.split == "val").values & have & ~np.isnan(y)
        if a.dev:
            is_va, is_tr = is_tr & (fold == 4), is_tr & (fold != 4)
        fold = np.where(is_tr, fold, -1)
        y = np.where(is_tr | is_va, y, np.nan)
        if is_tr.sum() < 30 or is_va.sum() < 5:
            print(t, "skipped: too few labelled cases", int(is_tr.sum()), int(is_va.sum()), flush=True)
            continue
        ow = logit(C[f"owl_{d}"].values)
        Cs = CS if kind == "cls" else [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
        pg, _ = nested(pf.global_X(r), y, fold, is_tr, is_va, Cs, "lr" if kind == "cls" else "ridge")
        pu, _ = nested(pf.unit_X(r, d), y, fold, is_tr, is_va, Cs, "lr" if kind == "cls" else "ridge")
        S = {"owl": ow, "pg": pg, "pu": pu}
        if t == "hs_grade":
            S["meas"] = -(A["liver_hu_med"].values - A["spleen_hu_med"].values)        # lower liver-spleen HU = fattier
        if meas:
            m = A[meas].values.astype(float) * (10 if meas.endswith("_cm") else 1)
            S["meas"] = m
        ev = is_va
        o = {"n_train": int(is_tr.sum()), "n_val": int(ev.sum())}
        if kind == "cls":
            o["pos_train"], o["pos_val"] = int(y[is_tr].sum()), int(y[ev].sum())
            Z = np.c_[ow, pu] if "meas" not in S else np.c_[ow, pu, S["meas"]]
            S["st"] = stack_lr(Z, y, fold, is_tr, is_va)
            if "meas" in S:
                S["st_owl_meas"] = stack_lr(np.c_[ow, S["meas"]], y, fold, is_tr, is_va)
            o["val"] = boot_case(y[ev], {k: v[ev] for k, v in S.items()}, pairs=[("pu", "owl"), ("st", "owl")]
                                 + ([("st", "st_owl_meas")] if "meas" in S else []), seed=11)
        else:
            from sklearn.linear_model import LinearRegression
            mt = is_tr & ~np.isnan(S["meas"])
            lr = LinearRegression().fit(S["meas"][mt, None], y[mt])
            S["meas_cal"] = np.where(np.isnan(S["meas"]), np.nan, lr.predict(np.nan_to_num(S["meas"])[:, None]))
            mf = np.where(np.isnan(S["meas"]), np.nanmedian(S["meas"][is_tr]), S["meas"])
            S["st"], _ = nested(np.c_[mf, pu], y, fold, is_tr, is_va, Cs, "ridge")
            fns = {}
            vi = np.where(ev)[0]
            for k in ("pg", "pu", "meas_cal", "st"):
                fns[f"mae_{k}"] = (lambda kk: lambda i: float(np.nanmean(np.abs(S[kk][vi][i] - y[vi][i]))))(k)
                fns[f"rho_{k}"] = (lambda kk: lambda i: float(spearmanr(S[kk][vi][i], y[vi][i],
                                                                       nan_policy="omit").correlation))(k)
            o["val"] = boot_cluster(len(vi), fns, pairs=[("mae_st", "mae_meas_cal"), ("rho_st", "rho_meas_cal"),
                                                         ("mae_pu", "mae_meas_cal")], seed=13)
        out[t] = o
        print(t, json.dumps({k: v for k, v in o.items() if k != "val"}),
              {k: np.round(v.get("auroc", v.get("value", v.get("delta"))), 3) for k, v in o["val"].items()}, flush=True)
    json.dump(out, open(W + f"out/cheap_metrics{'_dev' if a.dev else ''}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cls14", "cls14_eval", "kidney", "vert", "unit_eval", "cheap"])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--fsets", default="pglob,p64,p16,p64_16")
    a = ap.parse_args()
    os.makedirs(W + "out", exist_ok=True)
    if a.cmd == "cls14":
        cmd_cls14(a)
        cls14_eval(a.dev)
    elif a.cmd == "cls14_eval":
        cls14_eval(a.dev)
    elif a.cmd in ("kidney", "vert"):
        cmd_units(a, a.cmd)
    elif a.cmd == "unit_eval":
        unit_eval("kidney", a.dev)
        unit_eval("vert", a.dev)
    elif a.cmd == "cheap":
        cmd_cheap(a)


if __name__ == "__main__":
    main()
