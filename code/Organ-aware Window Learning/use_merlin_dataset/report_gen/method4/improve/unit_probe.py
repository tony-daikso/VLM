#!/usr/bin/env python3
"""Improvement 2 (agent F, 2026-11-11): unit-level probes on frozen OWL unit tokens -> per-kidney hydronephrosis and
per-vertebra (T8-L5) fracture scores for a structured prompt line.

extract : read ONLY the needed tokens of each train/val npz (m4/features/main) into small fp16 arrays
          kidney [N, 2 sides (right, left), 45 tokens (organ3 kidney_<side> x5 + ru_hydronephrosis k0=right/k1=left x40), 448]
          vert   [N, 10 levels, 12 tokens (vert15 x6 + bone3 x6), 448]
          (~3.5 GB fp16 in RAM at most; never the whole 19 GB; test is not in the jsonl files)
fit     : unit-level labels = build_unit_labels.py (report side / level) + case-level owl label negative -> every unit 0
          (not_mentioned counts as negative, the OWL binarisation). One model per task shared over units (+ unit one-hot).
          TRAIN: 5-fold OOF with the folds of the case-level OOF scores (hydronephrosis: region_units/hydronephrosis/out/
          oof_train.csv; fracture: outputs_fracture/spine_ft/in/labels.csv sitewise outer folds; cases without a fold ->
          crc32 % 5). VAL: model refit on all train. Model choice (logistic vs small MLP) by train OOF AUROC only.
          Val labels (unit_labels_val + labels_disease14 val rows) are read only to REPORT probe AUROC / side / level hits.
writes  : <out>/probe_scores.csv (case_id, split, kid_right, kid_left, T8..L5), probe_metrics.json, unit_cue.json
          (train unit labels, unit thresholds t_f1 from train OOF, pos/neg pools of train OOF scores -> gt-dropout unit
          cues). Holds case ids: stays on A6000, never in git.
usage: python unit_probe.py extract --out improve/probe ; python unit_probe.py fit --out improve/probe
"""
import argparse
import csv
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import DISEASES, owl_binary  # noqa: E402

M4 = Path("/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4")
MW = Path("/home/advantech/Desktop/Tony/VLM/OWL/mw_exp")
LEVELS = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
SIDES = ["right", "left"]
HY, FX = DISEASES.index("hydronephrosis"), DISEASES.index("fracture")


def token_index():
    meta = list(csv.DictReader(open(M4 / "features/token_meta.csv")))
    by = {}
    for r in meta:
        by.setdefault((r["group"], r["unit"]), []).append(int(r["tok"]))
    kid = [by[("organ3", f"kidney_{s}")] + by[("ru_hydronephrosis", k)] for s, k in (("right", "k0"), ("left", "k1"))]
    vert = [by[("vert15", lv)] + by[("bone3", lv)] for lv in LEVELS]
    assert all(len(x) == 45 for x in kid) and all(len(x) == 12 for x in vert)
    return np.array(kid), np.array(vert)


def cases():
    out = []
    for sp in ("train", "val"):
        for line in open(M4 / f"data/{sp}.jsonl"):
            r = json.loads(line)
            assert r["split"] == sp
            out.append((r["case_id"], sp, r["feature_path"]))
    return out


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    kid_i, vert_i = token_index()
    cs = cases()
    N = len(cs)
    K = np.zeros((N, 2, 45, 448), np.float16)
    V = np.zeros((N, 10, 12, 448), np.float16)
    KM = np.zeros((N, 2, 45), bool)
    VM = np.zeros((N, 10, 12), bool)
    t0 = time.time()
    for n, (cid, sp, fp) in enumerate(cs):
        z = np.load(fp)
        tok, m = z["tok"], z["mask"]
        K[n], KM[n] = tok[kid_i], m[kid_i]
        V[n], VM[n] = tok[vert_i], m[vert_i]
        if n % 2000 == 0:
            print(f"{n}/{N} {time.time() - t0:.0f}s", flush=True)
    np.save(Path(a.out) / "kid.npy", K)
    np.save(Path(a.out) / "vert.npy", V)
    np.savez(Path(a.out) / "meta.npz", ids=np.array([c[0] for c in cs]), split=np.array([c[1] for c in cs]), kid_mask=KM,
             vert_mask=VM)
    print(f"done {N} cases in {time.time() - t0:.0f}s; kidney present {KM.any(-1).mean():.3f}, vert present {VM.any(-1).mean():.3f}")


# ------------------------------------------------------------------ labels
def case_labels(ids, split):
    """owl-binarised case labels for hydronephrosis / fracture: train from cue_gt.json, val from labels_disease14 val rows."""
    z = json.load(open(M4 / "cue/cue_gt.json"))["labels_train"]
    val = {}
    for line in open("/home/advantech/Desktop/Tony/VLM/OWL/data/labels_disease14_all.jsonl"):
        r = json.loads(line)
        if r.get("split") != "val":
            continue                                             # test rows skipped unread
        ans = r["answer"] if isinstance(r["answer"], dict) else json.loads(r["answer"])
        val[str(r["study_id"])] = [owl_binary(ans[d]["category"]) for d in DISEASES]
    y = np.full((len(ids), 2), np.nan)
    for n, (cid, sp) in enumerate(zip(ids, split)):
        lab = z.get(cid) if sp == "train" else val.get(cid)
        if lab is not None:
            y[n] = [np.nan if lab[HY] is None else lab[HY], np.nan if lab[FX] is None else lab[FX]]
    return y


def unit_labels(ids, split, ycase):
    """[N,2] kidney (right, left) and [N,10] vertebra labels: 1 / 0 / nan (unknown)."""
    UK = np.full((len(ids), 2), np.nan)
    UV = np.full((len(ids), 10), np.nan)
    ul = {}
    for sp in ("train", "val"):
        for line in open(M4 / f"unit_labels/unit_labels_{sp}.jsonl"):
            r = json.loads(line)
            ul[r["case_id"]] = r["units"]
    for n, cid in enumerate(ids):
        u = ul.get(cid, {})
        for s, side in enumerate(SIDES):
            for d, v in u.get(f"organ3/kidney_{side}", []):
                if d == HY:
                    UK[n, s] = v
        for k, lv in enumerate(LEVELS):
            for d, v in u.get(f"vert15/{lv}", []) + u.get(f"bone3/{lv}", []):
                if d == FX:
                    UV[n, k] = v if np.isnan(UV[n, k]) else max(UV[n, k], v)
        if ycase[n, 0] == 0:
            UK[n] = 0
        if ycase[n, 1] == 0:
            UV[n] = 0
    return UK, UV


def folds(ids, split):
    f_hy, f_fx = {}, {}
    for r in csv.DictReader(open(MW / "outputs_final/region_units/hydronephrosis/out/oof_train.csv")):
        f_hy[r["study_id"]] = int(r["fold"])
    for r in csv.DictReader(open(MW / "outputs_fracture/spine_ft/in/labels.csv")):
        if r["split"] == "train":
            f_fx[r["study_id"]] = int(r["fold"])
    fh = np.array([f_hy.get(c, zlib.crc32(c.encode()) % 5) if s == "train" else -1 for c, s in zip(ids, split)])
    ff = np.array([f_fx.get(c, zlib.crc32(c.encode()) % 5) if s == "train" else -1 for c, s in zip(ids, split)])
    miss = (sum(c not in f_hy for c, s in zip(ids, split) if s == "train"),
            sum(c not in f_fx for c, s in zip(ids, split) if s == "train"))
    return fh, ff, miss


# ------------------------------------------------------------------ model
def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.min() == y.max():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def train_model(X, U, y, kind, seed=0, epochs=8, dev="cpu"):
    """X [n, T, 448] fp16, U [n] unit index, y [n] 0/1 -> fitted torch model (logistic or MLP over the flattened tokens
    + unit one-hot). Fixed hyper-parameters (no tuning on val)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    T, D = X.shape[1], X.shape[2]
    nU = int(U.max()) + 1

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(D)
            self.ue = nn.Embedding(nU, 64 if kind == "mlp" else 1)
            if kind == "mlp":
                self.proj = nn.Linear(D, 32)                     # shared per-token projection
                self.pos = nn.Parameter(torch.zeros(T, 32))
                self.head = nn.Sequential(nn.Linear(T * 32 + 64, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
            else:
                self.lin = nn.Linear(T * D, 1)

        def forward(self, x, u):
            x = self.norm(x.float())
            if kind == "mlp":
                h = (self.proj(x) + self.pos).flatten(1)
                return self.head(torch.cat([h, self.ue(u)], 1)).squeeze(1)
            return self.lin(x.flatten(1)).squeeze(1) + self.ue(u).squeeze(1)

    m = M().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05 if kind == "mlp" else 0.5)
    Xt, Ut, yt = torch.from_numpy(X), torch.from_numpy(U).long(), torch.from_numpy(y).float()
    pw = torch.tensor(min((1 - y.mean()) / max(y.mean(), 1e-6), 20.0) ** 0.5)   # mild positive weighting
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    bs = 512
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = lossf(m(Xt[b].to(dev), Ut[b].to(dev)), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def predict(m, X, U, dev="cpu"):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(torch.sigmoid(m(torch.from_numpy(X[i:i + 2048]).to(dev),
                                       torch.from_numpy(U[i:i + 2048]).long().to(dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run_task(name, F, MK, Y, fold, split, kinds, log):
    """F [N, nU, T, D]; MK [N, nU, T] mask; Y [N, nU] labels (nan unknown). Returns scores [N, nU] (train OOF, val refit;
    nan where the unit has no tokens) and the chosen kind."""
    N, nU = Y.shape
    present = MK.any(-1)                                       # [N, nU]
    rows = [(n, u) for n in range(N) for u in range(nU) if present[n, u]]
    rn = np.array([r[0] for r in rows])
    ru = np.array([r[1] for r in rows])
    yr = Y[rn, ru]
    tr = split[rn] == "train"
    best = None
    for kind in kinds:
        oof = np.full(len(rows), np.nan)
        for k in range(5):
            fit_i = np.where(tr & (fold[rn] != k) & ~np.isnan(yr))[0]
            pr_i = np.where(tr & (fold[rn] == k))[0]
            m = train_model(F[rn[fit_i], ru[fit_i]], ru[fit_i], yr[fit_i], kind, seed=k)
            oof[pr_i] = predict(m, F[rn[pr_i], ru[pr_i]], ru[pr_i])
            log(f"  {name} {kind} fold {k}: fit {len(fit_i)} ({int(yr[fit_i].sum())} pos), OOF AUROC so far "
                f"{auroc(yr[pr_i], oof[pr_i]):.3f}")
        a = auroc(yr[tr], oof[tr])
        log(f"{name} {kind}: train OOF unit AUROC {a:.4f}")
        if best is None or a > best[0]:
            best = (a, kind, oof)
    a, kind, oof = best
    fit_i = np.where(tr & ~np.isnan(yr))[0]
    va_i = np.where(~tr)[0]
    m = train_model(F[rn[fit_i], ru[fit_i]], ru[fit_i], yr[fit_i], kind, seed=99)
    sc = np.full((N, nU), np.nan)
    sc[rn[tr], ru[tr]] = oof[tr]
    sc[rn[va_i], ru[va_i]] = predict(m, F[rn[va_i], ru[va_i]], ru[va_i])
    import torch
    return sc, kind, a, m


def f1_threshold(y, s):
    ok = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[ok], s[ok]
    best = (0, 0.5)
    for t in np.unique(np.round(s, 2)):
        p = s >= t
        tp = (p & (y == 1)).sum()
        f1 = 2 * tp / max(p.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def cmd_fit(a):
    out = Path(a.out)
    logf = open(out / "fit.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    meta = np.load(out / "meta.npz")
    ids, split = meta["ids"], meta["split"]
    ycase = case_labels(ids, split)
    UK, UV = unit_labels(ids, split, ycase)
    fh, ff, miss = folds(ids, split)
    log(f"[{time.strftime('%m-%d %H:%M')}] fit: {len(ids)} cases; train cases without official fold (crc32 fallback): "
        f"hydronephrosis {miss[0]}, fracture {miss[1]}")
    for nm, Y in (("kidney", UK), ("vertebra", UV)):
        for sp in ("train", "val"):
            s = split == sp
            log(f"  {nm} {sp}: labelled units {int((~np.isnan(Y[s])).sum())}, positive {int(np.nansum(Y[s]))}")
    kinds = a.kinds.split(",")
    res = {}
    import torch
    torch.set_num_threads(a.threads)
    K = np.load(out / "kid.npy", mmap_mode="r")
    sk, kk, ak, mk = run_task("kidney", np.asarray(K), meta["kid_mask"], UK, fh, split, kinds, log)
    del K
    torch.save(mk.state_dict(), out / f"kidney_{kk}_trainfull.pt")
    V = np.load(out / "vert.npy", mmap_mode="r")
    sv, kv, av, mv = run_task("vertebra", np.asarray(V), meta["vert_mask"], UV, ff, split, kinds, log)
    del V
    torch.save(mv.state_dict(), out / f"vertebra_{kv}_trainfull.pt")
    tr, va = split == "train", split == "val"
    # ---- metrics (val = report only)
    res["kidney"] = {"model": kk, "train_oof_auroc": ak, "val_auroc": auroc(UK[va].ravel(), sk[va].ravel())}
    res["vertebra"] = {"model": kv, "train_oof_auroc": av, "val_auroc": auroc(UV[va].ravel(), sv[va].ravel())}
    # side: val cases with hydronephrosis and a known side; predicted side = sides with score >= t (else argmax)
    tk, f1k = f1_threshold(UK[tr].ravel(), sk[tr].ravel())
    tv, f1v = f1_threshold(UV[tr].ravel(), sv[tr].ravel())
    res["kidney"]["t_f1_train"], res["vertebra"]["t_f1_train"] = tk, tv
    for sp, msk in (("train_oof", tr), ("val", va)):
        idx = np.where(msk & (ycase[:, 0] == 1) & ~np.isnan(UK).any(1) & ~np.isnan(sk).any(1))[0]
        truth = [tuple(UK[n] == 1) for n in idx]
        pred_arg = [(sk[n, 0] >= sk[n, 1], sk[n, 1] > sk[n, 0]) for n in idx]
        pred_thr = [tuple(sk[n] >= tk) if (sk[n] >= tk).any() else (sk[n, 0] >= sk[n, 1], sk[n, 1] > sk[n, 0]) for n in idx]
        res["kidney"][f"{sp}_side"] = {"n": len(idx), "argmax_acc": float(np.mean([p == t for p, t in zip(pred_arg, truth)])),
                                       "thr_exact_acc": float(np.mean([p == t for p, t in zip(pred_thr, truth)])),
                                       "bilateral_share": float(np.mean([all(t) for t in truth]))}
        idx = np.where(msk & (ycase[:, 1] == 1) & (np.nan_to_num(UV, nan=0) == 1).any(1))[0]
        top1 = [np.nanargmax(sv[n]) for n in idx if not np.isnan(sv[n]).all()]
        hit = [UV[n, t] == 1 for n, t in zip([n for n in idx if not np.isnan(sv[n]).all()], top1)]
        res["vertebra"][f"{sp}_level"] = {"n": len(hit), "top1_hit": float(np.mean(hit)),
                                          "chance_top1": float(np.mean([np.nansum(UV[n] == 1) / 10 for n in idx]))}
        # case-level AUROC of max over units vs case label
        res["kidney"][f"{sp}_case_auroc_max"] = auroc(ycase[msk, 0], np.nanmax(np.where(np.isnan(sk[msk]), -1, sk[msk]), 1))
        res["vertebra"][f"{sp}_case_auroc_max"] = auroc(ycase[msk, 1], np.nanmax(np.where(np.isnan(sv[msk]), -1, sv[msk]), 1))
    (out / "probe_metrics.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))
    with open(out / "probe_scores.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "kid_right", "kid_left"] + LEVELS)
        for n in range(len(ids)):
            w.writerow([ids[n], split[n]] + ["" if np.isnan(v) else f"{v:.5f}" for v in list(sk[n]) + list(sv[n])])
    # gt-dropout unit cue file (train only): labels + thresholds + pools of train OOF scores
    pools = {"kidney": {"pos": sorted(round(float(v), 2) for v in sk[tr][(UK[tr] == 1) & (sk[tr] >= tk)]),
                        "neg": sorted(round(float(v), 2) for v in sk[tr][(UK[tr] == 0) & (sk[tr] < tk)])},
             "vertebra": {"pos": sorted(round(float(v), 2) for v in sv[tr][(UV[tr] == 1) & (sv[tr] >= tv)]),
                          "neg": sorted(round(float(v), 2) for v in sv[tr][(UV[tr] == 0) & (sv[tr] < tv)])}}
    lab = {str(ids[n]): {"kid": [None if np.isnan(v) else int(v) for v in UK[n]],
                         "vert": [None if np.isnan(v) else int(v) for v in UV[n]]} for n in np.where(tr)[0]}
    (out / "unit_cue.json").write_text(json.dumps({"labels_train": lab, "thresholds": {"kidney": tk, "vertebra": tv},
                                                   "pools": pools, "levels": LEVELS, "sides": SIDES}))
    log(f"pools: kidney pos {len(pools['kidney']['pos'])} neg {len(pools['kidney']['neg'])}; vertebra pos "
        f"{len(pools['vertebra']['pos'])} neg {len(pools['vertebra']['neg'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="logistic,mlp")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
