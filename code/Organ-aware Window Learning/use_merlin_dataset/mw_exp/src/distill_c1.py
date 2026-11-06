"""Distillation C1: ensemble/stack teacher -> ONE bone site-wise MIL probe (E3_late recipe of bone_expI.py 'ref').

Teacher = train-OOF logits of an ensemble (OOF member format; several files are averaged after z-scoring each within
its CV fold). The teacher's TRAIN rows only are used: Platt-calibrated on train (y ~ a*z + b, train rows), then
q = sigmoid((a*z + b) / T). Student any-fracture target on the fit rows: t = alpha * y + (1 - alpha) * q.
Everything else (site sub-labels, class weights, folds, inner early stop on the HARD label AUROC, HP) is the
unchanged E3_late / bone_expI 'ref' recipe, so the student is directly comparable with the hard-label single seed.
Val is never used to fit, stop or select (val logit = mean of the 5 fold models). Test never loaded.
Caveat: the teacher OOF of a fit row came from teacher models that saw the student's held-out fold, so the student
train OOF is slightly optimistic (same for every alpha / T; val is unaffected).
usage: python distill_c1.py --teacher 'glob' --tname ensI --alpha 0.5 --T 1 --seed-off 0 --outdir DIR
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bone_expI as BE  # noqa: E402
from bone_expI_eval import fold_of  # noqa: E402
import pandas as pd  # noqa: E402


def teacher_logit(spec, meta):
    fo = fold_of(meta).values
    L = []
    for f in sorted({f for g in spec.split(",") for f in glob.glob(g)}):
        z = np.load(f, allow_pickle=True)
        assert set(np.unique(z["split"])) <= {"train", "val"}
        assert [str(i) for i in z["ids"]] == list(meta.index)
        v = np.asarray(z["logit"], float).copy()
        for k in range(5):
            m = fo == k
            v[m] = (v[m] - v[m].mean()) / v[m].std()
        L.append(v)
    assert L, spec
    return np.mean(L, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--tname", required=True)
    ap.add_argument("--alpha", type=float, required=True); ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--seed-off", type=int, default=0); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    meta = pd.read_csv(BE.OUT / "meta.csv", index_col=0, keep_default_na=False, na_values=[""])
    tr = (meta.split == "train").values
    zt = teacher_logit(a.teacher, meta)
    from sklearn.linear_model import LogisticRegression
    yv = meta.y.values
    ok = tr & ~np.isnan(yv)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(zt[ok, None], yv[ok].astype(int))
    q = 1 / (1 + np.exp(-(lr.coef_[0, 0] * zt + lr.intercept_[0]) / a.T))
    q = np.where(tr, q, np.nan)                                  # val rows never get a teacher target
    print(f"teacher {a.tname}: platt a={lr.coef_[0,0]:.3f} b={lr.intercept_[0]:.3f}; T={a.T} alpha={a.alpha}; "
          f"q mean pos {np.nanmean(q[ok & (yv == 1)]):.3f} neg {np.nanmean(q[ok & (yv == 0)]):.3f}", flush=True)
    base_sw = BE.sample_weights

    def sw(cfg, y, m, fit):
        w, t = base_sw(cfg, y, m, fit)
        idx = list(m.index)
        assert idx == list(meta.index)
        t = np.where(tr & ~np.isnan(y), a.alpha * np.nan_to_num(y) + (1 - a.alpha) * np.nan_to_num(q), t).astype(np.float32)
        return w, t
    BE.sample_weights = sw
    tag = f"{a.tname}_a{a.alpha:g}_T{a.T:g}"
    od = Path(a.outdir); wd = od / f"work_{tag}_s{a.seed_off}"; wd.mkdir(parents=True, exist_ok=True)
    BE.run("ref", a.seed_off, torch.device("cpu"), str(wd))
    src = wd / f"oof_bone_E_ref_s{a.seed_off}.npz"
    src.rename(od / f"oof_distill_{tag}_s{a.seed_off}.npz")
    wd.rmdir()
    print("->", od / f"oof_distill_{tag}_s{a.seed_off}.npz", flush=True)


if __name__ == "__main__":
    main()
