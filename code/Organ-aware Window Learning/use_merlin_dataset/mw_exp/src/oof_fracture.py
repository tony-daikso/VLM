"""Out-of-fold (OOF) fracture logits for E0 / E3_late, so a downstream stack can be fit on honest train scores.

For each of K stratified folds of the TRAIN split:
  * standardize embeddings with statistics of the fold's training part only (as probe.py --standardize:
    per (window, dim), pooled over present organ units; absent organs stay 0),
  * train a fracture-only Probe (class-balanced BCE, lr/wd from results/std/<exp>_grid.json),
    early-stopping on an inner 10% holdout of the fold's training part (val is never used for selection),
  * predict the held-out fold (OOF logit) and the VAL split.
VAL logit = mean of the K fold models' logits. Test rows are dropped right after loading and never touched.
Note: Merlin metadata has no patient id, so folds are study-wise (the official split itself is patient-wise).
usage: python oof_fracture.py --config config_final.yaml --exp E0 --out outputs_fracture/oof_E0.npz [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import load_cfg
from probe import Probe, auroc, experiments, load_data


def standardize(E, P, fit_rows):
    rows = E[fit_rows][P[fit_rows] > 0]                       # [m,W,D] present organs of the fit rows
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    return ((E - mu) / sd) * (P[..., None, None] > 0)


def train_fracture(spec, E, P, y, tr_idx, es_idx, lr, wd, seed, max_epochs=100, patience=10, bs=256):
    """E,P: tensors for the single fracture unit (U=1). Returns the best-on-inner-holdout model."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = Probe(spec["mode"], E.shape[1], E.shape[2], E.shape[3], [[0]], spec.get("win_idx"), seed=seed)
    gate = [model.gate] if model.gate is not None else []
    head = [p for n, p in model.named_parameters() if n != "gate"]
    groups = [{"params": head, "lr": lr}] + ([{"params": gate, "lr": lr}] if gate else [])
    opt = torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[tr_idx], P[tr_idx], torch.from_numpy(y[tr_idx])[:, None]
    npos = float(Yt.sum()); nall = len(Yt); nneg = nall - npos
    wpos, wneg = nall / (2 * max(npos, 1)), nall / (2 * max(nneg, 1))
    best, best_state, bad, n = -1.0, None, 0, len(Et)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            yy = Yt[j]
            logit = model(Et[j], Pt[j])
            w = torch.where(yy == 1, wpos, wneg)
            loss = (F.binary_cross_entropy_with_logits(logit, yy, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], model(E[es_idx], P[es_idx])[:, 0].numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, best, best_ep


def predict(model, E, P, idx):
    with torch.no_grad():
        return torch.cat([model(E[idx[i:i + 2048]], P[idx[i:i + 2048]])[:, 0] for i in range(0, len(idx), 2048)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True, choices=["E0", "E3_late"])
    ap.add_argument("--grid-json", default=None, help="default: <out>/results/std/<exp>_grid.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_cfg(a.config)
    ids, E, P, Y, S, units, wins, findings = load_data(cfg)
    keep = np.isin(S, ["train", "val"])                       # drop test immediately
    ids, E, P, Y, S = np.array(ids)[keep], E[torch.from_numpy(keep)], P[torch.from_numpy(keep)], Y[keep], S[keep]
    fi = findings.index("fracture")
    y = Y[:, fi]
    u = units.index(cfg["finding_map"]["fracture"]["organs"][0]); assert len(cfg["finding_map"]["fracture"]["organs"]) == 1
    spec = experiments(cfg, units, wins)[a.exp]
    hp = json.loads(open(a.grid_json or cfg["out"] / "results" / "std" / f"{a.exp}_grid.json").read())["chosen"]
    print(f"{a.exp}: spec={spec} hp={hp} N(train,val)={(S == 'train').sum()},{(S == 'val').sum()} unit={units[u]}", flush=True)

    tr_all = np.where((S == "train") & ~np.isnan(y))[0]
    va_all = np.where(S == "val")[0]
    oof = np.full(len(ids), np.nan, np.float32)
    val_logits, info = [], []
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=cfg["seed"])
    for k, (fit_pos, hold_pos) in enumerate(skf.split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fit_pos], tr_all[hold_pos]
        Ek = standardize(E, P, torch.from_numpy(np.isin(np.arange(len(ids)), fit)))[:, u:u + 1]   # stats: fold-train only
        Pk = P[:, u:u + 1]
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=cfg["seed"] + k)
        model, es_auc, ep = train_fracture(spec, Ek, Pk, y, inner, es, hp["lr"], hp["wd"], cfg["seed"] + k)
        oof[hold] = predict(model, Ek, Pk, hold)
        val_logits.append(predict(model, Ek, Pk, va_all))
        h_auc = auroc(y[hold], oof[hold]); v_auc = auroc(y[va_all], val_logits[-1])
        info.append({"fold": k, "inner_es_auc": es_auc, "best_epoch": ep, "heldout_auc": h_auc, "val_auc": v_auc})
        print(f"  fold {k}: es_auc={es_auc:.4f} ep={ep} heldout_auc={h_auc:.4f} val_auc={v_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del Ek
    vmean = np.mean(val_logits, 0)
    oof[va_all] = vmean
    sel = np.r_[tr_all, va_all]
    sel.sort()
    print(f"{a.exp}: OOF train AUROC={auroc(y[tr_all], oof[tr_all]):.4f}  val AUROC (5-fold mean)={auroc(y[va_all], vmean):.4f}", flush=True)
    np.savez(a.out, ids=ids[sel], split=S[sel], logit=oof[sel], y=y[sel], exp=a.exp, hp=json.dumps(hp),
             folds=json.dumps(info), val_fold_logits=np.stack(val_logits))


if __name__ == "__main__":
    main()
