"""Experiment E: training-label refinements for the bone-unit fracture model (best Exp B config site_max/mean/all3).

Only the TRAINING targets / weights change; early stopping, OOF train AUROC and every val metric use the unchanged
hard target y = yes/probably_yes vs no/probably_no/not_mentioned (unclear = NaN, never scored).
Test ids are never loaded (bone_matrix.npz has train+val only; the extra unclear rows are asserted train-only).

variants (--variant):
  ref           hard labels (Exp B recipe, rerun with this code on CPU)
  excl_amb      probably_yes / probably_no get weight 0 (unclear already excluded)
  soft_cat      probably_yes -> 0.7, probably_no -> 0.2
  soft_p        any-target = LLM confidence p/100 (yes ~.95, no ~.02-.05, not_mentioned ~.05-.10)
  soft_p_unc    soft_p + train 'unclear' cases added with target p/100 (fit only, never scored)
  soft_p_site   soft_p and the site aux targets soft too (site listed -> p/100, negatives -> p/100)
  acute_w2/w4   acute (yes / mixed) fracture positives weighted x2 / x4
  head_acute    separate model: acute positives vs negatives (other positives weight 0)
  head_old      separate model: old (acute = no) positives vs negatives (other positives weight 0)
  nm_drop       not_mentioned negatives weight 0
  nm_w03        not_mentioned negatives weight 0.3
usage: python bone_label_exp.py --variant soft_p [--model site_max --pool mean --win all3] [--seed-off K]
       [--device cuda] [--outdir DIR]   (non-default model/pool/win -> file name gets __<model>__<pool>__<win>)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bone_probe import EMB, LABELS, SPLIT, WINS, BoneModel, load, predict, prep_X, site_targets  # noqa: E402
from probe import auroc  # noqa: E402

VARIANTS = ["ref", "excl_amb", "soft_cat", "soft_p", "soft_p_unc", "soft_p_site", "acute_w2", "acute_w4",
            "head_acute", "head_old", "nm_drop", "nm_w03"]


def raw_fracture(ids):
    """category, p, acute for the given ids (train/val only)."""
    want, rows = set(ids), {}
    for l in open(LABELS):
        r = json.loads(l)
        if r["study_id"] in want:
            a = r["answer"]["fracture"]
            rows[r["study_id"]] = {"cat": a["category"], "p": float(a.get("p", np.nan)),
                                   "acute": (a.get("detail") or {}).get("acute", "not_stated")}
    return pd.DataFrame.from_dict(rows, orient="index").loc[list(ids)]


def unclear_rows(out):
    """Cached [K,17,3,2,448] embeddings of TRAIN cases whose fracture category is 'unclear'."""
    f = out / "bone_matrix_unclear_train.npz"
    if not f.exists():
        spl = pd.read_csv(SPLIT).set_index("study_id")["split"]
        tr = set(spl.index[spl == "train"])
        cand = []
        for l in open(LABELS):
            r = json.loads(l)
            if r["study_id"] in tr and r["answer"]["fracture"]["category"] == "unclear":
                cand.append(r["study_id"])
        cand = sorted(c for c in cand if all((EMB / w / f"{c}.npz").exists() for w in WINS))
        E = np.zeros((len(cand), 17, 3, 2, 448), np.float16); P = np.zeros((len(cand), 17), bool)
        for i, cid in enumerate(cand):
            for w, wn in enumerate(WINS):
                z = np.load(EMB / wn / f"{cid}.npz")
                E[i, :, w, 0], E[i, :, w, 1] = z["emb"], z["emb_max"]
                if w == 0:
                    P[i] = z["organ_present"]
        np.savez(f, E=E, P=P, ids=np.array(cand))
    z = np.load(f, allow_pickle=True)
    spl = pd.read_csv(SPLIT).set_index("study_id")["split"]
    assert (spl.loc[list(z["ids"])] == "train").all()
    return list(z["ids"]), z["E"], z["P"]


def make_targets(variant, raw, y, ys):
    """-> soft any-target t, sample weight w, site targets (NaN = ignore)."""
    t, w, ys = y.copy(), np.ones_like(y), ys.copy()
    cat, p, ac = raw["cat"].values, raw["p"].values / 100.0, raw["acute"].values
    pos = y == 1
    if variant == "excl_amb":
        w[np.isin(cat, ["probably_yes", "probably_no"])] = 0
    elif variant == "soft_cat":
        t[cat == "probably_yes"], t[cat == "probably_no"] = 0.7, 0.2
    elif variant in ("soft_p", "soft_p_unc", "soft_p_site"):
        t = np.where(np.isnan(p), y, p)
        if variant == "soft_p_site":
            ys = np.where(np.isnan(ys), np.nan, np.where(ys == 1, t[:, None], t[:, None]))
    elif variant in ("acute_w2", "acute_w4"):
        w[pos & np.isin(ac, ["yes", "mixed"])] = 2.0 if variant == "acute_w2" else 4.0
    elif variant == "head_acute":
        w[pos & ~np.isin(ac, ["yes", "mixed"])] = 0
    elif variant == "head_old":
        w[pos & (ac != "no")] = 0
    elif variant == "nm_drop":
        w[cat == "not_mentioned"] = 0
    elif variant == "nm_w03":
        w[cat == "not_mentioned"] = 0.3
    elif variant != "ref":
        raise ValueError(variant)
    ys[w == 0] = np.nan
    return t.astype(np.float32), w.astype(np.float32), ys.astype(np.float32)


def train_e(kind, X, M, t, w, ys, yhard, tr_idx, es_idx, lr, wd, seed, dev, max_epochs=100, patience=10, bs=512):
    """bone_probe.train with soft targets + sample weights; identical to it for hard t and w = 1.
    Class balance from (soft) counts: per-sample weight = w * (t*wp + (1-t)*wn). Early stop on hard y."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = BoneModel(kind, X.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, Mt = X[tr_idx].to(dev), M[tr_idx].to(dev)
    tt, wt = torch.from_numpy(t[tr_idx]).to(dev), torch.from_numpy(w[tr_idx]).to(dev)
    yst = torch.from_numpy(ys[tr_idx]).to(dev)
    vs = ~torch.isnan(yst)
    act = wt > 0
    n, n1 = act.sum(), (tt * act).sum()
    wp, wn = n / (2 * n1.clamp(min=1)), n / (2 * (n - n1).clamp(min=1))
    yz = yst.nan_to_num(0)
    ns, ns1 = vs.sum(0), (yz * vs).sum(0)
    swp, swn = ns / (2 * ns1.clamp(min=1)), ns / (2 * (ns - ns1).clamp(min=1))
    Xe, Me = X[es_idx].to(dev), M[es_idx].to(dev)
    best, state, bad, best_ep = -1.0, None, 0, 0
    N = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, bs):
            j = perm[i:i + bs]
            la, ls = model(Xt[j], Mt[j])
            cw = (tt[j] * wp + (1 - tt[j]) * wn) * wt[j]
            loss = (F.binary_cross_entropy_with_logits(la, tt[j], reduction="none") * cw).mean()
            if ls is not None and not kind.endswith("_noaux"):
                m = vs[j]
                sw = (yz[j] * swp + (1 - yz[j]) * swn) * m
                loss = loss + (F.binary_cross_entropy_with_logits(ls, yz[j], reduction="none") * sw).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(yhard[es_idx], model(Xe, Me)[0].cpu().numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def run(out, dev, variant, kind, pk, wk, folds=5, seed_off=0, outdir=None):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    ids, S, E, P, units, lab, T = load(out)
    y, ys = T["any"], site_targets(T)
    raw = raw_fracture(ids)
    t, w, yss = make_targets(variant, raw, y, ys)
    N0 = len(ids)
    extra = np.array([], int)
    if variant == "soft_p_unc":
        uid, Eu, Pu = unclear_rows(out)
        ru = raw_fracture(uid)
        E, P = np.concatenate([E, Eu]), np.concatenate([P, Pu])
        S = np.concatenate([S, np.array(["train"] * len(uid))])
        y = np.concatenate([y, np.full(len(uid), np.nan)])
        t = np.concatenate([t, (ru["p"].values / 100).astype(np.float32)])
        w = np.concatenate([w, np.ones(len(uid), np.float32)])
        yss = np.concatenate([yss, np.full((len(uid), 3), np.nan, np.float32)])
        extra = np.arange(N0, N0 + len(uid))
        print(f"added {len(uid)} unclear train cases (mean p={t[extra].mean():.3f})")
    M = torch.from_numpy(P)
    name = f"{kind}__{pk}__{wk}"
    hp = json.loads((out / "agg" / f"{name}.json").read_text())["hp"]           # Exp B HP, unchanged
    tr_all, va_all = np.where(S[:N0] == "train")[0], np.where(S[:N0] == "val")[0]
    o = np.full(N0, np.nan, np.float32)
    vl, info = [], []
    print(f"{variant}: train fit weight-0 rows={int((w[tr_all] == 0).sum())}, soft rows={int(((t[tr_all] > 0) & (t[tr_all] < 1)).sum())}")
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fp], tr_all[hpos]
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        inner = np.concatenate([inner[w[inner] > 0], extra])
        X = prep_X(E, P, wk, pk, np.isin(np.arange(len(S)), np.concatenate([fit, extra])))
        m, s, ep = train_e(kind, X, M, t, w, yss, y, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * seed_off, dev)
        o[hold] = predict(m, X, M, hold, dev)[0]
        vl.append(predict(m, X, M, va_all, dev)[0])
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold]), "val": auroc(y[va_all], vl[-1])})
        print(info[-1], f"{time.time() - t0:.0f}s", flush=True)
    o[va_all] = np.mean(vl, 0)
    cv = auroc(y[tr_all], o[tr_all])
    print(f"{variant} {name}: OOF train AUROC={cv:.4f} val={auroc(y[va_all], o[va_all]):.4f}")
    od = Path(outdir) if outdir else out / "labelE"
    od.mkdir(parents=True, exist_ok=True)
    cfg = "" if name == "site_max__mean__all3" else f"__{name}"           # default config keeps the Exp E file names
    np.savez(od / f"oof_bone_E_{variant}{cfg}{f'_s{seed_off}' if seed_off else ''}.npz", ids=np.array(ids), split=S[:N0], logit=o, y=y[:N0],
             exp=f"E_{variant}:{name}", hp=json.dumps(hp), folds=json.dumps(info), oof_train_auc=cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--out", default=str(Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones"))
    ap.add_argument("--model", default="site_max"); ap.add_argument("--pool", default="mean"); ap.add_argument("--win", default="all3")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed-off", type=int, default=0, help="extra seed repeat (model init only; folds unchanged)")
    ap.add_argument("--device", default="cpu", help="cpu (Exp E default) or cuda")
    ap.add_argument("--outdir", default=None, help="output dir (default <out>/labelE)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(Path(a.out), torch.device(a.device), a.variant, a.model, a.pool, a.win, seed_off=a.seed_off, outdir=a.outdir)


if __name__ == "__main__":
    main()
