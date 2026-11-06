"""Experiment G: rib-level units (organ_map_ribs.yaml, emb_ribs/) inside the bone-unit site-wise MIL. TRAIN fit, VAL report.

The 2 big rib units (ribs_left / ribs_right) of the Exp B bone model are replaced by (or extended with) the 18 rib-level
units (rib_<side>_1_4 merged, rib_<side>_5 .. rib_<side>_12 single); the rib site head is a shared linear head over the
rib units + per-unit bias, site logit = max (site_max) / logsumexp (site_lse) over present rib units. Vertebra and
pelvis heads unchanged. Training recipe, folds (StratifiedKFold(5, rs=42) on train), early stopping, HP (Exp B
site_max__mean__all3) and the label variants (ref / excl_amb) are those of bone_label_exp.py, so the OOF files are
drop-in members for bone_ens.py / final_stack.py.

stages:
  build  rib matrix for the ids of bone_matrix.npz (train+val only; test never loaded): [N,18,3 windows,2 pools,448]
         fp16 + presence -> <out>/rib_matrix.npz ; ids without rib embeddings -> all rib units absent (reported)
  oof    --ribset rib18|rib18big --variant ref|excl_amb --model site_max|site_lse --seed-off K
         -> <outdir>/oof_bone_E_<variant>__<model>-<ribset>__<pool>__<win>[_s<K>].npz (+ per-site train OOF / val info)
usage: python rib_probe.py --stage build ; python rib_probe.py --stage oof --ribset rib18 --variant ref --seed-off 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from bone_label_exp import make_targets, raw_fracture  # noqa: E402
from bone_probe import POOLS, WINS, WSHORT, load, predict, site_targets, std_unit  # noqa: E402
from probe import auroc  # noqa: E402

ROOT = Path.home() / "Desktop/Tony/VLM/OWL"
EMBR = ROOT / "mw_exp/outputs/emb_ribs"
BONES = ROOT / "mw_exp/outputs_fracture/bones"
VERT_B, RIBS_B, PELV_B = list(range(10)), [10, 11], [12, 13, 14, 15, 16]


def build(out):
    z = np.load(out / "bone_matrix.npz", allow_pickle=True)
    ids = list(z["ids"]); assert set(z["split"]) <= {"train", "val"}
    E = np.zeros((len(ids), 18, 3, 2, 448), np.float16)
    P = np.zeros((len(ids), 18), bool)
    miss, names, t0 = [], None, time.time()
    for i, cid in enumerate(ids):
        fs = [EMBR / w / f"{cid}.npz" for w in WINS]
        if not all(f.exists() for f in fs):
            miss.append(cid); continue
        for w, f in enumerate(fs):
            r = np.load(f)
            E[i, :, w, 0], E[i, :, w, 1] = r["emb"], r["emb_max"]
            if w == 0:
                P[i] = r["organ_present"]; names = list(r["organ_names"])
        if i % 2000 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out / "rib_matrix.npz", E=E, P=P, ids=np.array(ids), units=np.array(names), missing=np.array(miss))
    print(f"N={len(ids)}, missing rib embeddings: {len(miss)}")
    print("presence rate per rib unit:", dict(zip(names, P.mean(0).round(3))))


class RibSiteModel(nn.Module):
    """site heads over arbitrary unit index lists; any-logit = logsumexp(site logits) + b (BoneModel 'site_*')."""

    def __init__(self, kind, D, U, sites):
        super().__init__()
        self.kind, self.sites = kind, sites
        self.heads = nn.ModuleList([nn.Linear(D, 1) for _ in sites])
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(len(sites)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X, M):
        out = []
        for k, idx in enumerate(self.sites):
            s = (self.heads[k](X[:, idx])[..., 0] + self.ub[idx]).masked_fill(~M[:, idx], -1e4)
            r = s.max(1).values if self.kind.startswith("site_max") else torch.logsumexp(s, 1)
            out.append(torch.where(M[:, idx].any(1), r, self.none[k].expand_as(r)))
        St = torch.stack(out, 1)
        return torch.logsumexp(St, 1) + self.b, St


def assemble(ribset, Eb, Pb, Er, Pr):
    """-> E [N,U,3,2,448], P [N,U], site index lists."""
    if ribset == "rib18":
        keep = VERT_B + PELV_B
        E, P = np.concatenate([Eb[:, keep], Er], 1), np.concatenate([Pb[:, keep], Pr], 1)
        sites = [list(range(10)), list(range(15, 33)), list(range(10, 15))]
    elif ribset == "rib18big":          # 18 rib units + the 2 big side units in the rib site
        E, P = np.concatenate([Eb, Er], 1), np.concatenate([Pb, Pr], 1)
        sites = [VERT_B, RIBS_B + list(range(17, 35)), PELV_B]
    else:
        raise ValueError(ribset)
    return E, P, sites


def prep_X(E, P, wk, pk, fit_mask):
    return torch.from_numpy(np.stack([np.concatenate([std_unit(E[:, u, w, p], P[:, u], fit_mask)
                                                      for w in WSHORT[wk] for p in POOLS[pk]], 1)
                                      for u in range(E.shape[1])], 1))


def train_g(kind, sites, X, M, t, w, ys, yhard, tr_idx, es_idx, lr, wd, seed, dev, max_epochs=100, patience=10, bs=512):
    """bone_label_exp.train_e with RibSiteModel (same loss, class balance, early stop on hard y)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = RibSiteModel(kind, X.shape[2], X.shape[1], sites).to(dev)
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


def site_auc(y, ysite, s, rows):
    """site positives vs all negatives (ysite NaN = fracture elsewhere -> excluded)."""
    m = rows & ~np.isnan(ysite)
    return auroc(ysite[m], s[m])


def run_oof(out, dev, ribset, variant, kind, pk, wk, seed_off, outdir, hp_from, folds=5):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    ids, S, Eb, Pb, units, lab, T = load(out)
    zr = np.load(out / "rib_matrix.npz", allow_pickle=True)
    assert list(zr["ids"]) == list(ids)
    pl = POOLS[pk]                                   # keep only the pooling(s) used (memory)
    E, P, sites = assemble(ribset, Eb[:, :, :, pl], Pb, zr["E"][:, :, :, pl], zr["P"])
    del Eb, zr
    pk_idx = {"mean": "mean", "max": "mean", "meanmax": "meanmax"}[pk]   # after slicing: index 0 (,1)
    y, ys = T["any"], site_targets(T)
    t, w, yss = make_targets(variant, raw_fracture(ids), y, ys)
    M = torch.from_numpy(P)
    hp = json.loads((out / "agg" / f"{hp_from}.json").read_text())["hp"]
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    o = np.full(len(ids), np.nan, np.float32); osite = np.full((len(ids), 3), np.nan, np.float32)
    vl, vls, info = [], [], []
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fp], tr_all[hpos]
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        inner = inner[w[inner] > 0]
        X = prep_X(E, P, wk, pk_idx, np.isin(np.arange(len(ids)), fit))
        m, s, ep = train_g(kind, sites, X, M, t, w, yss, y, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * seed_off, dev)
        o[hold], osite[hold] = predict(m, X, M, hold, dev)
        a, st = predict(m, X, M, va_all, dev); vl.append(a); vls.append(st)
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold]), "val": auroc(y[va_all], a)})
        print(info[-1], f"{time.time() - t0:.0f}s", flush=True)
        del X
    o[va_all], osite[va_all] = np.mean(vl, 0), np.mean(vls, 0)
    tr, va = S == "train", S == "val"
    cv = auroc(y[tr], o[tr])
    rib_cv, rib_val = site_auc(y, T["rib"], o, tr), site_auc(y, T["rib"], o, va)
    name = f"{kind}-{ribset}__{pk}__{wk}"
    print(f"G {variant} {name} s{seed_off}: OOF train AUROC={cv:.4f} (rib {rib_cv:.4f}) | val (info) {auroc(y[va], o[va]):.4f} (rib {rib_val:.4f})")
    od = Path(outdir); od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"oof_bone_E_{variant}__{name}{f'_s{seed_off}' if seed_off else ''}.npz", ids=np.array(ids), split=S,
             logit=o, site_logit=osite, y=y, exp=f"G_{variant}:{name}", hp=json.dumps(hp), folds=json.dumps(info),
             oof_train_auc=cv, oof_train_rib_auc=rib_cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "oof"])
    ap.add_argument("--out", default=str(BONES))
    ap.add_argument("--ribset", default="rib18", choices=["rib18", "rib18big"])
    ap.add_argument("--variant", default="ref", choices=["ref", "excl_amb"])
    ap.add_argument("--model", default="site_max", choices=["site_max", "site_lse"])
    ap.add_argument("--pool", default="mean"); ap.add_argument("--win", default="all3")
    ap.add_argument("--hp-from", default="site_max__mean__all3", help="Exp B agg json whose lr/wd are reused")
    ap.add_argument("--seed-off", type=int, default=0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--outdir", default=str(BONES / "final_G/members"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    if a.stage == "build":
        build(out)
    else:
        run_oof(out, torch.device(a.device), a.ribset, a.variant, a.model, a.pool, a.win, a.seed_off, a.outdir, a.hp_from)


if __name__ == "__main__":
    main()
