"""Experiment J: 1.5 mm per-vertebra units (vert15_full.py emb15/) inside the bone site-wise MIL (bone_probe.py).

stages
  build : bone_matrix.npz (17 units, 3 mm; mean pool only) + emb15 (T8..L5, 1.5 mm, mean pool, 3 windows)
          -> <out>/v15_matrix.npz  E [N,27,3,1,448] f16, P [N,27], ids, split (train/val only; test never loaded)
          units 0-16 = bone_matrix units, 17-26 = v15_T8..v15_L5 (absent if the case has no emb15 yet)
  oof   : E3_late-style site MIL (bone_probe.ModeModel recipe generalised to K sites), 5-fold OOF on TRAIN
          (inner 10 % early stop; val never used), val = mean of the fold models.
          configs: ref  = 17 units, sites (vert3, ribs, pelvis)                     [same as bone E3_late]
                   add  = 27 units, sites (vert3, ribs, pelvis, vert15)               4th site aux = vertebra label
                   addc = as add, 4th site aux target = compression sub-label
                   repl = 17 units with T8..L5 taken from 1.5 mm (falls back to 3 mm per level if 1.5 mm absent)
          --grid: lr x wd grid with seed 0, chosen on TRAIN OOF AUROC -> <out>/hp_<cfg>.json; then --seeds k..
          --subset tier0: train rows restricted to tier0 ids (interim evaluation while extraction runs)
outputs <out>/oof/<subset>/oof_v15_<cfg>_s<k>.npz (ids, split, logit, y, oof_train_auc) — final_stack.py format.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import bone_probe as B  # noqa: E402
from probe import auroc  # noqa: E402

LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
CFG_SITES = {
    "ref": [list(range(10)), [10, 11], [12, 13, 14, 15, 16]],
    "repl": [list(range(10)), [10, 11], [12, 13, 14, 15, 16]],
    "add": [list(range(10)), [10, 11], [12, 13, 14, 15, 16], list(range(17, 27))],
    "addc": [list(range(10)), [10, 11], [12, 13, 14, 15, 16], list(range(17, 27))],
    "v15only": [list(range(17, 27))],
}


class KModel(nn.Module):
    """bone_probe.ModeModel 'late' mode with an arbitrary list of sites (B.SITES, set before use)."""

    def __init__(self, spec, W, D, U=None, seed=0):
        super().__init__()
        K = len(B.SITES)
        U = NU[0]
        bd = 1 / D ** 0.5
        self.mode = "late"
        self.hw = nn.Parameter(torch.empty(K, W, D).uniform_(-bd, bd))
        self.hb = nn.Parameter(torch.empty(K, W).uniform_(-bd, bd))
        self.gate = nn.Parameter(torch.zeros(U, W))
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X, M):
        sites = []
        for k, idx in enumerate(B.SITES):
            L = torch.einsum("nuwd,wd->nuw", X[:, idx], self.hw[k]) + self.hb[k]
            s = (L * torch.softmax(self.gate[idx], -1)[None]).sum(-1) + self.ub[idx]
            r, any_ = B.BoneModel.pool(s, M[:, idx], "max")
            sites.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(sites, 1)
        return torch.logsumexp(St, 1) + self.b, St


NU = [27]


def build(a, out):
    z = np.load(a.bone_matrix, allow_pickle=True)
    ids, S = [str(i) for i in z["ids"]], z["split"].astype(str)
    assert set(S) <= {"train", "val"}
    E0 = z["E"]
    N = len(ids)
    E = np.zeros((N, 27, 3, 1, 448), np.float16)
    E[:, :17, :, 0] = E0[:, :, :, 0]
    del E0
    P = np.zeros((N, 27), bool)
    P[:, :17] = z["P"]
    found = 0
    files = {}
    for d in a.emb_dirs:
        for p in Path(d).glob("*.npz"):
            if ".tmp" not in p.name:
                files[p.stem] = p
    for i, cid in enumerate(ids):
        p = files.get(cid)
        if p is None:
            continue
        e = np.load(p)
        assert list(e["levels"]) == LV and list(e["windows"]) == ["baseline", "soft_tissue", "bone"]
        E[i, 17:, :, 0] = e["emb"]
        P[i, 17:] = e["present"]
        found += 1
    units = list(z["units"]) + [f"v15_{l}" for l in LV]
    np.savez(out / "v15_matrix.npz", E=E, P=P, ids=np.array(ids), split=S, units=np.array(units),
             has15=P[:, 17:].any(1))
    print(f"N={N}, with emb15: {found}; presence v15:", P[:, 17:].mean(0).round(3), flush=True)


def load(out, subset_file):
    z = np.load(out / "v15_matrix.npz", allow_pickle=True)
    ids, S = list(z["ids"]), z["split"].astype(str)
    assert set(S) <= {"train", "val"}
    lab = B.load_labels(B.LABELS, ids).loc[ids]
    T = B.targets(lab)
    keep = np.ones(len(ids), bool)
    if subset_file:
        sub = set(pd.read_csv(subset_file).query("tier == 0").study_id)
        keep = np.array([i in sub for i in ids]) & z["has15"]
    keep &= ~np.isnan(T["any"])
    return ids, S, z["E"], z["P"], T, keep


def cfg_data(cfg, E, P):
    if cfg == "repl":
        E2, P2 = E[:, :17].copy(), P[:, :17].copy()
        has = P[:, 17:]
        E2[:, :10][has] = E[:, 17:][has]
        return E2, P2
    if cfg == "ref":
        return E[:, :17], P[:, :17]
    return E, P


def site_y(cfg, T):
    cols = [T["vertebra"], T["rib"], T["pelvic"]]
    if cfg == "add":
        cols.append(T["vertebra"])
    if cfg == "addc":
        cols.append(T["compression"])
    if cfg == "v15only":
        cols = [T["vertebra"]]
    return np.stack(cols, 1)


def run_oof(cfg, E, P, S, T, keep, lr, wd, so, dev, folds=5):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    B.SITES = CFG_SITES[cfg]
    B.ModeModel = KModel
    Ec, Pc = cfg_data(cfg, E, P)
    NU[0] = Ec.shape[1]
    y, ys = T["any"], site_y(cfg, T)
    M = torch.from_numpy(Pc)
    tr_all = np.where((S == "train") & keep)[0]
    va_all = np.where((S == "val") & keep)[0]
    o = np.full(len(S), np.nan, np.float32); vl, info = [], []
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        fit, hold = tr_all[fp], tr_all[hpos]
        S2 = np.where(np.isin(np.arange(len(S)), fit), "train", "other")
        X, _ = B.mode_X(Ec, Pc, S2, True, None)
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        m, s, ep = B.train_m({"mode": "late"}, X, M, y, ys, inner, es, lr, wd, 42 + k + 100 * so, dev, None)
        o[hold] = B.predict(m, X, M, hold, dev)[0]
        vl.append(B.predict(m, X, M, va_all, dev)[0])
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold])})
        del X
    o[va_all] = np.mean(vl, 0)
    return o, auroc(y[tr_all], o[tr_all]), auroc(y[va_all], o[va_all]), info


def oof_stage(a, out, dev):
    ids, S, E, P, T, keep = load(out, a.subset_file)
    sub = "tier0" if a.subset_file else "full"
    od = out / "oof" / sub; od.mkdir(parents=True, exist_ok=True)
    print(f"[{sub}] train rows {int(((S == 'train') & keep).sum())}, val rows {int(((S == 'val') & keep).sum())}",
          flush=True)
    for cfg in a.cfgs:
        hpf = od / f"hp_{cfg}.json"
        if a.grid and not hpf.exists():
            grid = []
            for lr, wd in itertools.product((1e-3, 1e-2), (1e-4, 1e-3, 1e-2)):
                t0 = time.time()
                o, cv, _, info = run_oof(cfg, E, P, S, T, keep, lr, wd, 0, dev)
                grid.append({"lr": lr, "wd": wd, "train_oof": cv, "eps": [i["ep"] for i in info]})
                print(f"[{sub}] {cfg} lr={lr} wd={wd}: train OOF {cv:.4f} ({time.time() - t0:.0f}s)", flush=True)
            hp = max(grid, key=lambda g: g["train_oof"])
            hpf.write_text(json.dumps({"hp": hp, "grid": grid}, indent=1))
        hp = json.loads(hpf.read_text())["hp"] if hpf.exists() else {"lr": a.lr, "wd": a.wd}
        for so in a.seeds:
            f = od / f"oof_v15_{cfg}_s{so}.npz"
            if f.exists():
                continue
            t0 = time.time()
            o, cv, va, info = run_oof(cfg, E, P, S, T, keep, hp["lr"], hp["wd"], so, dev)
            sel = keep
            np.savez(f, ids=np.array(ids)[sel], split=S[sel], logit=o[sel], y=T["any"][sel], exp=f"v15:{cfg}",
                     hp=json.dumps(hp), folds=json.dumps(info), oof_train_auc=cv)
            print(f"[{sub}] {cfg} s{so}: train OOF {cv:.4f} (val {va:.4f}, info only) ({time.time() - t0:.0f}s)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "oof"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bone-matrix", default=None)
    ap.add_argument("--emb-dirs", nargs="*", default=[])
    ap.add_argument("--cfgs", nargs="*", default=["ref", "add", "addc", "repl"])
    ap.add_argument("--subset-file", default=None, help="ids_prio.csv: restrict to tier0 (interim)")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    if a.stage == "build":
        build(a, out)
    else:
        oof_stage(a, out, torch.device(a.device))


if __name__ == "__main__":
    main()
