"""Experiment B: fracture probes on the 17 bone units (emb_bones/, src/extract_bones.py). TRAIN fit, VAL report.

Test rows are dropped while building the matrix (only train/val ids are ever loaded).
stages:
  build   cache [N,17,3 windows,2 pools,448] fp16 + presence for train+val ids -> <out>/bone_matrix.npz
  screen  per-unit logistic probe (sklearn, class-balanced, standardized per unit; C grid picked on val)
          for each unit x window x pooling x target (any / vertebra / compression / rib / pelvic)
  agg     torch aggregations (probe.py recipe: Adam, class-balanced BCE, lr x wd grid + early stop on val, 3 seeds):
            concat  : all 17 units concatenated (+ presence indicators)
            milv_*  : shared vertebra head over T8..L5 (+ level bias), case = max / logsumexp over present levels
            site_*  : site heads (vertebra MIL, ribs MIL, pelvis/sacrum/hip/femur MIL) trained multi-task on
                      site sub-labels, any-fracture logit = logsumexp(site logits)  (+ any-fracture loss)
          for pooling in {mean, max, meanmax} x windows in {baseline, soft_tissue, bone, all3}
  oof     5-fold OOF on TRAIN (inner 10% holdout early stop; val never used) for one agg config -> npz like
          oof_fracture.py (ids, split, logit, y); val logit = mean of fold models
  modes   probe.py window experiments E0/E1_*/E2/E3/E3_late/E4/E0_rf (+ _trap/_trap2/_inlp) with site_max MIL on the
          3 real bone windows (agg recipe; --nostd = raw embeddings) -> <out>/rerun/<std|nostd>/<exp>.{npz,json}
  wablate leave-one-window-out for --model E3_late|E2 (chosen HP, 3 seeds)
  modes_oof  5-fold train OOF x --seed-offs for --model <exp> (val unused) -> rerun/<sub>/oof/
usage: python bone_probe.py --stage build|screen|agg|oof [--model site_lse --pool max --win all3]
       python bone_probe.py --stage modes [--exps E0 E2 ...] [--nostd] | --stage wablate --model E3_late
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from probe import auroc  # noqa: E402

ROOT = Path.home() / "Desktop/Tony/VLM/OWL"
EMB = ROOT / "mw_exp/outputs/emb_bones"
LABELS = Path.home() / "Desktop/Tony/VLM/OWL/data/labels_disease14_all.jsonl"
SPLIT = ROOT / "data/split.csv"
WINS = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
WSHORT = {"baseline": [0], "soft": [1], "bone": [2], "all3": [0, 1, 2]}
POOLS = {"mean": [0], "max": [1], "meanmax": [0, 1]}
VERT = list(range(10))            # T8..L5
RIBS = [10, 11]
PELV = [12, 13, 14, 15, 16]       # sacrum, hip_l, hip_r, femur_l, femur_r
TARGETS = ["any", "vertebra", "compression", "rib", "pelvic"]


_SID = re.compile(r'"study_id":\s*"([^"]+)"')
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}


def load_labels(path, ids):
    """fracture labels for the given (train/val) ids only: a line is json-parsed only after its study_id has been
    matched against `ids`, so test label lines are never parsed (same fields as fracture_eval.load_labels)."""
    want, rows = set(ids), {}
    for l in open(path):
        m = _SID.search(l[:120])
        if not m or m.group(1) not in want:
            continue
        r = json.loads(l)
        a = r["answer"]["fracture"]
        d = a.get("detail", {}) or {}
        cat = a["category"]
        rows[r["study_id"]] = {"y": 1.0 if cat in POS else 0.0 if cat in NEG else np.nan,
                               "compression": d.get("vertebral_compression") == "yes",
                               "sites": d.get("sites", []) or [], "acute": d.get("acute", "not_stated")}
    return pd.DataFrame.from_dict(rows, orient="index")


def trval_ids():
    spl = pd.read_csv(SPLIT).set_index("study_id")["split"]
    return spl[spl.isin(["train", "val"])]


def targets(lab):
    """any: y; site targets: 1 if the site is reported, 0 if no fracture, NaN for fractures elsewhere."""
    y = lab["y"].values.astype(float)
    def site(f):
        s = lab.apply(f, axis=1).values
        return np.where(y == 0, 0.0, np.where((y == 1) & s, 1.0, np.nan))
    return {"any": y,
            "vertebra": site(lambda r: "vertebra" in r["sites"]),
            "compression": site(lambda r: bool(r["compression"])),
            "rib": site(lambda r: "rib" in r["sites"]),
            "pelvic": site(lambda r: any(x in r["sites"] for x in ("pelvis", "sacrum", "hip_femur")))}


def build(out):
    spl = trval_ids()                                            # test ids never loaded
    lab = load_labels(LABELS, spl.index)
    sets = [{p.stem for p in (EMB / w).glob("*.npz")} for w in WINS]
    print("per-window counts:", [len(s) for s in sets])
    ids = sorted(set.intersection(*sets) & set(spl.index) & set(lab.index[~lab["y"].isna()]))
    print(f"train+val ids with all 3 windows and a fracture label: {len(ids)}")
    E = np.zeros((len(ids), 17, 3, 2, 448), np.float16)
    P = np.zeros((len(ids), 17), bool)
    t0 = time.time()
    for i, cid in enumerate(ids):
        for w, wn in enumerate(WINS):
            z = np.load(EMB / wn / f"{cid}.npz")
            E[i, :, w, 0], E[i, :, w, 1] = z["emb"], z["emb_max"]
            if w == 0:
                P[i] = z["organ_present"]; units = list(z["organ_names"])
        if i % 2000 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out / "bone_matrix.npz", E=E, P=P, ids=np.array(ids), split=spl.loc[ids].values.astype(str), units=np.array(units))
    print("presence rate per unit:", dict(zip(units, P.mean(0).round(3))))


def load(out):
    z = np.load(out / "bone_matrix.npz", allow_pickle=True)
    ids, S = list(z["ids"]), z["split"]
    assert set(S) <= {"train", "val"}
    lab = load_labels(LABELS, ids).loc[ids]
    return ids, S, z["E"], z["P"], list(z["units"]), lab, targets(lab)


def std_unit(X, present, fit):
    """X [N,D] one unit; z-score with fit-row present stats; absent -> 0."""
    r = X[fit & present].astype(np.float32)
    mu, sd = r.mean(0), r.std(0) + 1e-6
    return ((X.astype(np.float32) - mu) / sd) * present[:, None]


def screen(out, n_jobs):
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    ids, S, E, P, units, lab, T = load(out)
    tr, va = S == "train", S == "val"

    def one(u, wk, pk, tname):
        y = T[tname]
        X = np.concatenate([std_unit(E[:, u, w, p], P[:, u], tr) for w in WSHORT[wk] for p in POOLS[pk]], 1)
        X = np.c_[X, P[:, u].astype(np.float32)]
        mt, mv = tr & ~np.isnan(y), va & ~np.isnan(y)
        best = (-1, None, None)
        for C in (1e-3, 1e-2, 1e-1):
            clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced").fit(X[mt], y[mt])
            s = clf.decision_function(X[va])
            a = auroc(y[va], s)
            if a > best[0]:
                best = (a, C, s)
        return {"unit": units[u], "win": wk, "pool": pk, "target": tname, "C": best[1], "val_auc_target": best[0],
                "val_auc_any": auroc(T["any"][va], best[2])}

    jobs = [(u, wk, pk, t) for t in TARGETS for wk in ("baseline", "soft", "bone") for pk in ("mean", "max")
            for u in range(17)]
    jobs += [(u, "all3", "meanmax", "any") for u in range(17)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(res)
    df.to_csv(out / "screen_per_unit.csv", index=False)
    print(df[df.target == "any"].pivot_table(index="unit", columns=["win", "pool"], values="val_auc_any").round(3).to_string())


class BoneModel(nn.Module):
    def __init__(self, kind, D, U=17):
        super().__init__()
        self.kind = kind
        if kind == "concat":
            self.lin = nn.Linear(U * D + U, 1)
        elif kind.startswith("milv"):
            self.h = nn.Linear(D, 1); self.lb = nn.Parameter(torch.zeros(10)); self.none = nn.Parameter(torch.zeros(1))
        elif kind.startswith("site"):
            self.heads = nn.ModuleList([nn.Linear(D, 1) for _ in range(3)])
            self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(3))
            self.b = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(kind)

    @staticmethod
    def pool(s, m, how):
        """s [N,K] level logits, m [N,K] presence; returns [N] and whether any level is present."""
        s = s.masked_fill(~m, -1e4)
        r = s.max(1).values if how == "max" else torch.logsumexp(s, 1)
        return r, m.any(1)

    def forward(self, X, M):
        """X [N,17,D] standardized (absent = 0), M [N,17] bool. Returns any-logit [N] and site logits [N,3] or None."""
        if self.kind == "concat":
            return self.lin(torch.cat([X.flatten(1), M.float()], 1))[:, 0], None
        how = "max" if "_max" in self.kind else "lse"
        if self.kind.startswith("milv"):
            s = self.h(X[:, VERT])[..., 0] + self.lb
            r, any_ = self.pool(s, M[:, VERT], how)
            return torch.where(any_, r, self.none.expand_as(r)), None
        sites = []
        for k, idx in enumerate((VERT, RIBS, PELV)):
            s = self.heads[k](X[:, idx])[..., 0] + self.ub[idx]
            r, any_ = self.pool(s, M[:, idx], how)
            sites.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(sites, 1)
        return torch.logsumexp(St, 1) + self.b, St


def prep_X(E, P, wk, pk, fit_mask):
    """-> float32 tensor [N,17,D'] standardized per unit (fit-row stats), absent = 0."""
    parts = []
    for u in range(17):
        parts.append(np.concatenate([std_unit(E[:, u, w, p], P[:, u], fit_mask) for w in WSHORT[wk] for p in POOLS[pk]], 1))
    return torch.from_numpy(np.stack(parts, 1))


def site_targets(T):
    return np.stack([T["vertebra"], T["rib"], T["pelvic"]], 1)


def train(kind, X, M, y, ys, tr_idx, es_idx, lr, wd, seed, dev, max_epochs=100, patience=10, bs=512):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BoneModel(kind, X.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, Mt = X[tr_idx].to(dev), M[tr_idx].to(dev)
    yt = torch.from_numpy(y[tr_idx]).float().to(dev)
    yst = torch.from_numpy(ys[tr_idx]).float().to(dev)
    vs = ~torch.isnan(yst)
    def wts(v, valid):
        n1 = (v.nan_to_num(0) * valid).sum(0); n = valid.sum(0)
        return n / (2 * n1.clamp(min=1)), n / (2 * (n - n1).clamp(min=1))
    wp, wn = wts(yt, torch.ones_like(yt, dtype=torch.bool))
    swp, swn = wts(yst, vs)
    Xe, Me = X[es_idx].to(dev), M[es_idx].to(dev)
    best, state, bad, best_ep = -1.0, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            la, ls = model(Xt[j], Mt[j])
            loss = (F.binary_cross_entropy_with_logits(la, yt[j], reduction="none") * torch.where(yt[j] == 1, wp, wn)).mean()
            if ls is not None and not kind.endswith("_noaux"):     # *_noaux: site heads, any-fracture loss only
                m = vs[j]
                w = torch.where(yst[j] == 1, swp, swn) * m
                loss = loss + (F.binary_cross_entropy_with_logits(ls, yst[j].nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], model(Xe, Me)[0].cpu().numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def predict(model, X, M, idx, dev):
    out_a, out_s = [], []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            a, s = model(X[j].to(dev), M[j].to(dev))
            out_a.append(a.cpu()); out_s.append(s.cpu() if s is not None else torch.zeros(len(j), 0))
    return torch.cat(out_a).numpy(), torch.cat(out_s).numpy()


def agg(out, dev, kinds, pools, wins, seeds):
    ids, S, E, P, units, lab, T = load(out)
    tr, va = S == "train", S == "val"
    y, ys = T["any"], site_targets(T)
    M = torch.from_numpy(P)
    tr_idx, va_idx = np.where(tr)[0], np.where(va)[0]
    (out / "agg").mkdir(exist_ok=True)
    old = out / "agg_summary.json"                     # legacy single file -> per-model json
    if old.exists():
        for k, v in json.loads(old.read_text()).items():
            if not (out / "agg" / f"{k}.json").exists():
                (out / "agg" / f"{k}.json").write_text(json.dumps(v, indent=1))
    for pk, wk in itertools.product(pools, wins):
        X = prep_X(E, P, wk, pk, tr)
        for kind in kinds:
            name = f"{kind}__{pk}__{wk}"
            if (out / "agg" / f"{name}.json").exists():
                continue
            t0 = time.time()
            grid = []
            for lr, wd in itertools.product((1e-3, 1e-2), (1e-4, 1e-3, 1e-2)):
                _, s, ep = train(kind, X, M, y, ys, tr_idx, va_idx, lr, wd, 42, dev)
                grid.append({"lr": lr, "wd": wd, "val": s, "ep": ep})
            hp = max(grid, key=lambda g: g["val"])
            la, ls, vals = [], [], []
            for sd in range(seeds):
                m, s, _ = train(kind, X, M, y, ys, tr_idx, va_idx, hp["lr"], hp["wd"], 42 + sd, dev)
                a, st = predict(m, X, M, va_idx, dev)
                la.append(a); ls.append(st); vals.append(s)
            a = np.mean(la, 0)
            np.savez(out / "agg" / f"{name}.npz", ids=np.array(ids)[va_idx], logit=a, site_logit=np.mean(ls, 0))
            r = {"hp": hp, "seed_val": vals, "val_auc_any_seedmean": auroc(y[va_idx], a)}
            (out / "agg" / f"{name}.json").write_text(json.dumps(r, indent=1))
            print(f"{name}: val={r['val_auc_any_seedmean']:.4f} hp={hp} ({time.time() - t0:.0f}s)", flush=True)


def oof(out, dev, kind, pk, wk, folds=5, tag=None):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    ids, S, E, P, units, lab, T = load(out)
    y, ys = T["any"], site_targets(T)
    M = torch.from_numpy(P)
    name = f"{kind}__{pk}__{wk}"
    hp = json.loads((out / "agg" / f"{name}.json").read_text())["hp"]
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    o = np.full(len(ids), np.nan, np.float32)
    vl, info = [], []
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fp], tr_all[hpos]
        X = prep_X(E, P, wk, pk, np.isin(np.arange(len(ids)), fit))
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        m, s, ep = train(kind, X, M, y, ys, inner, es, hp["lr"], hp["wd"], 42 + k, dev)
        o[hold] = predict(m, X, M, hold, dev)[0]
        vl.append(predict(m, X, M, va_all, dev)[0])
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold]), "val": auroc(y[va_all], vl[-1])})
        print(info[-1], f"{time.time() - t0:.0f}s", flush=True)
    o[va_all] = np.mean(vl, 0)
    print(f"{name}: OOF train AUROC={auroc(y[tr_all], o[tr_all]):.4f} val={auroc(y[va_all], o[va_all]):.4f}")
    np.savez(out / f"oof_bone_{tag or name}.npz", ids=np.array(ids), split=S, logit=o, y=y, exp=name, hp=json.dumps(hp),
             folds=json.dumps(info))


# ---------------------------------------------------------------------------------------------------------------
# Rerun of the probe.py window experiments on the bone units (stage `modes`, `wablate`, `modes_oof`).
# Same site-wise MIL as site_max (3 site heads + per-unit bias, site = max over present units, any = logsumexp
# over sites, site sub-label aux loss), same optimiser / grid / early stopping / 3 seeds as `agg`; only the window
# handling changes, as in probe.py:
#   single (E0 = baseline, E1_soft, E1_bone) | concat (E2) | gate (E3: per-unit softmax gate, weighted sum of the
#   window embeddings) | late (E3_late: one site head per window, per-unit softmax gate mixes the window logits) |
#   preset (E4) | randfeat (E0_rf: baseline + fixed random ReLU features, width = concat width)
# extras: trap (training-only a[age_bin] + g(metadata), N(0, sd^2) prior, empirical-Bayes init, zero-init g),
#   trap2 (prior sd 1.0, metadata + continuous age, age^2), inlp (probe.inlp_remove_age, 20 iterations).
# Only the 3 real bone windows exist (baseline / soft_tissue / bone); no liver / narrow_abd substitution.
SITES = (VERT, RIBS, PELV)
WIN3 = ["baseline", "soft", "bone"]
MODE_EXPS = {
    "E0": {"mode": "single", "win_idx": 0},
    "E1_soft": {"mode": "single", "win_idx": 1},
    "E1_bone": {"mode": "single", "win_idx": 2},
    "E2": {"mode": "concat"},
    "E3": {"mode": "gate"},
    "E3_late": {"mode": "late"},
    "E4": {"mode": "preset"},          # presets: bone units inherit lumbar_spine -> bone (== E1_bone, see notes)
    "E0_rf": {"mode": "randfeat", "win_idx": 0, "n_rf": 2 * 448},
}
for _b in ("E0", "E2", "E3_late"):
    MODE_EXPS[f"{_b}_trap"] = dict(MODE_EXPS[_b], trap=True)
    MODE_EXPS[f"{_b}_trap2"] = dict(MODE_EXPS[_b], trap=True, trap_plus=True, trap_prior_sd=1.0)
    MODE_EXPS[f"{_b}_inlp"] = dict(MODE_EXPS[_b], inlp=True)
    MODE_EXPS[f"{_b}_trap2_inlp"] = dict(MODE_EXPS[_b], trap=True, trap_plus=True, trap_prior_sd=1.0, inlp=True)


class ModeModel(nn.Module):
    def __init__(self, spec, W, D, U=17, seed=0):
        super().__init__()
        self.mode = mode = spec["mode"]
        self.win_idx = spec.get("win_idx", 0)
        if mode == "late":
            bd = 1 / D ** 0.5                                            # nn.Linear default init range
            self.hw = nn.Parameter(torch.empty(3, W, D).uniform_(-bd, bd))
            self.hb = nn.Parameter(torch.empty(3, W).uniform_(-bd, bd))
        else:
            din = {"concat": W * D, "randfeat": D + spec.get("n_rf", 0)}.get(mode, D)
            self.heads = nn.ModuleList([nn.Linear(din, 1) for _ in range(3)])
        self.gate = nn.Parameter(torch.zeros(U, W)) if mode in ("gate", "late") else None
        if mode == "randfeat":
            g = torch.Generator().manual_seed(1234 + seed)
            self.register_buffer("R", torch.randn(D, spec["n_rf"], generator=g) / D ** 0.5)
        if mode == "preset":
            self.register_buffer("preset_idx", torch.tensor(spec["preset_idx"]))
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(3))
        self.b = nn.Parameter(torch.zeros(1))

    def unit_repr(self, X):
        N, U, W, D = X.shape
        if self.mode == "single":
            return X[:, :, self.win_idx]
        if self.mode == "concat":
            return X.reshape(N, U, W * D)
        if self.mode == "gate":
            return torch.einsum("nuwd,uw->nud", X, torch.softmax(self.gate, -1))
        if self.mode == "preset":
            return X[:, torch.arange(U, device=X.device), self.preset_idx]
        if self.mode == "randfeat":
            x = X[:, :, self.win_idx]
            return torch.cat([x, torch.relu(x @ self.R)], -1)
        raise ValueError(self.mode)

    def forward(self, X, M):
        Z = None if self.mode == "late" else self.unit_repr(X)
        sites = []
        for k, idx in enumerate(SITES):
            if self.mode == "late":
                L = torch.einsum("nuwd,wd->nuw", X[:, idx], self.hw[k]) + self.hb[k]
                s = (L * torch.softmax(self.gate[idx], -1)[None]).sum(-1)
            else:
                s = self.heads[k](Z[:, idx])[..., 0]
            s = s + self.ub[idx]
            r, any_ = BoneModel.pool(s, M[:, idx], "max")
            sites.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(sites, 1)
        return torch.logsumexp(St, 1) + self.b, St


class TrapB(nn.Module):
    """training-only bias branch for the single fracture target: a[age_bin] + g(metadata)."""

    def __init__(self, a_init, k):
        super().__init__()
        self.a = nn.Parameter(a_init.clone())
        self.g = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)

    def forward(self, b, X):
        return self.a[b] + self.g(X)[:, 0]


def mode_X(E, P, S, standardize, inlp_age=None):
    """E [N,17,3,2,448] fp16 -> mean-pooled [N,17,3,448] float32; per (unit, window) z-score with train present rows
    (bone_probe recipe, == probe.py --standardize for bone units); absent units = 0. Optional INLP age removal."""
    tr = S == "train"
    X = np.zeros(E.shape[:3] + E.shape[4:], np.float32)
    for u in range(E.shape[1]):
        for w in range(3):
            X[:, u, w] = std_unit(E[:, u, w, 0], P[:, u], tr) if standardize else E[:, u, w, 0].astype(np.float32) * P[:, u, None]
    X = torch.from_numpy(X)
    if inlp_age is not None:
        from probe import inlp_remove_age
        X, r2b, r2a = inlp_remove_age(X, torch.from_numpy(P.astype(np.float32)), S, inlp_age, n_iter=20)
        print(f"INLP x20: val R^2 age ~ unit embedding {r2b:.3f} -> {r2a:.3f}", flush=True)
        return X, (r2b, r2a)
    return X, None


def trap_inputs(ids):
    from probe import build_trap_inputs
    return build_trap_inputs({"metadata_file": str(Path.home() / "Desktop/Tony/VLM/merlin_disease14/metadata.csv")}, ids)


def train_m(spec, X, M, y, ys, tr_idx, es_idx, lr, wd, seed, dev, TR=None, max_epochs=100, patience=10, bs=512):
    """bone_probe.train with ModeModel (+ optional training-only trap); identical loss / early stopping."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ModeModel(spec, X.shape[2], X.shape[3], seed=seed).to(dev)
    groups = [{"params": list(model.parameters()), "lr": lr}]
    trap = None
    Xt, Mt = X[tr_idx].to(dev), M[tr_idx].to(dev)
    yt = torch.from_numpy(y[tr_idx]).float().to(dev)
    if spec.get("trap"):
        btr = TR["bin"][tr_idx]
        pf = yt.mean().cpu()
        pos_b = torch.zeros(TR["n_bins"]).index_add_(0, btr, yt.cpu())
        n_b = torch.zeros(TR["n_bins"]).index_add_(0, btr, torch.ones(len(btr)))
        rate = (pos_b + 20.0 * pf) / (n_b + 20.0)
        lg = lambda q: torch.log(q.clamp(1e-4, 1 - 1e-4) / (1 - q.clamp(1e-4, 1 - 1e-4)))
        key = "X_plus" if spec.get("trap_plus") else "X"
        trap = TrapB(lg(rate) - lg(pf), TR[key].shape[1]).to(dev)
        prior_sd = float(spec.get("trap_prior_sd", 0.5))
        Bt, Mt_meta = btr.to(dev), TR[key][tr_idx].to(dev)
        groups.append({"params": list(trap.parameters()), "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    yst = torch.from_numpy(ys[tr_idx]).float().to(dev)
    vs = ~torch.isnan(yst)
    def wts(v, valid):
        n1 = (v.nan_to_num(0) * valid).sum(0); n = valid.sum(0)
        return n / (2 * n1.clamp(min=1)), n / (2 * (n - n1).clamp(min=1))
    wp, wn = wts(yt, torch.ones_like(yt, dtype=torch.bool))
    swp, swn = wts(yst, vs)
    Xe, Me = X[es_idx].to(dev), M[es_idx].to(dev)
    best, state, bad, best_ep = -1.0, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            la, ls = model(Xt[j], Mt[j])
            if trap is not None:
                la = la + trap(Bt[j], Mt_meta[j])
            loss = (F.binary_cross_entropy_with_logits(la, yt[j], reduction="none") * torch.where(yt[j] == 1, wp, wn)).mean()
            m = vs[j]
            w = torch.where(yst[j] == 1, swp, swn) * m
            loss = loss + (F.binary_cross_entropy_with_logits(ls, yst[j].nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
            if trap is not None:
                loss = loss + (trap.a ** 2).sum() / (2 * prior_sd ** 2) / n
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], torch.cat([model(Xe[i:i + 2048], Me[i:i + 2048])[0] for i in range(0, len(Xe), 2048)]).cpu().numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def _mode_setup(out, standardize, need_inlp):
    ids, S, E, P, units, lab, T = load(out)
    TR = trap_inputs(ids)
    X, r2 = mode_X(E, P, S, standardize, TR["age"] if need_inlp else None)
    del E
    return ids, S, X, torch.from_numpy(P), units, T, TR, r2


def _spec(name, units):
    spec = dict(MODE_EXPS[name])
    if spec["mode"] == "preset":
        spec["preset_idx"] = [2] * len(units)        # windows.yaml: lumbar_spine -> bone; bone units inherit it
    return spec


def modes(out, dev, exps, standardize, seeds, sub):
    od = out / "rerun" / sub; od.mkdir(parents=True, exist_ok=True)
    exps = [e for e in exps if not (od / f"{e}.json").exists()]
    if not exps:
        return
    groups = {}
    for e in exps:
        groups.setdefault(bool(MODE_EXPS[e].get("inlp")), []).append(e)
    for inl, es in groups.items():
        ids, S, X, M, units, T, TR, r2 = _mode_setup(out, standardize, inl)
        y, ys = T["any"], site_targets(T)
        tr_idx, va_idx = np.where(S == "train")[0], np.where(S == "val")[0]
        for name in es:
            spec = _spec(name, units)
            t0 = time.time()
            grid = []
            for lr, wd in itertools.product((1e-3, 1e-2), (1e-4, 1e-3, 1e-2)):
                _, s, ep = train_m(spec, X, M, y, ys, tr_idx, va_idx, lr, wd, 42, dev, TR)
                grid.append({"lr": lr, "wd": wd, "val": s, "ep": ep})
            hp = max(grid, key=lambda g: g["val"])
            la, ls, vals, gates = [], [], [], []
            for sd in range(seeds):
                m, s, _ = train_m(spec, X, M, y, ys, tr_idx, va_idx, hp["lr"], hp["wd"], 42 + sd, dev, TR)
                a, st = predict(m, X, M, va_idx, dev)
                la.append(a); ls.append(st); vals.append(s)
                if m.gate is not None:
                    gates.append(torch.softmax(m.gate, -1).detach().cpu().numpy())
            a = np.mean(la, 0)
            np.savez(od / f"{name}.npz", ids=np.array(ids)[va_idx], logit=a, site_logit=np.mean(ls, 0),
                     gate=np.stack(gates) if gates else np.zeros(0))
            r = {"spec": {k: v for k, v in spec.items() if k != "preset_idx"}, "standardize": standardize, "grid": grid,
                 "hp": hp, "seed_val": vals, "val_auc_any_seedmean": auroc(y[va_idx], a), "inlp_r2": r2}
            (od / f"{name}.json").write_text(json.dumps(r, indent=1))
            print(f"[{sub}] {name}: val={r['val_auc_any_seedmean']:.4f} hp={hp} ({time.time() - t0:.0f}s)", flush=True)


def wablate(out, dev, base, sub, seeds):
    """leave-one-window-out for `base` (E3_late or E2) with its chosen HP; 3 seeds; val."""
    od = out / "rerun" / sub
    hp = json.loads((od / f"{base}.json").read_text())["hp"]
    ids, S, X, M, units, T, TR, _ = _mode_setup(out, sub.startswith("std"), False)
    y, ys = T["any"], site_targets(T)
    tr_idx, va_idx = np.where(S == "train")[0], np.where(S == "val")[0]
    spec = _spec(base, units)
    for wi, wn in enumerate(WIN3):
        name = f"{base}_without_{wn}"
        if (od / f"{name}.json").exists():
            continue
        keep = [j for j in range(3) if j != wi]
        Xk = X[:, :, keep].contiguous()
        la, ls = [], []
        for sd in range(seeds):
            m, s, _ = train_m(spec, Xk, M, y, ys, tr_idx, va_idx, hp["lr"], hp["wd"], 42 + sd, dev, TR)
            a, st = predict(m, Xk, M, va_idx, dev)
            la.append(a); ls.append(st)
        a = np.mean(la, 0)
        np.savez(od / f"{name}.npz", ids=np.array(ids)[va_idx], logit=a, site_logit=np.mean(ls, 0), gate=np.zeros(0))
        (od / f"{name}.json").write_text(json.dumps({"hp": hp, "val_auc_any_seedmean": auroc(y[va_idx], a)}, indent=1))
        print(f"[{sub}] {name}: val={auroc(y[va_idx], a):.4f}", flush=True)


def modes_oof(out, dev, name, sub, seed_offs, folds=5):
    """bone_label_exp `ref` recipe for a mode experiment: 5-fold OOF on TRAIN (inner 10% early stop; val unused),
    HP = the val-chosen HP of stage `modes` (as Exp B/F). -> rerun/<sub>/oof/oof_bone_E_ref__<name>_s<k>.npz"""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    od = out / "rerun" / sub / "oof"; od.mkdir(parents=True, exist_ok=True)
    hp = json.loads((out / "rerun" / sub / f"{name}.json").read_text())["hp"]
    spec = MODE_EXPS[name]
    ids, S, E, P, units, lab, T = load(out)
    y, ys = T["any"], site_targets(T)
    TR = trap_inputs(ids)
    M = torch.from_numpy(P)
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    for so in seed_offs:
        f = od / f"oof_bone_E_ref__{name}_s{so}.npz"
        if f.exists():
            continue
        o = np.full(len(ids), np.nan, np.float32); vl, info = [], []
        for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
            fit, hold = tr_all[fp], tr_all[hpos]
            S2 = np.where(np.isin(np.arange(len(ids)), fit), "train", "other")
            X, _ = mode_X(E, P, S2, sub.startswith("std"), TR["age"] if spec.get("inlp") else None)
            inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
            m, s, ep = train_m(_spec(name, units), X, M, y, ys, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * so, dev, TR)
            o[hold] = predict(m, X, M, hold, dev)[0]
            vl.append(predict(m, X, M, va_all, dev)[0])
            info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold])})
        o[va_all] = np.mean(vl, 0)
        cv = auroc(y[tr_all], o[tr_all])
        print(f"OOF {name} s{so}: train OOF={cv:.4f} val={auroc(y[va_all], o[va_all]):.4f}", flush=True)
        np.savez(f, ids=np.array(ids), split=S, logit=o, y=y, exp=f"ref:{name}", hp=json.dumps(hp), folds=json.dumps(info),
                 oof_train_auc=cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default=str(ROOT / "mw_exp/outputs_fracture/bones"))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--kinds", nargs="*", default=["concat", "milv_max", "milv_lse", "site_max", "site_lse"])
    ap.add_argument("--pools", nargs="*", default=["mean", "max", "meanmax"])
    ap.add_argument("--wins", nargs="*", default=["baseline", "soft", "bone", "all3"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default=None); ap.add_argument("--pool", default=None); ap.add_argument("--win", default=None)
    ap.add_argument("--exps", nargs="*", default=["E0", "E1_soft", "E1_bone", "E2", "E3", "E3_late", "E0_rf"])
    ap.add_argument("--nostd", action="store_true", help="modes: raw embeddings (no per-unit z-score)")
    ap.add_argument("--sub", default=None)
    ap.add_argument("--seed-offs", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(a.threads)
    if a.stage == "build":
        build(out)
    elif a.stage == "screen":
        screen(out, a.jobs)
    elif a.stage == "agg":
        agg(out, dev, a.kinds, a.pools, a.wins, a.seeds)
    elif a.stage == "oof":
        oof(out, dev, a.model, a.pool, a.win)
    elif a.stage == "modes":
        modes(out, dev, a.exps, not a.nostd, a.seeds, a.sub or ("nostd" if a.nostd else "std"))
    elif a.stage == "wablate":
        wablate(out, dev, a.model, a.sub or "std", a.seeds)
    elif a.stage == "modes_oof":
        modes_oof(out, dev, a.model, a.sub or "std", a.seed_offs)


if __name__ == "__main__":
    main()
