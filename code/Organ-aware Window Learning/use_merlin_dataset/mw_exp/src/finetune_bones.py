"""Experiment H: finetune the TotalSegmentator --fast nnU-Net encoder on ALL 17 bone units with the site-wise MIL
head (bone_probe.BoneModel 'site_max': vertebra / rib / pelvis heads, max within a site, logsumexp across sites),
optionally with the training-only trap branch (age bins + metadata).

Exactness: per bone unit the tiles of extract_bones.py are forwarded once per window (baseline, soft_tissue, bone;
the shared single-channel encoder, batch = tiles x 3 windows), stage 1-3 mask mean pooling, per-stage L2, concat,
tile average weighted by unit voxels -> [3 x 448] per unit, standardized with the frozen per-unit train statistics.
With the encoder frozen this reproduces emb_bones/ (mean pool) and therefore the frozen probe (stage `check`).

Training (stage `train`):
  * head initialised by training the frozen probe (site_max, mean, all3, lr 1e-3, wd 1e-2) on the cached frozen
    embeddings of TRAIN minus a 10% stratified inner-validation set (seed 42); its early stopping uses inner-val.
  * per case only a few units are forwarded with gradients (per site the top-`sel_top` units by the frozen head's
    unit logit + `sel_rand` random others; max-pooling within a site sends gradient to the arg-max unit only);
    the remaining units use their cached frozen embeddings (detached).
  * encoder: last `train_stages` stages trainable (default 2 = stages 2, 3), earlier stages run without grad.
  * epoch = all positive fit cases + `neg_per_pos` x as many random negatives (new negatives every epoch).
  * model selection / early stopping: any-fracture AUROC on inner-val (full forward of every unit); val is
    predicted once, at the end, with the selected checkpoint. The test split is never loaded: split.csv is filtered
    to train/val first and only label lines whose study_id is in that set are parsed.
Stages: units (bone-unit volumes + tiles per case from TS masks), check, train, predict.
usage (Spark):
  python finetune_bones.py --stage units --jobs 8
  python finetune_bones.py --stage check --check-n 30
  python finetune_bones.py --stage train --run H1_trap --trap
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from bone_probe import PELV, RIBS, VERT, BoneModel, targets  # noqa: E402
from bone_probe import train as train_frozen_head  # noqa: E402
from common import ROOT, load_cfg  # noqa: E402
from extract import crop, pool_stage  # noqa: E402
from probe import Trap, auroc, build_trap_inputs  # noqa: E402
from windowing import apply_window  # noqa: E402

HOME = Path.home()
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}
WIN_NAMES = ["baseline", "soft_tissue", "bone"]          # cache order of bone_matrix.npz
SITES = [VERT, RIBS, PELV]
U = 17


# ----------------------------------------------------------------------------------------------- labels / split
def load_split(path):
    spl = pd.read_csv(path).set_index("study_id")["split"]
    return spl[spl.isin(["train", "val"])]                                   # test ids dropped before anything else


def load_labels_trval(path, keep):
    """fracture labels for train/val ids only: the id is matched on the raw line before the line is parsed."""
    rx, rows = re.compile(r'"study_id"\s*:\s*"([^"]+)"'), {}
    for line in open(path):
        m = rx.search(line)
        if not m or m.group(1) not in keep:
            continue
        r = json.loads(line)
        a = r["answer"]["fracture"]
        d = a.get("detail", {}) or {}
        cat = a["category"]
        rows[r["study_id"]] = {"y": 1.0 if cat in POS else 0.0 if cat in NEG else np.nan,
                               "compression": d.get("vertebral_compression") == "yes",
                               "sites": d.get("sites", []) or [], "acute": d.get("acute", "not_stated")}
    return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------------------------------------- stage: units
def _units_one(args):
    cid, cfg_path, mask_dir, out_dir = args
    dst = Path(out_dir) / f"{cid}.npz"
    if dst.exists():
        return cid, "skip"
    try:
        import nibabel as nib
        from extract_bones import unit_tiles
        from patches import label_lookup, units_from_labels
        cfg = _cfg(cfg_path)
        prep = np.load(prep_path(cfg, cid))
        hu, aff = prep["hu"], prep["affine"]
        msk = nib.load(str(Path(mask_dir) / f"{cid}.nii.gz"))
        regrid = msk.shape != hu.shape[::-1] or not np.allclose(msk.affine, aff, atol=1e-3)
        if regrid:                                       # identical to extract_bones.load_units
            from nibabel.processing import resample_from_to
            from totalsegmentator.alignment import as_closest_canonical
            msk = resample_from_to(as_closest_canonical(msk), (hu.shape[::-1], aff), order=0)
        lab = np.asanyarray(msk.dataobj).astype(np.int32).transpose(2, 1, 0)
        units = units_from_labels(cfg, lab, label_lookup(cfg))
        P = json.loads((Path(cfg["ts_model_dir"]) / "plans.json").read_text())["configurations"]["3d_fullres"]["patch_size"]
        tiles, nvox = unit_tiles(cfg, units, U, P)
        T = np.array([[k, *s, n] for k, tl in enumerate(tiles) for s, n in tl], np.int64).reshape(-1, 5)
        tmp = dst.with_name(f"{cid}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, units=units, tiles=T, nvox=nvox, regrid=np.array(regrid), shape=np.array(hu.shape))
        tmp.rename(dst)
        return cid, "ok"
    except Exception as e:  # noqa: BLE001
        return cid, f"ERR {e!r}"


def stage_units(a):
    from multiprocessing import Pool
    out = Path(a.data) / "bone_units"; out.mkdir(parents=True, exist_ok=True)
    spl = load_split(a.split)
    cfg = _cfg(a.config)
    have = {p.name[:-7] for p in Path(a.mask_dir).glob("*.nii.gz")}
    ids = [i for i in spl.index if i in have and prep_path(cfg, i) is not None]
    todo = [i for i in ids if not (out / f"{i}.npz").exists()]
    print(f"train/val ids {len(spl)}, with mask+prep {len(ids)}, to do {len(todo)}", flush=True)
    t0, n = time.time(), 0
    with Pool(a.jobs) as pool:
        for cid, st in pool.imap_unordered(_units_one, [(c, a.config, a.mask_dir, str(out)) for c in todo], chunksize=4):
            n += 1
            if st.startswith("ERR"):
                print(cid, st, flush=True)
            if n % 500 == 0:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"units done: {n} in {time.time() - t0:.0f}s", flush=True)


_CFG = {}


def _cfg(path):
    if path not in _CFG:
        cfg = load_cfg(path)
        cfg["organ_map"] = yaml.safe_load(open(ROOT / "organ_map_bones.yaml"))
        assert list(cfg["organ_map"])[:2] == ["T8", "T9"] and len(cfg["organ_map"]) == U
        _CFG[path] = cfg
    return _CFG[path]


# ----------------------------------------------------------------------------------------------- data
PREP_DIRS = []


def prep_path(cfg, cid):
    """first existing prep/{id}.npz in --prep-dirs (default: <output_dir>/prep)."""
    for d in PREP_DIRS or [cfg["out"] / "prep"]:
        p = Path(d) / f"{cid}.npz"
        if p.exists():
            return p
    return None


class BoneCases(Dataset):
    """CPU side: crop the tiles of the requested units (HU int16 + unit mask)."""

    def __init__(self, cfg, data_dir, ids, P, S0=None, pres=None, sel_top=1, sel_rand=1, train=False, seed=0):
        self.cfg, self.dd, self.ids, self.P = cfg, Path(data_dir), ids, P
        self.S0, self.pres, self.sel_top, self.sel_rand, self.train, self.seed = S0, pres, sel_top, sel_rand, train, seed
        self.items, self.epoch = None, 0             # items: list of case rows (indices into ids)

    def __len__(self):
        return len(self.items)

    def select(self, r, rng):
        """per site: top-k units by frozen unit logit + random others (present units only)."""
        sel = []
        for site in SITES:
            pr = [u for u in site if self.pres[r, u]]
            if not pr:
                continue
            order = sorted(pr, key=lambda u: -self.S0[r, u])
            top = order[:self.sel_top]
            rest = order[self.sel_top:]
            if rest and self.sel_rand:
                top += list(rng.choice(rest, min(self.sel_rand, len(rest)), replace=False))
            sel += top
        return sorted(sel)

    def __getitem__(self, i):
        t0 = time.time()
        r = self.items[i]
        cid = self.ids[r]
        hu = np.load(prep_path(self.cfg, cid))["hu"]
        z = np.load(self.dd / "bone_units" / f"{cid}.npz")
        units, T = z["units"], z["tiles"]
        assert tuple(z["shape"]) == hu.shape, (cid, z["shape"], hu.shape)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, int(r)))
            sel = set(self.select(r, rng))
            jit = (float(rng.uniform(0.97, 1.03)), float(rng.uniform(-20, 20)))
        else:
            sel, jit = set(np.unique(T[:, 0]).tolist()), None
        hs, ms, tu, nv = [], [], [], []
        for k, z0, y0, x0, n in T:
            if k in sel:
                hs.append(crop(hu, [z0, y0, x0], self.P, self.cfg["pad_hu"]))
                ms.append(crop(units, [z0, y0, x0], self.P, 0) == k + 1)
                tu.append(k); nv.append(n)
        return {"r": int(r), "id": cid, "jit": jit,
                "hu": torch.from_numpy(np.stack(hs)) if hs else torch.zeros((0, *self.P), dtype=torch.int16),
                "mask": torch.from_numpy(np.stack(ms)) if ms else torch.zeros((0, *self.P), dtype=torch.bool),
                "tile_unit": tu, "n_vox": nv, "load_s": time.time() - t0}


def collate_list(b):
    return b


# ----------------------------------------------------------------------------------------------- encoder
class Enc(torch.nn.Module):
    """nnU-Net encoder stages 0..3 (single input channel); the first n_frozen stages run without grad."""

    def __init__(self, nnunet_encoder, upto):
        super().__init__()
        self.stem = getattr(nnunet_encoder, "stem", None)
        self.stages = torch.nn.ModuleList(list(nnunet_encoder.stages[:upto + 1]))
        self.n_frozen = len(self.stages)
        self.grad_ckpt = False

    def set_trainable(self, k):
        for p in self.parameters():
            p.requires_grad_(False)
        self.n_frozen = len(self.stages) - k
        for st in self.stages[self.n_frozen:]:
            for p in st.parameters():
                p.requires_grad_(True)

    def forward(self, x):
        out = []
        with torch.no_grad():
            if self.stem is not None:
                x = self.stem(x)
            for st in self.stages[:self.n_frozen]:
                x = st(x); out.append(x)
        for st in self.stages[self.n_frozen:]:
            x = checkpoint(st, x, use_reentrant=False) if (self.grad_ckpt and torch.is_grad_enabled()) else st(x)
            out.append(x)
        return out


def embed_units(enc, case, wins, norm, stages, min_w, dev, amp_dtype, tile_batch):
    """-> {unit: [3, 448] float32} (window-major, same layout as emb_bones), differentiable if enc trains."""
    hu = case["hu"]
    T = hu.shape[0]
    W = len(wins)
    acc, wsum = {}, {}
    for i0 in range(0, T, tile_batch):
        h = hu[i0:i0 + tile_batch].to(dev, non_blocking=True).float()
        if case["jit"] is not None:
            h = h * case["jit"][0] + case["jit"][1]
        x = torch.stack([apply_window(h, w, norm) for w in wins], 1)                # [t, W, Z, Y, X]
        x = x.reshape(-1, 1, *x.shape[2:])                                          # tile-major, window-minor
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            feats = enc(x)
        m = case["mask"][i0:i0 + tile_batch].to(dev, non_blocking=True)
        for b in range(h.shape[0]):
            ti = i0 + b
            k, n = case["tile_unit"][ti], float(case["n_vox"][ti])
            mb = m[b][None, None].float()
            vs = [F.normalize(pool_stage(feats[s][b * W:(b + 1) * W], mb, min_w)[0], dim=1) for s in stages]
            e = torch.cat(vs, 1) * n                                                 # [W, 448]
            acc[k] = acc[k] + e if k in acc else e
            wsum[k] = wsum.get(k, 0.0) + n
    return {k: acc[k] / wsum[k] for k in acc}


# ----------------------------------------------------------------------------------------------- setup
def setup(a, dev):
    cfg = _cfg(a.config)
    spl = load_split(a.split)
    z = np.load(Path(a.data) / "bone_cache_mean.npz", allow_pickle=True)
    assert set(np.unique(z["split"])) <= {"train", "val"}
    assert list(z["units"]) == list(cfg["organ_map"])
    cids = [str(i) for i in z["ids"]]
    have = {p.stem for p in (Path(a.data) / "bone_units").glob("*.npz") if ".tmp" not in p.name}
    keep_rows = np.array([c in have and c in spl.index and prep_path(cfg, c) is not None for c in cids])
    lab = load_labels_trval(a.labels, set(np.array(cids)[keep_rows]))
    yv = lab["y"].reindex(cids).values.astype(float)
    keep_rows &= ~np.isnan(yv)
    rows = np.where(keep_rows)[0]
    ids = [cids[r] for r in rows]
    S = spl.reindex(ids).values.astype(str)
    assert set(S) <= {"train", "val"}
    E, P = z["E"][rows], z["P"][rows]                                       # [N,17,3,448] fp16, [N,17]
    lab = lab.loc[ids]
    T = targets(lab)
    y = T["any"]; ys = np.stack([T["vertebra"], T["rib"], T["pelvic"]], 1)
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    if a.max_train:
        rng = np.random.default_rng(0)
        tr_all = np.sort(rng.choice(tr_all, min(a.max_train, len(tr_all)), replace=False))
    from sklearn.model_selection import train_test_split
    fit, inner = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all], random_state=42)
    fit, inner = np.sort(fit), np.sort(inner)
    # per-unit, per-window standardization with fit-row statistics (= bone_probe.std_unit)
    mu = np.zeros((U, 3, 448), np.float32); sd = np.ones((U, 3, 448), np.float32)
    fm = np.zeros(len(ids), bool); fm[fit] = True
    X = np.zeros((len(ids), U, 3 * 448), np.float32)
    for u in range(U):
        for w in range(3):
            r = E[fm & P[:, u], u, w].astype(np.float32)
            mu[u, w], sd[u, w] = r.mean(0), r.std(0)
    # ~21 dead channels per unit/window have sd ~ 0 (constant ~5e-5): any drift of the finetuned encoder would explode
    # their z-score, so they are zeroed; the remaining sd are floored at 1e-3 (the head is trained on the same inputs)
    keep = (sd >= 1e-4).astype(np.float32)
    sd = np.maximum(sd, 1e-3)
    for u in range(U):
        for w in range(3):
            X[:, u, w * 448:(w + 1) * 448] = ((E[:, u, w].astype(np.float32) - mu[u, w]) / sd[u, w]) * keep[u, w] * P[:, u, None]
    print(f"standardization: {int((keep == 0).sum())} constant dims zeroed (of {keep.size})", flush=True)
    print(f"cases: train {len(tr_all)} (fit {len(fit)}, inner-val {len(inner)}; pos {int(y[fit].sum())}/{int(y[inner].sum())})"
          f" val {len(va_all)} (pos {int(y[va_all].sum())})", flush=True)
    return dict(cfg=cfg, ids=ids, S=S, P=P, X=torch.from_numpy(X), M=torch.from_numpy(P), y=y, ys=ys, lab=lab,
                fit=fit, inner=inner, val=va_all, mu=torch.from_numpy(mu).to(dev), sd=torch.from_numpy(sd).to(dev),
                keep=torch.from_numpy(keep).to(dev))


def init_head(a, d, dev, outdir):
    p = outdir / "head_init.pt"
    if p.exists():
        ck = torch.load(p, map_location=dev, weights_only=False)
        head = BoneModel("site_max", 3 * 448).to(dev); head.load_state_dict(ck["head"])
        return head, ck["inner_auc"]
    t = time.time()
    head, best, ep = train_frozen_head("site_max", d["X"], d["M"], d["y"], d["ys"], d["fit"], d["inner"],
                                       a.init_lr, a.init_wd, a.seed, dev)
    torch.save({"head": head.state_dict(), "inner_auc": best, "ep": ep}, p)
    print(f"frozen head: inner-val AUROC {best:.4f} (ep {ep}, {time.time() - t:.0f}s)", flush=True)
    return head, best


@torch.no_grad()
def unit_logits(head, X, M, dev, bs=2048):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev)
        s = torch.zeros(x.shape[0], U, device=dev)
        for k, idx in enumerate(SITES):
            s[:, idx] = head.heads[k](x[:, idx])[..., 0] + head.ub[idx]
        out.append(s.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def head_on_cache(head, X, M, idx, dev):
    a, s = head(X[idx].to(dev), M[idx].to(dev))
    return a.cpu().numpy(), s.cpu().numpy()


def std_rows(emb, d):
    """{u: [3,448]} -> {u: standardized [1344]}."""
    return {u: ((e - d["mu"][u]) / d["sd"][u] * d["keep"][u]).reshape(-1) for u, e in emb.items()}


def loader(ds, items, shuffle, bs, workers):
    ds.items = list(items)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate_list,
                      pin_memory=False, persistent_workers=False, prefetch_factor=2 if workers else None)


@torch.no_grad()
def predict(enc, head, ds, items, d, a, dev, amp_dtype, wins, norm, stages, tag=""):
    """full forward of every present unit -> any logit, site logits, max |emb - cache| (standardized units)."""
    enc.eval(); head.eval()
    la, ls, rows, diffs = [], [], [], []
    t0 = time.time()
    for bi, batch in enumerate(loader(ds, items, False, 2, a.workers)):
        for c in batch:
            r = c["r"]
            em = std_rows(embed_units(enc, c, wins, norm, stages, d["cfg"]["min_stage_weight"], dev, amp_dtype, a.tile_batch), d)
            x = torch.zeros(1, U, 3 * 448, device=dev)
            for u, e in em.items():
                x[0, u] = e
            diffs.append(float((x[0] - d["X"][r].to(dev)).abs().max()))
            o = head(x, d["M"][r:r + 1].to(dev))
            la.append(float(o[0][0])); ls.append(o[1][0].cpu().numpy()); rows.append(r)
        if bi % 100 == 0:
            print(f"  predict{tag} {len(rows)}/{len(items)} {(time.time() - t0) / max(1, len(rows)):.2f}s/case", flush=True)
    return np.array(rows), np.array(la), np.stack(ls), np.array(diffs), time.time() - t0


def build(a, dev):
    from encoder import NnUNetEncoder
    cfg = _cfg(a.config)
    ne = NnUNetEncoder(cfg["ts_model_dir"], device=str(dev))
    enc = Enc(ne.encoder, max(cfg["stages_used"])).to(dev)
    ne.network.decoder = None
    wins = [cfg["windows"][w] for w in WIN_NAMES]
    assert wins[0].get("type") == "baseline" and wins[1] == {"level": 40, "width": 400} and wins[2] == {"level": 400, "width": 1800}
    return enc, list(ne.patch_size), ne.norm, wins


# ----------------------------------------------------------------------------------------------- stage: check
def stage_check(a, dev):
    """frozen encoder: pipeline embeddings vs cached emb_bones on a few inner-val cases (+ head logits)."""
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    outdir = Path(a.data) / "runs" / a.run; outdir.mkdir(parents=True, exist_ok=True)
    d = setup(a, dev)
    head, _ = init_head(a, d, dev, outdir)
    enc, P, norm, wins = build(a, dev)
    ds = BoneCases(d["cfg"], a.data, d["ids"], P)
    items = d["inner"][:a.check_n]
    rows, la, ls, diffs, tv = predict(enc, head, ds, items, d, a, dev, amp_dtype, wins, norm, d["cfg"]["stages_used"])
    lc, _ = head_on_cache(head, d["X"], d["M"], rows, dev)
    rep = {"n": len(rows), "amp": a.amp, "max_abs_diff_std_emb": float(diffs.max()), "median_max_abs_diff": float(np.median(diffs)),
           "max_abs_diff_any_logit": float(np.abs(la - lc).max()), "s_per_case": tv / len(rows)}
    print(json.dumps(rep, indent=1), flush=True)
    (outdir / f"check_{a.amp}.json").write_text(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------------------------- stage: train
def stage_train(a, dev):
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    outdir = Path(a.data) / "runs" / a.run; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(a), indent=1))
    d = setup(a, dev)
    head, inner_cache_auc = init_head(a, d, dev, outdir)
    enc, P, norm, wins = build(a, dev)
    enc.set_trainable(a.train_stages); enc.grad_ckpt = a.grad_ckpt
    stages = d["cfg"]["stages_used"]
    S0 = unit_logits(head, d["X"], d["M"], dev)
    y, ys, fit, inner = d["y"], d["ys"], d["fit"], d["inner"]

    trap = None
    if a.trap:
        TI = build_trap_inputs({"metadata_file": a.metadata}, d["ids"])
        Yt = torch.from_numpy(np.c_[y, ys][fit]).float(); vt = ~torch.isnan(Yt); yt = Yt.nan_to_num(0)
        btr = TI["bin"][torch.from_numpy(fit)]
        pf = (yt * vt).sum(0) / vt.sum(0).clamp(min=1)
        pos_b = torch.zeros(TI["n_bins"], 4).index_add_(0, btr, yt * vt)
        n_b = torch.zeros(TI["n_bins"], 4).index_add_(0, btr, vt.float())
        rate = (pos_b + 20.0 * pf) / (n_b + 20.0)
        lg = lambda q: torch.log(q.clamp(1e-4, 1 - 1e-4) / (1 - q.clamp(1e-4, 1 - 1e-4)))
        trap = Trap(TI["n_bins"], TI["X"].shape[1], 4, lg(rate) - lg(pf)).to(dev)
        Tb, TX = TI["bin"].to(dev), TI["X"].to(dev)
        n_fit = len(fit)

    # step-0 inner-val with the frozen encoder through the full pipeline = equivalence check + baseline score
    ds_ev = BoneCases(d["cfg"], a.data, d["ids"], P)
    ev_items = inner if not a.eval_max else inner[:a.eval_max]
    lc, _ = head_on_cache(head, d["X"], d["M"], ev_items, dev)
    auc_cache = auroc(y[ev_items], lc)

    def evaluate(tag):
        rows, la, ls, diffs, tv = predict(enc, head, ds_ev, ev_items, d, a, dev, amp_dtype, wins, norm, stages, tag)
        assert (rows == ev_items).all()
        return auroc(y[rows], la), la, float(np.median(diffs)), tv

    log = []
    p0 = outdir / "eval0.json"
    if p0.exists() and not a.redo_eval0:
        e0 = json.loads(p0.read_text())
    else:
        auc0, la0, dm0, tv0 = evaluate(" step0")
        e0 = {"inner_auc_pipeline_frozen": auc0, "inner_auc_cache": auc_cache, "diff": auc0 - auc_cache,
              "median_max_abs_std_emb_diff": dm0, "max_abs_logit_diff": float(np.abs(la0 - lc).max()),
              "n": len(ev_items), "s_per_case": tv0 / len(ev_items)}
        p0.write_text(json.dumps(e0, indent=1))
    print("step-0 (frozen) inner-val:", json.dumps(e0), flush=True)
    best = e0["inner_auc_pipeline_frozen"]
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "step": 0, "inner_auc": best}, outdir / "best.pt")
    log.append({"step": 0, "epoch": 0, "inner_auc": best})

    enc_params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": a.lr_head, "weight_decay": a.wd_head},
              {"params": enc_params, "lr": a.lr_enc, "weight_decay": a.wd_enc}]
    if trap is not None:
        groups.append({"params": list(trap.parameters()), "lr": a.lr_head, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    base_lr = [g["lr"] for g in groups]
    pos_fit, neg_fit = fit[y[fit] == 1], fit[y[fit] == 0]
    n_epoch = len(pos_fit) + min(len(neg_fit), int(a.neg_per_pos * len(pos_fit)))
    steps_per_epoch = math.ceil(n_epoch / a.cases_per_step)
    total = steps_per_epoch * a.epochs
    eval_every = a.eval_every or steps_per_epoch
    print(f"trainable encoder params {sum(p.numel() for p in enc_params):,} | epoch {n_epoch} cases, {steps_per_epoch} steps,"
          f" eval every {eval_every} steps, total {total}", flush=True)
    ds_tr = BoneCases(d["cfg"], a.data, d["ids"], P, S0=S0, pres=d["P"], sel_top=a.sel_top, sel_rand=a.sel_rand,
                      train=True, seed=a.seed)
    Xd = d["X"]; Md = d["M"]
    yT = torch.from_numpy(y).float(); ysT = torch.from_numpy(ys).float()
    step, bad, stop = 0, 0, False
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for ep in range(a.epochs):
        rng = np.random.default_rng((a.seed, ep))
        items = np.r_[pos_fit, rng.choice(neg_fit, n_epoch - len(pos_fit), replace=False)]
        # class-balanced weights on this epoch's composition (any + 3 site targets)
        ye, yse = y[items], ys[items]
        wp = len(ye) / (2 * ye.sum()); wn = len(ye) / (2 * (len(ye) - ye.sum()))
        sv = ~np.isnan(yse); s1 = np.nansum(yse, 0); sn = sv.sum(0)
        swp = torch.tensor(sn / (2 * np.maximum(s1, 1)), device=dev).float()
        swn = torch.tensor(sn / (2 * np.maximum(sn - s1, 1)), device=dev).float()
        ds_tr.epoch = ep
        enc.train(); head.train()
        t_wait = time.time()
        for batch in loader(ds_tr, items, True, a.cases_per_step, a.workers):
            t_got = time.time(); wait = t_got - t_wait
            for g, lr0 in zip(opt.param_groups, base_lr):
                f = min(1.0, (step + 1) / max(1, a.warmup_steps)) * 0.5 * (1 + math.cos(math.pi * min(step, total) / max(1, total)))
                g["lr"] = lr0 * f
            opt.zero_grad(set_to_none=True)
            ltot, ntiles = 0.0, 0
            B = len(batch)
            for c in batch:
                r = c["r"]; ntiles += c["hu"].shape[0]
                em = std_rows(embed_units(enc, c, wins, norm, stages, d["cfg"]["min_stage_weight"], dev, amp_dtype, a.tile_batch), d)
                x = Xd[r].to(dev)
                x = torch.stack([em[u].float() if u in em else x[u] for u in range(U)])[None]
                la, ls = head(x, Md[r:r + 1].to(dev))
                if trap is not None:
                    t = trap(Tb[r:r + 1], TX[r:r + 1])
                    la, ls = la + t[:, 0], ls + t[:, 1:]
                yy = yT[r:r + 1].to(dev)
                loss = F.binary_cross_entropy_with_logits(la, yy) * (wp if y[r] == 1 else wn)
                yss = ysT[r:r + 1].to(dev); m = ~torch.isnan(yss)
                if m.any():
                    w = torch.where(yss == 1, swp, swn) * m
                    loss = loss + (F.binary_cross_entropy_with_logits(ls, yss.nan_to_num(0), reduction="none") * w).sum() / m.sum()
                loss = loss / B
                loss.backward()
                ltot += float(loss.detach())
            if trap is not None:
                pr = (trap.a ** 2).sum() / (2 * a.trap_prior_sd ** 2) / n_fit
                pr.backward(); ltot += float(pr)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
            opt.step()
            step += 1
            if not math.isfinite(ltot):
                raise RuntimeError(f"non-finite loss at step {step}")
            dt = time.time() - t_got
            if step % 20 == 0 or step <= 3:
                el = time.time() - t_start
                print(f"ep{ep} step{step}/{total} loss {ltot:.4f} {dt / B:.2f}s/case ({ntiles / B:.1f} tiles) wait {wait / B:.3f}s/case"
                      f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB eta_train {(total - step) * el / step / 3600:.1f}h", flush=True)
            if step % eval_every == 0 or (a.max_steps and step >= a.max_steps):
                sc, _, dm, tv = evaluate(f" step{step}")
                rec = {"step": step, "epoch": ep, "inner_auc": sc, "median_max_abs_std_emb_drift": dm, "eval_s": tv}
                if sc > best + 1e-5:
                    best, bad = sc, 0
                    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "step": step, "inner_auc": sc}, outdir / "best.pt")
                else:
                    bad += 1
                rec["best"] = best; rec["bad"] = bad; log.append(rec)
                (outdir / "log.json").write_text(json.dumps(log, indent=1))
                print(f"  inner-val AUROC {sc:.4f} (best {best:.4f}, bad {bad}) emb drift {dm:.3f} eval {tv:.0f}s", flush=True)
                enc.train(); head.train()
                if bad >= a.patience or (a.max_steps and step >= a.max_steps):
                    stop = True
            t_wait = time.time()
            if stop:
                break
        if stop:
            break
    (outdir / "log.json").write_text(json.dumps(log, indent=1))
    final_predict(a, dev, d=d, enc=enc, head=head, P=P, norm=norm, wins=wins)


def final_predict(a, dev, d=None, enc=None, head=None, P=None, norm=None, wins=None):
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    outdir = Path(a.data) / "runs" / a.run
    if d is None:
        d = setup(a, dev)
        head = BoneModel("site_max", 3 * 448).to(dev)
        enc, P, norm, wins = build(a, dev)
    ck = torch.load(outdir / "best.pt", map_location=dev, weights_only=False)
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    ds = BoneCases(d["cfg"], a.data, d["ids"], P)
    for name, items in (("inner", d["inner"]), ("val", d["val"])):
        if name == "inner" and ck["step"] == 0 and (outdir / "pred_inner.npz").exists():
            continue
        if a.pred_max:
            items = items[:a.pred_max]
        rows, la, ls, diffs, tv = predict(enc, head, ds, items, d, a, dev, amp_dtype, wins, norm, d["cfg"]["stages_used"], f" {name}")
        np.savez(outdir / f"pred_{name}.npz", ids=np.array(d["ids"])[rows], split=d["S"][rows], logit=la, site_logit=ls,
                 best_step=ck["step"], inner_auc_best=ck["inner_auc"])
        print(f"{name}: {len(rows)} cases predicted in {tv:.0f}s (best step {ck['step']})", flush=True)
    (outdir / "DONE").write_text(time.strftime("%F %T"))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["units", "check", "train", "predict"])
    ap.add_argument("--config", default=str(ROOT / "config_sparks_ft.yaml"))
    ap.add_argument("--data", default=str(HOME / "OWL/data_h"))
    ap.add_argument("--mask-dir", default=str(HOME / "OWL/data_h/masks"))
    ap.add_argument("--split", default=str(HOME / "OWL/data_ft/split.csv"))
    ap.add_argument("--labels", default=str(HOME / "OWL/data_ft/labels_disease14_all.jsonl"))
    ap.add_argument("--metadata", default=str(HOME / "OWL/data_ft/metadata.csv"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--prep-dirs", nargs="*", default=None, help="prep/ folders searched in order (default <output_dir>/prep)")
    ap.add_argument("--run", default="H1_trap")
    ap.add_argument("--trap", action="store_true"); ap.add_argument("--trap-prior-sd", type=float, default=0.5)
    ap.add_argument("--train-stages", type=int, default=2)
    ap.add_argument("--sel-top", type=int, default=1); ap.add_argument("--sel-rand", type=int, default=1)
    ap.add_argument("--neg-per-pos", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=0, help="steps (0 = once per epoch)")
    ap.add_argument("--cases-per-step", type=int, default=8)
    ap.add_argument("--tile-batch", type=int, default=4, help="tiles per encoder forward (x 3 windows)")
    ap.add_argument("--lr-enc", type=float, default=2e-5); ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd-enc", type=float, default=1e-4); ap.add_argument("--wd-head", type=float, default=1e-2)
    ap.add_argument("--init-lr", type=float, default=1e-3); ap.add_argument("--init-wd", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None, help="smoke: subsample train before the inner split")
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--pred-max", type=int, default=None); ap.add_argument("--check-n", type=int, default=30)
    ap.add_argument("--redo-eval0", action="store_true")
    a = ap.parse_args()
    if a.prep_dirs:
        PREP_DIRS[:] = a.prep_dirs
    if a.stage == "units":
        return stage_units(a)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    {"check": stage_check, "train": stage_train, "predict": final_predict}[a.stage](a, dev)


if __name__ == "__main__":
    main()
