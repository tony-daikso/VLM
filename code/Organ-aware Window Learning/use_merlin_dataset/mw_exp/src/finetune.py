"""Method 2 (spec 8.4): finetune the TotalSegmentator nnU-Net encoder with multi-window input channels.

Model
  * nnU-Net encoder (Dataset297, 3 mm) loaded as in encoder.py; decoder and the unused bottleneck stage dropped.
  * First conv of stage 0 expanded 1 -> N input channels (N = windows in windows.yaml, baseline first);
    every channel gets weight/N, bias unchanged -> N identical baseline channels reproduce the original features.
  * Input per organ tile = the N windowed versions of the HU patch (windowing.apply_window) stacked as channels.
  * Organ embedding = extract.py multi-stage mask pooling (stages_used, per-stage L2, concat, D=448),
    tiles of a unit averaged weighted by organ voxels (differentiable copy of extract.embed_case).
  * Heads = probe.Probe mode "single" (one linear head per finding over the mean of its present organs'
    embeddings + present-fraction feature). Optional --trap adds probe.Trap during training only.
  * Class-balanced BCE, missing labels masked.

Never touches the test split: only cases with split train/val are loaded.

usage (Spark):
  python finetune.py --config ../config_sparks_ft.yaml --check-equiv
  python finetune.py --config ../config_sparks_ft.yaml --run ft2 --train-stages 2 [--trap] [--max-train 300 --max-val 100]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from common import load_cfg, log_progress, unit_names
from extract import crop, pool_stage, window_key
from probe import Probe, Trap, auroc, build_trap_inputs
from windowing import apply_window


# ----------------------------------------------------------------------------------------------- model
class MultiWindowEncoder(nn.Module):
    """encoder stages 0..max(stages_used) of the nnU-Net, first conv widened to n_in channels."""

    def __init__(self, nnunet_encoder, n_in: int, upto: int):
        super().__init__()
        self.stages = nn.ModuleList(list(nnunet_encoder.stages[:upto + 1]))
        blk = self.stages[0][0].convs[0]                      # ConvDropoutNormReLU
        old = blk.conv
        new = nn.Conv3d(n_in, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None).to(old.weight.device)
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(1, n_in, 1, 1, 1) / n_in)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        blk.conv = new
        blk.all_modules[0] = new                              # forward() uses all_modules
        self.first_conv = new
        self.grad_ckpt = False

    def forward(self, x):
        out = []
        for st in self.stages:
            if self.grad_ckpt and torch.is_grad_enabled():
                x = checkpoint(st, x, use_reentrant=False)
            else:
                x = st(x)
            out.append(x)
        return out

    def set_trainable(self, k: int, first_conv: bool = True):
        """last k of the kept stages (+ the widened first conv) trainable; the rest frozen."""
        for p in self.parameters():
            p.requires_grad_(False)
        n = len(self.stages)
        for st in self.stages[n - k:] if k > 0 else []:
            for p in st.parameters():
                p.requires_grad_(True)
        if first_conv:
            for p in self.first_conv.parameters():
                p.requires_grad_(True)


class FTHead(Probe):
    """Probe mode 'single'; averaging uses the sampled units, the present-fraction feature the true presence."""

    def forward(self, E, Ps, Pt):
        A = Ps[:, None, :] * self.Mu[None]
        cnt = A.sum(-1, keepdim=True)
        frac = (Pt[:, None, :] * self.Mu[None]).sum(-1, keepdim=True) / self.Mu.sum(-1)[None, :, None]
        H = torch.einsum("nfu,nud->nfd", A, E) / cnt.clamp(min=1)
        H = torch.cat([H, frac], -1)
        return (H * self.w[None]).sum(-1) + self.b


# ----------------------------------------------------------------------------------------------- data
class CaseDataset(Dataset):
    """CPU side: load prep + patch json, crop every tile of the chosen units. Windowing happens on the GPU."""

    def __init__(self, cfg, ids, Y, units_list, P, units_per_case=None, train=False, seed=0, active=None):
        self.cfg, self.ids, self.Y, self.ul, self.P = cfg, ids, Y, units_list, P
        self.active = np.ones(len(units_list), bool) if active is None else np.asarray(active, bool)
        self.upc, self.train, self.seed, self.epoch = units_per_case, train, seed, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        t0 = time.time()
        cid = self.ids[i]
        prep = np.load(self.cfg["out"] / "prep" / f"{cid}.npz")
        hu, units = prep["hu"], prep["units"]
        info = json.loads((self.cfg["out"] / "patches" / f"{cid}.json").read_text())
        present = np.array([bool(info["units"][u]["present"]) for u in self.ul])
        sampled = present & self.active          # only units used by the selected findings are forwarded
        if self.train and self.upc and sampled.sum() > self.upc:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            keep = rng.choice(np.where(sampled)[0], self.upc, replace=False)
            sampled[:] = False
            sampled[keep] = True
        hs, ms, tu, nv = [], [], [], []
        for k, u in enumerate(self.ul):
            if not sampled[k]:
                continue
            for t in info["units"][u]["tiles"]:
                hs.append(crop(hu, t["start"], self.P, self.cfg["pad_hu"]))
                ms.append(crop(units, t["start"], self.P, 0) == k + 1)
                tu.append(k); nv.append(t["n_vox"])
        return {"id": cid, "idx": i,
                "hu": torch.from_numpy(np.stack(hs)) if hs else torch.zeros((0, *self.P), dtype=torch.int16),
                "mask": torch.from_numpy(np.stack(ms)) if ms else torch.zeros((0, *self.P), dtype=torch.bool),
                "tile_unit": torch.tensor(tu, dtype=torch.long), "n_vox": torch.tensor(nv, dtype=torch.float32),
                "present": torch.from_numpy(present.astype(np.float32)),
                "sampled": torch.from_numpy(sampled.astype(np.float32)),
                "y": torch.from_numpy(self.Y[i]), "load_s": time.time() - t0}


def collate_list(b):
    return b


# ----------------------------------------------------------------------------------------------- embedding
def embed(model, case, wins, norm, cfg, dev, n_units, tile_batch, amp_dtype, baseline_only=False):
    """differentiable version of extract.embed_case for one case -> [U, D] (zeros for unsampled units)."""
    stages = cfg["stages_used"]
    hu = case["hu"].to(dev, non_blocking=True)
    T = hu.shape[0]
    D = None
    acc, wsum = {}, torch.zeros(n_units, device=dev)
    for i0 in range(0, T, tile_batch):
        h = hu[i0:i0 + tile_batch].float()
        if baseline_only:
            xb = apply_window(h, wins[0][1], norm)
            x = torch.stack([xb] * len(wins), 1)
        else:
            x = torch.stack([apply_window(h, w, norm) for _, w in wins], 1)   # [t, N, Z, Y, X]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            feats = model(x)
        for b in range(x.shape[0]):
            ti = i0 + b
            k = int(case["tile_unit"][ti])
            m = case["mask"][ti].to(dev)[None, None].float()
            vs = []
            for s in stages:
                v, _, _ = pool_stage(feats[s][b:b + 1], m, cfg["min_stage_weight"])
                vs.append(F.normalize(v, dim=1))
            e = torch.cat(vs, 1)[0] * case["n_vox"][ti]
            acc[k] = acc[k] + e if k in acc else e
            wsum[k] += case["n_vox"][ti]
            D = e.shape[0]
    D = D or sum(int(model.stages[s][-1].convs[-1].conv.out_channels) for s in stages)
    rows = [acc[k] / wsum[k] if k in acc else torch.zeros(D, device=dev) for k in range(n_units)]
    return torch.stack(rows)


# ----------------------------------------------------------------------------------------------- setup
def load_split(cfg, args):
    lab = pd.read_csv(cfg["labels_file"]).set_index("study_id")
    spl = pd.read_csv(cfg["split_file"]).set_index("study_id")["split"]
    findings = [f for f in cfg["finding_map"] if f in lab.columns and (not args.findings or f in args.findings)]
    have = {p.stem for p in (cfg["out"] / "prep").glob("*.npz")} & {p.stem for p in (cfg["out"] / "patches").glob("*.json")}
    ids = sorted(have & set(lab.index) & set(spl.index))
    ids = [i for i in ids if spl[i] in ("train", "val")]          # the test split is never loaded
    rng = np.random.default_rng(args.seed)
    tr = [i for i in ids if spl[i] == "train"]
    va = [i for i in ids if spl[i] == "val"]
    if args.max_train:
        tr = sorted(rng.choice(tr, min(args.max_train, len(tr)), replace=False).tolist())
    if args.max_val:
        va = sorted(rng.choice(va, min(args.max_val, len(va)), replace=False).tolist())
    assert not any(spl[i] == "test" for i in tr + va)
    Y = lambda L: lab.loc[L, findings].values.astype(np.float32)
    return tr, va, Y(tr), Y(va), findings


def build_model(cfg, args, dev, n_win):
    from encoder import NnUNetEncoder
    enc = NnUNetEncoder(cfg["ts_model_dir"], device=str(dev))
    upto = max(cfg["stages_used"])
    mw = MultiWindowEncoder(enc.encoder, n_win, upto).to(dev)
    enc.network.decoder = None                              # decoder dropped
    return enc, mw


def init_trap(cfg, ids_tr, Ytr, findings, dev, plus=False):
    """same empirical-Bayes init as probe.train_one."""
    T = build_trap_inputs(cfg, ids_tr)
    Yt = torch.from_numpy(Ytr); vt = ~torch.isnan(Yt); yt = Yt.nan_to_num(0)
    btr = T["bin"]
    pf = (yt * vt).sum(0) / vt.sum(0).clamp(min=1)
    k_shrink = 20.0
    pos_b = torch.zeros(T["n_bins"], len(findings)).index_add_(0, btr, yt * vt)
    n_b = torch.zeros(T["n_bins"], len(findings)).index_add_(0, btr, vt.float())
    rate = (pos_b + k_shrink * pf) / (n_b + k_shrink)
    lg = lambda q: torch.log(q.clamp(1e-4, 1 - 1e-4) / (1 - q.clamp(1e-4, 1 - 1e-4)))
    key = "X_plus" if plus else "X"
    trap = Trap(T["n_bins"], T[key].shape[1], len(findings), lg(rate) - lg(pf)).to(dev)
    return trap, T["bin"].to(dev), T[key].to(dev), int(vt.sum())


@torch.no_grad()
def predict(model, head, std, loader, wins, norm, cfg, dev, U, args, amp_dtype):
    model.eval(); head.eval()
    probs, E_all, t_load = [], [], 0.0
    for batch in loader:
        for c in batch:
            t_load += c["load_s"]
            E = embed(model, c, wins, norm, cfg, dev, U, args.tile_batch, amp_dtype)
            Pt = c["present"].to(dev)[None]; Ps = c["sampled"].to(dev)[None]
            E = standardize(E[None], Ps, std)
            probs.append(torch.sigmoid(head(E, Ps, Pt)).float().cpu().numpy()[0])
    return np.stack(probs)


def out_dir(cfg, args):
    return cfg["out"] / "finetune" / args.run


def standardize(E, P, std):
    if std is None:
        return E
    return ((E - std[0]) / std[1]) * (P[..., None] > 0)


# ----------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def check_equiv(cfg, args, dev):
    """(1) widened conv == original conv on random input (fp32); (2) frozen widened encoder with N copies of the
    baseline input reproduces the cached emb/baseline/{id}.npz (fp16 autocast like extract.py)."""
    wins = list(cfg["windows"].items())
    enc, mw = build_model(cfg, args, dev, len(wins))
    mw.eval()
    # (1) layer-level, fp32
    from encoder import NnUNetEncoder
    ref = NnUNetEncoder(cfg["ts_model_dir"], device=str(dev))
    x = torch.randn(1, 1, *ref.patch_size, device=dev)
    a = ref.stage_features(x, upto=max(cfg["stages_used"]))
    b = mw(x.repeat(1, len(wins), 1, 1, 1))
    rep = {"random_input_fp32_max_abs_diff_per_stage": [float((u - v).abs().max()) for u, v in zip(a, b)],
           "random_input_fp32_max_abs_per_stage": [float(u.abs().max()) for u in a]}
    del ref
    # (2) cached embeddings
    U = len(unit_names(cfg))
    ds = CaseDataset(cfg, [], None, unit_names(cfg), list(enc.patch_size))
    cached = sorted(p.stem for p in (cfg["out"] / "emb" / "baseline").glob("*.npz") if ".tmp" not in p.name)[:args.equiv_n]
    diffs, coss = [], []
    for cid in cached:
        ds.ids, ds.Y = [cid], np.zeros((1, 1), np.float32)
        c = ds[0]
        E = embed(mw, c, wins, enc.norm, cfg, dev, U, args.tile_batch, torch.float16 if cfg["amp"] else None,
                  baseline_only=True).float().cpu().numpy()
        z = np.load(cfg["out"] / "emb" / "baseline" / f"{cid}.npz")
        Ec, pr = z["emb"].astype(np.float32), z["organ_present"]
        d = np.abs(E[pr] - Ec[pr]).max()
        cs = (E[pr] * Ec[pr]).sum(1) / (np.linalg.norm(E[pr], axis=1) * np.linalg.norm(Ec[pr], axis=1))
        diffs.append(float(d)); coss.append(float(cs.min()))
        print(f"{cid}: max|diff|={d:.2e} min cos={cs.min():.6f} present={int(pr.sum())}", flush=True)
    rep.update({"n_cases": len(cached), "emb_max_abs_diff": max(diffs) if diffs else None,
                "emb_mean_max_abs_diff": float(np.mean(diffs)) if diffs else None,
                "emb_min_cos": min(coss) if coss else None})
    print(json.dumps(rep, indent=1))
    out = cfg["out"] / "finetune"; out.mkdir(parents=True, exist_ok=True)
    (out / "equiv_check.json").write_text(json.dumps(rep, indent=1))


def align_refs(cfg, args):
    """copy frozen-probe result npz files (e.g. results/std2/E0.npz, E3_late.npz) into results/<subdir>/, restricted
    to (and ordered like) the val ids of results/<subdir>/E5_<run>.npz, so evaluate.py --split val --results-subdir
    <subdir> and age_shortcut.py compare frozen vs finetuned on identical cases."""
    rd = cfg["out"] / "results" / args.results_subdir
    ft = np.load(rd / f"E5_{args.run}.npz", allow_pickle=True)
    ids = list(ft["ids"])
    od = rd / args.align_out if args.align_out else rd
    od.mkdir(parents=True, exist_ok=True)
    if args.align_intersect:           # refs covering only part of val (e.g. interim std2): common ids only
        common = set(ids)
        for spec in args.align_ref:
            common &= set(np.load(cfg["out"] / "results" / spec.partition(":")[0], allow_pickle=True)["ids"])
        sel = np.array([k for k, c in enumerate(ids) if c in common])
        ids = [ids[k] for k in sel]
        e5 = dict(ft); e5.update(ids=np.array(ids), split=ft["split"][sel], prob=ft["prob"][:, sel])
        np.savez(od / f"E5_{args.run}.npz", **e5)
        print(f"intersection: {len(ids)} val ids -> {od}")
    for spec in args.align_ref:                     # "path[:new_name.npz]", path absolute or relative to results/
        ref, _, newname = spec.partition(":")
        z = dict(np.load(cfg["out"] / "results" / ref, allow_pickle=True))
        pos = {c: i for i, c in enumerate(z["ids"])}
        miss = [c for c in ids if c not in pos]
        if miss:
            raise SystemExit(f"{ref}: {len(miss)} finetune val ids missing")
        sel = np.array([pos[c] for c in ids])
        fidx = np.array([list(z["findings"]).index(f) for f in ft["findings"]])
        z.update(ids=np.array(ids), split=z["split"][sel], prob=z["prob"][:, sel][:, :, fidx], findings=ft["findings"])
        assert (z["split"] == "val").all()
        name = newname or Path(ref).name
        np.savez(od / name, **z)
        print("->", od / name, z["prob"].shape)


# ----------------------------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default="ft")
    ap.add_argument("--results-subdir", default="finetune", help="prediction npz -> results/<subdir>/E5_<run>.npz")
    ap.add_argument("--check-equiv", action="store_true")
    ap.add_argument("--equiv-n", type=int, default=20)
    ap.add_argument("--train-stages", type=int, default=2, help="last k of stages 0..max(stages_used); 0 = frozen")
    ap.add_argument("--no-train-first-conv", action="store_true")
    ap.add_argument("--trap", action="store_true"); ap.add_argument("--trap-plus", action="store_true")
    ap.add_argument("--trap-prior-sd", type=float, default=0.5)
    ap.add_argument("--units-per-case", type=int, default=None, help="random subset of present units per train case")
    ap.add_argument("--cases-per-step", type=int, default=4)
    ap.add_argument("--tile-batch", type=int, default=12, help="tiles per encoder forward")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3, help="in validations")
    ap.add_argument("--val-every", type=int, default=0, help="steps between validations (0 = once per epoch)")
    ap.add_argument("--lr-enc", type=float, default=2e-5); ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--standardize", action="store_true",
                    help="z-score embeddings with stats of the frozen-init encoder on --std-cases train cases")
    ap.add_argument("--std-cases", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=None); ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (smoke tests)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--findings", nargs="*", default=None,
                    help="only these findings' heads, and only the organ units they map to are forwarded")
    ap.add_argument("--select-finding", default=None, help="early stopping / best ckpt on this finding's val AUROC "
                    "(default: mean over findings)")
    ap.add_argument("--predict-split", default=None, choices=["train", "val"],
                    help="no training: load finetune/<run>/best.pt and write results/<subdir>/E5_<run>_<split>.npz")
    ap.add_argument("--align-out", default="", help="sub-folder of results/<subdir>/ for the aligned files")
    ap.add_argument("--align-intersect", action="store_true", help="restrict to val ids common to all refs")
    ap.add_argument("--align-ref", nargs="*", default=None,
                    help="results-relative probe npz files to restrict to this run's val ids (no training)")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    if args.check_equiv:
        return check_equiv(cfg, args, dev)
    if args.align_ref:
        return align_refs(cfg, args)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    wins = list(cfg["windows"].items())
    assert wins[0][1].get("type") == "baseline", "baseline window must be first in windows.yaml"
    ul = unit_names(cfg); U = len(ul)
    tr, va, Ytr, Yva, findings = load_split(cfg, args)
    print(f"train={len(tr)} val={len(va)} findings={len(findings)} windows={[w for w, _ in wins]}", flush=True)

    enc, model = build_model(cfg, args, dev, len(wins))
    model.set_trainable(args.train_stages, not args.no_train_first_conv)
    model.grad_ckpt = args.grad_ckpt
    P = list(enc.patch_size); norm = enc.norm
    D = sum(enc.stage_channels(cfg["stages_used"]))
    fu = [[ul.index(u) for u in cfg["finding_map"][f]["organs"]] for f in findings]
    active = np.zeros(U, bool)
    for us in fu:
        active[us] = True
    print(f"active units: {[u for u, a in zip(ul, active) if a]}", flush=True)
    head = FTHead("single", U, 1, D, fu, win_idx=0).to(dev)

    ds_tr = CaseDataset(cfg, tr, Ytr, ul, P, args.units_per_case, train=True, seed=args.seed, active=active)
    ds_va = CaseDataset(cfg, va, Yva, ul, P, active=active)
    mk = lambda ds, sh, bs: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=args.workers, collate_fn=collate_list,
                                       pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    dl_tr, dl_va = mk(ds_tr, True, args.cases_per_step), mk(ds_va, False, 4)

    if args.predict_split:
        ck = torch.load(out_dir(cfg, args) / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
        std = ck["std"]
        ids_p, Y_p = (tr, Ytr) if args.predict_split == "train" else (va, Yva)
        t = time.time()
        pv = predict(model, head, std, mk(CaseDataset(cfg, ids_p, Y_p, ul, P, active=active), False, 4),
                     wins, norm, cfg, dev, U, args, amp_dtype)
        rd = cfg["out"] / "results" / args.results_subdir; rd.mkdir(parents=True, exist_ok=True)
        np.savez(rd / f"E5_{args.run}_{args.predict_split}.npz", ids=np.array(ids_p),
                 split=np.array([args.predict_split] * len(ids_p)), prob=pv[None], findings=np.array(findings),
                 note=np.array("in-sample" if args.predict_split == "train" else "val"))
        print(f"predicted {len(ids_p)} {args.predict_split} cases in {time.time()-t:.0f}s; "
              f"AUROC {[round(auroc(Y_p[:, f], pv[:, f]), 4) for f in range(len(findings))]}", flush=True)
        return

    std = None
    if args.standardize:   # frozen-init embeddings of a train subset -> per-dim mean/std over present organs
        t = time.time()
        sub = CaseDataset(cfg, tr[:args.std_cases], Ytr[:args.std_cases], ul, P, active=active)
        rows = []
        model.eval()
        with torch.no_grad():
            for batch in mk(sub, False, 4):
                for c in batch:
                    E = embed(model, c, wins, norm, cfg, dev, U, args.tile_batch, amp_dtype).float()
                    rows.append(E[c["sampled"].bool()])
        R = torch.cat(rows)
        std = (R.mean(0), R.std(0).clamp(min=1e-6))
        print(f"standardize stats from {len(rows)} cases ({len(R)} organ rows) in {time.time()-t:.0f}s", flush=True)

    trap = None
    if args.trap:
        trap, Btr, Xtr, n_valid_tr = init_trap(cfg, tr, Ytr, findings, dev, args.trap_plus)

    Yt = torch.from_numpy(Ytr); valid = ~torch.isnan(Yt)
    npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = (valid.sum(0) / (2 * npos.clamp(min=1))).to(dev); wneg = (valid.sum(0) / (2 * nneg.clamp(min=1))).to(dev)

    enc_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc})
    if trap is not None:
        groups.append({"params": list(trap.parameters()), "lr": args.lr_head, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lr = [g["lr"] for g in groups]
    steps_per_epoch = math.ceil(len(tr) / args.cases_per_step)
    total = steps_per_epoch * args.epochs
    print(f"trainable encoder params {sum(p.numel() for p in enc_params):,} | head {sum(p.numel() for p in head.parameters()):,}"
          f" | steps/epoch {steps_per_epoch}", flush=True)

    out = out_dir(cfg, args); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    logf = open(out / "log.csv", "w", newline=""); lw = csv.writer(logf)
    lw.writerow(["epoch", "step", "loss", "s_per_case", "load_wait_s_per_case", "worker_load_s_per_case", "peak_gib", "val_auroc"])
    best, bad, step, stop = -1.0, 0, 0, False
    stats = {"s_per_case": [], "wait": [], "wload": []}

    def validate():
        t = time.time()
        pv = predict(model, head, std, dl_va, wins, norm, cfg, dev, U, args, amp_dtype)
        aucs = [auroc(Yva[:, f], pv[:, f]) for f in range(len(findings))]
        sc = aucs[findings.index(args.select_finding)] if args.select_finding else float(np.nanmean(aucs))
        return sc, pv, aucs, time.time() - t

    def val_and_ckpt():
        nonlocal best, bad
        sc, _, _, tv = validate()
        if sc > best + 1e-5:
            best, bad = sc, 0
            torch.save({"encoder": model.state_dict(), "head": head.state_dict(),
                        "trap": trap.state_dict() if trap is not None else None, "std": std,
                        "args": vars(args), "step": step, "val_auroc": sc}, out / "best.pt")
        else:
            bad += 1
        print(f"  val {'AUROC ' + args.select_finding if args.select_finding else 'mean AUROC'} {sc:.4f} (best {best:.4f}, bad {bad}) {tv/len(va):.2f}s/case", flush=True)
        model.train(); head.train()
        return f"{sc:.4f}"

    torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        ds_tr.epoch = ep
        model.train(); head.train()
        t_wait0 = time.time()
        for batch in dl_tr:
            t_got = time.time()
            wait = t_got - t_wait0
            for g, lr0 in zip(opt.param_groups, base_lr):   # linear warmup then cosine
                f = min(1.0, (step + 1) / max(1, args.warmup_steps))
                f *= 0.5 * (1 + math.cos(math.pi * min(step, total) / max(1, total)))
                g["lr"] = lr0 * f
            y = torch.stack([c["y"] for c in batch]).to(dev)
            m = ~torch.isnan(y)
            Ps = torch.stack([c["sampled"] for c in batch]).to(dev)
            Pt = torch.stack([c["present"] for c in batch]).to(dev)
            # a finding counts only if one of its present units was sampled, or none of them is present
            Mu = head.Mu
            has_s = (Ps @ Mu.T) > 0; has_p = (Pt @ Mu.T) > 0
            m = m & (has_s | ~has_p)
            denom = m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss_tot = 0.0
            for j, c in enumerate(batch):       # one case at a time -> activations freed after each backward
                E = embed(model, c, wins, norm, cfg, dev, U, args.tile_batch, amp_dtype).float()
                E = standardize(E[None], Ps[j:j + 1], std)
                logit = head(E, Ps[j:j + 1], Pt[j:j + 1])
                if trap is not None:
                    gi = c["idx"]
                    logit = logit + trap(Btr[gi:gi + 1], Xtr[gi:gi + 1])
                yy, mm = y[j:j + 1], m[j:j + 1]
                w = torch.where(yy == 1, wpos, wneg) * mm
                loss = (F.binary_cross_entropy_with_logits(logit, yy.nan_to_num(0), reduction="none") * w).sum() / denom
                loss.backward()
                loss_tot += float(loss)
            if trap is not None:
                pr = (trap.a ** 2).sum() / (2 * args.trap_prior_sd ** 2) / n_valid_tr
                pr.backward(); loss_tot += float(pr)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
            opt.step()
            step += 1
            dt = time.time() - t_got
            if not math.isfinite(loss_tot):
                raise RuntimeError(f"non-finite loss at step {step}")
            stats["s_per_case"].append(dt / len(batch)); stats["wait"].append(wait / len(batch))
            stats["wload"].append(np.mean([c["load_s"] for c in batch]))
            pk = torch.cuda.max_memory_allocated() / 2**30
            row = [ep, step, f"{loss_tot:.4f}", f"{dt/len(batch):.2f}", f"{wait/len(batch):.3f}",
                   f"{stats['wload'][-1]:.3f}", f"{pk:.1f}", ""]
            if step % 10 == 0:
                print(f"ep{ep} step{step} loss {loss_tot:.4f} {dt/len(batch):.2f}s/case wait {wait/len(batch):.3f}s/case "
                      f"peak {pk:.1f}GiB", flush=True)
            do_val = args.val_every and step % args.val_every == 0
            if args.max_steps and step >= args.max_steps:
                stop = True; do_val = True
            if do_val:
                row[-1] = val_and_ckpt()
            lw.writerow(row); logf.flush()
            t_wait0 = time.time()
            if stop or bad >= args.patience:
                break
        if stop or bad >= args.patience:
            break
        if not args.val_every:
            lw.writerow([ep, step, "", "", "", "", "", val_and_ckpt()]); logf.flush()
            if bad >= args.patience:
                break
    # ---- final: best checkpoint -> val predictions in probe.py npz format
    ck = torch.load(out / "best.pt", map_location=dev)
    model.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    sc, pv, aucs, tv = validate()
    rd = cfg["out"] / "results" / args.results_subdir; rd.mkdir(parents=True, exist_ok=True)
    np.savez(rd / f"E5_{args.run}.npz", ids=np.array(va), split=np.array(["val"] * len(va)), prob=pv[None],
             findings=np.array(findings), units=np.array(ul), windows=np.array([w for w, _ in wins]),
             val_auc=np.array([sc]), gate=np.zeros(0))
    summ = {"run": args.run, "best_val_auroc": best, "final_val_auroc": sc, "per_finding": dict(zip(findings, aucs)),
            "steps": step, "s_per_case_train_median": float(np.median(stats["s_per_case"][2:] or stats["s_per_case"])),
            "load_wait_s_per_case_median": float(np.median(stats["wait"][2:] or stats["wait"])),
            "worker_load_s_per_case_median": float(np.median(stats["wload"])),
            "val_s_per_case": tv / max(1, len(va)), "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "n_train": len(tr), "n_val": len(va)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
    log_progress(f"finetune.py run={args.run}: train={len(tr)} val={len(va)} steps={step} best val AUROC {best:.4f}")


if __name__ == "__main__":
    main()
