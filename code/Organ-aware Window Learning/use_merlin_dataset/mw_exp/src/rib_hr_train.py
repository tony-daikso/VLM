"""Experiment K (rib high-res path), step 2: rib-centric 3D CNN + attention MIL on the straightened ribs.

Input per case: up to 16 straightened ribs (L5..L12, R5..R12) from rib_hr_prep.py, each (L, 32, 32) at 1 mm.
Model: shared small 3D CNN over each rib (channels: bone window L400/W1800 and soft-tissue window L40/W400; the in-plane
       32x32 is pooled away, the rib length is kept at 4 mm resolution) -> 1D conv along the rib -> gated attention MIL
       along the rib length -> rib feature (+ rib-identity embedding) -> rib logit; case logit = logsumexp over ribs.
Supervision (train only):
  case BCE (rib fracture vs no fracture; non-rib fracture positives = rib-negative with weight 0.5)
  rib-level BCE: reported ribs (side+level, parsed by analysis_rib) = 1; all ribs of negatives = 0;
                 contralateral ribs of unilateral positives = 0 (weight 0.3)
  side MIL: unilateral positives without levels -> max over the reported side = 1
  left-right contrast: reported rib logit > same-level contralateral rib logit (logistic pairwise loss)
  hard negatives come from the sampling (rib_hr_ids.py: >=75 y / osteopenia / hardware).
Protocol: 5-fold CV on the train ids (stratified); inside each training fold a 15% inner hold-out picks the epoch
          (case rib AUROC). Outer-fold predictions = train OOF; val predictions = mean over the 5 fold models.
          Val labels are never read here (val ids are only scored).
Outputs (--out): fold{k}.pt, preds_{tag}.npz (ids, split, logit, rib_logit[N,16], rib_present[N,16], att_peak[N,16])
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RIBS = [f"L{k}" for k in range(5, 13)] + [f"R{k}" for k in range(5, 13)]
RIX = {r: i for i, r in enumerate(RIBS)}
LMAX = 336


# ------------------------------------------------------------------------------------------------ labels
def rib_targets(row):
    """-> tgt (16,) in {1,0,-1}, w (16,), side_mil ('L'|'R'|None), pairs [(pos_idx, contra_idx)]"""
    tgt = -np.ones(16, np.float32)
    w = np.zeros(16, np.float32)
    side_mil, pairs = None, []
    if row.rib_t == 0:
        tgt[:] = 0; w[:] = 1.0 if row.y == 0 else 0.5
        return tgt, w, side_mil, pairs
    if row.rib_t != 1:
        return tgt, w, side_mil, pairs
    side = row.side if isinstance(row.side, str) else "?"
    lv = re.findall(r"([LRB])(\d+)", row.rib_levels or "") if isinstance(row.rib_levels, str) else []
    pos = set()
    for s, k in lv:
        k = int(k)
        if not 5 <= k <= 12:
            continue
        for ss in (("L", "R") if s == "B" else (s,)):
            pos.add(f"{ss}{k}")
    for r in pos:
        tgt[RIX[r]] = 1; w[RIX[r]] = 1.0
    if side in ("L", "R"):
        other = "R" if side == "L" else "L"
        for k in range(5, 13):
            if tgt[RIX[f"{other}{k}"]] < 0:
                tgt[RIX[f"{other}{k}"]] = 0; w[RIX[f"{other}{k}"]] = 0.3
        for r in pos:
            c = ("R" if r[0] == "L" else "L") + r[1:]
            if c not in pos:
                pairs.append((RIX[r], RIX[c]))
        if not pos:
            side_mil = side
    return tgt, w, side_mil, pairs


def load_table(path, cpr_dir):
    t = pd.read_pickle(path)
    assert set(t.split) <= {"train", "val"}
    have = {p.stem for p in Path(cpr_dir).glob("*.npz") if not p.stem.endswith(".tmp")}
    t = t[t.index.isin(have)].copy()
    t["rib_t"] = np.where(t.rib == 1, 1, np.where(t.y == 0, 0, np.where(t.y == 1, 0, -1)))  # non-rib pos -> 0 (w .5)
    return t, have


# ------------------------------------------------------------------------------------------------ data
class CaseDS(torch.utils.data.Dataset):
    def __init__(self, ids, cpr_dir, train, table=None):
        self.ids, self.dir, self.train, self.t = list(ids), Path(cpr_dir), train, table

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        z = np.load(self.dir / f"{cid}.npz")
        X = np.zeros((16, LMAX, 32, 32), np.uint8)
        lens = np.zeros(16, np.int64)
        rng = np.random.default_rng() if self.train else None
        for r in RIBS:
            k = f"vol_{r}"
            if k not in z.files:
                continue
            v = z[k]
            if len(v) > LMAX:
                s = rng.integers(0, len(v) - LMAX + 1) if self.train else (len(v) - LMAX) // 2
                v = v[s:s + LMAX]
            if self.train and len(v) > 40:  # random trim of up to 8 mm at each end
                a0, a1 = rng.integers(0, 9, 2)
                v = v[a0:len(v) - a1]
            X[RIX[r], :len(v)] = v
            lens[RIX[r]] = len(v)
        out = {"x": torch.from_numpy(X), "len": torch.from_numpy(lens), "i": i}
        if self.t is not None:
            row = self.t.loc[cid]
            tgt, w, sm, pairs = rib_targets(row)
            out |= {"y": torch.tensor(float(row.rib_t == 1)), "yw": torch.tensor(0.5 if (row.rib_t == 0 and row.y == 1) else 1.0),
                    "rt": torch.from_numpy(tgt), "rw": torch.from_numpy(w),
                    "sm": torch.tensor({"L": 0, "R": 1}.get(sm, -1)),
                    "pairs": torch.tensor((pairs + [(-1, -1)] * 8)[:8])}
        return out


# ------------------------------------------------------------------------------------------------ model
def cbr(ci, co, k=3, s=1):
    return nn.Sequential(nn.Conv3d(ci, co, k, s, k // 2, bias=False), nn.BatchNorm3d(co), nn.ReLU(inplace=True))


class RibNet(nn.Module):
    def __init__(self, c=(16, 32, 48, 64, 96), drop=0.2, crop=32, lstride=1):
        super().__init__()
        self.crop = crop
        self.enc = nn.Sequential(
            cbr(2, c[0]), nn.Conv3d(c[0], c[0], 3, (lstride, 2, 2), 1, bias=False), nn.BatchNorm3d(c[0]), nn.ReLU(inplace=True),
            # L, 16, 16 (stride-2 conv instead of conv+maxpool at full resolution)
            cbr(c[0], c[1]), cbr(c[1], c[1]), nn.MaxPool3d((3 - lstride, 2, 2)),         # L/2, 8, 8
            cbr(c[1], c[2]), cbr(c[2], c[2]), nn.MaxPool3d((2, 2, 2)),         # L/4, 4, 4
            cbr(c[2], c[3]), cbr(c[3], c[3]))
        self.seq = nn.Sequential(nn.Conv1d(c[3] * 2, c[4], 5, padding=2), nn.BatchNorm1d(c[4]), nn.ReLU(inplace=True),
                                 nn.Conv1d(c[4], c[4], 5, padding=2), nn.BatchNorm1d(c[4]), nn.ReLU(inplace=True))
        self.att_v = nn.Linear(c[4], 64)
        self.att_u = nn.Linear(c[4], 64)
        self.att_w = nn.Linear(64, 1)
        self.rib_emb = nn.Embedding(16, 16)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(c[4] + 16 + 1, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    @staticmethod
    def window(x):
        hu = x.float() * 12.0 - 1000.0
        return torch.stack([((hu + 500.0) / 1800.0).clamp(0, 1), ((hu + 160.0) / 400.0).clamp(0, 1)], 1)

    def forward_ribs(self, x, lens, rib_idx):
        """x (n, L, 32, 32) uint8; lens (n,); rib_idx (n,) -> rib logits (n,), attention (n, L/4)"""
        if self.crop < x.shape[-1]:
            o = (x.shape[-1] - self.crop) // 2
            x = x[..., o:o + self.crop, o:o + self.crop]
        h = self.enc(self.window(x))                 # n, C, L/4, 4, 4 (3, 3 with crop 24)
        h = torch.cat([h.mean((3, 4)), h.amax((3, 4))], 1)   # n, 2C, L/4
        h = self.seq(h).transpose(1, 2)              # n, L/4, C
        Lq = h.shape[1]
        m = torch.arange(Lq, device=h.device)[None] < ((lens + 3) // 4)[:, None]
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        a = a.masked_fill(~m, -1e4).softmax(1)
        f = (a[..., None] * h).sum(1)
        ln = (lens.float() / 300.0)[:, None]
        return self.head(torch.cat([f, self.rib_emb(rib_idx), ln], 1)).squeeze(-1), a


def run_batch(model, b, dev, train, aug=True):
    x, lens = b["x"].to(dev, non_blocking=False), b["len"].to(dev)
    B = x.shape[0]
    pres = lens > 0
    bi, ri = pres.nonzero(as_tuple=True)
    xr, lr = x[bi, ri], lens[bi, ri]
    Lb = int(((lr.max() + 7) // 8 * 8).item())
    xr = xr[:, :Lb]
    if train and aug:  # in-plane shift +-2 voxels, intensity jitter (in uint8 units)
        dx, dy = np.random.randint(-2, 3, 2)
        xr = torch.roll(xr, (int(dx), int(dy)), (2, 3))
        xr = (xr.float() * (1 + 0.03 * torch.randn(len(xr), 1, 1, 1, device=dev)) + torch.randn(len(xr), 1, 1, 1, device=dev)
              ).clamp(0, 255)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        rl, att = model.forward_ribs(xr, lr, ri)
    rl = rl.float()
    R = torch.full((B, 16), -1e4, device=dev)
    R[bi, ri] = rl
    case = torch.logsumexp(R, 1)
    peak = torch.full((B, 16), -1, device=dev, dtype=torch.long)
    peak[bi, ri] = att.float().argmax(1) * 4
    return case, R, pres, peak


def losses(case, R, pres, b, dev, wr=0.5, wlr=0.3):
    y, yw = b["y"].to(dev), b["yw"].to(dev)
    lc = (F.binary_cross_entropy_with_logits(case, y, reduction="none") * yw).mean()
    rt, rw = b["rt"].to(dev), b["rw"].to(dev) * pres.float()
    ok = (rt >= 0) & (rw > 0)
    lr_ = (F.binary_cross_entropy_with_logits(R.clamp(-30, 30), rt.clamp(0, 1), reduction="none") * rw * ok).sum() / ok.sum().clamp(min=1)
    sm = b["sm"].to(dev)
    ls = torch.zeros((), device=dev)
    if (sm >= 0).any():
        idx = (sm >= 0).nonzero(as_tuple=True)[0]
        Rs = R[idx].view(-1, 2, 8)
        side_max = torch.logsumexp(Rs[torch.arange(len(idx)), sm[idx]], 1)
        ls = F.binary_cross_entropy_with_logits(side_max, torch.ones_like(side_max))
    pr = b["pairs"].to(dev)
    lp = torch.zeros((), device=dev)
    okp = (pr[..., 0] >= 0)
    if okp.any():
        bb, kk = okp.nonzero(as_tuple=True)
        p, c = pr[bb, kk, 0], pr[bb, kk, 1]
        both = pres[bb, p] & pres[bb, c]
        if both.any():
            lp = F.softplus(-(R[bb, p] - R[bb, c]))[both].mean()
    return lc + wr * lr_ + wr * ls + wlr * lp, (lc.item(), lr_.item(), ls.item(), lp.item())


def predict(model, ids, cpr_dir, dev, bs, workers):
    dl = torch.utils.data.DataLoader(CaseDS(ids, cpr_dir, False), batch_size=bs, num_workers=workers, shuffle=False)
    model.eval()
    C, RR, P, K = [], [], [], []
    with torch.no_grad():
        for b in dl:
            case, R, pres, peak = run_batch(model, b, dev, False)
            C.append(case.cpu()); RR.append(R.cpu()); P.append(pres.cpu()); K.append(peak.cpu())
    return torch.cat(C).numpy(), torch.cat(RR).numpy(), torch.cat(P).numpy(), torch.cat(K).numpy()


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def train_fold(tr_ids, ho_ids, t, a, dev, log):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = RibNet(crop=a.crop, lstride=a.lstride).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    ds = CaseDS(tr_ids, a.cpr, True, t)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True)
    steps = a.epochs * len(dl)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    yho = (t.loc[ho_ids, "rib_t"] == 1).values.astype(int)
    best, best_ep, best_state, bad = -1, -1, None, 0
    for ep in range(a.epochs):
        model.train()
        t0, L = time.time(), []
        for b in dl:
            case, R, pres, _ = run_batch(model, b, dev, True)
            loss, parts = losses(case, R, pres, b, dev, a.w_rib, a.w_lr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sch.step()
            L.append(parts)
        c, *_ = predict(model, ho_ids, a.cpr, dev, a.bs, a.workers)
        au = auroc(yho, c)
        Lm = np.mean(L, 0)
        log(f"  ep{ep} loss case={Lm[0]:.3f} rib={Lm[1]:.3f} side={Lm[2]:.3f} lr={Lm[3]:.3f} holdout AUROC={au:.3f} "
            f"({time.time()-t0:.0f}s)")
        if au > best:
            best, best_ep, bad = au, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience and ep >= a.min_epochs:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--cpr", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-ids", default=None, help="restrict training ids (csv with study_id) e.g. tier0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=24)
    ap.add_argument("--cudnn-bench", action="store_true")
    ap.add_argument("--lstride", type=int, default=1, help="stride along the rib in the 2nd conv (1 or 2)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--w-rib", type=float, default=0.5)
    ap.add_argument("--w-lr", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time N training steps and exit")
    ap.add_argument("--predict-ids", default=None, help="score these ids (val) with saved fold models -> preds_val_fold{k}.npz")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"=== {time.strftime('%F %T')} {vars(a)}")
    t, have = load_table(a.table, a.cpr)
    tr = t[(t.split == "train") & (t.rib_t >= 0)]
    if a.train_ids:
        keep = set(pd.read_csv(a.train_ids).study_id)
        tr = tr[tr.index.isin(keep)]
    spl = pd.read_csv(a.split).set_index("study_id")["split"]
    val_ids = sorted(i for i in have if spl.get(i) == "val")
    assert not any(spl.get(i) == "test" for i in have)
    dev = a.device
    only = set(map(int, a.only_folds.split(","))) if a.only_folds else None
    torch.backends.cudnn.benchmark = a.cudnn_bench
    log(f"train cases {len(tr)} (rib pos {(tr.rib_t==1).sum()}), val cases with cpr {len(val_ids)}")

    if a.bench:
        torch.manual_seed(0)
        model = RibNet(crop=a.crop, lstride=a.lstride).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        dl = torch.utils.data.DataLoader(CaseDS(tr.index[:a.bench * a.bs], a.cpr, True, t), batch_size=a.bs, num_workers=a.workers)
        t0 = time.time(); tl = 0
        for k, b in enumerate(dl):
            tl += time.time() - t0 if k > 1 else 0
            case, R, pres, _ = run_batch(model, b, dev, True)
            loss, parts = losses(case, R, pres, b, dev)
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            if k == 1:
                T0 = time.time()
            t0 = time.time()
        torch.cuda.synchronize()
        log(f"bench: {(time.time()-T0)/(k-1):.3f}s/step (data wait {tl/(k-1):.3f}s) bs={a.bs} peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB loss={parts}")
        return

    if a.predict_ids:
        pids = [l.strip().split(",")[0] for l in open(a.predict_ids) if l.strip() and not l.startswith("study_id")]
        pids = [i for i in pids if i in have and spl.get(i) == "val"]
        for f in sorted(out.glob("fold*.pt")):
            k = int(f.stem[4:])
            if only is not None and k not in only:
                continue
            model = RibNet(crop=a.crop, lstride=a.lstride).to(dev)
            model.load_state_dict(torch.load(f, map_location=dev))
            c, R, P, K = predict(model, pids, a.cpr, dev, a.bs, a.workers)
            np.savez(out / f"preds_val_fold{k}.npz", ids=np.array(pids), split=np.array(["val"] * len(pids)), logit=c,
                     rib_logit=R, rib_present=P, att_peak=K)
            log(f"predicted {len(pids)} val ids with fold {k}")
        return

    from sklearn.model_selection import StratifiedKFold
    ids = tr.index.values
    y = (tr.rib_t == 1).values.astype(int)
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    only = set(map(int, a.only_folds.split(","))) if a.only_folds else None
    for k, (itr, ite) in enumerate(skf.split(ids, y)):
        if only is not None and k not in only:
            continue
        if (out / f"preds_fold{k}.npz").exists():
            log(f"fold {k} done, skip"); continue
        rng = np.random.default_rng(100 + k)
        sub = ids[itr]; ysub = y[itr]
        ho = np.concatenate([rng.choice(sub[ysub == c], int(round(0.15 * (ysub == c).sum())), replace=False) for c in (0, 1)])
        trk = np.setdiff1d(sub, ho)
        log(f"fold {k}: train {len(trk)} holdout {len(ho)} outer {len(ite)}")
        model, best, bep = train_fold(trk, ho, t, a, dev, log)
        torch.save(model.state_dict(), out / f"fold{k}.pt")
        pid = list(ids[ite]) + ([] if a.no_val else val_ids)
        c, R, P, K = predict(model, pid, a.cpr, dev, a.bs, a.workers)
        sp = np.array(["train"] * len(ite) + ["val"] * (len(pid) - len(ite)))
        np.savez(out / f"preds_fold{k}.npz", ids=np.array(pid), split=sp, logit=c, rib_logit=R, rib_present=P, att_peak=K,
                 holdout_auroc=best, best_ep=bep)
        log(f"fold {k}: holdout AUROC {best:.3f} @ep{bep}; outer-fold AUROC {auroc(y[ite], c[:len(ite)]):.3f}")
    json.dump({"ribs": RIBS}, open(out / "ribs.json", "w"))


if __name__ == "__main__":
    main()
