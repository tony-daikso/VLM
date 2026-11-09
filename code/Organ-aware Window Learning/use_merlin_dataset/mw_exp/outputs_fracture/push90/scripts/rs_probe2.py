"""push90 rs_probe2 = rib_v3 rib_probe3 + (--blocks column selection from rs_feat2 / pe_feat cols, --nbmode tile
(mirror = same cell index on the contralateral rib/side instead of the mirrored angular sector), --wcsv per-case train
weights (0 = excluded from training; OOF still scored on ALL train rows with official labels), --target column).
Original doc:
rib_v3 MIL probe on frozen rib x sector instance features (feats3: Dataset295 stage 0-1, baseline+bone, 24 sectors) with
  --ctx    contrast features: each instance's hidden vector minus the mean of (a) the contralateral mirror instances
           (other side, same rib level, mirrored sector +-1), (b) the adjacent ribs (same side, level +-1, same sector),
           (c) the adjacent sectors of the same rib -> asymmetry / local-anomaly cues.
  --side L weak side supervision (train reports only, rib_side.csv): pooled left / right bag logits trained on the
           report side (negatives: both 0; side unknown: ignored), weight L.
  --lev L  weak rib-level supervision: max instance logit over the mentioned rib (+-1 level, that side) -> 1, weight L;
           negatives: max instance logit over all instances -> 0.
  --ema D  evaluate / keep an EMA of the weights (decay D) instead of the raw weights.
Outputs per fold: score_fold{k}.csv with ft (= --out-mode score) + ft_head, ft_side; summary.json with train OOF (t_rib)
of each score. Train/val only (asserted); target t_rib; folds = labels.fold; best epoch by inner 12 % holdout (train);
val AUROC never computed.
usage: rib_probe3.py LABELS FEATS OUT [--side-csv rib_side.csv] [--ctx] [--side 0.5] [--lev 0.3] [--seeds 3] ..."""
import os, json, argparse, copy, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("feats"); ap.add_argument("out")
ap.add_argument("--side-csv", default=""); ap.add_argument("--ctx", action="store_true")
ap.add_argument("--ctxin", action="store_true", help="precomputed input-space contrast (x - group mean) concatenated to x (fast)")
ap.add_argument("--side", type=float, default=0.0); ap.add_argument("--lev", type=float, default=0.0)
ap.add_argument("--hid", type=int, default=512); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--ema", type=float, default=0.0)
ap.add_argument("--out-mode", default="head", choices=["head", "side", "mix"]); ap.add_argument("--sel", default="head", choices=["head", "side", "mix"])
ap.add_argument("--rows", default="all", help="all | tier0 (train tier==0 + val rows only; quick comparisons)")
ap.add_argument("--nsec", type=int, default=24)
ap.add_argument("--blocks", default="", help="stats:stages:wins e.g. mean,max,sec:0,1:base,bone (needs cols in feats)")
ap.add_argument("--nbmode", default="sector", choices=["sector", "tile"]); ap.add_argument("--wcsv", default="", help="study_id,w")
ap.add_argument("--target", default="t_rib")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NS = a.nsec; NI = 24 * NS
lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
have = {f[:-4] for f in os.listdir(a.feats) if f.endswith(".npz") and ".tmp" not in f}
lab = lab[lab.study_id.isin(have)]
if a.rows == "tier0":
    lab = lab[(lab.split == "train") & (lab.tier == 0)]
lab = lab.reset_index(drop=True)
# weak side / level labels (train rows only by construction)
sideL = np.full(len(lab), np.nan, np.float32); sideR = sideL.copy(); levmask = np.zeros((len(lab), 24), bool)
if a.side_csv:
    sc = pd.read_csv(a.side_csv).set_index("study_id")
    for n, sid in enumerate(lab.study_id):
        if sid in sc.index and lab.split[n] == "train":
            r = sc.loc[sid]; sideL[n], sideR[n] = r.sideL, r.sideR
            for side, col, base in (("L", "levL", 0), ("R", "levR", 12)):
                if isinstance(r[col], str) and r[col]:
                    for l in map(int, r[col].split(";")):
                        for ll in (l - 1, l, l + 1):
                            if 1 <= ll <= 12: levmask[n, base + ll - 1] = True


SEL = None
if a.blocks:
    import json as _j
    _c0 = lab.study_id.iloc[0]
    with np.load(os.path.join(a.feats, _c0 + ".npz")) as z: cols = _j.loads(str(z["cols"]))
    bs, bg, bw = a.blocks.split(":"); bs, bg, bw = bs.split(","), [int(x) for x in bg.split(",")], bw.split(",")
    SEL = np.concatenate([np.arange(c[3], c[4]) for c in cols if c[0] in bs and c[1] in bg and c[2] in bw])
    print("blocks", a.blocks, "D", len(SEL), flush=True)


def _ld(p):
    with np.load(p) as z: return (z["X"][:, SEL] if SEL is not None else z["X"]), z["inst"]


Z = [_ld(os.path.join(a.feats, c + ".npz")) for c in lab.study_id]
IM = max(len(z[1]) for z in Z); D = Z[0][0].shape[1]
X = np.zeros((len(Z), IM, D), np.float16); P = np.zeros((len(Z), IM), bool); II = np.full((len(Z), IM), -1, np.int64)
for n, (x, i) in enumerate(Z):
    k = len(i); X[n, :k] = x; P[n, :k] = True; II[n, :k] = i
del Z
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int); yt = lab[a.target].values.astype(np.float32)
if "tier" not in lab: lab["tier"] = 0
Wt = np.ones(len(lab), np.float32)
if a.wcsv:
    _w = pd.read_csv(a.wcsv).set_index("study_id")["w"]; Wt = lab.study_id.map(_w).fillna(1.0).values.astype(np.float32)
    print("weights: excluded", int(((Wt == 0) & tr).sum()), "down-weighted", int(((Wt > 0) & (Wt < 1) & tr).sum()), flush=True)
Xs = X[tr][P[tr]].astype(np.float32); mu = Xs.mean(0); sd = Xs.std(0) + 1e-4; del Xs
for n in range(0, len(X), 2000):
    X[n:n + 2000] = np.where(P[n:n + 2000, :, None], (X[n:n + 2000].astype(np.float32) - mu) / sd, 0).astype(np.float16)
# neighbour slots: mirror (3), adjacent ribs (2), adjacent sectors (2); IM = "missing"
rib = np.where(P, II // NS, 0); sec = np.where(P, II % NS, 0)            # rib 0..23 (0-11 left, 12-23 right)
slot = np.full((len(X), NI), IM, np.int64)
rr, cc = np.nonzero(P); slot[rr, II[rr, cc]] = cc


def nb(r_, s_):
    ok = (r_ >= 0) & (r_ < 24)
    v = np.take_along_axis(slot, np.where(ok, r_ * NS + s_ % NS, 0), 1)
    return np.where(ok & P, v, IM)


lvl = rib % 12; sd_ = rib // 12; mr = np.where(sd_ == 0, rib + 12, rib - 12); ms = sec if a.nbmode == "tile" else (NS // 2 - 1 - sec) % NS
NB = np.stack([nb(mr, ms - 1), nb(mr, ms), nb(mr, ms + 1),
               nb(np.where(lvl > 0, rib - 1, -1), sec), nb(np.where(lvl < 11, rib + 1, -1), sec),
               nb(rib, sec - 1), nb(rib, sec + 1)], 2)
LEFT = P & (rib < 12); RIGHT = P & (rib >= 12)
if a.ctxin:  # [x, x - mean(mirror), x - mean(adjacent ribs), x - mean(adjacent sectors), 3 presence flags]
    Xc = np.zeros((len(X), IM, 4 * D + 3), np.float16)
    for n0 in range(0, len(X), 1000):
        x = X[n0:n0 + 1000].astype(np.float32); xp = np.concatenate([x, np.zeros_like(x[:, :1])], 1); nb_ = NB[n0:n0 + 1000]
        parts = [x]; fl = []
        for lo, hi in ((0, 3), (3, 5), (5, 7)):
            g = np.stack([np.take_along_axis(xp, nb_[:, :, j:j + 1], 1) for j in range(lo, hi)], 2)
            m = (nb_[:, :, lo:hi] < IM)[..., None]; c = m.sum(2)
            mean = (g * m).sum(2) / np.maximum(c, 1)
            parts.append(np.where(c > 0, x - mean, 0)); fl.append((c > 0).astype(np.float32))
        Xc[n0:n0 + 1000] = np.where(P[n0:n0 + 1000, :, None], np.concatenate(parts + fl, 2), 0).astype(np.float16)
    X = Xc; del Xc; D = X.shape[2]
print(f"N {len(lab)} train {tr.sum()} rib+ {int((yt[tr] == 1).sum())} IM {IM} D {D} side-known {int(np.isfinite(sideL).sum())} "
      f"lev-rows {int(levmask.any(1).sum())} nb-present {[(NB[..., g] < IM)[P].mean().round(3) for g in range(7)]}", flush=True)
T = lambda v: torch.from_numpy(v).to(dev)
Xt, Pt, NBt, Lt, Rt, RIBt = T(X), T(P), T(NB), T(LEFT), T(RIGHT), T(rib); del X
LMt = T(levmask); SLt, SRt = T(sideL), T(sideR); WT = T(Wt)


class M(nn.Module):
    def __init__(s):
        super().__init__(); H = a.hid
        s.f = nn.Sequential(nn.Dropout(a.drop), nn.Linear(D, H), nn.GELU(), nn.Dropout(a.drop))
        if a.ctx: s.g = nn.Sequential(nn.Linear(4 * H + 3, H), nn.GELU(), nn.Dropout(a.drop))
        s.att = nn.Linear(H, 1); s.inst = nn.Linear(H, 1); s.head = nn.Linear(2 * H, 1)

    def pool(s, h, p):
        at = s.att(h)[..., 0].masked_fill(~p, -1e4); w = torch.softmax(at, 1)[..., None]
        z = torch.cat([(w * h).sum(1), h.masked_fill(~p[..., None], -1e4).amax(1)], 1)
        return s.head(z)[:, 0]

    def forward(s, x, p, nb_):
        h = s.f(x.float()).masked_fill(~p[..., None], 0)
        if a.ctx:
            B, I, H = h.shape
            hp = torch.cat([h, torch.zeros(B, 1, H, device=h.device)], 1)
            g = torch.gather(hp, 1, nb_.reshape(B, -1, 1).expand(-1, -1, H)).reshape(B, I, 7, H)
            m = (nb_ < I).float()[..., None]; cs = []
            for lo, hi in ((0, 3), (3, 5), (5, 7)):
                c = m[:, :, lo:hi].sum(2); mean = (g[:, :, lo:hi] * m[:, :, lo:hi]).sum(2) / c.clamp(min=1)
                cs.append(torch.where(c > 0, h - mean, torch.zeros_like(h)))
            flags = torch.cat([(m[:, :, lo:hi].sum(2) > 0).float() for lo, hi in ((0, 3), (3, 5), (5, 7))], 2)
            h = s.g(torch.cat([h] + cs + [flags], 2)).masked_fill(~p[..., None], 0)
        return h

    def scores(s, h, p, l, r):
        o = {"head": s.pool(h, p)}
        if a.side or a.out_mode != "head" or a.sel != "head":
            # empty side -> very negative logit (pool over an all-masked bag gives garbage)
            sl = torch.where(l.any(1), s.pool(h, l), torch.full_like(o["head"], -8.0))
            sr = torch.where(r.any(1), s.pool(h, r), torch.full_like(o["head"], -8.0))
            o["sl"], o["sr"] = sl, sr; o["side"] = torch.logaddexp(sl, sr)
            o["mix"] = 0.5 * (o["head"] + o["side"])
        if a.lev:
            o["u"] = s.inst(h)[..., 0].masked_fill(~p, -1e4)
        return o


def batch(idx):
    return Xt[idx], Pt[idx], NBt[idx], Lt[idx], Rt[idx]


def pred(m, idx, key):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]; x, p, nb_, l, r = batch(b); o = m.scores(m(x, p, nb_), p, l, r)
            out.append(torch.stack([o.get(k, o["head"]) for k in ("head", "side", "mix")], 1))
    o = torch.cat(out).cpu().numpy(); return {"head": o[:, 0], "side": o[:, 1], "mix": o[:, 2]}[key] if key else o


def fit(itr, ies, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = M().to(dev)
    ema = copy.deepcopy(m) if a.ema else None
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    yy = T(np.nan_to_num(yt)); pw = torch.tensor(float((yt[itr] == 0).sum() / (yt[itr] == 1).sum()), device=dev)
    best, bst = -1, None; iesT = T(ies)
    for ep in range(a.epochs):
        m.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // a.bs)):
            b = T(b); x, p, nb_, l, r = batch(b); o = m.scores(m(x, p, nb_), p, l, r)
            wb = WT[b]; loss = (F.binary_cross_entropy_with_logits(o["head"] if a.out_mode == "head" else o[a.out_mode], yy[b], pos_weight=pw,
                                                         reduction="none") * wb).sum() / wb.sum().clamp(min=1e-6)
            if a.side:
                for key, tgt in (("sl", SLt[b]), ("sr", SRt[b])):
                    k_ = torch.isfinite(tgt)
                    if k_.any():
                        loss = loss + a.side * F.binary_cross_entropy_with_logits(o[key][k_], tgt[k_], pos_weight=pw)
            if a.lev:
                lm = LMt[b]; u = o["u"]; ribb = RIBt[b]
                pos = lm.any(1)
                if pos.any():
                    inmask = torch.gather(lm[pos], 1, ribb[pos]) & p[pos]
                    mx = u[pos].masked_fill(~inmask, -1e4).amax(1)
                    loss = loss + a.lev * F.binary_cross_entropy_with_logits(mx, torch.ones_like(mx))
                neg = yy[b] == 0
                if neg.any():
                    loss = loss + a.lev * F.binary_cross_entropy_with_logits(u[neg].amax(1), torch.zeros(int(neg.sum()), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        mm = ema if ema is not None else m
        sc = auc(yt[ies], pred(mm, iesT, a.sel))
        if sc > best: best, bst = sc, copy.deepcopy(mm.state_dict())
    m.load_state_dict(bst); return m, best


trainable = tr & ~np.isnan(yt); oof = np.full((len(lab), 3), np.nan)
for k in range(5):
    a_ = np.where(trainable & (fold != k) & (Wt > 0))[0]; idx = np.where((tr & (fold == k)) | ~tr)[0]; ps = []
    fk = f"{a.out}/score_fold{k}.csv"
    if os.path.exists(fk):  # resume: fold already done
        d_ = pd.read_csv(fk).set_index("study_id").loc[lab.study_id.values[idx]]
        p = d_[["ft_head", "ft_side", "ft_mix"]].values; oof[idx[tr[idx]]] = p[tr[idx]]; print(f"fold {k} resumed", flush=True); continue
    for sd0 in range(a.seed0, a.seed0 + a.seeds):
        r = np.random.default_rng(100 * sd0 + k).permutation(a_); ne = int(0.12 * len(r))
        m, b = fit(r[ne:], r[:ne], 100 * sd0 + k); ps.append(pred(m, T(idx), None))
        print(f"fold {k} seed {sd0} inner {b:.4f}", flush=True)
    p = np.mean(ps, 0); oof[idx[tr[idx]]] = p[tr[idx]]
    sel = {"head": 0, "side": 1, "mix": 2}[a.out_mode]
    pd.DataFrame({"study_id": lab.study_id.values[idx], "split": lab.split.values[idx], "fold": k, "ft": p[:, sel],
                  "ft_head": p[:, 0], "ft_side": p[:, 1], "ft_mix": p[:, 2]}).to_csv(f"{a.out}/score_fold{k}.csv", index=False)
m_ = trainable & ~np.isnan(oof[:, 0]); t0 = m_ & (lab.tier.values == 0)
res = {f"oof_t_rib_{n}": float(auc(yt[m_], oof[m_, j])) for j, n in enumerate(("head", "side", "mix"))}
res.update({"oof_t_rib": res[f"oof_t_rib_{a.out_mode}"], "oof_t_rib_tier0": float(auc(yt[t0], oof[t0, {"head": 0, "side": 1, "mix": 2}[a.out_mode]])) if t0.sum() else None,
            "n_train_scored": int(m_.sum()), "args": vars(a)})
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("PROBE", a.out, json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)
