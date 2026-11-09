"""push90b pv15_probe2 = pv15_probe + --mirw W (bilateral mirror cells mirrored x +-W; 0 = exact same anatomical cell only),
--nonbr (drop the 6-neighbour group: mirror contrast only), per-fold resume (score_fold{k}.csv) and --folds.
Original doc:
pelvis15 MIL probe on frozen pelvis instance features (pv15_feat.py: bone x 18 mm cell, stage 0-1 mean/max).
--ctx adds contrast features: hidden vector minus mean of (a) the bilateral mirror cells (other-side bone, same z/y cell,
mirrored x +-1; sacrum mirrors onto itself) and (b) the 6-connected neighbour cells of the same bone.
Target = site label (default t_pel); folds = labels.fold (sitewise outer folds); best epoch by inner 12 % holdout of the
training folds; fold models score their held-out train fold + val (val AUROC never computed). Train/val only (asserted).
Out: score_fold{k}.csv (study_id, split, fold, ft) + summary.json (train OOF AUROC on the target).
usage: pv15_probe.py LABELS FEATS OUT [--ids ids.txt (restrict rows)] [--ctx] [--seeds 3] [--hid 512] [--epochs 60]"""
import os, json, argparse, copy, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("feats"); ap.add_argument("out")
ap.add_argument("--ids", default=""); ap.add_argument("--target", default="t_pel"); ap.add_argument("--ctx", action="store_true")
ap.add_argument("--hid", type=int, default=512); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--blocks", default="all")
ap.add_argument("--side-csv", default=""); ap.add_argument("--side", type=float, default=0.0)
ap.add_argument("--mirw", type=int, default=1); ap.add_argument("--nonbr", action="store_true")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--maxinst", type=int, default=0, help="keep the N largest instances per case (0 = all)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
have = {f[:-4] for f in os.listdir(a.feats) if f.endswith(".npz") and ".tmp" not in f}
lab = lab[lab.study_id.isin(have)]
if a.ids:
    keep = {l.strip() for l in open(a.ids) if l.strip()}; lab = lab[lab.study_id.isin(keep)]
lab = lab.reset_index(drop=True)
MIR = np.array([0, 2, 1, 3, 5, 4])


def _ld(p):
    with np.load(p) as z:
        X, M, nv = z["X"], z["meta"].astype(np.int64), z["nvox"]
    if a.maxinst and len(nv) > a.maxinst:
        o = np.sort(np.argsort(-nv)[:a.maxinst]); X, M = X[o], M[o]
    return X, M


Z = [_ld(os.path.join(a.feats, c + ".npz")) for c in lab.study_id]
IM = max(len(z[1]) for z in Z); D0 = Z[0][0].shape[1]
B = {"s0": (0, D0 // 3), "s1": (D0 // 3, D0)}  # (2 windows x (32 mean + 32 max)) for s0 = D/3 when maxst=1
cols = np.arange(D0) if a.blocks == "all" else np.r_[tuple(np.arange(*B[b]) for b in a.blocks.split(","))]
X = np.zeros((len(Z), IM, len(cols)), np.float16); P = np.zeros((len(Z), IM), bool); RT = np.zeros((len(Z), IM), bool); NM = 2 * a.mirw + 1; NK = NM + (0 if a.nonbr else 6); NB = np.full((len(Z), IM, NK), IM, np.int64)
key = lambda b, z, y, x: ((b * 64 + z) * 64 + y) * 64 + (x + 32)
for n, (x, M) in enumerate(Z):
    k = len(M); X[n, :k] = x[:, cols]; P[n, :k] = True
    b, cz, cy, cx = M.T; RT[n, :k] = (b == 1) | (b == 4) | ((b == 3) & (cx >= 0)); kk = key(b, cz, cy, cx); o = np.argsort(kk); ks = kk[o]  # RT = patient right (+x)
    cand = [key(MIR[b], cz, cy, -cx - 1 + d) for d in range(-a.mirw, a.mirw + 1)] + \
           ([] if a.nonbr else [key(b, cz + dz, cy + dy, cx + dx) for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))])
    for j, c in enumerate(cand):
        pos = np.clip(np.searchsorted(ks, c), 0, k - 1); hit = ks[pos] == c
        NB[n, :k, j] = np.where(hit, o[pos], IM)
del Z
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int); yt = lab[a.target].values.astype(np.float32)
Xs = X[tr][P[tr]].astype(np.float32); mu = Xs.mean(0); sd = Xs.std(0) + 1e-4; del Xs
for n in range(0, len(X), 1000):
    X[n:n + 1000] = np.where(P[n:n + 1000, :, None], (X[n:n + 1000].astype(np.float32) - mu) / sd, 0).astype(np.float16)
print(f"N {len(lab)} train {tr.sum()} pos {int((yt[tr] == 1).sum())} IM {IM} D {X.shape[2]} "
      f"nb {[(NB[..., g] < IM)[P].mean().round(3) for g in range(NK)]}", flush=True)
sR = np.full(len(lab), np.nan, np.float32); sL = sR.copy()
if a.side_csv:
    sc = pd.read_csv(a.side_csv).set_index("study_id")
    for n, sid in enumerate(lab.study_id):
        if lab.split[n] == "train" and sid in sc.index: sR[n], sL[n] = sc.loc[sid, "sideR"], sc.loc[sid, "sideL"]
T = lambda v: torch.from_numpy(v).to(dev)
Xt, Pt, NBt = T(X), T(P), T(NB); D = X.shape[2]; del X
RTt = T(RT); SRt, SLt = T(sR), T(sL)
GR = ((0, NM),) if a.nonbr else ((0, NM), (NM, NK)); print("groups", GR, flush=True)


class M(nn.Module):
    def __init__(s):
        super().__init__(); H = a.hid
        s.f = nn.Sequential(nn.Dropout(a.drop), nn.Linear(D, H), nn.GELU(), nn.Dropout(a.drop))
        if a.ctx: s.g = nn.Sequential(nn.Linear((len(GR) + 1) * H + len(GR), H), nn.GELU(), nn.Dropout(a.drop))
        s.att = nn.Linear(H, 1); s.head = nn.Linear(2 * H, 1)

    def forward(s, x, p, nb_):
        h = s.f(x.float()).masked_fill(~p[..., None], 0)
        if a.ctx:
            Bn, I, H = h.shape
            hp = torch.cat([h, torch.zeros(Bn, 1, H, device=h.device)], 1)
            g = torch.gather(hp, 1, nb_.reshape(Bn, -1, 1).expand(-1, -1, H)).reshape(Bn, I, NK, H)
            m = (nb_ < I).float()[..., None]; cs, fl = [], []
            for lo, hi in GR:
                c = m[:, :, lo:hi].sum(2); mean = (g[:, :, lo:hi] * m[:, :, lo:hi]).sum(2) / c.clamp(min=1)
                cs.append(torch.where(c > 0, h - mean, torch.zeros_like(h))); fl.append((c > 0).float())
            h = s.g(torch.cat([h] + cs + fl, 2)).masked_fill(~p[..., None], 0)
        return h

    def pool(s, h, p):
        at = s.att(h)[..., 0].masked_fill(~p, -1e4); w = torch.softmax(at, 1)[..., None]
        z = torch.cat([(w * h).sum(1), h.masked_fill(~p[..., None], -1e4).amax(1)], 1)
        return s.head(z)[:, 0]

    def scores(s, x, p, nb_, rt):
        h = s(x, p, nb_); o = {"head": s.pool(h, p)}
        if a.side:
            r, l = p & rt, p & ~rt
            o["sr"] = torch.where(r.any(1), s.pool(h, r), torch.full_like(o["head"], -8.0))
            o["sl"] = torch.where(l.any(1), s.pool(h, l), torch.full_like(o["head"], -8.0))
            o["side"] = torch.logaddexp(o["sr"], o["sl"])
        else:
            o["side"] = o["head"]
        o["mix"] = 0.5 * (o["head"] + o["side"]); return o


def pred(m, idx, key=None):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]; o = m.scores(Xt[b], Pt[b], NBt[b], RTt[b]); out.append(torch.stack([o["head"], o["side"], o["mix"]], 1))
    o = torch.cat(out).cpu().numpy(); return o if key is None else o[:, {"head": 0, "side": 1, "mix": 2}[key]]


def fit(itr, ies, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = M().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    yy = T(np.nan_to_num(yt)); pw = torch.tensor(float((yt[itr] == 0).sum() / (yt[itr] == 1).sum()), device=dev)
    best, bst = -1, None; iesT = T(ies)
    for ep in range(a.epochs):
        m.train()
        for b in np.array_split(rng.permutation(itr), max(1, len(itr) // a.bs)):
            b = T(b); o = m.scores(Xt[b], Pt[b], NBt[b], RTt[b])
            loss = F.binary_cross_entropy_with_logits(o["head"], yy[b], pos_weight=pw)
            if a.side:
                for key_, tgt in (("sr", SRt[b]), ("sl", SLt[b])):
                    k_ = torch.isfinite(tgt)
                    if k_.any(): loss = loss + a.side * F.binary_cross_entropy_with_logits(o[key_][k_], tgt[k_], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sc = auc(yt[ies], pred(m, iesT, "head"))
        if bst is None or sc > best: best, bst = sc, copy.deepcopy(m.state_dict())
    m.load_state_dict(bst); return m, best


trainable = tr & ~np.isnan(yt); oof = np.full((len(lab), 3), np.nan)
for k in range(5):
    a_ = np.where(trainable & (fold != k))[0]; idx = np.where((tr & (fold == k)) | ~tr)[0]; ps = []
    fk = f"{a.out}/score_fold{k}.csv"
    if os.path.exists(fk):
        d_ = pd.read_csv(fk).set_index("study_id").loc[lab.study_id.values[idx]]
        p = d_[["ft", "ft_side", "ft_mix"]].values; oof[idx[tr[idx]]] = p[tr[idx]]; print(f"fold {k} resumed", flush=True); continue
    if str(k) not in a.folds.split(","): continue
    for sd0 in range(a.seed0, a.seed0 + a.seeds):
        r = np.random.default_rng(100 * sd0 + k).permutation(a_); ne = int(0.12 * len(r))
        m, b = fit(r[ne:], r[:ne], 100 * sd0 + k); ps.append(pred(m, T(idx)))
        print(f"fold {k} seed {sd0} inner {b:.4f}", flush=True)
    p = np.mean(ps, 0); oof[idx[tr[idx]]] = p[tr[idx]]
    pd.DataFrame({"study_id": lab.study_id.values[idx], "split": lab.split.values[idx], "fold": k, "ft": p[:, 0], "ft_side": p[:, 1],
                  "ft_mix": p[:, 2]}).to_csv(f"{a.out}/score_fold{k}.csv", index=False)
if not all(os.path.exists(f"{a.out}/score_fold{k}.csv") for k in range(5)):
    print("FOLDS_PART_DONE", a.folds, flush=True); raise SystemExit(0)
m_ = trainable & ~np.isnan(oof[:, 0])
res = {"oof": float(auc(yt[m_], oof[m_, 0])), "oof_side": float(auc(yt[m_], oof[m_, 1])), "oof_mix": float(auc(yt[m_], oof[m_, 2])), "target": a.target, "n_train_scored": int(m_.sum()), "n_pos": int((yt[m_] == 1).sum()), "args": vars(a)}
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("PROBE", a.out, json.dumps({k: v for k, v in res.items() if k != "args"}), flush=True)
