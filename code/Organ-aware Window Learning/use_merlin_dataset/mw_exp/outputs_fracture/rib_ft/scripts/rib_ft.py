"""rib_ft: fine-tune the TotalSegmentator Dataset295 (part5_ribs, 1.5 mm) nnU-Net encoder on whole 1.5 mm rib-cage ROIs
(Exp L grid; roi/{id}.npz from rib_prep.py) with rib-masked pooling and attention MIL over the 24 ribs -> case-level
rib-fracture logit.  gs15_ft recipe: stages < --train-from frozen (tf2 = train stages 2-4; deeper stages unused),
windows baseline + bone (+ soft tissue with --win3), per rib k and stage s in {2,3,4}: rib-masked mean & max
(mask = rib k present in the stage cell) -> LayerNorm -> gated attention + max over ribs -> dropout -> linear.
Labels: labels.csv (train/val rows only, asserted; test never read).  Target t_rib (rib fx = 1, fracture-free = 0,
other-site fx excluded from training but scored).  Folds = labels.fold (sitewise outer folds).  Each epoch: all
training rib positives + NEGR x as many random negatives.  Best epoch = AUROC on an inner 12 % holdout of the
training folds.  Fold model predicts its held-out train fold (OOF) and, unless --no-val, the val rows (val score =
mean over fold models; val AUROC is NEVER computed here).
usage: rib_ft.py LABELS ROI_DIR OUT --folds 0,1 [--train-from 2] [--epochs 6] [--seed 0] [--negr 3] [--no-val]"""
import sys, os, time, json, argparse, threading, queue
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("roi"); ap.add_argument("out")
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src"))
ap.add_argument("--train-from", type=int, default=2); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--negr", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--no-val", action="store_true")
ap.add_argument("--win3", action="store_true"); ap.add_argument("--win1", action="store_true"); ap.add_argument("--vram-gb", type=float, default=0)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--stop", default=""); ap.add_argument("--s1", action="store_true", help="also pool the frozen stage-1 (3 mm) cache features"); ap.add_argument("--score", action="store_true", help="load model_fold{k}.pt; score fold-k train rows + val rows -> score_fold{k}.csv"); ap.add_argument("--cache", default="", help="stage-1 cache dir (rib_cache.py); baseline window only"); ap.add_argument("--bench", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
torch.backends.cudnn.benchmark = False; dev = torch.device("cuda")
if a.vram_gb:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
STOP = a.stop or os.path.join(a.out, "STOP")

lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
have = {f[:-4] for f in os.listdir(a.cache or a.roi) if f.endswith(".npz") and ".tmp" not in f}
lab = lab[lab.study_id.isin(have)].reset_index(drop=True)
if a.limit: lab = pd.concat([lab[lab.t_rib == 1].head(a.limit), lab[lab.t_rib == 0].head(a.limit * 3), lab[lab.split == "val"].head(a.limit)]).reset_index(drop=True)
ids = lab.study_id.values; tr = (lab.split == "train").values; fold = lab.fold.values.astype(int)
yt = lab.t_rib.values; trainable = tr & ~np.isnan(yt)
print(f"cases with roi: {len(lab)} (train {tr.sum()}, trainable {trainable.sum()}, rib+ {(yt == 1).sum()}, val {(~tr).sum()})", flush=True)

enc0 = NnUNetEncoder(a.ts, device="cuda"); NORM = enc0.norm
BASE_SD = {k: v.detach().clone() for k, v in enc0.encoder.state_dict().items()}
WINS = [{"type": "baseline"}] + ([] if (a.win1 or a.cache) else [{"level": 400, "width": 1800}]) + ([{"level": 40, "width": 400}] if a.win3 else [])
NW = len(WINS); ST = [2, 3, 4]; CH = enc0.stage_channels(ST); STR = [4, 8, 16]; K = 24
D = NW * 2 * sum(CH) + (2 * enc0.stage_channels([1])[0] if a.s1 else 0)


def load(cid, train, rng):
    if a.cache:
        z = np.load(os.path.join(a.cache, cid + ".npz"))
        m2 = np.unpackbits(z["m2"])[:int(np.prod(z["m2shape"]))].reshape(z["m2shape"]).astype(bool)
        return (torch.from_numpy(z["idx"].astype(np.int64)), torch.from_numpy(z["f"].view(np.int16)), torch.from_numpy(z["grid"])), torch.from_numpy(m2)
    z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb = z["hu"], z["lab"]
    if train:  # random crop jitter (rois carry an 8-voxel margin)
        lo = rng.integers(0, 5, 3); hi = rng.integers(0, 5, 3)
        sl = tuple(slice(l, s - h) for l, h, s in zip(lo, hi, hu.shape)); hu, lb = hu[sl], lb[sl]
    P = [(-s) % 32 for s in hu.shape]
    hu = np.pad(hu, [(0, p) for p in P], constant_values=-1024); lb = np.pad(lb, [(0, p) for p in P])
    lb = np.where(lb > K, 0, lb).astype(np.uint8)
    Z, Y, X = (s // 4 for s in lb.shape); nb = Z * Y * X
    bl = lb.reshape(Z, 4, Y, 4, X, 4).transpose(0, 2, 4, 1, 3, 5).reshape(nb, 64)
    pres = np.zeros((nb, K + 1), bool); pres[np.arange(nb)[:, None], bl] = True
    m2 = np.ascontiguousarray(pres[:, 1:].T.reshape(K, Z, Y, X))
    return torch.from_numpy(hu), torch.from_numpy(m2)


class DS(torch.utils.data.Dataset):
    def __init__(s, order, train, seed): s.order = list(order); s.train = train; s.seed = seed
    def __len__(s): return len(s.order)
    def __getitem__(s, i):
        try: return (i, *load(ids[s.order[i]], s.train, np.random.default_rng(s.seed * 100003 + i)))
        except Exception as e: print("load fail", ids[s.order[i]], e, flush=True); return (i, None, None)


def Prefetch(order, train, seed):
    return torch.utils.data.DataLoader(DS(order, train, seed), batch_size=None, shuffle=False, num_workers=a.workers,
                                       prefetch_factor=3, persistent_workers=False, pin_memory=False)


class FT(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc0.encoder; s.enc.load_state_dict(BASE_SD)
        for i, st in enumerate(s.enc.stages):
            for p in st.parameters(): p.requires_grad_(i >= a.train_from)
        s.norm = nn.LayerNorm(D); s.V = nn.Linear(D, 128); s.U = nn.Linear(D, 128); s.w = nn.Linear(128, 1)
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * D, 1))
    def forward(s, hu, lb, train):
        if a.cache:
            idx, f1, g = hu
            h = torch.zeros(int(torch.prod(g)), f1.shape[1], device=dev, dtype=torch.bfloat16)
            h[idx.to(dev)] = f1.to(dev).view(torch.bfloat16)
            if train: h = h * (torch.rand(1, f1.shape[1], device=dev) > 0.1).to(h.dtype) / 0.9      # channel dropout
            h = h.T.reshape(1, f1.shape[1], *g.tolist()); feats = []; s.h1 = h if a.s1 else None
            for i, st in enumerate(s.enc.stages[2:max(ST) + 1], start=2):
                if i < a.train_from:
                    with torch.no_grad(): h = st(h)
                else: h = st(h)
                if i in ST: feats.append(h)
            return s.pool(feats, lb)
        if train:
            hu = hu.float() * (1 + 0.04 * (torch.rand(1, device=dev) - .5)) + 20 * (torch.rand(1, device=dev) - .5)
        x = torch.stack([apply_window(hu, w, NORM) for w in WINS])[:, None]       # [NW,1,Z,Y,X]
        h = x; feats = []
        if getattr(s.enc, "stem", None) is not None: h = s.enc.stem(h)
        for i, st in enumerate(s.enc.stages[:max(ST) + 1]):
            if i < a.train_from:
                with torch.no_grad(): h = st(h)
            else: h = st(h)
            if i in ST: feats.append(h)
        return s.pool(feats, lb)

    def pool(s, feats, lb):
        mk = lb[None].float(); vs = []; present = None                             # lb = stride-4 presence [K,z,y,x]
        for j, f in enumerate(feats):
            if j: mk = F.max_pool3d(mk, 2)
            m = mk[0].flatten(1) > 0                                               # [K,N]
            if present is None: present = m.sum(1) >= 2
            f = f.flatten(2).float()                                               # [NW,C,N]
            mf = m.float(); cnt = mf.sum(1).clamp(min=1)
            mean = torch.einsum("wcn,kn->kwc", f, mf) / cnt[:, None, None]
            mx = torch.stack([f[:, :, m[k]].amax(-1) if present[k] else f.new_zeros(f.shape[:2]) for k in range(K)])
            vs += [mean, mx]
        if a.s1 and getattr(s, "h1", None) is not None:
            m = F.interpolate(lb[None].float(), scale_factor=2, mode="nearest")[0].flatten(1) > 0
            f = s.h1.flatten(2).float(); mf = m.float(); cnt = mf.sum(1).clamp(min=1)
            vs += [torch.einsum("wcn,kn->kwc", f, mf) / cnt[:, None, None],
                   torch.stack([f[:, :, m[k]].amax(-1) if present[k] else f.new_zeros(f.shape[:2]) for k in range(K)])]
        v = torch.cat(vs, 2).flatten(1)                                            # [K,D]
        v = s.norm(v)[present]
        if len(v) == 0: v = s.norm(torch.zeros(1, D, device=dev))
        att = s.w(torch.tanh(s.V(v)) * torch.sigmoid(s.U(v)))                     # [k,1]
        z = torch.cat([(torch.softmax(att, 0) * v).sum(0), v.amax(0)])
        return s.head(z[None])[0, 0]


def predict(m, idx, tag=""):
    m.eval(); out = np.full(len(idx), np.nan)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, hu, lb in Prefetch(idx, False, 0):
            if hu is None: continue
            out[i] = m(hu if a.cache else hu.to(dev, non_blocking=True), lb.to(dev, non_blocking=True), False).float().item()
    return out


def fit(idx_tr, idx_es, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = FT().to(dev)
    ep_ = [p for i, st in enumerate(m.enc.stages) if i >= a.train_from for p in st.parameters()]
    hp = [p for n, p in m.named_parameters() if not n.startswith("enc.")]
    groups = [{"params": hp, "lr": a.lr * 10}] + ([{"params": ep_, "lr": a.lr}] if ep_ else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    pos = idx_tr[yt[idx_tr] == 1]; neg = idx_tr[yt[idx_tr] == 0]
    nneg = min(len(neg), int(a.negr * len(pos))); per_ep = len(pos) + nneg
    steps = a.epochs * int(np.ceil(per_ep / a.accum))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps + 1, pct_start=0.15)
    pw = torch.tensor(nneg / max(len(pos), 1), device=dev)
    best, bst = -1, None
    for ep in range(a.epochs):
        if os.path.exists(STOP): print("STOP seen", flush=True); sys.exit(3)
        m.train(); order = rng.permutation(np.r_[pos, rng.choice(neg, nneg, replace=False)])
        tl = 0; t1 = time.time(); n = 0; opt.zero_grad()
        for i, hu, lb in Prefetch(order, True, seed * 1000 + ep):
            if hu is None: continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(hu if a.cache else hu.to(dev, non_blocking=True), lb.to(dev, non_blocking=True), True)
            t = torch.tensor(float(yt[order[i]]), device=dev)
            loss = F.binary_cross_entropy_with_logits(lg.float(), t, pos_weight=pw) / a.accum
            loss.backward(); tl += loss.item() * a.accum; n += 1
            if n % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step(); opt.zero_grad()
            if a.bench and n == a.bench:
                dt = time.time() - t1
                print(f"BENCH {n} train cases {dt:.1f}s = {dt / n:.3f}s/case peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
                t2 = time.time(); predict(m, idx_es[:a.bench]); dt = time.time() - t2
                print(f"BENCH predict {a.bench} {dt:.1f}s = {dt / a.bench:.3f}s/case", flush=True); sys.exit(0)
        sc = auc(yt[idx_es], predict(m, idx_es))
        print(f"  seed {seed} ep {ep} n {n} loss {tl / max(n, 1):.4f} inner AUROC {sc:.4f} {time.time() - t1:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if sc > best: best, bst = sc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst); return m, best


vai = np.where(~tr)[0]
if a.score:
    for k in [int(x) for x in a.folds.split(",")]:
        mf = f"{a.out}/model_fold{k}.pt"
        if not os.path.exists(mf): print("no model", k, flush=True); continue
        m = FT().to(dev); miss = m.load_state_dict(torch.load(mf), strict=False)
        assert not miss.unexpected_keys, miss.unexpected_keys
        idx = np.where((tr & (fold == k)) | (~tr if not a.no_val else np.zeros(len(tr), bool)))[0]
        t0 = time.time(); p = predict(m, idx)
        pd.DataFrame({"study_id": ids[idx], "split": lab.split.values[idx], "fold": k, "ft": p}).to_csv(f"{a.out}/score_fold{k}.csv", index=False)
        print(f"scored fold {k}: {len(idx)} rows {time.time() - t0:.0f}s", flush=True); del m
    print("RIB_FT_SCORE_DONE", flush=True); sys.exit(0)
for k in [int(x) for x in a.folds.split(",")]:
    of = f"{a.out}/oof_fold{k}.csv"
    if os.path.exists(of): print(f"fold {k} done, skip", flush=True); continue
    t0 = time.time()
    a_ = np.where(trainable & (fold != k))[0]; b_ = np.where(tr & (fold == k))[0]
    a_ = np.random.default_rng(1000 * a.seed + k).permutation(a_); nes = int(0.12 * len(a_))
    m, b = fit(a_[nes:], a_[:nes], 100 * a.seed + k)
    p = predict(m, b_); hb = ~np.isnan(yt[b_]) & ~np.isnan(p)
    print(f"fold {k} inner {b:.4f} OOF AUROC(t_rib) {auc(yt[b_][hb], p[hb]):.4f} {time.time() - t0:.0f}s", flush=True)
    if not a.no_val:
        pv = predict(m, vai)
        pd.DataFrame({"study_id": ids[vai], "fold": k, "ft": pv}).to_csv(f"{a.out}/val_fold{k}.csv", index=False)
    torch.save({kk: v.cpu() for kk, v in m.state_dict().items() if not kk.startswith("enc.") or int(kk.split(".")[2]) >= a.train_from}, f"{a.out}/model_fold{k}.pt")
    pd.DataFrame({"study_id": ids[b_], "fold": k, "ft": p}).to_csv(of, index=False)
    json.dump({"inner": b, "args": vars(a)}, open(f"{a.out}/fold{k}.json", "w"))
    del m; torch.cuda.empty_cache()
print("RIB_FT_DONE", flush=True)
