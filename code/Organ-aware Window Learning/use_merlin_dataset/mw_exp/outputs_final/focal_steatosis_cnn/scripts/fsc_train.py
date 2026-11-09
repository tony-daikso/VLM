"""Small multi-region 3D CNN for 局部脂肪浸潤 (y_focal) on the Exp N fst crops (3 liver sites x 48^3, 3 mm).
Shared 3D ResNet-ish encoder per site -> per-site embedding (+ site embedding) -> gated-attention + max pooling over the
present sites -> logit. Input channels per site: 5 windows (windows.yaml: baseline~z-score, soft, liver, narrow, bone),
liver-relative HU ((HU - case liver median)/40), site mask r0, crop-liver mask r2.
Heavy aug (random 40^3 crop, flips, HU scale/shift, noise, site dropout), balanced sampling (pos 25 %), EMA weights.
Per seed: 5 train folds (labels_trval fold column = y_focal OOF folds) -> OOF logits (+ val preds of each fold model),
then full-train model -> val. Never sees test (labels file is train/val only).  Resource caps: GPU <= --vram GB,
torch threads 2.   usage: fsc_train.py --work W --out OUT --seed S [--epochs E] [--cfg base]"""
import argparse, os, zlib, time, json, threading, queue, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--spe", type=int, default=4000); ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--pos-frac", type=float, default=0.25)
ap.add_argument("--width", type=int, default=16); ap.add_argument("--crop", type=int, default=40); ap.add_argument("--vram", type=float, default=7.5)
ap.add_argument("--arch", default="3d", choices=["3d", "aniso"])
ap.add_argument("--folds", default="0,1,2,3,4,full"); ap.add_argument("--max-steps", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(2); torch.set_num_interop_threads(1)
dev = "cuda"; torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
torch.backends.cudnn.benchmark = True
os.makedirs(a.out, exist_ok=True)
M = pd.read_csv(f"{a.work}/meta.csv"); assert set(M.split) <= {"train", "val"}
HU = np.load(f"{a.work}/hu.npy", mmap_mode="r"); MK = np.load(f"{a.work}/msk.npy", mmap_mode="r")
PRES = M[["present0", "present1", "present2"]].values.astype(bool); MED = M.liver_med.fillna(M.liver_med.median()).values.astype(np.float32)
WINS = [(None, None), (40, 400), (60, 150), (40, 250), (400, 1800)]  # baseline handled as z-score
S, CR = 48, a.crop


def load(idx):
    idx = np.asarray(idx); o = np.argsort(idx); r = np.empty_like(o); r[o] = np.arange(len(o)); s = idx[o]
    h = torch.from_numpy(np.ascontiguousarray(HU[s]))[r]; m = torch.from_numpy(np.ascontiguousarray(MK[s]))[r]
    return h, m, torch.from_numpy(PRES[idx]), torch.from_numpy(MED[idx])


def to_input(h, m, pres, med, train):
    """h int16 [B,3,S,S,S] -> x [B,3,8,CR,CR,CR] on GPU (augment if train)."""
    h = h.to(dev, non_blocking=True).float(); m = m.to(dev, non_blocking=True); pres = pres.to(dev).clone(); med = med.to(dev)
    B = h.shape[0]
    if train:
        sc = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.97, 1.03); sh = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-10, 10)
        h = h * sc + sh + torch.randn_like(h) * torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0, 8)
        med = med * sc.view(B) + sh.view(B)
        drop = (torch.rand(B, 3, device=dev) < 0.15) & pres
        keep1 = (pres & ~drop).any(1); drop[~keep1] = False; pres = pres & ~drop
        ox, oy, oz = (np.random.randint(0, S - CR + 1, 3) for _ in range(3))
    pres[~pres.any(1)] = True  # cases with no present site: attend over all (avoid -1e4 max-pool blow-up)
    xs = []
    for k in range(3):
        if train:
            o0, o1, o2 = ox[k], oy[k], oz[k]
        else:
            o0 = o1 = o2 = (S - CR) // 2
        hk = h[:, k, o0:o0 + CR, o1:o1 + CR, o2:o2 + CR]; mk = m[:, k, o0:o0 + CR, o1:o1 + CR, o2:o2 + CR]
        ch = []
        for L, Wd in WINS:
            if L is None: ch.append(((hk.clamp(-1024, 2000) - 40) / 300))
            else: ch.append(((hk - (L - Wd / 2)) / Wd).clamp(0, 1) * 2 - 1)
        ch.append(((hk - med.view(B, 1, 1, 1)) / 40).clamp(-4, 4))
        ch.append((mk & 1).float() * 2 - 1); ch.append(((mk >> 2) & 1).float() * 2 - 1)
        xs.append(torch.stack(ch, 1))
    x = torch.stack(xs, 1)  # B,3,8,D,H,W
    if train:
        for ax in (3, 4, 5):
            f = torch.rand(B, device=dev) < 0.5
            if f.any(): x[f] = x[f].flip(ax)
    return x, pres


class Res(nn.Module):
    def __init__(s, ci, co, st, k=3):
        super().__init__()
        ks, pd_ = ((1, 3, 3), (0, 1, 1)) if k == "a" else (3, 1)
        s.c1 = nn.Conv3d(ci, co, ks, st, pd_, bias=False); s.b1 = nn.BatchNorm3d(co); s.c2 = nn.Conv3d(co, co, ks, 1, pd_, bias=False); s.b2 = nn.BatchNorm3d(co)
        s.sk = nn.Sequential(nn.Conv3d(ci, co, 1, st, bias=False), nn.BatchNorm3d(co)) if (st != 1 or ci != co) else nn.Identity()
    def forward(s, x): return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class Net(nn.Module):
    def __init__(s, w=16, d=128, arch="3d"):
        super().__init__()
        k = "a" if arch == "aniso" else 3  # aniso = 2.5D-style: in-plane (1x3x3) convs in stem + layer1, 3D afterwards
        ks, pd_ = ((1, 3, 3), (0, 1, 1)) if k == "a" else (3, 1)
        s.stem = nn.Sequential(nn.Conv3d(8, w, ks, 1, pd_, bias=False), nn.BatchNorm3d(w), nn.ReLU())
        s.l1 = nn.Sequential(Res(w, w, 1, k), Res(w, 2 * w, 2, k)); s.l2 = Res(2 * w, 4 * w, 2); s.l3 = nn.Sequential(Res(4 * w, 8 * w, 2), Res(8 * w, 8 * w, 1))
        s.proj = nn.Sequential(nn.Linear(8 * w * 3, d), nn.ReLU(), nn.Dropout(0.3)); s.site = nn.Parameter(torch.zeros(3, d))
        s.att_v = nn.Linear(d, 64); s.att_u = nn.Linear(d, 64); s.att_w = nn.Linear(64, 1)
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, 1))
    def forward(s, x, pres):
        B, K = x.shape[:2]; liv = (x[:, :, 7:8] > 0).flatten(0, 1).float()
        f = s.l3(s.l2(s.l1(s.stem(x.flatten(0, 1)))))
        lm = F.adaptive_avg_pool3d(liv, f.shape[2:]); lm = (lm > 0.2).float()
        gap = f.mean((2, 3, 4)); gmp = f.amax((2, 3, 4)); lmp = (f * lm).sum((2, 3, 4)) / lm.sum((2, 3, 4)).clamp(min=1)
        e = s.proj(torch.cat([gap, gmp, lmp], 1)).view(B, K, -1) + s.site
        a_ = s.att_w(torch.tanh(s.att_v(e)) * torch.sigmoid(s.att_u(e))).squeeze(-1).masked_fill(~pres, -1e4)
        att = torch.softmax(a_, 1); ea = (att.unsqueeze(-1) * e).sum(1)
        em = e.masked_fill(~pres.unsqueeze(-1), -1e4).amax(1)
        return s.head(torch.cat([ea, em], 1)).squeeze(-1)


def predict(net, rows):
    net.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), 48):
            h, m, p, md = load(rows[i:i + 48]); x, pr = to_input(h, m, p, md, False)
            pr[~pr.any(1)] = True
            l = 0
            for fl in [(), (5,), (4,), (3, 5)]:
                l = l + net(x.flip(fl) if fl else x, pr).float()
            out.append((l / 4).cpu().numpy())
    return np.concatenate(out)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def train_one(tr_rows, ev_sets, tag):
    torch.manual_seed(a.seed * 100 + zlib.crc32(tag.encode()) % 97); np.random.seed(a.seed * 100 + zlib.crc32(tag.encode()) % 97)
    y = M.y.values[tr_rows]; pos, neg = tr_rows[y == 1], tr_rows[y == 0]
    net = Net(a.width, arch=a.arch).to(dev).to(memory_format=torch.channels_last_3d)
    ema = Net(a.width, arch=a.arch).to(dev); ema.load_state_dict(net.state_dict()); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (a.spe // a.bs); 
    if a.max_steps: steps = min(steps, a.max_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    q = queue.Queue(maxsize=3)
    def producer():
        rng = np.random.default_rng(a.seed * 7 + len(tag))
        for _ in range(steps):
            npos = rng.binomial(a.bs, a.pos_frac)
            b = np.concatenate([rng.choice(pos, npos), rng.choice(neg, a.bs - npos)])
            q.put((load(b), torch.from_numpy(M.y.values[b].astype(np.float32))))
    th = threading.Thread(target=producer, daemon=True); th.start()
    t0 = time.time(); lsum = 0
    for st in range(steps):
        (h, m, p, md), yb = q.get(); x, pr = to_input(h, m, p, md, True)
        net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = F.binary_cross_entropy_with_logits(net(x, pr).float(), yb.to(dev) * 0.95 + 0.025)
        opt.zero_grad(set_to_none=True); lo.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sch.step()
        with torch.no_grad():
            dcy = min(0.998, (1 + st) / (10 + st))
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(dcy).add_(pn.detach(), alpha=1 - dcy)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        lsum += lo.item()
        if (st + 1) % 200 == 0:
            print(f"[{tag}] step {st + 1}/{steps} loss {lsum / 200:.4f} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True); lsum = 0
    th.join()
    return {k: predict(ema, r) for k, r in ev_sets.items()}


tr = M[(M.split == "train") & M.has_crop]; va = M[M.split == "val"]; vrows = va.row.values
for fd in a.folds.split(","):
    fo = f"{a.out}/pred_{fd}_s{a.seed}.csv"
    if os.path.exists(fo): print("skip", fo, flush=True); continue
    if fd == "full":
        P = train_one(tr.row.values, {"val": vrows}, f"full_s{a.seed}")
        d = pd.DataFrame({"study_id": va.study_id, "split": "val", "fold": -1, "y": va.y, "logit": P["val"], "src": "full"})
        print(f"full s{a.seed} done", flush=True)
    else:
        k = int(fd); te = tr[tr.fold == k]
        P = train_one(tr[tr.fold != k].row.values, {"oof": te.row.values, "val": vrows}, f"f{k}_s{a.seed}")
        d = pd.concat([pd.DataFrame({"study_id": te.study_id, "split": "train", "fold": k, "y": te.y, "logit": P["oof"], "src": f"f{k}"}),
                       pd.DataFrame({"study_id": va.study_id, "split": "val", "fold": -1, "y": va.y, "logit": P["val"], "src": f"f{k}"})])
        print(f"fold {k} s{a.seed}: inner OOF AUROC {auc(te.y, P['oof']):.4f}", flush=True)
    d.to_csv(fo + ".tmp", index=False); os.replace(fo + ".tmp", fo)
print("TRAIN_DONE", a.seed, flush=True)
