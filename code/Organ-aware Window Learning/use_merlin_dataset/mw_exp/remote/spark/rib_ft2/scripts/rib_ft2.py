"""rib_ft2: end-to-end fine-tune of the TS Dataset295 encoder (stages 2-3 trainable, 0-1 frozen) on the full 1.5 mm rib
ROI with REAL context (no masked cache), bone window, instance MIL over rib x 24-sector instances (same instances as
rib_feat3). Per instance: trainable stage-2/3 masked mean+max  ++ frozen feats3 vector (stage 0-1, baseline+bone)
-> MLP -> gated attention + max -> logit. Target t_rib (train/val only, asserted; test never read).
Folds = labels.fold (sitewise). Each fold: train on other folds (all rib+ and NEGR x negatives resampled per epoch),
best epoch on inner 12 % holdout, score held-out fold -> oof_fold{k}.csv. --full: one model on all train -> val_scores.csv.
usage: rib_ft2.py LABELS ROI FEATS3 OUT [--folds 0,1,2,3,4 | --full] [--epochs 3] [--negr 3] [--lr 1e-4]"""
import os, sys, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("labels"); ap.add_argument("roi"); ap.add_argument("feats"); ap.add_argument("out")
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--full", action="store_true")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--negr", type=float, default=3); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hid", type=int, default=512); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bench", type=int, default=0)
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src")); ap.add_argument("--nsec", type=int, default=24)
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda"); STOP = os.path.join(a.out, "STOP"); NS = a.nsec; NI = 24 * NS
lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
have = {f[:-4] for f in os.listdir(a.feats) if f.endswith(".npz")} & {f[:-4] for f in os.listdir(a.roi) if f.endswith(".npz")}
lab = lab[lab.study_id.isin(have)].reset_index(drop=True)
if a.limit: lab = pd.concat([lab[lab.t_rib == 1].head(a.limit), lab[lab.t_rib == 0].head(3 * a.limit)]).reset_index(drop=True)
ids = lab.study_id.values; tr = (lab.split == "train").values; fold = lab.fold.values.astype(int); yt = lab.t_rib.values
trainable = tr & ~np.isnan(yt)
# feats3 normalisation (train instances)
smp = [np.load(os.path.join(a.feats, c + ".npz"))["X"].astype(np.float32) for c in ids[np.where(tr)[0][::20]]]
smp = np.concatenate(smp); MU = torch.tensor(smp.mean(0), device=dev); SD = torch.tensor(smp.std(0) + 1e-4, device=dev); D3 = smp.shape[1]; del smp
enc = NnUNetEncoder(a.ts); NORM = enc.norm; BASE = {k: v.detach().clone() for k, v in enc.encoder.state_dict().items()}
CH = enc.stage_channels([2, 3]); DT = 2 * sum(CH)
print(f"N {len(lab)} trainable {trainable.sum()} rib+ {(yt[tr] == 1).sum()} val {(~tr).sum()} D3 {D3} DT {DT}", flush=True)


def load(cid, train, rng):
    z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb = z["hu"], z["lab"]
    f3 = np.load(os.path.join(a.feats, cid + ".npz")); X3, inst = f3["X"].astype(np.float32), f3["inst"].astype(np.int64)
    P = [(-s) % 16 for s in hu.shape]
    hu = np.pad(hu, [(0, p) for p in P], constant_values=-1024); lb = np.pad(lb, [(0, p) for p in P])
    rib = (lb >= 1) & (lb <= 24); idx = np.argwhere(rib)                         # instance map identical to rib_feat3
    cy, cx = idx[:, 1].mean(), idx[:, 2].mean()
    sec = ((np.arctan2(idx[:, 1] - cy, idx[:, 2] - cx) + np.pi) / (2 * np.pi) * NS).astype(int) % NS
    im = np.full(lb.shape, -1, np.int16); im[rib] = (lb[rib].astype(np.int16) - 1) * NS + sec
    return torch.from_numpy(hu), torch.from_numpy(im), torch.from_numpy(X3), torch.from_numpy(inst)


class DS(torch.utils.data.Dataset):
    def __init__(s, order, train, seed): s.o = list(order); s.t = train; s.seed = seed
    def __len__(s): return len(s.o)
    def __getitem__(s, i): return (i, *load(ids[s.o[i]], s.t, np.random.default_rng(s.seed * 100003 + i)))


def loader(order, train, seed):
    return torch.utils.data.DataLoader(DS(order, train, seed), batch_size=None, num_workers=a.workers, prefetch_factor=2)


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc.encoder; s.enc.load_state_dict(BASE)
        for i, st in enumerate(s.enc.stages): [p.requires_grad_(i in (2, 3)) for p in st.parameters()]
        s.nt = nn.LayerNorm(DT)
        s.f = nn.Sequential(nn.Dropout(0.3), nn.Linear(DT + D3, a.hid), nn.GELU(), nn.Dropout(0.3))
        s.att = nn.Linear(a.hid, 1); s.head = nn.Linear(2 * a.hid, 1)
    def forward(s, hu, im, X3, inst, train):
        hu = hu.float()
        if train: hu = hu * (1 + 0.04 * (torch.rand(1, device=dev) - .5)) + 20 * (torch.rand(1, device=dev) - .5)
        x = apply_window(hu, {"level": 400, "width": 1800}, NORM)[None, None]
        with torch.no_grad(): h = s.enc.stages[1](s.enc.stages[0](x))
        ii = F.max_pool3d(F.max_pool3d(im[None, None].float(), 2), 2)                # stage-2 grid instance id
        out = []
        for st in (2, 3):
            h = s.enc.stages[st](h)
            if st == 3: ii = F.max_pool3d(ii, 2)
            f = h[0].flatten(1).float(); i_ = ii.flatten().long(); m = i_ >= 0; f = f[:, m]; i_ = i_[m]; C = f.shape[0]
            cnt = torch.zeros(NI, device=dev).index_add_(0, i_, torch.ones_like(i_, dtype=torch.float))
            mean = torch.zeros(NI, C, device=dev).index_add_(0, i_, f.T) / cnt.clamp(min=1)[:, None]
            mx = torch.full((NI, C), -1e4, device=dev).scatter_reduce(0, i_[:, None].expand(-1, C), f.T, "amax", include_self=True)
            mx = torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx)); out += [mean[inst], mx[inst]]
        t = s.nt(torch.cat(out, 1)); v = torch.cat([t, (X3 - MU) / SD], 1)             # [I, DT+D3]
        hh = s.f(v); w = torch.softmax(s.att(hh)[:, 0], 0)[:, None]
        return s.head(torch.cat([(w * hh).sum(0), hh.amax(0)]))[0]


def run(m, it, train):
    i, hu, im, X3, inst = it
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(hu.to(dev), im.to(dev), X3.to(dev), inst.to(dev), train)


def predict(m, idx):
    m.eval(); out = np.full(len(idx), np.nan)
    with torch.no_grad():
        for it in loader(idx, False, 0): out[it[0]] = run(m, it, False).float().item()
    return out


def fit(itr, ies, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = M().to(dev)
    ip, ineg = ies[yt[ies] == 1], ies[yt[ies] == 0]; ies = np.r_[ip, rng.permutation(ineg)[:max(600 - len(ip), 2 * len(ip))]]
    ep_ = [p for n, p in m.named_parameters() if n.startswith("enc.") and p.requires_grad]
    hp = [p for n, p in m.named_parameters() if not n.startswith("enc.")]
    g = [{"params": hp, "lr": a.lr * 10}, {"params": ep_, "lr": a.lr}]; opt = torch.optim.AdamW(g, weight_decay=1e-2)
    pos = itr[yt[itr] == 1]; neg = itr[yt[itr] == 0]; nn_ = min(len(neg), int(a.negr * len(pos)))
    steps = a.epochs * int(np.ceil((len(pos) + nn_) / a.accum)) + 1
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [x["lr"] for x in g], total_steps=steps, pct_start=0.15)
    pw = torch.tensor(nn_ / len(pos), device=dev); best, bst = -1, None
    for ep in range(a.epochs):
        if os.path.exists(STOP): print("STOP", flush=True); sys.exit(3)
        m.train(); order = rng.permutation(np.r_[pos, rng.choice(neg, nn_, replace=False)]); t1 = time.time(); n = 0; tl = 0
        opt.zero_grad()
        for it in loader(order, True, seed * 1000 + ep):
            lg = run(m, it, True)
            loss = F.binary_cross_entropy_with_logits(lg.float(), torch.tensor(float(yt[order[it[0]]]), device=dev), pos_weight=pw) / a.accum
            loss.backward(); tl += loss.item() * a.accum; n += 1
            if n % a.accum == 0: torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step(); opt.zero_grad()
            if a.bench and n == a.bench:
                print(f"BENCH {(time.time() - t1) / n:.3f}s/case peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
                t2 = time.time(); predict(m, ies[:a.bench]); print(f"BENCH pred {(time.time() - t2) / a.bench:.3f}s/case", flush=True); sys.exit(0)
        sc = auc(yt[ies], predict(m, ies))
        print(f"  seed {seed} ep {ep} n {n} loss {tl / n:.4f} inner AUROC {sc:.4f} {time.time() - t1:.0f}s", flush=True)
        if sc > best: best, bst = sc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst); return m, best


if a.full:
    a_ = np.random.default_rng(999 + a.seed).permutation(np.where(trainable)[0]); ne = int(0.12 * len(a_))
    m, b = fit(a_[ne:], a_[:ne], 900 + a.seed); vi = np.where(~tr)[0]
    pd.DataFrame({"study_id": ids[vi], "ft": predict(m, vi)}).to_csv(f"{a.out}/val_scores_seed{a.seed}.csv", index=False)
    print(f"FULL inner {b:.4f} -> val scored (AUROC not computed)", flush=True); sys.exit(0)
for k in [int(x) for x in a.folds.split(",")]:
    of = f"{a.out}/oof_fold{k}.csv"
    if os.path.exists(of): continue
    t0 = time.time(); a_ = np.random.default_rng(1000 * a.seed + k).permutation(np.where(trainable & (fold != k))[0]); ne = int(0.12 * len(a_))
    m, b = fit(a_[ne:], a_[:ne], 100 * a.seed + k); bi = np.where(tr & (fold == k))[0]; p = predict(m, bi)
    h = ~np.isnan(yt[bi]); print(f"fold {k} inner {b:.4f} OOF AUROC(t_rib) {auc(yt[bi][h], p[h]):.4f} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"study_id": ids[bi], "fold": k, "ft": p}).to_csv(of, index=False); del m; torch.cuda.empty_cache()
print("RIB_FT2_DONE", flush=True)
