"""gs15 small 3D CNN finetuned on the 1.5 mm GB crops (train folds only; GS1 5-fold OOF -> stack input; val = full-train
models, scored once). Test is never read: the labels csv holds train/val rows only (asserted) and crops exist for train/val only.
Input per case: central 64^3 (96 mm) of the 96^3 x 1.5 mm crop, 3 channels = soft-tissue window (40/400), narrow window
(60/150), GB mask (r0 = GB dilated 3 mm). Augmentation: random shift +-6 vox, intensity jitter, no flips (anatomy).
Net: 3D ResNet-ish (stride-2 stem 32 -> 48-96-192, BN, mean|max pool, dropout 0.3), AdamW 1e-3 / wd 1e-4, cosine, class-balanced BCE,
EPOCHS epochs, best epoch by AUROC on an inner 12 % train holdout (from the training folds only), 2 seeds averaged.
usage: gs15_cnn_train.py ROOT LABELS_CSV OUT [--epochs 14] [--seeds 2]"""
import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("labels"); ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--pack", default=None, help="npz from gs15_cnn_pack.py instead of ROOT/crops")
ap.add_argument("--vram-gb", type=float, default=14)
ap.add_argument("--train-only", action="store_true", help="tuning: drop val rows entirely (no val model, no val score)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(4); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
lab = lab[lab.split.isin(["train", "val"])].reset_index(drop=True)
if a.train_only: lab = lab[lab.split == "train"].reset_index(drop=True)
if a.limit: lab = pd.concat([lab[lab.split == "train"].head(a.limit), lab[lab.split == "val"].head(a.limit // 4)]).reset_index(drop=True)
C, S0 = 64, 96; o0 = (S0 - C) // 2
N = len(lab); HU = np.full((N, C, C, C), -1024, np.int16); GB = np.zeros((N, C, C, C), np.uint8); ok = np.zeros(N, bool)
t0 = time.time()
if a.pack:
    pk = np.load(a.pack); pos = {s: i for i, s in enumerate(pk["ids"])}; Hp, Gp = pk["hu"], pk["gb"]
    for i, s in enumerate(lab.study_id):
        if s in pos:
            j = pos[s]; HU[i] = Hp[j]; GB[i] = np.unpackbits(Gp[j])[:C ** 3].reshape(C, C, C); ok[i] = True
    del Hp, Gp, pk
else:
  for i, s in enumerate(lab.study_id):
      f = f"{a.root}/crops/{s}.npz"
      if not os.path.exists(f): continue
      z = np.load(f)
      if "hu" not in z: continue
      sl = (slice(o0, o0 + C),) * 3
      HU[i] = z["hu"][0][sl]
      rs = tuple(z["rshape"]); R = np.unpackbits(z["regions"])[:int(np.prod(rs))].reshape(rs)
      GB[i] = R[0, 0][sl]; ok[i] = True
print(f"loaded {ok.sum()}/{N} crops in {time.time() - t0:.0f}s", flush=True)
y = lab.y.values; has = ~np.isnan(y) & ok; y = np.nan_to_num(y).astype(np.float32)
tr = (lab.split == "train").values; fold = lab.fold.values


def win(h, L, W):
    return ((h - (L - W / 2)) / W).clamp(0, 1) * 2 - 1


def batch(idx, train):
    h = torch.from_numpy(HU[idx]).to(dev).float(); g = torch.from_numpy(GB[idx]).to(dev).float()
    x = torch.stack([win(h, 40, 400), win(h, 60, 150), g * 2 - 1], 1)
    if train:
        sh = np.random.randint(-6, 7, 3)
        x = torch.roll(x, shifts=tuple(int(v) for v in sh), dims=(2, 3, 4))
        x[:, :2] = x[:, :2] * (1 + 0.1 * (torch.rand(len(idx), 1, 1, 1, 1, device=dev) - .5)) + 0.1 * (torch.rand(len(idx), 1, 1, 1, 1, device=dev) - .5)
    return x


class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__(); s.c1 = nn.Conv3d(i, o, 3, st, 1, bias=False); s.b1 = nn.BatchNorm3d(o); s.c2 = nn.Conv3d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm3d(o)
        s.sk = nn.Sequential(nn.Conv3d(i, o, 1, st, bias=False), nn.BatchNorm3d(o)) if (i != o or st != 1) else nn.Identity()
    def forward(s, x): return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv3d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm3d(32), nn.ReLU())
        s.l = nn.Sequential(Block(32, 48, 2), Block(48, 48, 1), Block(48, 96, 2), Block(96, 96, 1), Block(96, 192, 2))
        s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(192 * 2, 1))
    def forward(s, x):
        f = s.l(s.stem(x)); return s.fc(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], 1))[:, 0]


def predict(m, idx):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), 64):
            out.append(m(batch(idx[i:i + 64], False)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(idx_tr, idx_es, idx_pred, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(idx_tr) / a.bs)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    npos = y[idx_tr].sum(); pw = torch.tensor((len(idx_tr) - npos) / max(npos, 1), device=dev)
    best, bst = -1, None
    for ep in range(a.epochs):
        m.train(); perm = np.random.permutation(idx_tr); tl = 0
        for i in range(0, len(perm), a.bs):
            j = np.sort(perm[i:i + a.bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(batch(j, True))
            loss = F.binary_cross_entropy_with_logits(lg.float(), torch.from_numpy(y[j]).to(dev), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(j)
        sc = auc(y[idx_es], predict(m, idx_es))
        print(f"  seed {seed} ep {ep} loss {tl / len(perm):.4f} inner AUROC {sc:.4f}", flush=True)
        if sc > best: best, bst = sc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst); p = predict(m, idx_pred); del m, opt; torch.cuda.empty_cache(); return p, best


tri = np.where(tr & has)[0]; vai = np.where(~tr & ok)[0]
oof = np.full(N, np.nan); log = {}
for k in sorted(np.unique(fold[tri].astype(int))):
    a_ = tri[fold[tri] != k]; b_ = np.where(tr & ok & (fold == k))[0]
    a_ = np.random.default_rng(int(k)).permutation(a_); nes = int(0.12 * len(a_))
    ps = []
    for sd in range(a.seeds):
        t1 = time.time(); p, b = fit(a_[nes:], a_[:nes], b_, 100 + sd); ps.append(p); log[f"f{k}_s{sd}"] = b
        print(f"fold {k} seed {sd} inner {b:.4f} {time.time() - t1:.0f}s", flush=True)
    oof[b_] = np.mean(ps, 0)
    hb = has[b_]; print(f"fold {k} OOF AUROC {auc(y[b_][hb], oof[b_][hb]):.4f}", flush=True)
k_ = tr & has & ~np.isnan(oof)
print(f"CNN train OOF {auc(y[k_], oof[k_]):.4f}", flush=True)
if a.train_only:
    json.dump(dict(oof=float(auc(y[k_], oof[k_])), inner=log, epochs=a.epochs, seeds=a.seeds, lr=a.lr, n_train=int(len(tri))), open(f"{a.out}/summary.json", "w"), indent=1)
    print("GS15_CNN_TUNE_DONE", flush=True); sys.exit(0)
rr = np.random.default_rng(0).permutation(tri); nes = int(0.12 * len(rr))
vv = np.mean([fit(rr[nes:], rr[:nes], vai, 200 + sd)[0] for sd in range(a.seeds)], 0)
hv = has[vai]; print(f"CNN val {auc(y[vai][hv], vv[hv]):.4f}", flush=True)
ti = np.where(tr)[0]
pd.DataFrame({"study_id": lab.study_id.values[ti], "fold": fold[ti].astype(int), "cnn": oof[ti]}).to_csv(f"{a.out}/oof_train.csv", index=False)
pd.DataFrame({"study_id": lab.study_id.values[vai], "cnn": vv}).to_csv(f"{a.out}/val_scores.csv", index=False)
json.dump(dict(oof=float(auc(y[k_], oof[k_])), val=float(auc(y[vai][hv], vv[hv])), inner=log, epochs=a.epochs, seeds=a.seeds), open(f"{a.out}/summary.json", "w"), indent=1)
print("GS15_CNN_DONE", flush=True)
