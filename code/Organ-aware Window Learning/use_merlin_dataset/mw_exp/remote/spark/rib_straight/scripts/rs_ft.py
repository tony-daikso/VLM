"""rib_straight FT: fine-tune the late stages (>= --train-from, default 2) of a TS nnU-Net encoder on straightened rib tiles
(rs_tile.py: per rib a dense [L, 32, 32] 1.5 mm curved-planar tile, L multiple of 16, >= 64) with MIL over rib segments.
spine15/pelvis15 tf2lse recipe: each rib tile is cut into fixed 64-row chunks (stride 48, last chunk aligned to the end;
every chunk is dense real anatomy -> InstanceNorm sees normal statistics) -> windows -> encoder stages 0..4 (stages <
train_from frozen) -> per 16-row arc cell (24 mm; each cell taken from the chunk whose centre is closest): stage-3
[mean|max|rib-mask mean] + stage-4 [mean|max] per window -> layer-norm -> dropout -> linear cell logit + rib bias (24) +
arc-position bias (4 bins) -> logsumexp MIL -> case logit (BCE, pos_weight). Only cells overlapping the true rib are used.
Weak supervision as rib_v3: --side w: left (ribs 1-12) / right (13-24) bag lse vs train-report side (rib_side.csv);
--lev w: max cell logit over the mentioned rib levels (+-1, that side) -> 1; negatives max cell -> 0.
--maxch caps the chunks per case in training (random subset); --ckpt = gradient checkpointing of the trained stages.
Train/val only (asserted); test never read. Target t_rib (NaN rows not trained). Folds = labels.fold (sitewise outer).
Pool (--pool-ids) = training rows; --neg-epoch N: each epoch uses all pool positives + N fresh random pool negatives.
--tune: held-out POOL rows scored (selection metric = train OOF on pool rows); no val; no models saved.
final: held-out fold rows (all train with tiles) scored + val unless --no-val; fold models saved; --score fills later.
usage: rs_ft.py --labels L --tiles DIR --out DIR --ts MODEL_DIR [--side-csv S --side .5 --lev .3] [--tune] [--pool-ids F]"""
import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
from torch.utils.checkpoint import checkpoint
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--tiles", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ts", required=True); ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src"))
ap.add_argument("--train-from", type=int, default=2); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--vram-gb", type=float, default=20)
ap.add_argument("--threads", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--tune", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--folds", default="")
ap.add_argument("--pool-ids", default=""); ap.add_argument("--neg-epoch", type=int, default=0)
ap.add_argument("--es-max-neg", type=int, default=0, help="inner holdout: all positives + at most this many negatives (0 = all)")
ap.add_argument("--side-csv", default=""); ap.add_argument("--side", type=float, default=0.0); ap.add_argument("--lev", type=float, default=0.0)
ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--stride", type=int, default=48); ap.add_argument("--maxch", type=int, default=40)
ap.add_argument("--cs", type=int, default=32); ap.add_argument("--shift", type=int, default=2); ap.add_argument("--ckpt", action="store_true")
ap.add_argument("--amp", default="none", choices=["none", "bf16"]); ap.add_argument("--no-val", action="store_true")
ap.add_argument("--target", default="t_rib"); ap.add_argument("--wins", default="base,bone"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--padc", type=int, default=8, help="pad the chunk count to a multiple of this (fixed cudnn shapes)")
ap.add_argument("--no-cudnn-bench", action="store_true")
ap.add_argument("--pin", action="store_true", help="pin_memory (never on the A6000)"); ap.add_argument("--bench", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
torch.set_num_threads(a.threads); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = not a.no_cudnn_bench
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")
STOP = os.path.join(os.path.dirname(os.path.abspath(a.out)), "STOP")
CK, CS, NCELL = a.chunk, a.cs, a.chunk // 16

lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
if a.limit:
    lab = pd.concat([lab[lab.split == "train"].head(a.limit), lab[lab.split == "val"].head(a.limit // 4)]).reset_index(drop=True)
N = len(lab); SID = np.array(lab.study_id.astype(str).tolist(), dtype=str)
ok = np.array([os.path.exists(f"{a.tiles}/{s}.npz") for s in SID])
print(f"tiles present {ok.sum()}/{N} (train {ok[lab.split.values == 'train'].sum()}, val {ok[lab.split.values == 'val'].sum()})", flush=True)
yt = lab[a.target].values.astype(float); has = ~np.isnan(yt); y = np.nan_to_num(yt).astype(np.float32)
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int)
if a.pool_ids:
    pid = {l.strip() for l in open(a.pool_ids) if l.strip()}; pool_rows = tr & has & np.isin(SID, list(pid))
else:
    pool_rows = tr & has
sR = np.full(N, np.nan, np.float32); sL = sR.copy(); LEV = np.zeros((N, 24), bool)
if a.side_csv:
    sc = pd.read_csv(a.side_csv); sc = sc[sc.study_id.isin(set(SID[tr]))].set_index("study_id")  # train rows only
    ix = {s: i for i, s in enumerate(SID)}
    for s_, r_ in sc.iterrows():
        i = ix.get(s_)
        if i is None or not tr[i]: continue
        sR[i], sL[i] = r_.sideR, r_.sideL
        for col, base in (("levL", 0), ("levR", 12)):
            if isinstance(r_[col], str) and r_[col]:
                for l in map(int, r_[col].split(";")):
                    for ll in (l - 1, l, l + 1):
                        if 1 <= ll <= 12: LEV[i, base + ll - 1] = True
print(f"pool rows {pool_rows.sum()} (pos {int(y[pool_rows].sum())}); side known in pool {np.isfinite(sR[pool_rows]).sum()}; "
      f"lev rows in pool {int(LEV[pool_rows].any(1).sum())}", flush=True)


def load_case(i, train, rng):
    z = np.load(f"{a.tiles}/{SID[i]}.npz"); hu, mk, tab = z["hu"], z["m"], z["tab"]
    chunks, cells = [], []  # chunk: (row0); cell: (chunk idx, q, rib 0..23, posbin)
    for k, s0, L, m0, n, nv in tab:
        st = list(range(0, L - CK + 1, a.stride))
        if st[-1] != L - CK: st.append(L - CK)
        cen = np.array(st) + CK / 2; base = len(chunks); chunks += [s0 + s for s in st]
        for c in range(L // 16):
            lo_, hi_ = 16 * c, 16 * c + 16
            if min(hi_, m0 + n) - max(lo_, m0) < 4: continue  # cell must overlap the true rib by >= 4 rows
            cand = [q for q in range(len(st)) if st[q] <= lo_ and hi_ <= st[q] + CK]
            j = min(cand, key=lambda q: abs(cen[q] - (lo_ + 8)))
            pb = min(int(4 * ((lo_ + 8 - m0) / max(n, 1))), 3); pb = max(pb, 0)
            cells.append((base + j, (lo_ - st[j]) // 16, k - 1, pb))
    cells = np.array(cells, np.int64).reshape(-1, 4); ch = np.arange(len(chunks))
    if train and len(chunks) > a.maxch:
        ch = np.sort(rng.choice(len(chunks), a.maxch, replace=False)); cells = cells[np.isin(cells[:, 0], ch)]
    remap = -np.ones(len(chunks), np.int64); remap[ch] = np.arange(len(ch))
    if len(cells): cells[:, 0] = remap[cells[:, 0]]
    r0 = [chunks[j] for j in ch]
    if a.padc > 1 and len(r0) % a.padc: r0 += [r0[0]] * (a.padc - len(r0) % a.padc)  # pad chunks never referenced by a cell
    rows = np.array(r0, np.int64)[:, None] + np.arange(CK)[None]
    h = hu[rows]; m = np.unpackbits(mk[rows], -1)[..., :CS]
    if train and a.shift:
        s1, s2 = rng.integers(-a.shift, a.shift + 1, 2); h = np.roll(h, (s1, s2), (2, 3)); m = np.roll(m, (s1, s2), (2, 3))
    return dict(h=torch.from_numpy(np.ascontiguousarray(h)), m=torch.from_numpy(np.ascontiguousarray(m)),
                cell=torch.from_numpy(cells), idx=int(i))


class DS(torch.utils.data.Dataset):
    def __init__(s, idx, train): s.idx, s.train = idx, train

    def __len__(s): return len(s.idx)

    def __getitem__(s, k):
        rng = np.random.default_rng((int(s.idx[k]) * 7919 + int(time.time() * 1e6)) % 2 ** 32) if s.train else np.random.default_rng(0)
        return load_case(s.idx[k], s.train, rng)


def loader(idx, train):
    return torch.utils.data.DataLoader(DS(idx, train), batch_size=None, shuffle=False, num_workers=a.workers, pin_memory=a.pin,
                                       prefetch_factor=4 if a.workers else None)


enc0 = NnUNetEncoder(a.ts, device="cuda"); NORM = enc0.norm
BASE_SD = {k: v.detach().clone() for k, v in enc0.encoder.state_dict().items()}
WD = {"base": {"type": "baseline"}, "bone": {"level": 400, "width": 1800}, "soft": {"level": 40, "width": 400}}
WINS = [WD[w] for w in a.wins.split(",")]; NW = len(WINS)
CH = enc0.stage_channels([3, 4]); D = NW * (3 * CH[0] + 2 * CH[1])


class FT(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc0.encoder; s.enc.load_state_dict(BASE_SD)
        for i, st in enumerate(s.enc.stages):
            for p in st.parameters(): p.requires_grad_(i >= a.train_from)
        s.drop = nn.Dropout(0.3); s.head = nn.Linear(D, 1); s.rib = nn.Parameter(torch.zeros(24)); s.pos = nn.Parameter(torch.zeros(4))

    def trainable(s):
        sd = s.state_dict()
        keep = [k for k in sd if not k.startswith("enc.") or any(k.startswith(f"enc.stages.{i}.") for i in range(a.train_from, 9))]
        return {k: sd[k].detach().cpu() for k in keep}

    def feats(s, h, m, train):  # h [C,CK,CS,CS] int16, m uint8 -> per-cell features [C*NCELL, D]
        C = h.shape[0]; h = h.to(dev, non_blocking=True).float()
        if train: h = h * (1 + 0.04 * (torch.rand(1, device=dev) - .5)) + 20 * (torch.rand(1, device=dev) - .5)
        x = torch.cat([apply_window(h, w, NORM) for w in WINS], 0)[:, None]  # [NW*C,1,CK,CS,CS] window-major
        out = []
        if getattr(s.enc, "stem", None) is not None:
            with torch.no_grad(): x = s.enc.stem(x)
        for i, st in enumerate(s.enc.stages):
            if i < a.train_from:
                with torch.no_grad(): x = st(x)
            elif a.ckpt and train: x = checkpoint(st, x, use_reentrant=False)
            else: x = st(x)
            if i >= 3: out.append(x)
            if i >= 4: break
        f3, f4 = out  # [NW*C, C3, CK/8, CS/8, CS/8], [NW*C, C4, CK/16, CS/16, CS/16]
        g = F.adaptive_max_pool3d(m.to(dev).float()[:, None], f3.shape[2:]).repeat(NW, 1, 1, 1, 1).to(f3.dtype)
        B3, C3, T3 = f3.shape[0], f3.shape[1], f3.shape[2]
        f3c = f3.reshape(B3, C3, NCELL, T3 // NCELL, *f3.shape[3:]); g3c = g.reshape(B3, 1, NCELL, T3 // NCELL, *g.shape[3:])
        mean3 = f3c.mean((3, 4, 5)); max3 = f3c.amax((3, 4, 5))
        mm3 = (f3c * g3c).sum((3, 4, 5)) / g3c.sum((3, 4, 5)).clamp(min=1)
        f4c = f4.reshape(B3, f4.shape[1], NCELL, -1); mean4 = f4c.mean(3); max4 = f4c.amax(3)
        v = torch.cat([mean3, max3, mm3, mean4, max4], 1).float()  # [NW*C, D/NW, NCELL]
        v = v.reshape(NW, C, -1, NCELL).permute(1, 3, 0, 2).reshape(C * NCELL, -1)
        return F.layer_norm(v, v.shape[1:])

    def forward(s, c, train=False):
        cell = c["cell"].to(dev)
        if len(cell) == 0:
            z = torch.full((), -8.0, device=dev); return z, z, z, None, None
        if train:
            with torch.autocast(**AMP): V = s.feats(c["h"], c["m"], True)
        else:  # no grad: chunk the encoder pass
            Vs = []
            for j in range(0, c["h"].shape[0], 48):
                with torch.autocast(**AMP): Vs.append(s.feats(c["h"][j:j + 48], c["m"][j:j + 48], False))  # 48 = multiple of padc
            V = torch.cat(Vs)
        v = V[cell[:, 0] * NCELL + cell[:, 1]]
        L = s.head(s.drop(v))[:, 0].float() + s.rib[cell[:, 2]] + s.pos[cell[:, 3]]
        rt = cell[:, 2] >= 12; neg = torch.full_like(L, -1e4)
        sr = torch.logsumexp(torch.where(rt, L, neg), 0) if bool(rt.any()) else torch.full((), -8.0, device=dev)
        sl_ = torch.logsumexp(torch.where(~rt, L, neg), 0) if bool((~rt).any()) else torch.full((), -8.0, device=dev)
        return torch.logsumexp(L, 0), sr, sl_, L, cell[:, 2]


def predict(m, idx):
    m.eval(); out = np.zeros(len(idx), np.float32); pos = {int(j): k for k, j in enumerate(idx)}
    with torch.no_grad():
        for c in loader(idx, False):
            out[pos[c["idx"]]] = float(m(c)[0])
    return out


def epoch_rows(idx_tr, rng):
    if not a.neg_epoch: return rng.permutation(idx_tr)
    p_ = idx_tr[y[idx_tr] == 1]; n_ = idx_tr[y[idx_tr] == 0]
    return rng.permutation(np.r_[p_, rng.choice(n_, min(a.neg_epoch, len(n_)), replace=False)])


def fit(idx_tr, idx_es, seed, ck=None):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    m = FT().to(dev)
    ep_ = [p for i, st in enumerate(m.enc.stages) if i >= a.train_from for p in st.parameters()]
    hp = list(m.head.parameters()) + [m.rib, m.pos]
    groups = [{"params": hp, "lr": a.lr * 10}] + ([{"params": ep_, "lr": a.lr}] if ep_ else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    npe = len(epoch_rows(idx_tr, np.random.default_rng(0)))
    steps = a.epochs * int(np.ceil(npe / a.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    npos = y[idx_tr].sum(); pw = torch.tensor((npe - npos) / max(npos, 1), device=dev)
    best, bst, ep0 = -1, None, 0
    if ck and os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["m"], strict=False)
        opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"]); best, bst, ep0 = c["best"], c["bst"], c["ep"] + 1
        np.random.set_state(c["np"]); torch.set_rng_state(c["trng"].cpu()); torch.cuda.set_rng_state(c["crng"].cpu())
        rng = np.random.default_rng(seed + 1000 * ep0)
        print(f"  resumed {ck} at epoch {ep0} (best inner {best:.4f})", flush=True)
    nb = 0; t0 = time.time()
    for ep in range(ep0, a.epochs):
        if os.path.exists(STOP): print("STOP seen", flush=True); sys.exit(3)
        m.train(); perm = epoch_rows(idx_tr, rng); tl = 0; t1 = time.time(); k = 0; opt.zero_grad()
        for c in loader(perm, True):
            i = c["idx"]; lg, sr, sl_, L, rib = m(c, True)
            loss = F.binary_cross_entropy_with_logits(lg, torch.tensor(y[i], device=dev), pos_weight=pw)
            if a.side:
                for s_, t_ in ((sr, sR[i]), (sl_, sL[i])):
                    if np.isfinite(t_): loss = loss + a.side * F.binary_cross_entropy_with_logits(s_, torch.tensor(t_, device=dev), pos_weight=pw)
            if a.lev and L is not None:
                if LEV[i].any():
                    inm = torch.from_numpy(LEV[i]).to(dev)[rib]
                    if bool(inm.any()):
                        loss = loss + a.lev * F.binary_cross_entropy_with_logits(L[inm].max(), torch.ones((), device=dev))
                elif y[i] == 0:
                    loss = loss + a.lev * F.binary_cross_entropy_with_logits(L.max(), torch.zeros((), device=dev))
            (loss / a.bs).backward(); tl += float(loss); k += 1
            if k % a.bs == 0 or k == len(perm):
                opt.step(); sch.step(); opt.zero_grad()
            if a.bench:
                nb += 1
                if nb == 5: torch.cuda.synchronize(); t0 = time.time()
                if nb == a.bench + 5:
                    torch.cuda.synchronize(); dt = (time.time() - t0) / a.bench
                    print(f"BENCH train {dt:.3f} s/case peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
                    t2 = time.time(); predict(m, idx_es[:40]); print(f"BENCH pred {(time.time() - t2) / 40:.3f} s/case", flush=True); sys.exit(0)
        sc = auc(y[idx_es], predict(m, idx_es))
        print(f"  seed {seed} ep {ep} loss {tl / len(perm):.4f} inner AUROC {sc:.4f} {time.time() - t1:.0f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if sc > best: best, bst = sc, m.trainable()
        if ck:
            torch.save(dict(m=m.trainable(), opt=opt.state_dict(), sch=sch.state_dict(), best=best, bst=bst, ep=ep,
                            np=np.random.get_state(), trng=torch.get_rng_state(), crng=torch.cuda.get_rng_state()), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    m.load_state_dict(bst, strict=False); return m, best


def load_model(path):
    m = FT().to(dev); sd = torch.load(path, map_location=dev); miss = m.load_state_dict(sd, strict=False)
    assert not miss.unexpected_keys and all(k.startswith("enc.") for k in miss.missing_keys); return m


tri = np.where(pool_rows & ok)[0]; vai = np.where(~tr & ok)[0]
folds = [int(k) for k in a.folds.split(",")] if a.folds else sorted(np.unique(fold[tr]))
log = {}
for k in folds:
    fz = f"{a.out}/fold{k}.npz"
    if a.score:  # fill rows that were not scored on the training host
        if not os.path.exists(fz): print(f"fold {k}: no fold file, skip", flush=True); continue
        z = dict(np.load(fz)); done_b = set(z["b_ids"].astype(str)); done_v = set(z["v_ids"].astype(str)) if "v_ids" in z else set()
        nb_ = np.array([i for i in np.where(tr & ok & (fold == k))[0] if SID[i] not in done_b], int)
        nv = np.array([i for i in vai if SID[i] not in done_v], int) if not a.no_val else np.zeros(0, int)
        if len(nb_) + len(nv) == 0: continue
        mp = sorted(f for f in os.listdir(a.out) if f.startswith(f"model_f{k}_s"))
        pb = np.zeros(len(nb_)); pv = np.zeros(len(nv))
        for f in mp:
            m = load_model(f"{a.out}/{f}"); pb += predict(m, nb_) / len(mp); pv += predict(m, nv) / len(mp); del m; torch.cuda.empty_cache()
        z["b_ids"] = np.r_[z["b_ids"].astype(str), SID[nb_]]; z["pb"] = np.r_[z["pb"], pb]
        if len(nv) or "v_ids" in z:
            z["v_ids"] = np.r_[z.get("v_ids", np.zeros(0, "U16")).astype(str), SID[nv]]; z["pv"] = np.r_[z.get("pv", np.zeros(0)), pv]
        np.savez(fz + ".tmp.npz", **z); os.replace(fz + ".tmp.npz", fz)
        print(f"fold {k}: scored {len(nb_)} train + {len(nv)} val with {len(mp)} models", flush=True); continue
    if os.path.exists(fz):
        log.update(json.loads(str(np.load(fz)["log"]))); continue
    a_ = tri[fold[tri] != k]
    b_ = np.where(tr & ok & (fold == k) & (pool_rows if a.tune else True))[0]
    a_ = np.random.default_rng(k).permutation(a_); nes = int(0.12 * len(a_)); pb, pv, lg = [], [], {}
    es_ = a_[:nes]
    if a.es_max_neg:
        es_ = np.r_[es_[y[es_] == 1], es_[y[es_] == 0][:a.es_max_neg]]
    for sd in range(a.seed0, a.seed0 + a.seeds):
        t1 = time.time(); ck = f"{a.out}/ckpt_f{k}_s{sd}.pt"
        m, b = fit(a_[nes:], es_, 100 + sd, ck)
        pb.append(predict(m, b_)); lg[f"f{k}_s{sd}"] = b
        if not a.tune:
            if not a.no_val: pv.append(predict(m, vai))
            torch.save(m.trainable(), f"{a.out}/model_f{k}_s{sd}.pt")
        del m; torch.cuda.empty_cache()
        print(f"fold {k} seed {sd} inner {b:.4f} {time.time() - t1:.0f}s", flush=True)
    log.update(lg); ob = np.mean(pb, 0); hb = has[b_] & pool_rows[b_]
    print(f"fold {k} OOF AUROC (pool, {a.target}) {auc(y[b_][hb], ob[hb]):.4f}", flush=True)
    sv = dict(b_ids=SID[b_], pb=ob, log=json.dumps(lg))
    if pv: sv.update(v_ids=SID[vai], pv=np.mean(pv, 0))
    np.savez(fz + ".tmp.npz", **sv); os.replace(fz + ".tmp.npz", fz)
    for sd in range(a.seed0, a.seed0 + a.seeds):
        if os.path.exists(f"{a.out}/ckpt_f{k}_s{sd}.pt"): os.remove(f"{a.out}/ckpt_f{k}_s{sd}.pt")

# assemble (train OOF only; val written to file, never scored here)
oof = np.full(N, np.nan); vs = {}; nf = 0; pos = {s: i for i, s in enumerate(SID)}
for k in folds:
    fz = f"{a.out}/fold{k}.npz"
    if not os.path.exists(fz): continue
    z = np.load(fz); nf += 1
    for s, p in zip(z["b_ids"].astype(str), z["pb"]): oof[pos[s]] = p
    if "v_ids" in z.files:
        for s, p in zip(z["v_ids"].astype(str), z["pv"]): vs.setdefault(s, []).append(p)
    if "log" in z.files: log.update(json.loads(str(z["log"])))
ti = np.where(tr)[0]; k_ = pool_rows & ~np.isnan(oof); kall = tr & has & ~np.isnan(oof)
res = dict(oof=float(auc(y[k_], oof[k_])) if k_.sum() else None, n_oof_pool=int(k_.sum()),
           oof_all_train=float(auc(y[kall], oof[kall])) if kall.sum() > k_.sum() else None,
           n_oof_all=int((tr & ~np.isnan(oof)).sum()), n_train=int(tr.sum()), inner=log, args=vars(a), target=a.target,
           n_train_rows=int(len(tri)), n_pos=int(y[tri].sum()), folds_done=nf)
ka = tr & ~np.isnan(oof) & ~np.isnan(lab.y.values)
res["oof_overall_label_all_train"] = float(auc(lab.y.values[ka], oof[ka])) if ka.sum() else None
print(f"RSFT train OOF {res['oof']} (pool {a.target}, n={res['n_oof_pool']}); all-train {a.target} {res['oof_all_train']}; "
      f"overall label {res['oof_overall_label_all_train']}; train rows scored {res['n_oof_all']}/{res['n_train']}", flush=True)
pd.DataFrame({"study_id": SID[ti], "fold": fold[ti], "in_pool": pool_rows[ti], "ft": oof[ti]}).to_csv(f"{a.out}/oof_train.csv", index=False)
if not a.tune and vs:
    full = [s for s, v in vs.items() if len(v) == nf]
    vi = np.array([pos[s] for s in full]); vv = np.array([np.mean(vs[s]) for s in full])
    pd.DataFrame({"study_id": SID[vi], "ft": vv}).to_csv(f"{a.out}/val_scores.csv", index=False)
    res["n_val"] = int(len(vi)); res["n_val_total"] = int((~tr).sum())
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("RSFT_DONE", flush=True)
