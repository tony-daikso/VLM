"""pelvis15_ft: fine-tune the late stages (>= --train-from, default 2) of a TS nnU-Net encoder on the TRUE 1.5 mm pelvis ROIs
(pelvis15/roi/{id}.npz from pv15_crop.py: hu, lab 1 hip_R 2 hip_L 3 sacrum 4 femur_R 5 femur_L, org (x,y,z), xmid, sp) with MIL.
spine15_ft recipe (frac15_ft.py tf2lse): windows -> encoder stages 0..4 (stages < train_from frozen, no grad) -> per instance
[mean|max of stage-3] + [mean|max of stage-4 (nearest-upsampled to the stage-3 grid)] per window -> layer-norm -> dropout ->
linear instance logit + per-bone bias -> logsumexp MIL over instances -> case logit (BCE, pos_weight). 8 epochs OneCycle,
encoder lr 1e-4 (head 10x), best epoch by AUROC on an inner 12 % holdout of the training rows (train only).
Instances = bone x cell (cell = --cell stage-3 voxels = 12*cell mm; x measured from the body midline xmid so left/right cells
mirror) built on the stage-3 grid from per-block bone voxel counts.
Side weak supervision (L/R mirror MIL, as pelvis15 PV15): right bag = hip_R, femur_R, sacrum cells with x >= midline; left bag =
the rest; lse over each bag trained against train-report side labels (pel_side.csv; t_pel==0 rows: both sides 0) with weight --side.
--mir adds a mirror-contrast term (instance features minus its contralateral mirror cell's features).
Variable ROI size (padded to a multiple of 16, capped at --maxshape, centred on the bone bbox); one case per micro-batch,
--bs cases per optimiser step. Train/val only (asserted); test never read. Target default t_pel (NaN rows not trained).
Folds = labels.fold (sitewise outer folds). --pool-ids: training pool (default: tierA & target known).
--tune: fold models score only held-out POOL rows (selection metric = train OOF on pool rows); no val; no models saved.
final: score held-out fold rows (train, with ROI) + val (unless --no-val); fold models saved; --score fills missing rows later.
usage: pv15_ft.py --labels L --roi DIR --out DIR --ts MODEL_DIR [--side 0.5] [--mir] [--tune] [--pool-ids F] [--score]"""
import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--roi", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ts", required=True); ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src"))
ap.add_argument("--train-from", type=int, default=2); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--vram-gb", type=float, default=20)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--tune", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--folds", default="")
ap.add_argument("--pool-ids", default=""); ap.add_argument("--side-csv", default=""); ap.add_argument("--side", type=float, default=0.0)
ap.add_argument("--mir", action="store_true"); ap.add_argument("--cell", type=int, default=2); ap.add_argument("--minvox", type=int, default=64)
ap.add_argument("--maxshape", default="192,160,256"); ap.add_argument("--fixed", action="store_true", help="every case cropped/padded to --maxshape (one cudnn shape)"); ap.add_argument("--shift", type=int, default=6)
ap.add_argument("--amp", default="none", choices=["none", "bf16"]); ap.add_argument("--no-val", action="store_true")
ap.add_argument("--target", default="t_pel"); ap.add_argument("--wins", default="base,bone"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--pin", action="store_true", help="pin_memory (never on the A6000: corrupts batches there)")
ap.add_argument("--bench", type=int, default=0, help="time N training micro-steps and exit")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
torch.set_num_threads(a.threads); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")
STOP = os.path.join(os.path.dirname(os.path.abspath(a.out)), "STOP")
MAXS = np.array([int(v) for v in a.maxshape.split(",")]); NB = 5; MIRB = np.array([0, 2, 1, 3, 5, 4])

lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
if a.limit:
    lab = pd.concat([lab[lab.split == "train"].head(a.limit), lab[lab.split == "val"].head(a.limit // 4)]).reset_index(drop=True)
N = len(lab); SID = np.array(lab.study_id.astype(str).tolist(), dtype=str)
ok = np.array([os.path.exists(f"{a.roi}/{s}.npz") for s in SID])
print(f"ROIs present {ok.sum()}/{N} (train {ok[lab.split.values == 'train'].sum()}, val {ok[lab.split.values == 'val'].sum()})", flush=True)
yt = lab[a.target].values.astype(float); has = ~np.isnan(yt); y = np.nan_to_num(yt).astype(np.float32)
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int)
if a.pool_ids:
    pid = {l.strip() for l in open(a.pool_ids) if l.strip()}; pool_rows = tr & has & np.isin(SID, list(pid))
else:
    pool_rows = tr & has & lab.tierA.values.astype(bool)
sR = np.full(N, np.nan, np.float32); sL = sR.copy()
if a.side_csv:
    sc = pd.read_csv(a.side_csv); sc = sc[sc.study_id.isin(set(SID[tr]))].set_index("study_id")  # train rows only
    ix = {s: i for i, s in enumerate(SID)}
    for s_, r_ in sc.iterrows():
        if s_ in ix and tr[ix[s_]]: sR[ix[s_]], sL[ix[s_]] = r_.sideR, r_.sideL
print(f"pool rows {pool_rows.sum()} (pos {int(y[pool_rows].sum())}); side known in pool {np.isfinite(sR[pool_rows]).sum()} "
      f"(pos with side {int((np.isfinite(sR) & (y == 1) & pool_rows).sum())})", flush=True)


def load_case(i, train, rng):
    z = np.load(f"{a.roi}/{SID[i]}.npz"); hu, lb = z["hu"], z["lab"]; org, xmid, sp = z["org"], float(z["xmid"]), float(z["sp"])
    sh = np.array(hu.shape); T = MAXS.copy() if a.fixed else np.minimum(-(-sh // 16) * 16, MAXS)
    st = np.zeros(3, int); p0 = np.zeros(3, int)
    for d in range(3):
        if sh[d] > T[d]:  # crop around the bone bbox centre
            nzd = np.nonzero(lb.any(axis=tuple(e for e in range(3) if e != d)))[0]
            c = (nzd[0] + nzd[-1]) // 2 if len(nzd) else sh[d] // 2
            c += rng.integers(-a.shift, a.shift + 1) if train else 0
            st[d] = int(np.clip(c - T[d] // 2, 0, sh[d] - T[d]))
        else:
            room = T[d] - sh[d]; p0[d] = rng.integers(0, room + 1) if train else room // 2
    sl = tuple(slice(st[d], st[d] + min(sh[d], T[d])) for d in range(3))
    h = np.full(T, -1024, np.int16); g = np.zeros(T, np.uint8)
    dst = tuple(slice(p0[d], p0[d] + min(sh[d], T[d])) for d in range(3))
    h[dst] = hu[sl]; g[dst] = lb[sl]
    T8 = T // 8; V = int(np.prod(T8))
    blk = g.reshape(T8[0], 8, T8[1], 8, T8[2], 8).transpose(0, 2, 4, 1, 3, 5).reshape(V, 512)
    cnt = np.stack([(blk == b).sum(1) for b in range(1, NB + 1)], 1)  # [V, 5]
    v, b = np.nonzero(cnt >= 8); w = cnt[v, b].astype(np.float32)
    z3, y3, x3 = np.unravel_index(v, T8)
    xw = org[0] + ((x3 * 8 + 3.5) - p0[2] + st[2]) * sp
    cx = np.floor((xw - xmid) / (a.cell * 8 * sp)).astype(np.int64)
    key = (((b.astype(np.int64) * 64 + z3 // a.cell) * 64 + y3 // a.cell) * 128 + (cx + 64))
    uk, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=w, minlength=len(uk)); keep = tot >= a.minvox
    remap = np.cumsum(keep) - 1; sel = keep[inv]
    v, inv, w = v[sel], remap[inv[sel]], w[sel]; uk = uk[keep]
    mb = uk // (64 * 64 * 128) + 1; mz = (uk // (64 * 128)) % 64; my = (uk // 128) % 64; mx = uk % 128 - 64  # bone 1..5
    rt = (mb == 1) | (mb == 4) | ((mb == 3) & (mx >= 0))
    mk = ((((MIRB[mb] - 1).astype(np.int64) * 64 + mz) * 64 + my) * 128 + (-mx - 1 + 64))
    pos = np.clip(np.searchsorted(uk, mk), 0, max(len(uk) - 1, 0)); mir = np.where((len(uk) > 0) & (uk[pos] == mk), pos, -1) if len(uk) else np.zeros(0, int)
    return dict(h=torch.from_numpy(h), v=torch.from_numpy(v.astype(np.int64)), i=torch.from_numpy(inv.astype(np.int64)),
                w=torch.from_numpy(w), bone=torch.from_numpy((mb - 1).astype(np.int64)), rt=torch.from_numpy(rt),
                mir=torch.from_numpy(np.asarray(mir, np.int64)), n=len(uk), T8=tuple(int(t) for t in T8), idx=int(i))


class DS(torch.utils.data.Dataset):
    def __init__(s, idx, train): s.idx, s.train = idx, train

    def __len__(s): return len(s.idx)

    def __getitem__(s, k):
        rng = np.random.default_rng((int(s.idx[k]) * 7919 + int(time.time() * 1e6)) % 2 ** 32) if s.train else None
        return load_case(s.idx[k], s.train, rng)


def loader(idx, train):
    return torch.utils.data.DataLoader(DS(idx, train), batch_size=None, shuffle=False, num_workers=a.workers, pin_memory=a.pin,
                                       prefetch_factor=4 if a.workers else None)


enc0 = NnUNetEncoder(a.ts, device="cuda"); NORM = enc0.norm
BASE_SD = {k: v.detach().clone() for k, v in enc0.encoder.state_dict().items()}
WD = {"base": {"type": "baseline"}, "bone": {"level": 400, "width": 1800}, "soft": {"level": 40, "width": 400}}
WINS = [WD[w] for w in a.wins.split(",")]; NW = len(WINS)
CH = enc0.stage_channels([3, 4]); D = NW * (2 * CH[0] + 2 * CH[1])


class FT(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc0.encoder; s.enc.load_state_dict(BASE_SD)
        for i, st in enumerate(s.enc.stages):
            for p in st.parameters(): p.requires_grad_(i >= a.train_from)
        s.drop = nn.Dropout(0.3); s.head = nn.Linear(D * (2 if a.mir else 1), 1); s.lvl = nn.Parameter(torch.zeros(NB))

    def trainable(s):
        sd = s.state_dict()
        keep = [k for k in sd if not k.startswith("enc.") or any(k.startswith(f"enc.stages.{i}.") for i in range(a.train_from, 9))]
        return {k: sd[k].detach().cpu() for k in keep}

    def inst_logits(s, c, train):
        h = c["h"].to(dev, non_blocking=True).float()[None, None]
        if train:
            h = h * (1 + 0.04 * (torch.rand(1, device=dev) - .5)) + 20 * (torch.rand(1, device=dev) - .5)
        x = torch.cat([apply_window(h, w, NORM) for w in WINS], 0)  # [NW,1,Z,Y,X]
        out = []
        if getattr(s.enc, "stem", None) is not None:
            with torch.no_grad(): x = s.enc.stem(x)
        for i, st in enumerate(s.enc.stages):
            if i < a.train_from:
                with torch.no_grad(): x = st(x)
            else: x = st(x)
            if i >= 3: out.append(x)
            if i >= 4: break
        f3, f4 = out
        T8 = tuple(int(t) for t in c["T8"]); assert tuple(f3.shape[2:]) == T8, (f3.shape, T8)
        f4u = f4.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)[:, :, :T8[0], :T8[1], :T8[2]]
        Fm = torch.cat([f3, f4u], 1).float().flatten(2)  # [NW, C3+C4, V]
        Fm = Fm.permute(2, 0, 1).reshape(Fm.shape[2], -1)  # [V, NW*(C3+C4)]
        v, ii, w = c["v"].to(dev), c["i"].to(dev), c["w"].to(dev); n = c["n"]
        fv = Fm[v]; Cd = fv.shape[1]
        sm = torch.zeros(n, Cd, device=dev).index_add(0, ii, fv * w[:, None]); ws = torch.zeros(n, device=dev).index_add(0, ii, w)
        mean = sm / ws.clamp(min=1e-6)[:, None]
        mx = torch.full((n, Cd), -1e4, device=dev).scatter_reduce(0, ii[:, None].expand(-1, Cd), fv, "amax", include_self=True)
        vv = torch.cat([mean, mx], 1)
        vv = F.layer_norm(vv, vv.shape[1:])
        if a.mir:
            mi = c["mir"].to(dev); pm = torch.cat([vv, torch.zeros(1, vv.shape[1], device=dev)], 0)[torch.where(mi >= 0, mi, n)]
            vv = torch.cat([vv, torch.where((mi >= 0)[:, None], vv - pm, torch.zeros_like(vv))], 1)
        return s.head(s.drop(vv))[:, 0] + s.lvl[c["bone"].to(dev)]

    def forward(s, c, train=False):
        with torch.autocast(**AMP):
            L = s.inst_logits(c, train).float()
        if c["n"] == 0:
            z = torch.full((), -8.0, device=dev); return z, z, z
        rt = c["rt"].to(dev); neg = torch.full_like(L, -1e4)
        sr = torch.logsumexp(torch.where(rt, L, neg), 0) if bool(rt.any()) else torch.full((), -8.0, device=dev)
        sl_ = torch.logsumexp(torch.where(~rt, L, neg), 0) if bool((~rt).any()) else torch.full((), -8.0, device=dev)
        return torch.logsumexp(L, 0), sr, sl_


def predict(m, idx):
    m.eval(); out = np.zeros(len(idx), np.float32); pos = {int(j): k for k, j in enumerate(idx)}
    with torch.no_grad():
        for c in loader(idx, False):
            out[pos[c["idx"]]] = float(m(c)[0])
    return out


def fit(idx_tr, idx_es, seed, ck=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = FT().to(dev)
    ep_ = [p for i, st in enumerate(m.enc.stages) if i >= a.train_from for p in st.parameters()]
    hp = list(m.head.parameters()) + [m.lvl]
    groups = [{"params": hp, "lr": a.lr * 10}] + ([{"params": ep_, "lr": a.lr}] if ep_ else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(idx_tr) / a.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    npos = y[idx_tr].sum(); pw = torch.tensor((len(idx_tr) - npos) / max(npos, 1), device=dev)
    best, bst, ep0 = -1, None, 0
    if ck and os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False); m.load_state_dict(c["m"], strict=False)
        opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"]); best, bst, ep0 = c["best"], c["bst"], c["ep"] + 1
        np.random.set_state(c["np"]); torch.set_rng_state(c["trng"].cpu()); torch.cuda.set_rng_state(c["crng"].cpu())
        print(f"  resumed {ck} at epoch {ep0} (best inner {best:.4f})", flush=True)
    nb = 0; t0 = time.time()
    for ep in range(ep0, a.epochs):
        if os.path.exists(STOP): print("STOP seen", flush=True); sys.exit(3)
        m.train(); perm = np.random.permutation(idx_tr); tl = 0; t1 = time.time(); k = 0; opt.zero_grad()
        for c in loader(perm, True):
            i = c["idx"]; lg, sr, sl_ = m(c, True)
            loss = F.binary_cross_entropy_with_logits(lg, torch.tensor(y[i], device=dev), pos_weight=pw)
            if a.side:
                for s_, t_ in ((sr, sR[i]), (sl_, sL[i])):
                    if np.isfinite(t_): loss = loss + a.side * F.binary_cross_entropy_with_logits(s_, torch.tensor(t_, device=dev), pos_weight=pw)
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
        np.savez(fz, **z); print(f"fold {k}: scored {len(nb_)} train + {len(nv)} val with {len(mp)} models", flush=True); continue
    if os.path.exists(fz):
        log.update(json.loads(str(np.load(fz)["log"]))); continue
    a_ = tri[fold[tri] != k]
    b_ = np.where(tr & ok & (fold == k) & (pool_rows if a.tune else True))[0]
    a_ = np.random.default_rng(k).permutation(a_); nes = int(0.12 * len(a_)); pb, pv, lg = [], [], {}
    for sd in range(a.seed0, a.seed0 + a.seeds):
        t1 = time.time(); ck = f"{a.out}/ckpt_f{k}_s{sd}.pt"
        m, b = fit(a_[nes:], a_[:nes], 100 + sd, ck)
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
    np.savez(fz, **sv)
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
res = dict(oof=float(auc(y[k_], oof[k_])), n_oof_pool=int(k_.sum()), oof_all_train=float(auc(y[kall], oof[kall])) if kall.sum() > k_.sum() else None,
           n_oof_all=int((tr & ~np.isnan(oof)).sum()), n_train=int(tr.sum()), inner=log, args=vars(a), target=a.target,
           n_train_rows=int(len(tri)), n_pos=int(y[tri].sum()), folds_done=nf)
ka = tr & ~np.isnan(oof) & ~np.isnan(lab.y.values)
res["oof_overall_label_all_train"] = float(auc(lab.y.values[ka], oof[ka]))
print(f"PV15FT train OOF {res['oof']:.4f} (pool {a.target}, n={res['n_oof_pool']}); all-train {a.target} {res['oof_all_train']}; "
      f"overall label {res['oof_overall_label_all_train']:.4f}; train rows scored {res['n_oof_all']}/{res['n_train']}", flush=True)
pd.DataFrame({"study_id": SID[ti], "fold": fold[ti], "in_tier": pool_rows[ti], "ft": oof[ti]}).to_csv(f"{a.out}/oof_train.csv", index=False)
if not a.tune and vs:
    full = [s for s, v in vs.items() if len(v) == nf]
    vi = np.array([pos[s] for s in full]); vv = np.array([np.mean(vs[s]) for s in full])
    pd.DataFrame({"study_id": SID[vi], "ft": vv}).to_csv(f"{a.out}/val_scores.csv", index=False)
    res["n_val"] = int(len(vi)); res["n_val_total"] = int((~tr).sum())
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("PV15FT_DONE", flush=True)
