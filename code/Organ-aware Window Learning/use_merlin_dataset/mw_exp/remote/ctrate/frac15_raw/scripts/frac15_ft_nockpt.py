"""spine15_ft: fine-tune the late stages of the TS Dataset297 encoder on TRUE 1.5 mm vertebra crops (spine15_crop.py:
raw CT -> 10 units T8..L5, 40x48x40 at 1.5 mm) with MIL over vertebrae. Same recipe as spine_ft/frac_ft.py (3 windows,
stage-3 [mean|max|mask-mean] + stage-4 [mean|max] per window -> layer-norm -> dropout -> linear unit logit + per-level
bias -> lse/attn/max pool), but no x2 upsampling (crops are already 1.5 mm) and crops are read lazily per case
(DataLoader) so only the rows actually used need to be on the host.
Options: --ctx adds a neighbour-context term per unit (features minus mean of present adjacent vertebrae);
--mask m|m3 picks the seg15 (1.5 mm Dataset292) or the 3 mm Dataset297 vertebra mask; --amp bf16 (3080; NOT on the A6000).
Labels = spine_ft/in/labels.csv (train/val only, asserted; test never read). Target t_vert (1 vertebral fracture,
0 fracture-free, NaN other-site fracture -> not trained). Folds = sitewise outer folds. Training rows = train & tierA &
target known & crop present. Best epoch per fit = AUROC on an inner 12 % holdout of the training rows (train only).
--tune: each fold model scores only the held-out TIER rows (selection metric = train OOF on tier rows); no val.
final: scores held-out fold of ALL train rows with crops + all val rows with crops (val = mean over fold models);
fold models are saved so rows whose crops were not on this host can be scored later with --score (any host).
usage: frac15_ft.py --labels L --crops DIR --out DIR [--train-from 2] [--pool lse] [--ctx] [--tune] [--score]"""
import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--crops", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src"))
ap.add_argument("--train-from", type=int, default=2); ap.add_argument("--pool", default="lse", choices=["lse", "attn", "max"])
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (bs/accum cases each)")
ap.add_argument("--lr", type=float, default=1e-4, help="encoder lr; head lr = 10x"); ap.add_argument("--vram-gb", type=float, default=20)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--tune", action="store_true"); ap.add_argument("--score", action="store_true"); ap.add_argument("--folds", default="")
ap.add_argument("--ctx", action="store_true"); ap.add_argument("--mask", default="m", choices=["m", "m3"])
ap.add_argument("--amp", default="none", choices=["none", "bf16"]); ap.add_argument("--shift", type=int, default=2)
ap.add_argument("--no-val", action="store_true", help="final mode without val scoring (fill later with --score)")
ap.add_argument("--target", default="t_vert"); ap.add_argument("--wins", default="base,bone,soft"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); assert a.bs % a.accum == 0
sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
torch.set_num_threads(a.threads); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16")
STOP = os.path.join(os.path.dirname(os.path.abspath(a.out)), "STOP")
RES = os.path.expanduser("~/.totalsegmentator/nnunet/results")
ENC = "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
U, SH = 10, (40, 48, 40)

lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
if a.limit:
    lab = pd.concat([lab[lab.split == "train"].head(a.limit), lab[lab.split == "val"].head(a.limit // 4)]).reset_index(drop=True)
N = len(lab); SID = np.array(lab.study_id.astype(str).tolist(), dtype=str)
ok = np.array([os.path.exists(f"{a.crops}/{s}.npz") for s in SID])
for i in np.where(ok)[0]:  # a case with no vertebra unit at all is treated as missing
    ok[i] = bool(np.load(f"{a.crops}/{SID[i]}.npz")["present"].any())
print(f"crops present {ok.sum()}/{N} (train {ok[lab.split.values == 'train'].sum()}, val {ok[lab.split.values == 'val'].sum()})", flush=True)
yt = lab[a.target].values.astype(float); has = ~np.isnan(yt); y = np.nan_to_num(yt).astype(np.float32)
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int)
pool_rows = tr & has & lab.tierA.values.astype(bool)


class DS(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx

    def __len__(s): return len(s.idx)

    def __getitem__(s, i):
        z = np.load(f"{a.crops}/{SID[s.idx[i]]}.npz")
        m = np.unpackbits(z[a.mask], 1)[:, :int(np.prod(SH))].reshape((U,) + SH)
        return torch.from_numpy(z["hu"]), torch.from_numpy(m), torch.from_numpy(z["present"])


def loader(idx, bs, shuffle):
    return torch.utils.data.DataLoader(DS(idx), batch_size=bs, shuffle=False, num_workers=a.workers, pin_memory=True,
                                       persistent_workers=False, prefetch_factor=4 if a.workers else None, drop_last=False)


enc0 = NnUNetEncoder(f"{RES}/{ENC}", device="cuda"); NORM = enc0.norm
BASE_SD = {k: v.detach().clone() for k, v in enc0.encoder.state_dict().items()}
WD = {"base": {"type": "baseline"}, "bone": {"level": 400, "width": 1800}, "soft": {"level": 40, "width": 400}}
WINS = [WD[w] for w in a.wins.split(",")]; NW = len(WINS)
CH = enc0.stage_channels([3, 4]); D = NW * (3 * CH[0] + 2 * CH[1])


class FT(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc0.encoder; s.enc.load_state_dict(BASE_SD)
        for i, st in enumerate(s.enc.stages):
            for p in st.parameters(): p.requires_grad_(i >= a.train_from)
        s.drop = nn.Dropout(0.3); s.head = nn.Linear(D * (2 if a.ctx else 1), 1); s.lvl = nn.Parameter(torch.zeros(U))
        s.att = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, 1)) if a.pool == "attn" else None

    def trainable(s):
        sd = s.state_dict()
        keep = [k for k in sd if not k.startswith("enc.") or any(k.startswith(f"enc.stages.{i}.") for i in range(a.train_from, 9))]
        return {k: sd[k].detach().cpu() for k in keep}

    def forward(s, x, g, pr):  # x [B*U*NW,1,...] case-major/unit/window, g [B*U,1,...], pr [B,U] bool
        B = pr.shape[0]; out = []; h = x
        if getattr(s.enc, "stem", None) is not None:
            with torch.no_grad(): h = s.enc.stem(h)
        for i, st in enumerate(s.enc.stages):
            if i < a.train_from:
                with torch.no_grad(): h = st(h)
            else: h = st(h)
            if i >= 3: out.append(h)
            if i >= 4: break
        f3, f4 = out
        g3 = F.adaptive_max_pool3d(g, f3.shape[2:]).repeat_interleave(NW, 0).to(f3.dtype)
        gm = (f3 * g3).sum((2, 3, 4)) / g3.sum((2, 3, 4)).clamp(min=1)
        v = torch.cat([f3.mean((2, 3, 4)), f3.amax((2, 3, 4)), gm, f4.mean((2, 3, 4)), f4.amax((2, 3, 4))], 1)
        v = F.layer_norm(v.float(), v.shape[1:]).view(B, U, -1)
        if a.ctx:
            p = pr.float()[..., None]; z = torch.zeros_like(v[:, :1]); zp = torch.zeros_like(p[:, :1])
            vp = v * p; nb = torch.cat([z, vp[:, :-1]], 1) + torch.cat([vp[:, 1:], z], 1)
            nn_ = torch.cat([zp, p[:, :-1]], 1) + torch.cat([p[:, 1:], zp], 1)
            dif = torch.where(nn_ > 0, v - nb / nn_.clamp(min=1), torch.zeros_like(v))
            hv = torch.cat([v, dif], 2)
        else:
            hv = v
        L = s.head(s.drop(hv))[..., 0].float() + s.lvl[None]
        L = torch.where(pr, L, torch.full_like(L, -1e4))
        if a.pool == "lse":
            return torch.logsumexp(L, 1)
        if a.pool == "max":
            return L.max(1).values
        A = s.att(v)[..., 0].float(); A = torch.where(pr, A, torch.full_like(A, -1e4))
        return (torch.softmax(A, 1) * torch.where(pr, L, torch.zeros_like(L))).sum(1)


def prep(hu, m, pr, train):
    pr = pr.to(dev, non_blocking=True)
    h = hu.to(dev, non_blocking=True).reshape(-1, 1, *SH).float(); g = m.to(dev, non_blocking=True).reshape(-1, 1, *SH).float()
    if train:
        n = h.shape[0]
        h = h * (1 + 0.04 * (torch.rand(n, 1, 1, 1, 1, device=dev) - .5)) + 20 * (torch.rand(n, 1, 1, 1, 1, device=dev) - .5)
        sh = tuple(int(v) for v in np.random.randint(-a.shift, a.shift + 1, 3))
        h = torch.roll(h, sh, (2, 3, 4)); g = torch.roll(g, sh, (2, 3, 4))
    x = torch.stack([apply_window(h[:, 0], w, NORM) for w in WINS], 1).reshape(-1, 1, *SH)  # unit-major, windows inner
    return x, g, pr


def predict(m, idx):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast(**AMP):
        for hu, mm, pr in loader(idx, 8, False):
            out.append(m(*prep(hu, mm, pr, False)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(idx_tr, idx_es, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = FT().to(dev)
    ep_ = [p for i, st in enumerate(m.enc.stages) if i >= a.train_from for p in st.parameters()]
    hp = list(m.head.parameters()) + [m.lvl] + (list(m.att.parameters()) if m.att is not None else [])
    groups = [{"params": hp, "lr": a.lr * 10}] + ([{"params": ep_, "lr": a.lr}] if ep_ else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(idx_tr) / a.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    npos = y[idx_tr].sum(); pw = torch.tensor((len(idx_tr) - npos) / max(npos, 1), device=dev)
    best, bst = -1, None; mb = a.bs // a.accum
    for ep in range(a.epochs):
        if os.path.exists(STOP): print("STOP seen", flush=True); sys.exit(3)
        m.train(); perm = np.random.permutation(idx_tr); tl = 0; t1 = time.time()
        order = np.concatenate([np.sort(perm[i:i + a.bs]) for i in range(0, len(perm), a.bs)])
        it = iter(loader(order, mb, False)); pos_ = 0
        for i in range(0, len(perm), a.bs):
            j = order[i:i + a.bs]; opt.zero_grad()
            for q in range(0, len(j), mb):
                hu, mm, pr = next(it); jj = j[q:q + mb]
                with torch.autocast(**AMP):
                    lg = m(*prep(hu, mm, pr, True))
                loss = F.binary_cross_entropy_with_logits(lg.float(), torch.from_numpy(y[jj]).to(dev), pos_weight=pw) * len(jj) / len(j)
                loss.backward(); tl += loss.item() * len(j)
            opt.step(); sch.step()
        sc = auc(y[idx_es], predict(m, idx_es))
        print(f"  seed {seed} ep {ep} loss {tl / len(perm):.4f} inner AUROC {sc:.4f} {time.time() - t1:.0f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if sc > best: best, bst = sc, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bst); return m, best


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
        nb = np.array([i for i in np.where(tr & ok & (fold == k))[0] if SID[i] not in done_b], int)
        nv = np.array([i for i in vai if SID[i] not in done_v], int) if not a.tune else np.zeros(0, int)
        if len(nb) + len(nv) == 0: continue
        mp = sorted(f for f in os.listdir(a.out) if f.startswith(f"model_f{k}_s"))
        pb = np.zeros(len(nb)); pv = np.zeros(len(nv))
        for f in mp:
            m = load_model(f"{a.out}/{f}"); pb += predict(m, nb) / len(mp); pv += predict(m, nv) / len(mp); del m; torch.cuda.empty_cache()
        z["b_ids"] = np.r_[z["b_ids"].astype(str), SID[nb]]; z["pb"] = np.r_[z["pb"], pb]
        if len(nv) or "v_ids" in z:
            z["v_ids"] = np.r_[z.get("v_ids", np.zeros(0, "U16")).astype(str), SID[nv]]; z["pv"] = np.r_[z.get("pv", np.zeros(0)), pv]
        np.savez(fz, **z); print(f"fold {k}: scored {len(nb)} train + {len(nv)} val with {len(mp)} models", flush=True); continue
    if os.path.exists(fz):
        log.update(json.loads(str(np.load(fz)["log"]))); continue
    a_ = tri[fold[tri] != k]
    b_ = np.where(tr & ok & (fold == k) & (pool_rows if a.tune else True))[0]
    a_ = np.random.default_rng(k).permutation(a_); nes = int(0.12 * len(a_)); pb, pv, lg = [], [], {}
    for sd in range(a.seed0, a.seed0 + a.seeds):
        t1 = time.time()
        m, b = fit(a_[nes:], a_[:nes], 100 + sd)
        pb.append(predict(m, b_)); lg[f"f{k}_s{sd}"] = b
        if not a.tune:
            if not a.no_val: pv.append(predict(m, vai))
            torch.save(m.trainable(), f"{a.out}/model_f{k}_s{sd}.pt")
        del m; torch.cuda.empty_cache()
        print(f"fold {k} seed {sd} inner {b:.4f} {time.time() - t1:.0f}s", flush=True)
    log.update(lg); ob = np.mean(pb, 0); hb = has[b_] & pool_rows[b_]
    print(f"fold {k} OOF AUROC (tier, {a.target}) {auc(y[b_][hb], ob[hb]):.4f}", flush=True)
    sv = dict(b_ids=SID[b_], pb=ob, log=json.dumps(lg))
    if pv: sv.update(v_ids=SID[vai], pv=np.mean(pv, 0))
    np.savez(fz, **sv)

# assemble
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
res = dict(oof=float(auc(y[k_], oof[k_])), n_oof_tier=int(k_.sum()), oof_all_train=float(auc(y[kall], oof[kall])),
           n_oof_all=int((tr & ~np.isnan(oof)).sum()), n_train=int(tr.sum()), inner=log, args=vars(a), target=a.target,
           n_train_rows=int(len(tri)), n_pos=int(y[tri].sum()), folds_done=nf)
ka = tr & ~np.isnan(oof) & ~np.isnan(lab.y.values)
res["oof_overall_label_all_train"] = float(auc(lab.y.values[ka], oof[ka]))
print(f"FT train OOF {res['oof']:.4f} (tier {a.target}, n={res['n_oof_tier']}); all-train {a.target} {res['oof_all_train']:.4f}; "
      f"all-train overall label {res['oof_overall_label_all_train']:.4f}; train rows scored {res['n_oof_all']}/{res['n_train']}", flush=True)
pd.DataFrame({"study_id": SID[ti], "fold": fold[ti], "in_tier": pool_rows[ti], "ft": oof[ti]}).to_csv(f"{a.out}/oof_train.csv", index=False)
if not a.tune and vs:
    full = [s for s, v in vs.items() if len(v) == nf]
    vi = np.array([pos[s] for s in full]); vv = np.array([np.mean(vs[s]) for s in full])
    pd.DataFrame({"study_id": SID[vi], "ft": vv}).to_csv(f"{a.out}/val_scores.csv", index=False)
    res["n_val"] = int(len(vi)); res["n_val_total"] = int((~tr).sum())
    # val is only written here; it is NOT used for any selection
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1); print("FRAC15_FT_DONE", flush=True)
