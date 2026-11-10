"""spine_ft / pelvis_ft: fine-tune the late stages of a TotalSegmentator nnU-Net encoder on bone crops with MIL over units
(gallstones15_ft recipe: TS Dataset297 encoder, stages >= --train-from trained, 3 windows, mask pooling, linear head).
Input packs (crop_prep.py): 3 mm crops {sp,pv}_hu [N,U,z,y,x] int16, {sp,pv}_m masks, {sp,pv}_present; upsampled x2 on the
GPU (trilinear HU / nearest mask) -> 1.5 mm-spaced encoder input. Windows: baseline (nnU-Net norm), bone 400/1800, soft 40/400.
Per unit: stage-3 [mean|max|mask-mean] + stage-4 [mean|max] per window -> layer-norm -> dropout -> linear unit logit
(+ learnable per-level bias); case logit = MIL pool over present units (lse | attn | max).
Target = site label (spine: t_vert, pelvis: t_pel; 1 = fracture at the site, 0 = fracture-free, other-site fractures NaN
-> not trained on, still scored). Labels csv holds train/val rows only (asserted; test never read).
Folds = sitewise outer folds (labels.csv 'fold'). Training rows = train & tierA & target known (unless --all-train).
Per fold model: scores its held-out fold of ALL train rows (OOF: tier rows never trained on in that fold; non-tier rows are
never trained on at all) and, unless --tune, all val rows (val = mean over folds x seeds). Best epoch per fit = AUROC on an
inner 12 % holdout of the training rows (train only). Model selection between configs = train OOF on the target (tier rows).
usage: frac_ft.py --branch spine --labels L --packs P1 P2 --out DIR [--train-from 2] [--pool lse] [--enc 297] [--tune]

=== frac_ft_full.py (2026-11-10, CG10k fracture a6x10w2_tv7: FT_sp refit) ===
Copy of spine_ft/scripts/frac_ft.py with
  --mode full  : FULL-TRAIN refit per seed sd in [seed0, seed0+seeds): pool = all tierA train rows with a known target
                 (same rows as the CV runs' union), inner 12 % holdout = rng(5).permutation(pool)[:12 %] (5 = not a fold id),
                 torch/np seed 100+sd (as CV), same epochs / OneCycle / best-epoch-by-inner-AUROC. Saves OUT/model_full_s{sd}.pt
                 (trained stages >= train_from + head + lvl; base = TS encoder checkpoint) and scores all train + val rows
                 -> OUT/scores_full.csv (per seed + mean, raw logits).
  --mode score : --packs = packs to score (e.g. CG crop_prep pack), --labels ignored; loads OUT/model_full_s*.pt
                 -> --score-out csv (per seed + mean ft). Rows with no present unit are not scored (NaN), as in CV.
  --accum K    : gradient accumulation over K micro-batches of bs/K rows (loss scaled so the summed gradient equals the
                 bs-row mean; InstanceNorm encoder -> no batch statistics) - lets the fp32 recipe run on a 10 GB card."""
import sys, os, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score as auc
ap = argparse.ArgumentParser()
ap.add_argument("--branch", required=True, choices=["spine", "pelvis"]); ap.add_argument("--labels", required=True)
ap.add_argument("--packs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src"))
ap.add_argument("--enc", default="297"); ap.add_argument("--train-from", type=int, default=2)
ap.add_argument("--pool", default="lse", choices=["lse", "attn", "max"]); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4, help="encoder lr; head lr = 10x"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--vram-gb", type=float, default=20); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tune", action="store_true"); ap.add_argument("--folds", default=""); ap.add_argument("--all-train", action="store_true")
ap.add_argument("--target", default=""); ap.add_argument("--wins", default="base,bone,soft")
ap.add_argument("--mode", default="full", choices=["full", "score"]); ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--score-out", default=""); ap.add_argument("--pred-bs", type=int, default=16)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
torch.set_num_threads(a.threads); dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True  # bf16 3D conv is ~25x slower on this A6000 stack
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
STOP = os.path.join(os.path.dirname(os.path.abspath(a.out)), "STOP")
PFX = {"spine": "sp", "pelvis": "pv"}[a.branch]; TGT = a.target or {"spine": "t_vert", "pelvis": "t_pel"}[a.branch]
RES = os.path.expanduser("~/.totalsegmentator/nnunet/results")
ENC = {"297": "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres",
       "292": "Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"}[a.enc]

SCORE = a.mode == "score"
if SCORE:
    _ids = [str(i) for f in a.packs for i in np.load(f)["ids"]]
    lab = pd.DataFrame({"study_id": _ids, "split": "score", "fold": -1, "y": np.nan, "tierA": False})
    lab[a.target or {"spine": "t_vert", "pelvis": "t_pel"}[a.branch]] = np.nan
else:
    lab = pd.read_csv(a.labels); assert set(lab.split) <= {"train", "val"}
if a.limit:
    lab = pd.concat([lab[lab.split == "train"].head(a.limit), lab[lab.split == "val"].head(a.limit // 4)]).reset_index(drop=True)
N = len(lab); t0 = time.time()
HU = M = P = None; ok = np.zeros(N, bool); pos = {s: i for i, s in enumerate(lab.study_id)}
for f in a.packs:
    z = np.load(f); ids = z["ids"]; h, m, p = z[f"{PFX}_hu"], z[f"{PFX}_m"], z[f"{PFX}_present"]
    if HU is None:
        HU = np.full((N,) + h.shape[1:], -1024, np.int16); M = np.zeros((N,) + m.shape[1:], np.uint8); P = np.zeros((N, h.shape[1]), bool)
    for j, s in enumerate(ids):
        i = pos.get(str(s))
        if i is not None and not ok[i]:
            HU[i], M[i], P[i], ok[i] = h[j], m[j], p[j], True
    del z, h, m, p
ok &= P.any(1)
U = HU.shape[1]; SH = tuple(2 * s for s in HU.shape[2:])
print(f"{a.branch}: loaded {ok.sum()}/{N} cases, {U} units, crop {HU.shape[2:]} -> {SH}, {time.time() - t0:.0f}s", flush=True)
yt = lab[TGT].values.astype(float); has = ~np.isnan(yt) & ok; y = np.nan_to_num(yt).astype(np.float32)
tr = (lab.split == "train").values; fold = lab.fold.values.astype(int)
pool_rows = tr & has & (lab.tierA.values.astype(bool) | a.all_train)

enc0 = NnUNetEncoder(f"{RES}/{ENC}", device="cuda"); NORM = enc0.norm
BASE_SD = {k: v.detach().clone() for k, v in enc0.encoder.state_dict().items()}
WD = {"base": {"type": "baseline"}, "bone": {"level": 400, "width": 1800}, "soft": {"level": 40, "width": 400},
      "liver": {"level": 60, "width": 150}}
WINS = [WD[w] for w in a.wins.split(",")]; NW = len(WINS)
CH = enc0.stage_channels([3, 4]); D = NW * (3 * CH[0] + 2 * CH[1])


class FT(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc0.encoder; s.enc.load_state_dict(BASE_SD)
        for i, st in enumerate(s.enc.stages):
            for p in st.parameters(): p.requires_grad_(i >= a.train_from)
        s.drop = nn.Dropout(0.3); s.head = nn.Linear(D, 1); s.lvl = nn.Parameter(torch.zeros(U))
        s.att = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, 1)) if a.pool == "attn" else None

    def forward(s, x, g, uidx, cidx, pr):  # x [n*NW,1,...] unit-major, g [n,1,...]; uidx/cidx [n]
        n = g.shape[0]; B = pr.shape[0]; out = []; h = x
        if getattr(s.enc, "stem", None) is not None:
            with torch.no_grad(): h = s.enc.stem(h)
        for i, st in enumerate(s.enc.stages):
            if i < a.train_from:
                with torch.no_grad(): h = st(h)
            else: h = st(h)
            if i >= 3: out.append(h)
            if i >= 4: break
        f3, f4 = out
        g3 = F.adaptive_max_pool3d(g, f3.shape[2:]).repeat_interleave(NW, 0)
        gm = (f3 * g3).sum((2, 3, 4)) / g3.sum((2, 3, 4)).clamp(min=1)
        v = torch.cat([f3.mean((2, 3, 4)), f3.amax((2, 3, 4)), gm, f4.mean((2, 3, 4)), f4.amax((2, 3, 4))], 1)
        v = F.layer_norm(v.float(), v.shape[1:]).view(n, -1)
        ul = s.head(s.drop(v))[:, 0] + s.lvl[uidx]
        L = torch.full((B, U), -1e4, device=x.device, dtype=ul.dtype); L[cidx, uidx] = ul
        L = torch.where(pr, L, torch.full_like(L, -1e4))
        if a.pool == "lse":
            return torch.logsumexp(L, 1)
        if a.pool == "max":
            return L.max(1).values
        A = torch.full((B, U), -1e4, device=x.device, dtype=ul.dtype); A[cidx, uidx] = s.att(v)[:, 0]
        A = torch.where(pr, A, torch.full_like(A, -1e4))
        return (torch.softmax(A, 1) * torch.where(L > -1e3, L, torch.zeros_like(L))).sum(1)


def batch(idx, train):
    # fixed shapes (all B*U units; absent units masked in the pool) so cudnn.benchmark does not re-tune every step
    pr = torch.from_numpy(P[idx]).to(dev); cidx, uidx = [t.ravel() for t in np.indices((len(idx), U))]
    h = torch.from_numpy(HU[idx].reshape(-1, *HU.shape[2:])).to(dev).float()[:, None]
    g = torch.from_numpy(M[idx].reshape(-1, *M.shape[2:])).to(dev).float()[:, None]
    if train:
        n = h.shape[0]
        h = h * (1 + 0.04 * (torch.rand(n, 1, 1, 1, 1, device=dev) - .5)) + 20 * (torch.rand(n, 1, 1, 1, 1, device=dev) - .5)
        sh = tuple(int(v) for v in np.random.randint(-1, 2, 3))
        h = torch.roll(h, sh, (2, 3, 4)); g = torch.roll(g, sh, (2, 3, 4))
    h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
    g = F.interpolate(g, scale_factor=2, mode="nearest")
    x = torch.stack([apply_window(h[:, 0], w, NORM) for w in WINS], 1).reshape(-1, 1, *SH)  # unit-major
    return x, g, torch.from_numpy(uidx).to(dev), torch.from_numpy(cidx).to(dev), pr


def predict(m, idx):
    m.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        for i in range(0, len(idx), a.pred_bs):
            out.append(m(*batch(idx[i:i + a.pred_bs], False)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def SAVEK(k):  # trained part only: encoder stages >= train_from, head, level bias, attention
    if k.startswith("enc.stages."): return int(k.split(".")[2]) >= a.train_from
    return not k.startswith("enc.")


def fit(idx_tr, idx_es, preds, seed):
    global LAST_EP
    torch.manual_seed(seed); np.random.seed(seed)
    m = FT().to(dev)
    ep_ = [p for i, st in enumerate(m.enc.stages) if i >= a.train_from for p in st.parameters()]
    hp = list(m.head.parameters()) + [m.lvl] + (list(m.att.parameters()) if m.att is not None else [])
    groups = [{"params": hp, "lr": a.lr * 10}] + ([{"params": ep_, "lr": a.lr}] if ep_ else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(idx_tr) / a.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.15)
    npos = y[idx_tr].sum(); pw = torch.tensor((len(idx_tr) - npos) / max(npos, 1), device=dev)
    best, bst = -1, None
    for ep in range(a.epochs):
        if os.path.exists(STOP): print("STOP seen", flush=True); sys.exit(3)
        m.train(); perm = np.random.permutation(idx_tr); tl = 0; t1 = time.time()
        for i in range(0, len(perm), a.bs):
            j = np.sort(perm[i:i + a.bs]); opt.zero_grad(); lsum = 0.0
            for jj in np.array_split(j, min(a.accum, len(j))):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                    lg = m(*batch(jj, True))
                loss = F.binary_cross_entropy_with_logits(lg.float(), torch.from_numpy(y[jj]).to(dev), pos_weight=pw) * (len(jj) / len(j))
                loss.backward(); lsum += loss.item()
            opt.step(); sch.step(); tl += lsum * len(j)
        sc = auc(y[idx_es], predict(m, idx_es))
        print(f"  seed {seed} ep {ep} loss {tl / len(perm):.4f} inner AUROC {sc:.4f} {time.time() - t1:.0f}s "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if sc > best: best, bst, LAST_EP = sc, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep
    m.load_state_dict(bst); outp = [predict(m, ix) for ix in preds]
    global LAST_STATE
    LAST_STATE = {k: v.detach().cpu() for k, v in m.state_dict().items() if SAVEK(k)}
    del m, opt, bst; torch.cuda.empty_cache(); return outp, best


tri = np.where(pool_rows)[0]; vai = np.where(~tr & ok)[0]; LAST_EP = -1
okidx = np.where(ok)[0]
if not SCORE:
    o_ = pd.DataFrame({"study_id": lab.study_id.values, "split": lab.split.values}); PS = {}; info = {}
    for sd in range(a.seed0, a.seed0 + a.seeds):
        mp = f"{a.out}/model_full_s{sd}.pt"
        if os.path.exists(mp):
            meta = torch.load(mp, map_location="cpu", weights_only=False); m = FT().to(dev)
            miss = m.load_state_dict(meta["state"], strict=False); assert not miss.unexpected_keys
            PS[sd] = predict(m, okidx); info[sd] = meta["best_inner_auc"]; del m; torch.cuda.empty_cache(); print("seed", sd, "resumed", flush=True); continue
        a_ = np.random.default_rng(5).permutation(tri); nes = int(0.12 * len(a_)); t1 = time.time()
        (p_all,), b = fit(a_[nes:], a_[:nes], [okidx], 100 + sd)
        meta = {"state": LAST_STATE, "args": vars(a), "enc": ENC, "best_inner_auc": float(b), "best_epoch": int(LAST_EP),
                "n_fit": int(len(a_) - nes), "n_hold": int(nes), "hold_rng": 5, "torch_seed": 100 + sd, "target": TGT, "U": int(U)}
        torch.save(meta, mp + ".tmp"); os.replace(mp + ".tmp", mp); PS[sd] = p_all; info[sd] = float(b)
        print(f"full seed {sd} inner {b:.4f} best_ep {LAST_EP} {time.time() - t1:.0f}s", flush=True)
    for sd, p in PS.items():
        c = np.full(N, np.nan); c[okidx] = p; o_[f"s{sd}"] = c
    o_["ft"] = o_[[f"s{sd}" for sd in PS]].mean(1, skipna=False); o_.to_csv(f"{a.out}/scores_full.csv", index=False)
    kk = pool_rows & o_["ft"].notna().values
    json.dump({"inner": info, "train_resub_tier": float(auc(y[kk], o_["ft"].values[kk])), "args": vars(a)}, open(f"{a.out}/summary_full.json", "w"), indent=1)
    print("FULL_DONE", flush=True)
else:
    o_ = pd.DataFrame({"study_id": lab.study_id.values}); cols = []
    for f in sorted(x for x in os.listdir(a.out) if x.startswith("model_full_s") and x.endswith(".pt")):
        meta = torch.load(f"{a.out}/{f}", map_location="cpu", weights_only=False); m = FT().to(dev)
        miss = m.load_state_dict(meta["state"], strict=False); assert not miss.unexpected_keys
        assert all(not SAVEK(k) for k in miss.missing_keys)
        c = np.full(N, np.nan); c[okidx] = predict(m, okidx); o_[f[:-3]] = c; cols.append(f[:-3]); del m; torch.cuda.empty_cache()
    o_["ft"] = o_[cols].mean(1, skipna=False); o_.to_csv(a.score_out, index=False); print("SCORE_DONE", a.score_out, int(o_.ft.notna().sum()), flush=True)
