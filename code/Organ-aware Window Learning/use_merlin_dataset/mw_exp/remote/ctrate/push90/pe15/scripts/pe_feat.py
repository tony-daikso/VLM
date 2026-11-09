"""push90 task B: frozen posterior-element instance features from pe15 crops (T8..L5, (24,72,80) at 1.5 mm, axes S/A/R).
Encoder: TS Dataset292 (vertebrae, 1.5 mm) stages 0-2 frozen, windows base + bone(400/1800).
Regions per level from the (1-vox dilated) vertebra mask: midline x0 = mask centroid x of the anterior half; body back wall yb
= first row (scanning posteriorly from the anterior edge) where the midline (|x-x0|<=4) mask count drops below 15 % of its
peak (fallback: front - 20 rows); per side (x < x0 = patient left, RAS): body (y>=yb), arch (y<yb, |x-x0|<15),
transverse process TP (y<yb+4, |x-x0|>=15); spinous (y<yb-12, |x-x0|<=5) copied to both sides.
Instance id = (side*12 + level)*4 + region  (side 0=L, 1=R; level 0..9 = T8..L5; region 0 body, 1 arch, 2 TP, 3 spinous),
i.e. rib_probe3 layout with --nsec 4 --nbmode tile (mirror = same level/region, other side; adjacent 'ribs' = levels).
Per stage/window: region mean and region max  -> X float16 [I, 2*3*2*C...], cols json (stat, stage, win, start, end).
usage: pe_feat.py CROPS OUT --ids ids.txt [--part i/n] [--watch] [--amp bf16]"""
import os, sys, json, time, argparse, zlib, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("crops"); ap.add_argument("out"); ap.add_argument("--ids", required=True)
ap.add_argument("--part", default="0/1"); ap.add_argument("--watch", action="store_true", help="keep polling for new crops until CROP_DONE file")
ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src")); ap.add_argument("--vram-gb", type=float, default=6)
ap.add_argument("--amp", default=""); ap.add_argument("--done-flag", default="")
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; NORM = enc.norm
WN = ["base", "bone"]; W = [{"type": "baseline"}, {"level": 400, "width": 1800}]; STG = [0, 1, 2]; SH = (24, 72, 80); NV = int(np.prod(SH))


def regions(m):
    """m bool (24,72,80) -> list of (side, region, bool mask)"""
    md = F.max_pool3d(torch.from_numpy(m[None, None].astype(np.float32)), 3, 1, 1)[0, 0].numpy() > 0
    zz, yy, xx = np.nonzero(m)
    if len(yy) < 50: return []
    yf = np.percentile(yy, 99); ant = yy >= np.median(yy); x0 = xx[ant].mean()
    X = np.arange(SH[2])[None, None, :]; Y = np.arange(SH[1])[None, :, None]
    cnt = (m & (np.abs(X - x0) <= 4)).sum((0, 2))   # per y row
    yb = None; pk = cnt.max(); passed = False
    for y in range(int(yf), -1, -1):
        if cnt[y] >= 0.5 * pk: passed = True
        if passed and cnt[y] < 0.15 * pk: yb = y + 1; break
    if yb is None: yb = int(yf) - 20
    dx = X - x0; out = []
    sp = md & (Y < yb - 12) & (np.abs(dx) <= 5)
    for side, sm in ((0, dx < 0), (1, dx >= 0)):
        out.append((side, 0, md & (Y >= yb) & sm)); out.append((side, 1, md & (Y < yb) & (np.abs(dx) < 15) & sm))
        out.append((side, 2, md & (Y < yb + 4) & (np.abs(dx) >= 15) & sm)); out.append((side, 3, sp))
    return out


def one(cid):
    z = np.load(os.path.join(a.crops, cid + ".npz")); hu = z["hu"]; pres = z["present"]
    m = np.unpackbits(z["m"], 1)[:, :NV].reshape((10,) + SH).astype(bool)
    lv = [k for k in range(10) if pres[k]]
    insts, masks = [], []
    for k in lv:
        for side, rg, rm in regions(m[k]):
            if rm.sum() >= 5: insts.append((side * 12 + k) * 4 + rg); masks.append((lv.index(k), rm))
    if not insts: return "noinst"
    h = torch.from_numpy(hu[lv].astype(np.float32)).to(dev); B = len(lv)
    VX = [np.flatnonzero(r.ravel()) for _, r in masks]; UU = [u for u, _ in masks]
    feats = {"mean": [], "max": []}; cols = []
    with torch.no_grad():
        x = torch.cat([apply_window(h, w, NORM) for w in W], 0)[:, None]
        for s_ in range(max(STG) + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                x = st[s_](x)
            f = x.float(); C = f.shape[1]; fT, fY, fX = f.shape[2:]
            f = f.reshape(len(W), B, C, fT, fY, fX).permute(1, 0, 2, 3, 4, 5).reshape(B, len(W) * C, fT * fY * fX)
            sz, sy, sx = SH[0] // fT, SH[1] // fY, SH[2] // fX; Mn, Mx = [], []
            for u, v in zip(UU, VX):
                zz, yy, xx = np.unravel_index(v, SH); ci = (zz // sz) * fY * fX + (yy // sy) * fX + (xx // sx)
                uc, w = np.unique(ci, return_counts=True); uc = torch.from_numpy(uc).to(dev); w = torch.from_numpy(w).to(dev).float()
                g = f[u][:, uc]; Mn.append((g * w).sum(1) / w.sum()); Mx.append(g.amax(1))
            feats["mean"].append(torch.stack(Mn)); feats["max"].append(torch.stack(Mx)); del f
    nvox = np.array([len(v) for v in VX], np.int32)
    off = 0
    for stt in ("mean", "max"):
        for s_, P in zip(STG, feats[stt]):
            C = P.shape[1] // len(W)
            for wi, wn in enumerate(WN): cols.append((stt, s_, wn, off + wi * C, off + (wi + 1) * C))
            off += P.shape[1]
    Xo = torch.cat(feats["mean"] + feats["max"], 1).half().cpu().numpy()
    tmp = os.path.join(a.out, cid + ".tmp.npz")
    np.savez(tmp, X=Xo, inst=np.array(insts, np.int16), nvox=nvox, cols=json.dumps(cols)); os.replace(tmp, os.path.join(a.out, cid + ".npz"))
    return "ok"


pi, pn = (int(x) for x in a.part.split("/"))
order = [l.strip() for l in open(a.ids) if l.strip()]; order = [i for i in order if zlib.crc32(i.encode()) % pn == pi]
t0 = time.time(); nf = 0; nd = 0; bad = set()
while True:
    todo = [i for i in order if i not in bad and not os.path.exists(os.path.join(a.out, i + ".npz")) and os.path.exists(os.path.join(a.crops, i + ".npz"))]
    for cid in todo:
        if os.path.exists(os.path.join(a.out, "STOP")): print("STOP", flush=True); sys.exit(0)
        try:
            s = one(cid); nf = 0; nd += 1
            if s != "ok": bad.add(cid); print("SKIP", cid, s, flush=True)
        except Exception as e:
            nf += 1; bad.add(cid); print("FAIL", cid, repr(e)[:300], flush=True); torch.cuda.empty_cache()
            if nf >= 20: print("20 consecutive failures -> abort", flush=True); sys.exit(1)
        if nd % 500 == 0: print(f"{nd} done {time.time() - t0:.0f}s", flush=True)
    if not a.watch or (a.done_flag and os.path.exists(a.done_flag) and not todo): break
    time.sleep(60)
print("FEAT_DONE", nd, flush=True)
