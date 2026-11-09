"""rib_straight frozen features (rib_probe3 format) from straightened tiles: Dataset295 encoder stages 0-1 (frozen, bf16 off),
windows base+bone, on 64-row chunks (stride 48, last aligned to the end) of every rib tile. Instance = rib k x arc cell
(--cell rows = 12 mm by default, counted from the posterior true-rib start m0; cell index clipped to NS-1) -> per stage:
rib-mask mean, rib-mask max, cross-section mean (each cell taken from the chunk whose centre is closest).
Output feats/{id}.npz: X float16 [I, D], inst int16 = (k-1)*NS + cell, nvox int32 (rib-mask voxels in the cell).
usage: rs_feat.py TILES OUT [--ids order.txt] [--part i/n]"""
import os, sys, time, argparse, zlib, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("tiles"); ap.add_argument("out"); ap.add_argument("--ids", default="")
ap.add_argument("--part", default="0/1"); ap.add_argument("--cell", type=int, default=8); ap.add_argument("--ns", type=int, default=24)
ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src")); ap.add_argument("--vram-gb", type=float, default=6)
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; NORM = enc.norm; CK, CS, CL = 64, 32, a.cell
W = [{"type": "baseline"}, {"level": 400, "width": 1800}]


def one(cid):
    z = np.load(os.path.join(a.tiles, cid + ".npz")); hu, mk, tab = z["hu"], z["m"], z["tab"]
    chunks, cells = [], []
    for k, s0, L, m0, n, nv in tab:
        sts = list(range(0, L - CK + 1, 48))
        if sts[-1] != L - CK: sts.append(L - CK)
        cen = np.array(sts) + CK / 2; base = len(chunks); chunks += [s0 + s for s in sts]
        for c in range(int(np.ceil(n / CL))):
            lo_ = m0 + c * CL; hi_ = min(lo_ + CL, m0 + n)
            if hi_ - lo_ < 3: continue
            cand = [q for q in range(len(sts)) if sts[q] <= lo_ and hi_ <= sts[q] + CK]
            j = min(cand, key=lambda q: abs(cen[q] - (lo_ + hi_) / 2))
            cells.append((base + j, lo_ - sts[j], hi_ - sts[j], (k - 1) * a.ns + min(c, a.ns - 1)))
    r0 = chunks + [chunks[0]] * ((-len(chunks)) % 8)
    rows = np.array(r0)[:, None] + np.arange(CK)[None]
    h = torch.from_numpy(hu[rows]).to(dev).float(); m = torch.from_numpy(np.unpackbits(mk[rows], -1)[..., :CS]).to(dev).float()
    NCl = len(cells); cl = np.array(cells, np.int64).reshape(-1, 4)
    rmap = np.full((len(r0), CK), -1, np.int64)  # chunk row -> cell index
    for q, (ch, lo_, hi_, ii) in enumerate(cl): rmap[ch, lo_:hi_] = q
    rmap = torch.from_numpy(rmap).to(dev)
    S_m = None
    with torch.no_grad():
        for j in range(0, h.shape[0], 16):
            hb = h[j:j + 16]; B = hb.shape[0]; gb = m[j:j + 16][..., None]  # [B,CK,CS,CS,1]
            x = torch.cat([apply_window(hb, w, NORM) for w in W], 0)[:, None]; out = []
            for s_ in range(2):
                x = st[s_](x); out.append(x)
            fs = []
            for s_, f in enumerate(out):  # [2B, C, T, Y, X] -> [B, T, Y, X, 2C]
                f = f.float().reshape(2, B, *f.shape[1:]).permute(1, 3, 4, 5, 0, 2).flatten(4)
                if s_: f = f.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
                fs.append(f)
            Fm = torch.cat(fs, 4)  # [B, CK, CS, CS, D0]
            D0 = Fm.shape[-1]
            if S_m is None:
                S_m = torch.zeros(NCl, D0, device=dev); S_s = torch.zeros(NCl, D0, device=dev); Mx = torch.full((NCl, D0), -1e4, device=dev)
                Cn = torch.zeros(NCl, device=dev); Rn = torch.zeros(NCl, device=dev)
            rs = (Fm * gb).sum((2, 3)).reshape(-1, D0); rc = gb.sum((2, 3, 4)).reshape(-1)
            rx = Fm.masked_fill(gb == 0, -1e4).amax((2, 3)).reshape(-1, D0); rsec = Fm.mean((2, 3)).reshape(-1, D0)
            ri = rmap[j:j + B].reshape(-1); k_ = ri >= 0; ri = ri[k_]
            S_m.index_add_(0, ri, rs[k_]); Cn.index_add_(0, ri, rc[k_]); S_s.index_add_(0, ri, rsec[k_]); Rn.index_add_(0, ri, torch.ones_like(rc[k_]))
            Mx.scatter_reduce_(0, ri[:, None].expand(-1, D0), rx[k_], "amax", include_self=True)
            del Fm, fs, out, x
    sec = S_s / Rn.clamp(min=1)[:, None]; has = Cn > 0
    mean = torch.where(has[:, None], S_m / Cn.clamp(min=1)[:, None], sec); mx = torch.where(has[:, None], Mx, sec)
    X = torch.cat([mean, mx, sec], 1); inst = cl[:, 3]; nvox = Cn.int().cpu().numpy()
    X = X.half().cpu().numpy(); inst = inst.astype(np.int16); nvox = nvox.astype(np.int32)
    _, first = np.unique(inst, return_index=True)  # clipped duplicate cell ids at the far end: keep the first
    tmp = os.path.join(a.out, cid + ".tmp.npz"); np.savez(tmp, X=X[first], inst=inst[first], nvox=nvox[first])
    os.replace(tmp, os.path.join(a.out, cid + ".npz"))


pi, pn = (int(x) for x in a.part.split("/"))
ids = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else sorted(f[:-4] for f in os.listdir(a.tiles) if f.endswith(".npz") and ".tmp" not in f)
ids = [i for i in ids if zlib.crc32(i.encode()) % pn == pi and not os.path.exists(os.path.join(a.out, i + ".npz"))]
print(f"{len(ids)} to do", flush=True); t0 = time.time(); nf = 0
for n, cid in enumerate(ids):
    if os.path.exists(os.path.join(a.out, "STOP")): print("STOP", flush=True); break
    try: one(cid)
    except Exception as e:
        nf += 1; print("FAIL", cid, repr(e)[:200], flush=True)
        if nf >= 20: print("too many failures -> abort", flush=True); sys.exit(1)
    if (n + 1) % 500 == 0: print(f"{n + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True); torch.cuda.empty_cache()
print("FEAT_DONE", flush=True)
