"""push90 task A: multi-window / multi-scale frozen features from the straightened rib tiles (generalises rib_straight
rs_feat.py; identical chunking (64 rows, stride 48), 12 mm cells (--cell 8), instance ids and pooling).
Dataset295 encoder (frozen) stages --stages (default 0,1,2), windows --wins (default base, bone 400/1800, soft 50/400,
wide 500/3500). Per stage f (upsampled to the tile grid by repeat), per window: rib-mask mean, rib-mask max,
cross-section mean. Columns: for stat in (mean,max,sec): for stage: for window: C_stage  -> 'cols' json in the npz.
Output feats/{id}.npz: X float16 [I, D], inst int16, nvox int32, cols (json str).
usage: rs_feat2.py TILES OUT --ids ids.txt [--part i/n] [--bs 16] [--vram-gb 20]"""
import os, sys, json, time, argparse, zlib, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("tiles"); ap.add_argument("out"); ap.add_argument("--ids", required=True)
ap.add_argument("--part", default="0/1"); ap.add_argument("--cell", type=int, default=8); ap.add_argument("--ns", type=int, default=24)
ap.add_argument("--stages", default="0,1,2"); ap.add_argument("--wins", default="base,bone,soft,wide"); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--src", default=os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/src")); ap.add_argument("--vram-gb", type=float, default=20)
ap.add_argument("--amp", default="", help="bf16 to run the encoder in bf16 autocast")
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; NORM = enc.norm; CK, CS, CL = 64, 32, a.cell
WD = {"base": {"type": "baseline"}, "bone": {"level": 400, "width": 1800}, "soft": {"level": 50, "width": 400}, "wide": {"level": 500, "width": 3500}}
WN = a.wins.split(","); W = [WD[w] for w in WN]; STG = [int(s) for s in a.stages.split(",")]; NW = len(W); SMAX = max(STG)


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
    rmap = np.full((len(r0), CK), -1, np.int64)
    for q, (ch, lo_, hi_, ii) in enumerate(cl): rmap[ch, lo_:hi_] = q
    rmap = torch.from_numpy(rmap).to(dev); acc = {}
    with torch.no_grad():
        for j in range(0, h.shape[0], a.bs):
            hb = h[j:j + a.bs]; B = hb.shape[0]; gb = m[j:j + a.bs][..., None]
            x = torch.cat([apply_window(hb, w, NORM) for w in W], 0)[:, None]
            ri = rmap[j:j + B].reshape(-1); k_ = ri >= 0; ri = ri[k_]; rc = gb.sum((2, 3, 4)).reshape(-1)
            for s_ in range(SMAX + 1):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp == "bf16"):
                    x = st[s_](x)
                if s_ not in STG: continue
                f = x.float(); fT, fY, fX = f.shape[2:]
                f = f.reshape(NW, B, *f.shape[1:]).permute(1, 3, 4, 5, 0, 2).flatten(4)   # [B,T,Y,X,NW*C]
                f = f.repeat_interleave(CK // fT, 1).repeat_interleave(CS // fY, 2).repeat_interleave(CS // fX, 3)
                D0 = f.shape[-1]
                if s_ not in acc:
                    acc[s_] = dict(S=torch.zeros(NCl, D0, device=dev), Ss=torch.zeros(NCl, D0, device=dev),
                                   Mx=torch.full((NCl, D0), -1e4, device=dev), Cn=torch.zeros(NCl, device=dev), Rn=torch.zeros(NCl, device=dev))
                A = acc[s_]
                rs = (f * gb).sum((2, 3)).reshape(-1, D0); rx = f.masked_fill(gb == 0, -1e4).amax((2, 3)).reshape(-1, D0)
                rsec = f.mean((2, 3)).reshape(-1, D0)
                A["S"].index_add_(0, ri, rs[k_]); A["Cn"].index_add_(0, ri, rc[k_]); A["Ss"].index_add_(0, ri, rsec[k_])
                A["Rn"].index_add_(0, ri, torch.ones_like(rc[k_])); A["Mx"].scatter_reduce_(0, ri[:, None].expand(-1, D0), rx[k_], "amax", include_self=True)
                del f, rs, rx, rsec
            del x
    parts = {"mean": [], "max": [], "sec": []}; cols = []; off = 0
    for s_ in STG:
        A = acc[s_]; sec = A["Ss"] / A["Rn"].clamp(min=1)[:, None]; has = A["Cn"] > 0
        parts["mean"].append(torch.where(has[:, None], A["S"] / A["Cn"].clamp(min=1)[:, None], sec))
        parts["max"].append(torch.where(has[:, None], A["Mx"], sec)); parts["sec"].append(sec)
    for stt in ("mean", "max", "sec"):
        for s_, P in zip(STG, parts[stt]):
            C = P.shape[1] // NW
            for wi, wn in enumerate(WN): cols.append((stt, s_, wn, off + wi * C, off + (wi + 1) * C))
            off += P.shape[1]
    X = torch.cat(parts["mean"] + parts["max"] + parts["sec"], 1).half().cpu().numpy()
    inst = cl[:, 3].astype(np.int16); nvox = acc[STG[0]]["Cn"].int().cpu().numpy().astype(np.int32)
    _, first = np.unique(inst, return_index=True)
    tmp = os.path.join(a.out, cid + ".tmp.npz"); np.savez(tmp, X=X[first], inst=inst[first], nvox=nvox[first], cols=json.dumps(cols))
    os.replace(tmp, os.path.join(a.out, cid + ".npz"))


pi, pn = (int(x) for x in a.part.split("/"))
ids = [l.strip() for l in open(a.ids) if l.strip()]
ids = [i for i in ids if zlib.crc32(i.encode()) % pn == pi and not os.path.exists(os.path.join(a.out, i + ".npz"))
       and os.path.exists(os.path.join(a.tiles, i + ".npz"))]
print(f"{len(ids)} to do", flush=True); t0 = time.time(); nf = 0
for n, cid in enumerate(ids):
    if os.path.exists(os.path.join(a.out, "STOP")): print("STOP", flush=True); break
    try: one(cid); nf = 0
    except Exception as e:
        nf += 1; print("FAIL", cid, repr(e)[:300], flush=True); torch.cuda.empty_cache()
        if nf >= 20: print("20 consecutive failures -> abort", flush=True); sys.exit(1)
    if (n + 1) % 250 == 0: print(f"{n + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
print("FEAT_DONE", flush=True)
