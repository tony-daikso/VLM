"""pelvis15 frozen fine-scale features (rib_feat3 recipe on the pelvis): frozen TS nnU-Net encoder stages 0..maxst
(baseline [+ bone] window, bf16) on the whole 1.5 mm pelvis ROI (roi/{id}.npz from pv15_crop.py, real context).
Instances = bone (1 hip_R 2 hip_L 3 sacrum 4 femur_R 5 femur_L) x spatial cell (CELL voxels cube; x measured from the
midline xmid so left/right cells mirror) -> masked mean + max per stage. Output OUT/{id}.npz: X float16 [I, D],
meta int16 [I, 4] = (bone, cz, cy, cx_rel), nvox int32.
usage: pv15_feat.py ROI OUT --ts MODEL_DIR [--ids order.txt] [--part i/n] [--maxst 1] [--bone] [--cell 12] [--vram-gb G]
       polls for new ROIs until OUT/LAST exists (or --once)."""
import os, sys, time, argparse, zlib, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("roi"); ap.add_argument("out"); ap.add_argument("--once", action="store_true")
ap.add_argument("--ids", default=None); ap.add_argument("--part", default="0/1"); ap.add_argument("--vram-gb", type=float, default=0)
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src")); ap.add_argument("--maxst", type=int, default=1)
ap.add_argument("--bone", action="store_true"); ap.add_argument("--cell", type=int, default=12); ap.add_argument("--minvox", type=int, default=30)
ap.add_argument("--ts", required=True); ap.add_argument("--only-ids", action="store_true", help="process only ids listed in --ids")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda")
if a.vram_gb:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; NORM = enc.norm
pi, pn = (int(x) for x in a.part.split("/"))
order = [c.strip() for c in open(a.ids) if c.strip()] if a.ids else []
rank = {c: i for i, c in enumerate(order)}; only = set(order) if a.only_ids else None
W = [{"type": "baseline"}] + ([{"level": 400, "width": 1800}] if a.bone else [])


def pool(f, inst, NI):
    m = inst >= 0; f = f[:, m].float(); i = inst[m]; C = f.shape[0]
    cnt = torch.zeros(NI, device=dev).index_add_(0, i, torch.ones_like(i, dtype=torch.float))
    s = torch.zeros(NI, C, device=dev).index_add_(0, i, f.T)
    mx = torch.full((NI, C), -1e4, device=dev).scatter_reduce_(0, i[:, None].expand(-1, C), f.T, "amax", include_self=True)
    return s / cnt.clamp(min=1)[:, None], torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx)), cnt


def one(cid):
    z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb = z["hu"], z["lab"]; xmid, org, sp = float(z["xmid"]), z["org"], float(z["sp"])
    P = [(-s) % 16 for s in hu.shape]
    hu = np.pad(hu, [(0, p) for p in P], constant_values=-1024); lb = np.pad(lb, [(0, p) for p in P])
    zz, yy, xx = np.nonzero(lb)
    cx = np.floor((org[0] + xx * sp - xmid) / (a.cell * sp)).astype(np.int64)    # signed cell from the midline
    key = (((lb[zz, yy, xx].astype(np.int64) * 64 + zz // a.cell) * 64 + yy // a.cell) * 64 + (cx + 32))
    uk, inv = np.unique(key, return_inverse=True); NI = len(uk)
    inst = np.full(lb.shape, -1, np.int64); inst[zz, yy, xx] = inv
    meta = np.stack([uk // 64 ** 3, (uk // 64 ** 2) % 64, (uk // 64) % 64, uk % 64 - 32], 1).astype(np.int16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.stack([apply_window(torch.from_numpy(hu).to(dev), w, NORM) for w in W])[:, None]
        it = torch.from_numpy(inst).to(dev)
        out = []; cnt0 = None; ii = (it + 1)[None, None].float(); h = x
        for s_ in range(a.maxst + 1):
            h = st[s_](h)
            if s_: ii = F.max_pool3d(ii, 2)
            mean, mx, cnt = pool(h.flatten(0, 1).flatten(1), ii.flatten().long() - 1, NI)
            if cnt0 is None: cnt0 = cnt
            out += [mean, mx]
        X = torch.cat(out, 1); keep = torch.nonzero(cnt0 >= a.minvox)[:, 0]
    tmp = os.path.join(a.out, cid + ".tmp.npz"); kc = keep.cpu().numpy()
    np.savez(tmp, X=X[keep].half().cpu().numpy(), meta=meta[kc], nvox=cnt0[keep].int().cpu().numpy())
    os.rename(tmp, os.path.join(a.out, cid + ".npz"))


n = 0; t0 = time.time()
while True:
    have = {f[:-4] for f in os.listdir(a.out) if f.endswith(".npz") and ".tmp" not in f}
    todo = [f[:-4] for f in os.listdir(a.roi) if f.endswith(".npz") and ".tmp" not in f and f[:-4] not in have
            and zlib.crc32(f[:-4].encode()) % pn == pi and (only is None or f[:-4] in only)]
    todo.sort(key=lambda c: rank.get(c, 10 ** 9))
    if not todo:
        if a.once or os.path.exists(os.path.join(a.out, "LAST")): break
        time.sleep(60); continue
    for cid in todo[:200]:
        try: one(cid)
        except Exception as e: print("FAIL", cid, repr(e)[:200], flush=True)
        n += 1
        if n % 100 == 0: print(f"{n} {time.time() - t0:.0f}s", flush=True)
        if n % 50 == 0: torch.cuda.empty_cache()
        if os.path.exists(os.path.join(a.out, "STOP")): sys.exit(0)
print("FEAT_DONE", n, flush=True)
