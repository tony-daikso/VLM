"""rib_ft frozen fine-scale features with full real context: Dataset295 encoder stages 0-3 (baseline window, bf16) on the
whole 1.5 mm rib ROI (roi/{id}.npz). Instances = rib k (1-24) x angular sector j (12 x 30 deg around the rib-cage axis,
axial plane) -> per instance and stage: masked mean + max (stage 0 exact 1.5 mm labels; coarser stages: instance id of
the cell = max-pooled id of its rib voxels). Output feats2/{id}.npz: X float16 [I, 2*(32+64+128+256)=960], inst int16
(k-1)*12+j, nvox int32. usage: rib_feat2.py ROI OUT [--ids order] [--part i/n]; polls until ROI/../FEAT2_LAST or --once."""
import os, sys, time, argparse, zlib, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("roi"); ap.add_argument("out"); ap.add_argument("--once", action="store_true")
ap.add_argument("--ids", default=None); ap.add_argument("--part", default="0/1"); ap.add_argument("--vram-gb", type=float, default=0)
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src")); ap.add_argument("--nsec", type=int, default=12); ap.add_argument("--maxst", type=int, default=3); ap.add_argument("--bone", action="store_true")
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda")
if a.vram_gb:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; NORM = enc.norm; NS = a.nsec; NI = 24 * NS
pi, pn = (int(x) for x in a.part.split("/"))
rank = {c.strip(): i for i, c in enumerate(open(a.ids))} if a.ids else {}


def pool(f, inst):
    """f [C,N] float, inst [N] long (-1 = none) -> mean, max [NI,C], count [NI]"""
    m = inst >= 0; f = f[:, m]; i = inst[m]; C = f.shape[0]
    cnt = torch.zeros(NI, device=dev).index_add_(0, i, torch.ones_like(i, dtype=torch.float))
    s = torch.zeros(NI, C, device=dev).index_add_(0, i, f.T)
    mx = torch.full((NI, C), -1e4, device=dev).scatter_reduce_(0, i[:, None].expand(-1, C), f.T, "amax", include_self=True)
    return s / cnt.clamp(min=1)[:, None], torch.where(cnt[:, None] > 0, mx, torch.zeros_like(mx)), cnt


def one(cid):
    z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb = z["hu"], z["lab"]
    P = [(-s) % 16 for s in hu.shape]
    hu = np.pad(hu, [(0, p) for p in P], constant_values=-1024); lb = np.pad(lb, [(0, p) for p in P])
    rib = (lb >= 1) & (lb <= 24); idx = np.argwhere(rib)
    cy, cx = idx[:, 1].mean(), idx[:, 2].mean()
    ang = np.arctan2(idx[:, 1] - cy, idx[:, 2] - cx); sec = ((ang + np.pi) / (2 * np.pi) * NS).astype(int) % NS
    inst = np.full(lb.shape, -1, np.int16); inst[rib] = (lb[rib].astype(np.int16) - 1) * NS + sec
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        W = [{"type": "baseline"}] + ([{"level": 400, "width": 1800}] if a.bone else [])
        # LOW-MEMORY variant (2026-11-10, CG10k): the windows are run through the encoder one at a time (the original stacks
        # them as a batch of 2; conv / instance norm are per-sample, so the features are the same) and the per-stage pooled
        # mean / max are concatenated window-major, exactly the column order of the original h.flatten(0, 1).
        it = torch.from_numpy(inst).to(dev).long(); per = []
        for w in W:
            h = apply_window(torch.from_numpy(hu).to(dev), w, NORM)[None, None]; ii = it[None, None].float(); res = []
            for s_ in range(a.maxst + 1):
                h = st[s_](h)
                if s_: ii = F.max_pool3d(ii, 2)
                mean, mx, cnt = pool(h[0].flatten(1).float(), ii.flatten().long()); res.append((mean, mx, cnt))
            per.append(res); del h; torch.cuda.empty_cache()
        out = []; cnt0 = per[0][0][2]
        for s_ in range(a.maxst + 1):
            out += [torch.cat([p[s_][0] for p in per], 1), torch.cat([p[s_][1] for p in per], 1)]
        X = torch.cat(out, 1); keep = torch.nonzero(cnt0 >= 20)[:, 0]
    tmp = os.path.join(a.out, cid + ".tmp.npz")
    np.savez(tmp, X=X[keep].half().cpu().numpy(), inst=keep.short().cpu().numpy(), nvox=cnt0[keep].int().cpu().numpy())
    os.rename(tmp, os.path.join(a.out, cid + ".npz"))


n = 0; t0 = time.time()
while True:
    have = {f[:-4] for f in os.listdir(a.out) if f.endswith(".npz") and ".tmp" not in f}
    todo = [f[:-4] for f in os.listdir(a.roi) if f.endswith(".npz") and ".tmp" not in f and f[:-4] not in have
            and zlib.crc32(f[:-4].encode()) % pn == pi]
    todo.sort(key=lambda c: rank.get(c, 10 ** 9))
    if not todo:
        if a.once or os.path.exists(os.path.join(a.out, "LAST")): break
        time.sleep(60); continue
    for cid in todo[:300]:
        try: one(cid)
        except Exception as e: print("FAIL", cid, repr(e)[:200], flush=True)
        n += 1
        if n % 200 == 0: print(f"{n} {time.time() - t0:.0f}s", flush=True)
        if n % 50 == 0: torch.cuda.empty_cache()
print("FEAT2_DONE", n, flush=True)
