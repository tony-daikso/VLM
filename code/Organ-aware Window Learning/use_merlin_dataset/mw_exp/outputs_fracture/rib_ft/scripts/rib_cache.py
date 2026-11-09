"""rib_ft stage-1 cache: frozen Dataset295 stages 0-1 (baseline window, bf16) on the whole 1.5 mm rib ROI (padded to /32),
kept only in stage-1 cells (3 mm) within DIL cells of any rib (1-24) -> cache/{id}.npz:
  idx int32 (flat index into grid), f uint16 (bf16 bits) [n,64], grid (3,), m2 packbits of stride-4 rib presence [24,...].
Polls ROI_DIR for new rois until ROI_DIR/../CACHE_LAST exists and nothing is left. Splits work with --part i/n (hash).
usage: rib_cache.py ROI_DIR CACHE_DIR [--part 0/1] [--ids list] ; stop: touch CACHE_DIR/STOP"""
import os, sys, time, argparse, zlib
import numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("roi"); ap.add_argument("cache"); ap.add_argument("--part", default="0/1")
ap.add_argument("--ids", default=None); ap.add_argument("--dil", type=int, default=4); ap.add_argument("--vram-gb", type=float, default=0)
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src")); ap.add_argument("--once", action="store_true")
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
from windowing import apply_window
dev = torch.device("cuda"); K = 24
if a.vram_gb:
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)), 0)
enc = NnUNetEncoder(a.ts); NORM = enc.norm; st = enc.encoder.stages
pi, pn = (int(x) for x in a.part.split("/"))
prio = [l.strip() for l in open(a.ids)] if a.ids else None
rank = {c: i for i, c in enumerate(prio)} if prio else {}


def mine(c): return zlib.crc32(c.encode()) % pn == pi


def one(cid):
    z = np.load(os.path.join(a.roi, cid + ".npz")); hu, lb = z["hu"], z["lab"]
    P = [(-s) % 32 for s in hu.shape]
    hu = np.pad(hu, [(0, p) for p in P], constant_values=-1024); lb = np.pad(lb, [(0, p) for p in P])
    lb = np.where(lb > K, 0, lb).astype(np.uint8)
    Z, Y, X = (s // 4 for s in lb.shape); nb = Z * Y * X
    bl = lb.reshape(Z, 4, Y, 4, X, 4).transpose(0, 2, 4, 1, 3, 5).reshape(nb, 64)
    pres = np.zeros((nb, K + 1), bool); pres[np.arange(nb)[:, None], bl] = True
    m2 = pres[:, 1:].T.reshape(K, Z, Y, X)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = apply_window(torch.from_numpy(hu).to(dev), {"type": "baseline"}, NORM)[None, None]
        h = st[1](st[0](x))[0]                                                    # [64, Z/2, Y/2, X/2]
        rib = torch.from_numpy(lb > 0).to(dev)[None, None].float()
        keep = F.max_pool3d(F.max_pool3d(rib, 2), 2 * a.dil + 1, 1, a.dil)[0, 0] > 0
        idx = torch.nonzero(keep.flatten())[:, 0]
        f = h.flatten(1)[:, idx].T.contiguous().to(torch.bfloat16)
    tmp = os.path.join(a.cache, cid + ".tmp.npz")
    np.savez(tmp, idx=idx.int().cpu().numpy(), f=f.view(torch.int16).cpu().numpy().view(np.uint16),
             grid=np.array(h.shape[1:]), m2=np.packbits(m2), m2shape=np.array(m2.shape))
    os.rename(tmp, os.path.join(a.cache, cid + ".npz"))
    return len(idx) / keep.numel()


n = 0; t0 = time.time(); fr = []
while not os.path.exists(os.path.join(a.cache, "STOP")):
    have = {f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz") and ".tmp" not in f}
    todo = [f[:-4] for f in os.listdir(a.roi) if f.endswith(".npz") and ".tmp" not in f and f[:-4] not in have and mine(f[:-4])]
    todo.sort(key=lambda c: rank.get(c, 10 ** 9))
    if not todo:
        if a.once or os.path.exists(os.path.join(a.cache, "LAST")): break
        time.sleep(60); continue
    for cid in todo[:200]:
        try: fr.append(one(cid))
        except Exception as e: print("FAIL", cid, repr(e)[:200], flush=True)
        n += 1
        if n % 100 == 0: print(f"{n} done {time.time() - t0:.0f}s keep-frac {np.mean(fr[-100:]):.3f}", flush=True)
        torch.cuda.empty_cache() if n % 50 == 0 else None
print("CACHE_DONE", n, flush=True)
