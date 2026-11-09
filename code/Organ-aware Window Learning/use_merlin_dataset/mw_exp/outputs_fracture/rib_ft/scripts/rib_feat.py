"""frozen per-rib features from the stage-1 cache: Dataset295 stages 2-4 on the cached stage-1 map (+ stage 1 itself),
rib-masked mean & max per rib (24) -> feats/{id}.npy float16 [24, 2*(64+128+256+320)] (row of zeros = rib absent).
usage: rib_feat.py CACHE OUT [--ids order]; polls for new caches until CACHE/LAST or --once."""
import os, sys, time, argparse, numpy as np, torch, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("out"); ap.add_argument("--once", action="store_true")
ap.add_argument("--src", default=os.path.expanduser("~/OWL/mw_exp/src"))
ap.add_argument("--ts", default=os.path.expanduser("~/.totalsegmentator/nnunet/results/Dataset295_TotalSegmentator_part5_ribs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, a.src)
from encoder import NnUNetEncoder
enc = NnUNetEncoder(a.ts); st = enc.encoder.stages; dev = torch.device("cuda"); K = 24


def one(cid):
    z = np.load(os.path.join(a.cache, cid + ".npz"))
    g = [int(x) for x in z["grid"]]; m2 = np.unpackbits(z["m2"])[:int(np.prod(z["m2shape"]))].reshape(z["m2shape"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.zeros(int(np.prod(g)), 64, device=dev, dtype=torch.bfloat16)
        h[torch.from_numpy(z["idx"].astype(np.int64)).to(dev)] = torch.from_numpy(z["f"].view(np.int16)).to(dev).view(torch.bfloat16)
        h = h.T.reshape(1, 64, *g); feats = [h]
        for i in (2, 3, 4): h = st[i](h); feats.append(h)
        mk2 = torch.from_numpy(m2).to(dev)[None].float()
        masks = [F.interpolate(mk2, scale_factor=2, mode="nearest"), mk2, F.max_pool3d(mk2, 2), F.max_pool3d(mk2, 4)]
        out = []; present = (masks[1][0].flatten(1) > 0).sum(1) >= 2
        for f, mk in zip(feats, masks):
            m = mk[0].flatten(1) > 0; f = f[0].flatten(1).float(); mf = m.float()
            mean = (mf @ f.T) / mf.sum(1, keepdim=True).clamp(min=1)
            mx = torch.stack([f[:, m[k]].amax(-1) if present[k] else f.new_zeros(f.shape[0]) for k in range(K)])
            out += [mean, mx]
        v = torch.cat(out, 1) * present[:, None]
    np.save(os.path.join(a.out, cid + ".npy"), v.half().cpu().numpy())


n = 0; t0 = time.time()
while True:
    have = {f[:-4] for f in os.listdir(a.out) if f.endswith(".npy")}
    todo = [f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz") and ".tmp" not in f and f[:-4] not in have]
    if not todo:
        if a.once or os.path.exists(os.path.join(a.cache, "LAST")): break
        time.sleep(60); continue
    for cid in todo:
        try: one(cid)
        except Exception as e: print("FAIL", cid, repr(e)[:150], flush=True)
        n += 1
        if n % 500 == 0: print(f"{n} {time.time() - t0:.0f}s", flush=True)
print("FEAT_DONE", n, flush=True)
