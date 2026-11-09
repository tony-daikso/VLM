"""Exp O stage B (identical to Exp N nu_extract.py)
Exp N stage B (GPU; copy of Exp M ru_extract.py, region count taken from the crop file)
Exp M stage B (GPU): region-crop-unit embeddings with the frozen TS --fast (Dataset297) encoder.
Generalises rlq_extract.py / bd5_extract.py to K crops per case (ru_crop.py output) and optional extra windows.
Per crop: windows (windows.yaml 5 windows + --extra 'name:L:W' ...) -> encoder stages 1..3 -> per region
  mean pool (mask-fraction weighted, bbox fallback for tiny regions) and max pool (voxels with mask fraction > 0),
  each stage vector L2-normalised, stages concatenated (448) -> [mean | max] = 896.
Output out/{id}.npz: emb float16 [K, R, W, 896], present bool [K], region_vox int [K, R].
--consume: process every *.npz in the crops dir (loop until --done-flag file exists and the dir is empty), delete each crop
after its embedding is written. VRAM capped with --vram-gb (torch per-process memory fraction).
Usage: ru_extract.py --crops DIR --out DIR --ts-model-dir DIR --windows windows.yaml [--extra air:-700:1000]
       [--consume --done-flag FILE] [--ids ids.txt] [--device cuda] [--vram-gb 10] [--threads 4]"""
import argparse, sys, os, time, glob, yaml, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(os.environ.get("MW_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
from encoder import NnUNetEncoder
from windowing import apply_window

ap = argparse.ArgumentParser()
ap.add_argument("--crops", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ts-model-dir", required=True); ap.add_argument("--windows", default=str(ROOT / "windows.yaml"))
ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--ids", default=None)
ap.add_argument("--consume", action="store_true"); ap.add_argument("--done-flag", default=None)
ap.add_argument("--device", default="cuda"); ap.add_argument("--vram-gb", type=float, default=10.0)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
if a.device == "cuda":
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / tot), 0)
enc = NnUNetEncoder(a.ts_model_dir, device=a.device)
stages = [1, 2, 3]
wins = list(yaml.safe_load(open(a.windows))["windows"].values())
for e in a.extra:
    nm, L, W = e.split(":"); wins.append({"level": float(L), "width": float(W)})
os.makedirs(a.out, exist_ok=True)
D2 = 2 * sum(enc.stage_channels(stages))


def pool(feat, mask, min_w=1.0):
    """feat [W,C,z,y,x]; mask [1,1,Z,Y,X] -> mean [W,C], max [W,C]"""
    f = [m // s for m, s in zip(mask.shape[2:], feat.shape[2:])]
    w = F.avg_pool3d(mask, kernel_size=f, stride=f)
    if w.sum() < min_w:
        nz = torch.nonzero(mask[0, 0])
        lo = (nz.min(0).values // torch.tensor(f, device=nz.device)).tolist()
        hi = (nz.max(0).values // torch.tensor(f, device=nz.device)).tolist()
        w = torch.zeros_like(w); w[..., lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1.0
    ff = feat.float()
    mean = (ff * w).sum((2, 3, 4)) / w.sum()
    mx = ff.masked_fill(~(w > 0), -1e4).amax((2, 3, 4))
    return mean, mx


def do(src, fo):
    z = np.load(src)
    pres = np.asarray(z["present"], bool)
    if "hu" not in z or not pres.any():
        nr = int(z["rshape"][1]) if "rshape" in z else 4
        np.savez(fo + ".tmp.npz", present=pres, emb=np.zeros((len(pres), nr, len(wins), D2), np.float16),
                 region_vox=np.zeros((len(pres), nr), np.int32)); os.replace(fo + ".tmp.npz", fo); return
    H = z["hu"]; rs = tuple(z["rshape"]); R = np.unpackbits(z["regions"])[:int(np.prod(rs))].reshape(rs)
    K, Rn = rs[0], rs[1]
    emb = np.zeros((K, Rn, len(wins), D2), np.float32)
    for k in range(K):
        if not pres[k]:
            continue
        h = torch.from_numpy(H[k].astype(np.float32)).to(enc.device)
        x = torch.stack([apply_window(h, w, enc.norm) for w in wins])[:, None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(a.device == "cuda")):
            feats = enc.stage_features(x, upto=max(stages))
        Rk = torch.from_numpy(R[k]).to(enc.device).float()
        for r in range(Rn):
            if Rk[r].sum() < 1:
                continue
            ms, xs = [], []
            for s in stages:
                mean, mx = pool(feats[s], Rk[r][None, None])
                ms.append(F.normalize(mean, dim=1)); xs.append(F.normalize(mx, dim=1))
            emb[k, r] = torch.cat(ms + xs, 1).cpu().numpy()
    np.savez(fo + ".tmp.npz", present=pres, emb=emb.astype(np.float16), region_vox=R.sum((2, 3, 4)).astype(np.int32))
    os.replace(fo + ".tmp.npz", fo)


t0 = time.time(); n = 0
want = set(l.strip() for l in open(a.ids)) if a.ids else None
while True:
    todo = sorted(f for f in glob.glob(f"{a.crops}/*.npz") if ".tmp" not in f)
    if want is not None:
        todo = [f for f in todo if os.path.basename(f)[:-4] in want]
    for src in todo:
        sid = os.path.basename(src)[:-4]; fo = f"{a.out}/{sid}.npz"
        if not os.path.exists(fo):
            try:
                do(src, fo); n += 1
            except Exception as e:  # noqa: BLE001
                print("ERR", sid, repr(e)[:120], flush=True); continue
        if a.consume:
            os.remove(src)
        if n and n % 500 == 0:
            print(f"{n} cases {time.time() - t0:.0f}s", flush=True)
    if not a.consume:
        break
    if a.done_flag and os.path.exists(a.done_flag) and not glob.glob(f"{a.crops}/*.npz"):
        break
    time.sleep(20)
print("RU_EXTRACT_DONE", n, f"{time.time() - t0:.0f}s", flush=True)
