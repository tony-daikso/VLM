"""GS5 stage B (verbatim copy of biliary_improve/scripts/bd5_extract.py, used for the GB crops of gs5_crop.py; 3 regions)
BD5 stage B (GPU, or CPU for a smoke test): porta-hepatis crop-unit embeddings with the frozen TS --fast encoder.
Input : crops from bd5_crop.py (64^3 HU crop centred on the hilar portal vein + 4 region masks).
Per crop: 5 HU windows (windows.yaml, same apply_window as extract.py) -> encoder stages 1..3 -> for each region:
  mean pool  = mask-fraction weighted mean of the stage feature map (as extract.py pool_stage, bbox fallback)
  max pool   = max over stage voxels with mask fraction > 0
  each stage vector L2-normalised, stages concatenated (D=448) -> [mean | max] = 896 per region and window.
Output: out_dir/{id}.npz  emb float16 [4 regions, 5 windows, 896], present bool.
Usage: bd5_extract.py --crops DIR [DIR ...] --out DIR --ids ids.txt [--device cuda|cpu] [--limit N]
       (run from mw_exp/ so that src/ and the config resolve; config_final.yaml gives ts_model_dir)."""
import argparse, sys, os, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(os.environ.get("MW_ROOT", "."))
sys.path.insert(0, str(ROOT / "src"))
from common import load_cfg
from encoder import NnUNetEncoder
from windowing import apply_window

ap = argparse.ArgumentParser()
ap.add_argument("--crops", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ids", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--config", default=str(ROOT / "config_final.yaml")); ap.add_argument("--ts-model-dir", default=None)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
cfg = load_cfg(a.config)
enc = NnUNetEncoder(a.ts_model_dir or cfg["ts_model_dir"], device=a.device)
stages = cfg["stages_used"]; wins = list(cfg["windows"].values())
os.makedirs(a.out, exist_ok=True)
ids = [l.strip() for l in open(a.ids) if l.strip()]
if a.limit:
    ids = ids[:a.limit]


def pool(feat, mask, min_w=1.0):
    """feat [W,C,z,y,x]; mask [1,1,Z,Y,X] -> mean [W,C], max [W,C]"""
    f = [m // s for m, s in zip(mask.shape[2:], feat.shape[2:])]
    w = F.avg_pool3d(mask, kernel_size=f, stride=f)
    if w.sum() < min_w:                                  # tiny region: bbox fallback (as extract.py)
        nz = torch.nonzero(mask[0, 0])
        lo = (nz.min(0).values // torch.tensor(f, device=nz.device)).tolist()
        hi = (nz.max(0).values // torch.tensor(f, device=nz.device)).tolist()
        w = torch.zeros_like(w); w[..., lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1.0
    ff = feat.float()
    mean = (ff * w).sum((2, 3, 4)) / w.sum()
    mx = ff.masked_fill(~(w > 0), -1e4).amax((2, 3, 4))
    return mean, mx


t0 = time.time(); n = 0
with torch.no_grad():
    for sid in ids:
        fo = f"{a.out}/{sid}.npz"
        if os.path.exists(fo):
            continue
        src = next((f"{c}/{sid}.npz" for c in a.crops if os.path.exists(f"{c}/{sid}.npz")), None)
        if src is None:
            continue
        z = np.load(src)
        if not bool(z["present"]):
            np.savez(fo, present=False); continue
        h = torch.from_numpy(z["hu"].astype(np.float32)).to(enc.device)
        R = torch.from_numpy(z["regions"]).to(enc.device).float()          # [4,64,64,64]
        x = torch.stack([apply_window(h, w, enc.norm) for w in wins])[:, None]
        with torch.autocast("cuda", dtype=torch.float16, enabled=(a.device == "cuda")):
            feats = enc.stage_features(x, upto=max(stages))
        emb = np.zeros((R.shape[0], len(wins), 2 * sum(enc.stage_channels(stages))), np.float32)
        for r in range(R.shape[0]):
            if R[r].sum() < 1:
                continue
            ms, xs = [], []
            for s in stages:
                mean, mx = pool(feats[s], R[r][None, None])
                ms.append(F.normalize(mean, dim=1)); xs.append(F.normalize(mx, dim=1))
            emb[r] = torch.cat(ms + xs, 1).cpu().numpy()
        np.savez(fo, present=True, emb=emb.astype(np.float16), region_vox=R.sum((1, 2, 3)).cpu().numpy())
        n += 1
        if n % 200 == 0:
            print(f"{n} cases {time.time() - t0:.0f}s", flush=True)
print("GS5_EXTRACT_DONE", n, f"{time.time() - t0:.0f}s", flush=True)
