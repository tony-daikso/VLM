"""bd15 stage A (Spark CPU worker): 1.5 mm porta-hepatis / hepatoduodenal-ligament / pancreatic-head crop from the ORIGINAL CT.
Same centre rule and region definitions as BD5 (bd5_crop.py, 3 mm), redone at 1.5 mm isotropic:
  3 mm TotalSegmentator total mask (CT-rate, ssh cat) -> centre = centroid of portal-vein voxels within 2 voxels (6 mm) of the
  liver (fallback: PV voxel nearest the liver centroid) -> world (RAS) point via the mask affine.
  Original CT (CT-rate HDD, ssh cat) -> canonical RAS -> sub-box around the centre -> anti-alias Gaussian (only along axes with
  spacing < 1.5 mm) -> trilinear resample onto a 128^3 x 1.5 mm axis-aligned grid (192 mm cube, same FOV as BD5), pad -1024.
  3 mm labels -> nearest-neighbour onto the same grid.
Regions (bool, overlap allowed; distances in mm, = BD5 thresholds):
  r0 porta_ball  : <= 30 mm from the centre
  r1 hdl         : unlabelled voxels <= 24 mm from PV and (<= 30 mm from liver or pancreas)  (hepatoduodenal ligament / CBD)
  r2 hilum_liv   : liver <= 30 mm from PV (central intrahepatic ducts)
  r3 phead       : pancreas/unlabelled <= 9 mm from pancreas and <= 15 mm from duodenum (distal CBD)
  new at 1.5 mm (duct-like voxels, only meaningful at this resolution):
  r4 hdl_fluid   : r1 & -10 <= HU <= 35  (fluid density in the ligament: dilated CBD/CHD)
  r5 hil_dark    : r2 & -10 <= HU <= liver median - 30   (hypodense branching ducts in the hilar liver)
  r6 phead_fluid : r3 & -10 <= HU <= 35  (distal CBD in the pancreatic head)
Output crops/{id}.npz in the Exp N/O ru_extract format: hu int16 [1,128,128,128], regions packbits of bool [1,7,128,128,128],
rshape, present [1], center_world, ct_zooms, plus info line in logs/crop_info_w{wid}.jsonl.
Claims (O_EXCL files in claims/) make the N workers disjoint; STOP file stops after the current case.
Usage: bd15_crop.py --ids ids_prio.csv --maskmap maskmap.tsv --root ~/OWL/bd15 --wid 0 [--max-cases 300] [--threads 2]"""
import argparse, os, sys, time, json, subprocess, gzip
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True); ap.add_argument("--maskmap", required=True); ap.add_argument("--root", required=True)
ap.add_argument("--wid", type=int, default=0); ap.add_argument("--max-cases", type=int, default=300)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--limit-ids", default=None)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(a.threads)
import nibabel as nib
from scipy import ndimage

CT_HOST = "daikso@192.168.0.189"
CT_DIR = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
C = dict(liver=5, panc=7, duod=19, pv=64)
S, SP = 128, 1.5
R = Path(a.root).expanduser()
for d in ("crops", "claims", "ct", "logs"):
    (R / d).mkdir(parents=True, exist_ok=True)
mm = dict(l.rstrip("\n").split("\t") for l in open(a.maskmap) if "\t" in l)


def claim(cid):
    try:
        os.close(os.open(R / "claims" / cid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def fetch(remote, dst):
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["ssh", "-c", "aes128-gcm@openssh.com", "-o", "BatchMode=yes", CT_HOST, f"cat {remote}"],
                       stdout=fh, check=True, timeout=900)
    tmp.rename(dst); return dst


def centre_world(mpath):
    m3 = nib.load(str(mpath)); lab = np.asarray(m3.dataobj).astype(np.uint8)        # (x,y,z) index order
    L, PV = lab == C["liver"], lab == C["pv"]
    if PV.sum() < 20 or L.sum() < 500:
        return None, m3, lab
    dL = ndimage.distance_transform_edt(~L)
    hil = PV & (dL <= 2)
    if hil.sum() >= 5:
        cen = np.argwhere(hil).mean(0)
    else:
        lc = np.argwhere(L).mean(0); pv = np.argwhere(PV)
        cen = pv[np.argmin(((pv - lc) ** 2).sum(1))].astype(float)
    return (m3.affine @ np.r_[cen, 1])[:3], m3, lab


def grid_world(cw):
    off = (np.arange(S) - (S - 1) / 2) * SP
    X, Y, Z = np.meshgrid(cw[0] + off, cw[1] + off, cw[2] + off, indexing="ij")    # RAS, (x,y,z) axis order
    return np.stack([X, Y, Z, np.ones_like(X)], 0).reshape(4, -1)


def one(cid):
    t0 = time.time()
    mp = mm.get(cid)
    if mp is None:
        return dict(status="nomask")
    LOCAL_M = os.path.exists(mp)                       # A6000: 3 mm masks are local (outputs/masks, masks3_extra) -> read in place, never delete
    fm = Path(mp) if LOCAL_M else fetch(mp, R / "ct" / f"{cid}_m3.nii.gz")
    cw, m3, lab3 = centre_world(fm)
    if cw is None:
        np.savez(R / "crops" / f".{cid}.tmp.npz", present=np.array([False]), rshape=np.array([1, 7, S, S, S]))
        os.replace(R / "crops" / f".{cid}.tmp.npz", R / "crops" / f"{cid}.npz"); (None if LOCAL_M else fm.unlink(missing_ok=True))
        return dict(status="nopv")
    t1 = time.time()
    fc = fetch(f"{CT_DIR}/{cid}.nii.gz", R / "ct" / f"{cid}.nii.gz"); t2 = time.time()
    img = nib.as_closest_canonical(nib.load(str(fc)))
    zooms = np.array(img.header.get_zooms()[:3], float)
    G = grid_world(cw)
    vox = np.linalg.inv(img.affine) @ G                                            # canonical voxel coords of the grid
    lo = np.floor(vox[:3].min(1)).astype(int) - 4; hi = np.ceil(vox[:3].max(1)).astype(int) + 5
    lo = np.maximum(lo, 0); hi = np.minimum(hi, img.shape[:3])
    if np.any(hi - lo < 2):
        fc.unlink(missing_ok=True); (None if LOCAL_M else fm.unlink(missing_ok=True)); return dict(status="outside")
    sub = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    sig = np.array([0.5 * np.sqrt(max((SP / z) ** 2 - 1, 0)) if z < SP else 0 for z in zooms])
    if sig.max() > 0:
        sub = ndimage.gaussian_filter(sub, sig, mode="nearest")
    hu = ndimage.map_coordinates(sub, vox[:3] - lo[:, None], order=1, mode="constant", cval=-1024).reshape(S, S, S)
    hu = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)
    v3 = np.linalg.inv(m3.affine) @ G
    lab = ndimage.map_coordinates(lab3, v3[:3], order=0, mode="constant", cval=0).reshape(S, S, S).astype(np.uint8)
    t3 = time.time()
    # (x,y,z) RAS -> (z,y,x) like the 3 mm prep / nnU-Net order used by the encoder pipelines
    hu = hu.transpose(2, 1, 0).copy(); lab = lab.transpose(2, 1, 0).copy()
    Lc, Pc, Dc, PVc = (lab == C["liver"]), (lab == C["panc"]), (lab == C["duod"]), (lab == C["pv"])
    zz, yy, xx = np.indices((S, S, S)); c0 = (S - 1) / 2
    r0 = ((zz - c0) ** 2 + (yy - c0) ** 2 + (xx - c0) ** 2) * SP ** 2 <= 30 ** 2
    big = np.full(hu.shape, 999.)
    edt = lambda M: ndimage.distance_transform_edt(~M, sampling=SP) if M.any() else big
    dpv, dl, dp, dd = edt(PVc), edt(Lc), edt(Pc), edt(Dc)
    free = lab == 0
    r1 = free & (dpv <= 24) & ((dl <= 30) | (dp <= 30))
    r2 = Lc & (dpv <= 30)
    r3 = (Pc | free) & (dp <= 9) & (dd <= 15)
    fl = (hu >= -10) & (hu <= 35)
    lm = float(np.median(hu[Lc])) if Lc.sum() > 100 else 60.0
    r4 = r1 & fl; r5 = r2 & (hu >= -10) & (hu <= lm - 30); r6 = r3 & fl
    Rg = np.stack([r0, r1, r2, r3, r4, r5, r6])[None]
    tmp = R / "crops" / f".{cid}.tmp.npz"
    np.savez_compressed(tmp, present=np.array([True]), hu=hu[None], regions=np.packbits(Rg.ravel()), rshape=np.array(Rg.shape),
                        center_world=cw, ct_zooms=zooms, liver_med=lm)
    os.replace(tmp, R / "crops" / f"{cid}.npz")
    fc.unlink(missing_ok=True); (None if LOCAL_M else fm.unlink(missing_ok=True))
    return dict(status="ok", vox=[int(x) for x in Rg[0].sum((1, 2, 3))], zooms=zooms.round(3).tolist(), liver_med=lm,
                t_mask=round(t1 - t0, 2), t_ct=round(t2 - t1, 2), t_resample=round(t3 - t2, 2), t_regions=round(time.time() - t3, 2))


ids = [l.split(",")[0] for l in open(a.ids).read().splitlines()[1:] if l.strip()]
if a.limit_ids:
    keep = set(l.strip() for l in open(a.limit_ids) if l.strip()); ids = [i for i in ids if i in keep]
info = open(R / "logs" / f"crop_info_w{a.wid}.jsonl", "a"); n = 0
for cid in ids:
    if (R / "STOP").exists() or n >= a.max_cases:
        break
    if (R / "crops" / f"{cid}.npz").exists() or not claim(cid):
        continue
    t = time.time()
    try:
        r = one(cid)
    except Exception as e:  # noqa: BLE001
        r = dict(status="err", err=repr(e)[:200])
        for f in (R / "ct").glob(f"{cid}*"):
            f.unlink(missing_ok=True)
        with open(R / "failed.txt", "a") as fh:
            fh.write(f"{cid}\t{r['err']}\n")
    r.update(study_id=cid, wid=a.wid, t=round(time.time() - t, 2), ts=time.strftime("%H:%M:%S"))
    info.write(json.dumps(r) + "\n"); info.flush(); n += 1
else:
    if not (R / "STOP").exists():
        (R / f"CROP_DONE_w{a.wid}").touch()
print("CROP_WORKER_EXIT", a.wid, n, flush=True)
