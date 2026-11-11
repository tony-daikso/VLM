"""Pillar-0 (YalaLab/Pillar0-AbdomenCT) feature extraction helpers for Merlin abdominal CT.

Preprocessing reimplemented from the official code (verified 2026-11-11):
  * rad-vision-engine (rave) vision_engine/processing/ct_processor.py + pillar-pretrain configs/ct_abdomen.yaml:
      SimpleITK read (native voxel order, no reorientation) -> sitkLinear resample to 1.25 mm iso
      (new_size = round(size*spacing/1.25), same origin/direction, default pixel 0) -> int16
      -> middle 384 slices if z > 384 (start = (Z-384)//2)
      -> in-plane centre crop / pad to 384x384, pad value = volume.min() (rave _center_crop_pad_3d)
  * pillar-pretrain src/trainer/merlin_abd_ct.py load_rve_volume: z pad to 384 (before = diff//2) with -1024
  * pillar-pretrain train_multimodal.py + pillar-finetune engines/classifier.py: 11 channels =
      rve.get_available_windows('CT') = 10 anatomical windows (dict order below) + per-volume minmax,
      each clamp((v - (c - w/2)) / w, 0, 1).  NO mean/std normalisation is applied in the active
      training / finetune code (the 0.289/0.198 normalize path is commented out), so none here.
Model: CLIPMultimodalAtlas vision tower (MultiModalAtlas), global = cat of per-scale token max-pool
(3 x 384 = 1152, un-normalised; the CLIP image embedding is its L2-normalisation).
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path(os.environ.get("PILLAR_WEIGHTS", "/media/daikso/Tony/models/Pillar0-AbdomenCT"))
TARGET = 384
SPACING = 1.25
ZPAD_HU = -1024
MODALITY = "abdomen_ct"

# rave windowing_utils.ANATOMICAL_WINDOWS["CT"] (dict order = channel order), then minmax
WINDOWS = [
    ("lung", -600, 1500), ("mediastinum", 50, 400), ("abdomen", 40, 400), ("liver", 80, 150),
    ("bone", 400, 1800), ("brain", 40, 80), ("subdural", 75, 215), ("stroke", 40, 40),
    ("temporal_bone", 600, 2800), ("soft_tissue", 50, 350),
]
N_LABELS = 118  # TotalSegmentator 'total' v2: 0 = background, 1..117


# ----------------------------------------------------------------------------- CPU preprocessing
def _sitk():
    import SimpleITK as sitk
    return sitk


def resample_ct(img):
    """rave ct_processor._resample_volume (verbatim logic)."""
    sitk = _sitk()
    osp, osz = img.GetSpacing(), img.GetSize()
    new_size = [int(round(osz[i] * osp[i] / SPACING)) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([SPACING] * 3)
    r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(sitk.sitkLinear)
    return r.Execute(img)


def crop_pad_plan(shape_zyx):
    """Return the index plan that maps the resampled (Z,Y,X) array to the 384^3 model grid.
    For each axis: (src_start, src_stop, dst_start). Order of ops in the official code:
    z middle-crop (rave) -> in-plane centre crop/pad (rave) -> z centre-pad (loader)."""
    plan = []
    for ax, n in enumerate(shape_zyx):
        if n > TARGET:
            s = (n - TARGET) // 2
            plan.append((s, s + TARGET, 0))
        else:
            d = TARGET - n
            plan.append((0, n, d // 2))
    return plan


def apply_plan(arr, plan, inplane_pad, z_pad, dtype=None):
    out = np.empty((TARGET,) * 3, dtype=dtype or arr.dtype)
    out[...] = inplane_pad
    (z0, z1, zd), (y0, y1, yd), (x0, x1, xd) = plan
    nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
    # z padding uses a different value than in-plane padding (order: in-plane pad first, then z pad)
    if zd > 0 or nz < TARGET:
        out[:zd] = z_pad
        out[zd + nz:] = z_pad
    out[zd:zd + nz, yd:yd + ny, xd:xd + nx] = arr[z0:z1, y0:y1, x0:x1]
    return out


def preprocess_ct(path):
    """-> (vol int16 [384,384,384] in sitk z,y,x order, resampled sitk image (reference grid), info dict)"""
    sitk = _sitk()
    img = sitk.ReadImage(str(path))
    res = resample_ct(img)
    arr = sitk.GetArrayFromImage(res)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -32768, 32767).astype(np.int16)
    plan = crop_pad_plan(arr.shape)
    (z0, z1, _), (y0, y1, _), (x0, x1, _) = plan
    # rave pads in-plane with min of the z-cropped volume (before in-plane crop)
    inplane_pad = int(arr[z0:z1].min())
    vol = apply_plan(arr, plan, inplane_pad, ZPAD_HU)
    info = dict(orig_size=list(img.GetSize()), orig_spacing=[float(s) for s in img.GetSpacing()],
                res_shape_zyx=list(arr.shape), plan=[list(p) for p in plan], inplane_pad=inplane_pad,
                z_mm_total=float(arr.shape[0] * SPACING), z_mm_kept=float(min(arr.shape[0], TARGET) * SPACING))
    return vol, res, info


def preprocess_mask(mask_path, ref_img, plan):
    """TS-fast 3 mm multilabel mask -> nearest resample onto the 1.25 mm CT grid (physical space,
    handles the RAS/LPS direction difference) -> identical crop/pad (pad 0).
    Returns (mask384 uint8, total_count_1p25 per label [N_LABELS] (whole resampled volume))."""
    sitk = _sitk()
    m = sitk.ReadImage(str(mask_path))
    m = sitk.Cast(m, sitk.sitkUInt8)
    mr = sitk.Resample(m, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    marr = sitk.GetArrayFromImage(mr)
    total = np.bincount(marr.ravel(), minlength=N_LABELS)[:N_LABELS]
    m384 = apply_plan(marr, plan, 0, 0, dtype=np.uint8)
    return m384, total


def cell_counts(m384, f):
    """Per-cell label counts on the (384/f)^3 token grid. Returns dict label -> (cell_idx int32, count uint16)
    for labels present (label > 0). Cell index is raster (z,y,x) on the token grid."""
    g = TARGET // f
    zi = np.arange(TARGET) // f
    cell = (zi[:, None, None] * g * g + zi[None, :, None] * g + zi[None, None, :]).astype(np.int64)
    nz = m384 > 0
    lab = m384[nz].astype(np.int64)
    c = cell[nz]
    key = lab * (g ** 3) + c
    uk, cnt = np.unique(key, return_counts=True)
    out = {}
    labs = uk // (g ** 3)
    cells = uk % (g ** 3)
    for L in np.unique(labs):
        sel = labs == L
        out[int(L)] = (cells[sel].astype(np.int32), cnt[sel].astype(np.uint16))
    return out


def prep_case(cid, ct_path, mask_path):
    """CPU worker: everything except the network. Returns a dict (picklable)."""
    import time
    t0 = time.time()
    vol, ref, info = preprocess_ct(ct_path)
    t1 = time.time()
    out = dict(cid=cid, vol=vol, info=info, t_ct=t1 - t0)
    if mask_path is not None and Path(mask_path).exists():
        plan = [tuple(p) for p in info["plan"]]
        m384, total = preprocess_mask(mask_path, ref, plan)
        crop = np.bincount(m384.ravel(), minlength=N_LABELS)[:N_LABELS]
        out["mask_total"] = total.astype(np.int64)
        out["mask_crop"] = crop.astype(np.int64)
        out["cc64"] = cell_counts(m384, 6)
        out["mask_ok"] = True
    else:
        out["mask_ok"] = False
    out["t_mask"] = time.time() - t1
    return out


# ----------------------------------------------------------------------------- model
def load_model(device="cuda"):
    """Load CLIPMultimodalAtlas from the local weights folder only (no HF download)."""
    import torch
    from safetensors.torch import load_file
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.dont_write_bytecode = True
    pkg = "pillar0_hf"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(WEIGHTS_DIR)]
        sys.modules[pkg] = m
    cfg_mod = importlib.import_module(f"{pkg}.config_clip_mmatlas")
    mdl_mod = importlib.import_module(f"{pkg}.modeling_clip_mmatlas")
    cfg = cfg_mod.CLIPMultimodalAtlasConfig.from_pretrained(str(WEIGHTS_DIR))
    model = mdl_mod.CLIPMultimodalAtlas(cfg)
    sd = load_file(str(WEIGHTS_DIR / "model.safetensors"))
    # HF checkpoint keys are "model.<CustomTextCLIP key>"; load into CustomTextCLIP directly
    assert all(k.startswith("model.") for k in sd)
    sd = {k[len("model."):]: v for k, v in sd.items()}
    res = model.model.load_state_dict(sd, strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    visual = model.model.visual
    visual.eval().to(device)
    del model.model.text
    return visual, missing, unexpected


def window_gpu(vol_t):
    """vol_t: int16/float tensor [D,H,W] on GPU -> bf16 [1, 11, D, H, W] (rve batch_apply_windowing_vectorized)."""
    import torch
    v = vol_t.float()
    out = torch.empty((1, len(WINDOWS) + 1) + tuple(v.shape), device=v.device, dtype=torch.bfloat16)
    for i, (_, c, w) in enumerate(WINDOWS):
        out[0, i] = ((v - (c - w / 2)) / w).clamp_(0, 1)
    mn, mx = v.min(), v.max()
    out[0, len(WINDOWS)] = ((v - mn) / ((mx - mn) + 1e-8)).clamp_(0, 1)
    del v
    return out


def visual_forward(visual, x):
    """Mirror of MultiModalAtlas.forward that also returns per-scale spatial token grids.
    Returns (global [B,1152] float, grids list of [B,384,g,g,g] in input (D,H,W) order)."""
    import torch
    from einops import rearrange
    inp = {MODALITY: x}
    bsz = x.shape[0]
    xs, _ = visual.build_scales(inp)
    mc = visual.model_config["modalities"][MODALITY]
    layout = visual.prepare_multiscale_layout(img_size=mc["image_size"], merge_ratio=mc["merge_ratio"],
                                              local2global=mc["local2global"], patch_size=mc["patch_size"])
    grid_sizes = [l["grid_size"] for l in layout]
    feats = []
    for level, stage in enumerate(visual.atlas_models):
        is_last = len(xs) == 1 or level == len(visual.atlas_models) - 1
        xs = stage(xs, grid_sizes=grid_sizes[level:], multiscale_layout=layout[level:],
                   merge_ratio=mc["merge_ratio"], local2global=mc["local2global"], modality=MODALITY)
        if not is_last:
            feats.append(xs[0])
            xs = xs[1:]
    feats.append(xs[0])
    heads, grids = [], []
    for i, tok in enumerate(feats):
        if math.prod(grid_sizes[i]) > 1:
            tok = rearrange(tok, "(b nw) k c -> b (nw k) c", b=bsz)
        heads.append(tok)
        g0, g1, g2 = layout[i]["grid_size"]
        m0, m1, m2 = layout[i]["window_dims"]
        # windows raster over (d,h,w) of the input grid, tokens raster within window (m0,m1,m2)
        grids.append(rearrange(tok, "b (gd gh gw md mh mw) c -> b c (gd md) (gh mh) (gw mw)",
                               gd=g0, gh=g1, gw=g2, md=m0, mh=m1, mw=m2))
    glob = torch.cat([visual.maxpool(h.transpose(1, 2)).squeeze(2) for h in heads], dim=1)
    return glob, grids


def pool_units(grid64, grid16, cc64, device):
    """Mask-weighted mean of token features per TS label at 64^3 and 16^3.
    Weight of a token = fraction of its cell covered by the label (count / cell volume).
    Returns vec64 [N_LABELS,384], vec16 [N_LABELS,384] (float32 numpy, zeros if absent),
    w64, w16 [N_LABELS] (sum of weights = label volume in cell units, exact for composite merging)."""
    import torch
    C = grid64.shape[1]
    f64 = grid64[0].reshape(C, -1).T.float()  # [262144, C] raster (z,y,x)
    f16 = grid16[0].reshape(C, -1).T.float()  # [4096, C]
    vec64 = np.zeros((N_LABELS, C), np.float32)
    vec16 = np.zeros((N_LABELS, C), np.float32)
    w64 = np.zeros(N_LABELS, np.float64)
    w16 = np.zeros(N_LABELS, np.float64)
    g = 64
    for L, (cells, cnt) in cc64.items():
        if L <= 0 or L >= N_LABELS:
            continue
        ci = torch.from_numpy(cells.astype(np.int64)).to(device)
        w = torch.from_numpy(cnt.astype(np.float32) / 216.0).to(device)
        s = w.sum()
        vec64[L] = ((f64[ci] * w[:, None]).sum(0) / s).cpu().numpy()
        w64[L] = float(s)
        # 16^3 cell of a 64^3 cell: (z//4, y//4, x//4)
        z, y, x = ci // (g * g), (ci // g) % g, ci % g
        c16 = (z // 4) * 256 + (y // 4) * 16 + (x // 4)
        w16c = torch.zeros(4096, device=device).index_add_(0, c16, w / 64.0)
        nzc = torch.nonzero(w16c).squeeze(1)
        ww = w16c[nzc]
        vec16[L] = ((f16[nzc] * ww[:, None]).sum(0) / ww.sum()).cpu().numpy()
        w16[L] = float(ww.sum())
    return vec64, vec16, w64, w16
