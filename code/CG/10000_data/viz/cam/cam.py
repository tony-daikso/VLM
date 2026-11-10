"""Layer-2 OWL attribution: exact voxel CAM inside the organ units for E3_late (expB).

Why plain CAM is exact here
  embedding  E[u,w] = sum_t (n_t/N_u) * concat_s  v_ts / ||v_ts||,   v_ts = sum_j om_tsj f_tsj / sum_j om_tsj
             (f_tsj = frozen TS-fast encoder stage-s feature at stage voxel j of tile t, om = organ fraction,
              stages 1..3, D = 64+128+256 = 448; extract.py / extract_bones.py, bone_pool = mean)
  head       per finding, per window a linear head on standardized E, logits mixed by a fixed gate
             -> unit score s_u = sum_w G_fw [ w_fw . (E[u,w]-mu)/sd ] + const   (linear in E)
             mean findings: logit = mean over present units of s_u ; fracture (MIL): logit = LSE_site max_u s_u
  => s_u = sum_t (n_t/N_u) * mean_{x in organ, tile t} Q_t(x) + const, with the local score
     Q_t(x) = sum_w sum_s  (h_ws . f_ts(x)) / ||v_ts||,   h_ws = G_fw * w_fw,s / sd_w,s
  The per-tile norms ||v_ts|| are constants of the case, so this is an exact linear decomposition (CAM),
  no gradients needed. The voxel map value = Q(x) + const ("local logit"): its organ-weighted mean equals
  the unit's logit contribution, > 0 means "this voxel pushes the finding towards positive".

Per (EXAMID, finding): recompute encoder features only for the tiles of the finding's units, all windows,
upsample each stage map (trilinear) to the 3 mm prep grid, sum, mix windows with the gate, average over the
3 seeds; 0 outside the units. Writes cam_heatmap.nii.gz (float32, prep 3 mm canonical grid + prep affine,
same grid as make_itk.py ct.nii.gz) + cam_meta.json (unit scores, identity checks, hotspot).

usage: python cam.py --cache <dir with prep/ masks/ patches/ emb/ emb_bones/ E3_late.pt> --ids ids.txt
                     --finding fracture --out <root> [--device mps|cpu|cuda] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "remote" / "ctrate" / "src"
sys.path.insert(0, str(SRC))
from encoder import NnUNetEncoder  # noqa: E402
from extract import crop, window_key  # noqa: E402
from patches import tiles_for, units_from_labels  # noqa: E402
from windowing import apply_window  # noqa: E402

TS_MODEL = Path.home() / ".totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/" \
    "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
DIS_ZH = {"fracture": "骨折", "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤",
          "gallstones": "膽結石", "splenomegaly": "脾腫大", "ascites": "腹水", "hepatic_steatosis": "脂肪肝",
          "surgically_absent_gallbladder": "膽囊切除", "biliary_ductal_dilation": "膽管擴張",
          "atherosclerosis": "動脈粥狀硬化", "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞",
          "appendicitis": "闌尾炎", "free_air": "腹腔游離氣體"}
CFG = dict(stages_used=[1, 2, 3], min_stage_weight=1.0, tile_overlap=0.25, pad_hu=-1024, min_organ_voxels=100)


# ---------------------------------------------------------------- head -> per-unit linear weights
def head_weights(ck, finding):
    """-> units [list of unit idx], H [U_f, W, D] (already / sd, gate-mixed, seed-mean), C0 [U_f], Cf [U_f]:
    unit const = C0 + frac * Cf (frac = fraction of the finding's units present in the case)."""
    meta = ck["meta"]
    f = meta["findings"].index(finding)
    units = meta["finding_units"][f]
    mil = meta["finding_sites"][f] is not None
    mu, sd = ck["std_mu"].double(), ck["std_sd"].double()           # [U,W,D]
    Hs, Cs, Fs = [], [], []
    for st in ck["state_dicts"]:
        sm = torch.softmax(st["gate"].double(), -1)                  # [U,W]
        M = torch.zeros(sm.shape[0], dtype=torch.float64)
        M[units] = 1.0
        G = (M @ sm) / M.sum()                                       # [W] finding gate (mean over its units)
        H = torch.zeros(len(units), *mu.shape[1:], dtype=torch.float64)
        C = torch.zeros(len(units), dtype=torch.float64)
        Cf = torch.zeros(len(units), dtype=torch.float64)
        for i, u in enumerate(units):
            if mil:
                fi = [k for k, x in enumerate(meta["finding_sites"]) if x is not None].index(f)
                site = next(k for k, us in enumerate(meta["finding_sites"][f]) if u in us)
                wv = st["ws"][fi, site].double()                     # [W, D+1]
                bias = st["ub"][fi, u].double()
                bw = torch.zeros(wv.shape[0], dtype=torch.float64)
            else:
                wv = st["w"][f].double()                             # [W, D+1]
                bias = 0.0
                bw = st["b"][f].double()                             # [W]
            h = G[:, None] * wv[:, :-1] / sd[u]                      # [W,D]
            H[i] = h
            C[i] = -(h * mu[u]).sum() + (G * bw).sum() + bias
            Cf[i] = (G * wv[:, -1]).sum()
        Hs.append(H)
        Cs.append(C)
        Fs.append(Cf)
    return units, torch.stack(Hs).mean(0), torch.stack(Cs).mean(0), torch.stack(Fs).mean(0), mil


def unit_score_from_emb(E_uw, H_u, C_u):
    """E_uw [W,D] raw embedding -> unit score (same formula as the head)."""
    return float((H_u * torch.as_tensor(E_uw, dtype=torch.float64)).sum() + C_u)


# ---------------------------------------------------------------- data
def load_case(cache, cid, meta):
    prep = np.load(cache / "prep" / f"{cid}.npz")
    hu, aff, units_main = prep["hu"], prep["affine"], prep["units"]
    info = json.loads((cache / "patches" / f"{cid}.json").read_text())
    return hu, aff, units_main, info


def bone_units_volume(cache, cid, hu, aff):
    from totalsegmentator.map_to_binary import class_map
    lookup = {v: k for k, v in class_map["total"].items()}
    msk = nib.load(str(cache / "masks" / f"{cid}.nii.gz"))
    if msk.shape != hu.shape[::-1] or not np.allclose(msk.affine, aff, atol=1e-3):
        from nibabel.processing import resample_from_to
        from totalsegmentator.alignment import as_closest_canonical
        msk = resample_from_to(as_closest_canonical(msk), (hu.shape[::-1], aff), order=0)
    lab = np.asanyarray(msk.dataobj).astype(np.int32).transpose(2, 1, 0)
    om = yaml.safe_load(open(HERE / "organ_map_bones.yaml"))
    return units_from_labels({"organ_map": om}, lab, lookup), list(om.keys())


def bone_tiles(units, k, P):
    idx = np.argwhere(units == k + 1)
    if len(idx) < CFG["min_organ_voxels"]:
        return []
    lo, hi, cen = idx.min(0), idx.max(0), idx.mean(0)
    tl = []
    for s in tiles_for(lo, hi, cen, P, CFG["tile_overlap"]):
        sl = tuple(slice(max(s[a], 0), s[a] + P[a]) for a in range(3))
        n_in = int((units[sl] == k + 1).sum())
        if n_in:
            tl.append({"start": s, "n_vox": n_in})
    return tl


# ---------------------------------------------------------------- CAM for one unit
@torch.no_grad()
def unit_cam(enc, hu, unit_vol, label, tiles, wins, Hk, P, dev, mode="trilinear"):
    """wins: list of (window dict) actually encoded; Hk [K,D] gate-mixed head weights for those windows.
    Returns local-score volume (float32, full grid, NaN outside tiles), recomputed embedding [K,D], stats."""
    stages = CFG["stages_used"]
    dims = enc.stage_channels(stages)
    off = np.cumsum([0] + dims)
    acc = np.zeros(hu.shape, np.float64)
    cnt = np.zeros(hu.shape, np.float64)
    E = torch.zeros(len(wins), sum(dims), dtype=torch.float64)
    ident_num, nsum, n_fb = 0.0, 0, 0
    Hk_d = Hk.to(dev, torch.float32)
    for t in tiles:
        s0 = t["start"]
        h = torch.from_numpy(crop(hu, s0, P, CFG["pad_hu"])).to(dev)
        m = torch.from_numpy(crop(unit_vol, s0, P, 0) == label).float().to(dev)[None, None]
        x = torch.stack([apply_window(h, w, enc.norm) for w in wins])[:, None]
        feats = enc.stage_features(x, upto=max(stages))
        Q = torch.zeros(P, device=dev)
        parts = []
        for j, s in enumerate(stages):
            fm = feats[s].float()                                        # [K,C,z,y,x]
            fct = [a // b for a, b in zip(P, fm.shape[2:])]
            om = F.avg_pool3d(m, kernel_size=fct, stride=fct)             # [1,1,z,y,x]
            if float(om.sum()) < CFG["min_stage_weight"]:                 # bbox fallback (as extract.pool_stage)
                n_fb += 1
                nz = torch.nonzero(m[0, 0])
                ft = torch.tensor(fct, device=nz.device)
                lo, hi = (nz.min(0).values // ft).tolist(), (nz.max(0).values // ft).tolist()
                om = torch.zeros_like(om)
                om[..., lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = 1.0
            v = (fm * om).sum((2, 3, 4)) / om.sum()                       # [K,C]
            nrm = v.norm(dim=1)                                          # [K]
            parts.append(v / nrm[:, None])
            hs = Hk_d[:, off[j]:off[j + 1]]                              # [K,C]
            q = torch.einsum("kczyx,kc->kzyx", fm, hs / nrm[:, None]).sum(0)   # [z,y,x] local score, all windows
            Q += F.interpolate(q[None, None], size=P, mode=mode,
                               **({} if mode == "nearest" else {"align_corners": False}))[0, 0]
        E += torch.cat(parts, 1).cpu().double() * t["n_vox"]
        nsum += t["n_vox"]
        Qn = Q.cpu().numpy().astype(np.float64)
        mk = m[0, 0].cpu().numpy() > 0
        ident_num += Qn[mk].sum()
        # paste into the volume (crop may hang off the edge)
        src = tuple(slice(max(a, 0) - a, min(a + p, n) - a) for a, p, n in zip(s0, P, hu.shape))
        dst = tuple(slice(max(a, 0), min(a + p, n)) for a, p, n in zip(s0, P, hu.shape))
        acc[dst] += Qn[src]
        cnt[dst] += 1
    E /= nsum
    vol = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return vol, E, dict(n_tiles=len(tiles), n_fallback=n_fb, cam_mean_tilesum=ident_num / nsum)


# ---------------------------------------------------------------- mask-extended response (cam_ext)
def write_ext(d, cid, finding, ext_units, cam, lab, shape, aff, P, dt):
    """cam_ext: every voxel covered by >= 1 tile of a finding unit gets that unit's tile-wide response
    (head weights applied to the unmasked features, same 1/||v_ts|| and constant as the exact CAM); where tiles of
    several units overlap, the unit whose mask is nearest (EDT) wins. Not covered by any tile -> 0.
    cam_dist_mm: EDT (mm) to the union of the finding's unit masks (0 inside)."""
    from scipy.ndimage import distance_transform_edt
    sp = 3.0
    best = np.full(shape, np.inf, np.float32)
    ext = np.zeros(shape, np.float32)
    covered = np.zeros(shape, bool)
    union = np.zeros(shape, bool)
    urows = []
    for name, m, vol, tiles in ext_units:
        du = distance_transform_edt(~m, sampling=sp).astype(np.float32)
        ok = np.isfinite(vol)
        take = ok & (du < best)
        ext[take] = vol[take]
        best[take] = du[take]
        covered |= ok
        union |= m
        idx = np.argwhere(m)
        lo, hi = idx.min(0), idx.max(0)
        st = np.array([t["start"] for t in tiles])
        clo = np.maximum(st.min(0), 0)
        chi = np.minimum(st.max(0) + np.array(P) - 1, np.array(shape) - 1)
        ax = ["z(S-I)", "y(A-P)", "x(L-R)"]
        urows.append(dict(unit=name, n_tiles=len(tiles), tile_size_vox_zyx=P,
                          bbox_zyx=[lo.tolist(), hi.tolist()], tile_cover_zyx=[clo.tolist(), chi.tolist()],
                          bbox_expand_mm={ax[k]: [float((lo[k] - clo[k]) * sp), float((chi[k] - hi[k]) * sp)]
                                          for k in range(3)},
                          max_reach_mm=float(du[ok].max()),
                          full_cover_mm=float(du[~ok].min()) if (~ok).any() else None))
    dist = distance_transform_edt(~union, sampling=sp).astype(np.float32)
    inside = lab > 0
    diff = np.abs(ext[inside] - cam[inside])
    old = d / "cam_heatmap.nii.gz"
    diff_old = None
    if old.exists():
        o = np.asanyarray(nib.load(str(old)).dataobj).transpose(2, 1, 0)
        diff_old = float(np.abs(ext[inside] - o[inside]).max())
    for nm, arr, desc in (("cam_ext", ext, f"OWL CAM ext {finding} local logit"),
                          ("cam_dist_mm", dist, "EDT mm to finding unit masks")):
        img = nib.Nifti1Image(arr.transpose(2, 1, 0), aff)
        img.header.set_data_dtype(np.float32)
        img.header["descrip"] = desc[:79]
        nib.save(img, str(d / f"{nm}.nii.gz"))
    ring = lambda a, b: covered & (dist > a) & (dist <= b)
    json.dump(dict(
        id=cid, finding=finding, finding_zh=DIS_ZH.get(finding),
        meaning=("cam_ext: inside the unit masks identical to cam_heatmap (exact CAM, local logit). OUTSIDE the masks it is "
                 "the response of the same linear head (G*w/sd, same per-tile 1/||v_ts|| and constant) to features at "
                 "positions the model did NOT use (mask weight 0) - i.e. 'what the head would say if the organ looked "
                 "like this', NOT a faithful attribution of the prediction."),
        tile_scope=("Option A: full tile range kept (no distance cut). Tiles are the 112x112x128-voxel (336x336x384 mm) "
                    "encoder patches placed around each unit (organ <= patch: one patch centred on the organ; larger: "
                    "25 % overlapping grid), so reach beyond the mask differs per unit and per direction - see "
                    "units[].bbox_expand_mm [low side, high side] per axis, max_reach_mm, full_cover_mm (every voxel "
                    "within this EDT distance of the mask is covered)."),
        overlap_rule="voxel covered by tiles of several finding units -> unit with the nearest mask (EDT); ties -> first unit",
        outside_tiles="0 (not NaN); use covered fraction / cam_dist_mm + full_cover_mm to know where 0 means 'no data'",
        cam_dist_mm="float32 EDT (mm, 3 mm grid) to the union of the finding's unit masks, 0 inside",
        mask_check=dict(max_abs_diff_vs_cam_in_run=float(diff.max()) if diff.size else None,
                        max_abs_diff_vs_saved_cam_heatmap=diff_old),
        covered_frac_by_dist={f"{a}-{b}mm": float(ring(a, b).sum() / max(((dist > a) & (dist <= b)).sum(), 1))
                              for a, b in ((0, 10), (10, 30), (30, 60), (60, 120))},
        units=urows, seconds=round(dt, 1),
        grid="prep 3mm canonical (RAS), affine from prep npz (= OWL_ITK ct.nii.gz)"),
        open(d / "cam_ext_meta.json", "w"), indent=1, ensure_ascii=False)
    print(f"{cid} {finding} ext maskdiff={diff.max() if diff.size else 0:.2e} vs_saved={diff_old} "
          f"reach={[round(r['max_reach_mm']) for r in urows][:4]}", flush=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ids", required=True, help="file with EXAMIDs, or comma list")
    ap.add_argument("--finding", required=True)
    ap.add_argument("--out", required=True, help="root; writes <out>/<中文病名>/<id>/cam_heatmap.nii.gz")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--upsample", default="trilinear", choices=["trilinear", "nearest"],
                    help="nearest = exact identity (organ mean of map == unit score), blocky; trilinear = smooth")
    ap.add_argument("--ext", action="store_true",
                    help="write cam_ext.nii.gz (unmasked tile-wide response, nearest finding unit), cam_dist_mm.nii.gz, "
                         "cam_ext_meta.json; does NOT rewrite cam_heatmap.nii.gz (compares against it instead)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cache, outroot = Path(a.cache), Path(a.out)
    ids = [l.strip() for l in open(a.ids)] if os.path.exists(a.ids) else a.ids.split(",")
    ids = [i for i in ids if i]
    ck = torch.load(cache / "E3_late.pt", map_location="cpu", weights_only=False)
    meta = ck["meta"]
    units, H, C0, Cf, mil = head_weights(ck, a.finding)
    names = meta["units"]
    wkeys = meta["window_keys"]
    wdefs = yaml.safe_load(open(HERE / "windows.yaml"))["windows"]
    key2def = {window_key(n, w): w for n, w in wdefs.items()}
    dev = torch.device(a.device)
    enc = NnUNetEncoder(TS_MODEL, device="cpu")
    enc.network.to(dev)
    enc.device = dev
    P = list(enc.patch_size)
    for cid in ids:
        t0 = time.time()
        hu, aff, units_main, info = load_case(cache, cid, meta)
        bvol, bnames = None, None
        # presence of the finding's units from the cached embeddings (= what the head saw)
        pr = []
        for u in units:
            nm = names[u]
            dd, kk = ("emb_bones", meta["bone_window_keys"][0]) if nm in meta["bone_units"] else ("emb", wkeys[0])
            z = np.load(cache / dd / kk / f"{cid}.npz")
            pr.append(bool(z["organ_present"][list(z["organ_names"]).index(nm)]))
        frac = float(np.mean(pr))
        C = C0 + frac * Cf
        cam = np.zeros(hu.shape, np.float32)
        lab = np.zeros(hu.shape, np.uint8)
        rows = []
        ext_units = []                                                  # (unit name, label mask, response vol incl. const, tiles)
        for i, u in enumerate(units):
            name = names[u]
            is_bone = name in meta["bone_units"]
            if is_bone:
                if bvol is None:
                    bvol, bnames = bone_units_volume(cache, cid, hu, aff)
                label = bnames.index(name) + 1
                vol_u, tiles = bvol, bone_tiles(bvol, label - 1, P)
                keys_m = meta["bone_window_keys"]                   # model window -> cached window key
                emb_dir, emb_key = "emb_bones", meta["bone_emb_key"]
            else:
                ul = list(yaml.safe_load(open(HERE / "organ_map.yaml")).keys())
                label = ul.index(name) + 1
                vol_u, tiles = units_main, (info["units"][name].get("tiles", []) if info["units"][name]["present"] else [])
                keys_m = wkeys
                emb_dir, emb_key = "emb", "emb"
            if not tiles:
                rows.append(dict(unit=name, present=False))
                continue
            kuniq = list(dict.fromkeys(keys_m))                         # windows actually encoded
            Hk = torch.zeros(len(kuniq), H.shape[-1], dtype=torch.float64)
            for w, k in enumerate(keys_m):
                Hk[kuniq.index(k)] += H[i, w]
            tu = time.time()
            vol, E, st = unit_cam(enc, hu, vol_u, label, tiles, [key2def[k] for k in kuniq], Hk, P, dev, a.upsample)
            inside = vol_u == label
            loc = vol[inside] + C[i].item()
            cam[inside] = loc.astype(np.float32)
            lab[inside] = u + 1
            if a.ext:
                ext_units.append((name, inside, (vol + C[i].item()).astype(np.float32), tiles))
            # identity checks: (a) score from recomputed E, (b) from cached emb, (c) mean of the voxel map
            s_rec = float((Hk * E).sum() + C[i])
            cached = {k: np.load(cache / emb_dir / k / f"{cid}.npz") for k in kuniq}
            cn = list(next(iter(cached.values()))["organ_names"])
            Ec = torch.stack([torch.as_tensor(cached[k][emb_key][cn.index(name)].astype(np.float64)) for k in kuniq])
            s_cache = float((Hk * Ec).sum() + C[i])
            cos = float(F.cosine_similarity(E.float(), Ec.float(), dim=1).min())
            rows.append(dict(unit=name, present=True, score_cached_emb=s_cache, score_recomputed_emb=s_rec,
                             score_voxelmap_mean=float(loc.mean()), score_tile_identity=st["cam_mean_tilesum"] + C[i].item(),
                             emb_cos_min=cos, n_vox=int(inside.sum()), seconds=round(time.time() - tu, 1), **st))
        # finding logit from cached-emb unit scores (head semantics) for reference
        pres = [r for r in rows if r.get("present")]
        if mil:
            sites = meta["finding_sites"][meta["findings"].index(a.finding)]
            logit = None  # MIL: per-unit max is what we report
        else:
            logit = float(np.mean([r["score_cached_emb"] for r in pres]))
        nz = np.argwhere(lab > 0)
        hot = None
        if len(nz):
            vals = cam[lab > 0]
            k = int(np.argmax(vals))
            zyx = nz[k]
            hot = dict(zyx=zyx.tolist(), unit=names[int(lab[tuple(zyx)]) - 1], value=float(vals[k]),
                       ras_mm=(aff @ np.r_[zyx[::-1], 1])[:3].round(1).tolist())
        d = outroot / DIS_ZH.get(a.finding, a.finding) / cid
        d.mkdir(parents=True, exist_ok=True)
        if a.ext:
            write_ext(d, cid, a.finding, ext_units, cam, lab, hu.shape, aff, P, time.time() - t0)
            continue
        img = nib.Nifti1Image(cam.transpose(2, 1, 0), aff)
        img.header.set_data_dtype(np.float32)
        img.header["descrip"] = f"OWL CAM {a.finding} local logit"
        nib.save(img, str(d / ("cam_heatmap.nii.gz" if a.upsample == "trilinear" else f"cam_heatmap_{a.upsample}.nii.gz")))
        dt = time.time() - t0
        json.dump(dict(id=cid, finding=a.finding, finding_zh=DIS_ZH.get(a.finding), method="exact linear CAM",
                       units=rows, finding_logit_mean_units=logit, hotspot=hot, seconds=round(dt, 1),
                       device=str(dev), grid="prep 3mm canonical (RAS), affine from prep npz",
                       shape_xyz=list(cam.shape[::-1])),
                  open(d / ("cam_meta.json" if a.upsample == "trilinear" else f"cam_meta_{a.upsample}.json"), "w"), indent=1, ensure_ascii=False)
        print(f"{cid} {a.finding} {dt:.1f}s hot={hot and hot['unit']} " +
              " ".join(f"{r['unit']}:{r['score_cached_emb']:.2f}/{r['score_voxelmap_mean']:.2f}"
                       for r in pres), flush=True)


if __name__ == "__main__":
    main()
