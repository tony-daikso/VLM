"""Pilot: 1.5 mm vertebra-specialised TotalSegmentator encoder (Dataset292) vs current 3 mm --fast encoder
(Dataset297) as per-vertebra feature extractors for vertebral (compression) fracture.

Sub-commands (run on Spark, work dir ~/OWL/pilot_vert15):
  sample   : seeded id lists (train/val only; test is filtered out before anything else)
  extract  : per case: pull CT from CT-rate -> canonical
               3 mm : change_spacing 3.0 -> Dataset297 seg (seg_nnunet) -> units T8..L5 -> embeddings
               1.5mm: spine ROI from the 3 mm vertebra labels (x/y >= 210 mm around the spine, z +-100 mm) cut
                      from the canonical CT -> change_spacing 1.5 -> Dataset292 seg (its own vertebra labels)
                      -> units T8..L5 -> embeddings
             embeddings (both): one 128^3 tile per vertebra (patches.tiles_for), windows baseline / soft_tissue
             L40 W400 / bone L400 W1800 mapped to each model's own CT normalisation (windowing.apply_window),
             frozen encoder stages [1,2,3], mask-fraction mean pooling (extract.pool_stage, bbox fallback) and
             max pooling over stage voxels with mask fraction >= 0.5, per-stage L2, concat.
             Cache emb/{id}.npz; pulled CT deleted after use. Timing in timing_w{wid}.csv.
  probe    : shared per-vertebra logistic head + MIL max over levels; fit on train, scored on val only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("OWL_SRC", Path.home() / "OWL/mw_exp/src"))
sys.path.insert(0, str(SRC))

WORK = Path(os.environ.get("PILOT_WORK", Path.home() / "OWL/pilot_vert15"))
RES = Path.home() / ".totalsegmentator/nnunet/results"
M15 = RES / "Dataset292_TotalSegmentator_part2_vertebrae_1532subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
M3 = RES / "Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
CT_HOST = "daikso@192.168.0.189"
CT_DIR = "/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/merlin_data"
LEVELS = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
WINDOWS = [("baseline", {"type": "baseline"}), ("soft_tissue", {"level": 40, "width": 400}),
           ("bone", {"level": 400, "width": 1800})]
STAGES = [1, 2, 3]
# 1.5 mm net, 128^3 patch: stage feature maps 128/64/32/16/8/4 voxel = 1.5/3/6/12/24/48 mm.
# A vertebral body (~40x30x25 mm) covers ~13x10x8 / 7x5x4 / 3x3x2 voxels at stages 1/2/3 and <=1 voxel at
# stage 4+, so stages 1-3 are the deepest ones that still resolve the body (stage 0 = raw-intensity edges).
# The 3 mm net uses the same stage indices as the main run ([1,2,3] = 6/12/24 mm).
MIN_VOX = {"r15": 800, "r3": 100}       # 100 voxels on the 3 mm grid = 800 on the 1.5 mm grid
BASE_CFG = dict(pad_hu=-1024, amp=True, min_stage_weight=1.0, tile_overlap=0.25)


# ----------------------------------------------------------------------------------------------- sample
def load_groups():
    rows = []
    for line in open(WORK / "data/labels_disease14_all.jsonl"):
        r = json.loads(line)
        if r["split"] == "test":          # test split never enters the pilot
            continue
        fr = r["answer"].get("fracture", {})
        cat, d = fr.get("category"), fr.get("detail") or {}
        if cat in ("yes", "probably_yes"):
            g = "comp" if d.get("vertebral_compression") == "yes" else "oth"
        elif cat in ("no", "probably_no", "not_mentioned"):
            g = "neg"
        else:
            continue                      # unclear
        rows.append((r["study_id"], r["split"], g, d.get("acute")))
    return rows


def cmd_sample(a):
    import pandas as pd
    rows = pd.DataFrame(load_groups(), columns=["study_id", "split", "group", "acute"])
    spl = pd.read_csv(WORK / "data/split.csv").set_index("study_id")["split"]
    rows = rows[rows.study_id.map(spl).isin(["train", "val"]) & (rows.study_id.map(spl) == rows.split)]
    want = {"train": {"comp": 200, "oth": 100, "neg": 300}, "val": {"comp": 100, "oth": 50, "neg": 150}}
    rng = np.random.default_rng(42)
    out = []
    for s, gs in want.items():
        for g, n in gs.items():
            pool = rows[(rows.split == s) & (rows.group == g)].study_id.values
            out.append(rows[rows.study_id.isin(rng.choice(np.sort(pool), n, replace=False))])
    df = pd.concat(out).sort_values(["split", "group", "study_id"])
    assert not (df.split == "test").any()
    df.to_csv(WORK / "pilot_ids.csv", index=False)
    print(df.groupby(["split", "group"]).size())


# ----------------------------------------------------------------------------------------------- extract
def pull(cid, dst: Path):
    tmp = dst.with_suffix(".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["ssh", "-c", "aes128-gcm@openssh.com", "-o", "BatchMode=yes", CT_HOST,
                        f"cat {CT_DIR}/{cid}.nii.gz"], stdout=fh, check=True)
    tmp.rename(dst)


def max_stage(feat, mask, thr):
    import torch.nn.functional as F
    f = [m // s for m, s in zip(mask.shape[2:], feat.shape[2:])]
    sel = F.avg_pool3d(mask, kernel_size=f, stride=f)[0, 0] >= thr
    if not bool(sel.any()):
        return None
    return feat.float()[:, :, sel].amax(2)


def embed_units(enc, hu, units, U, min_vox, max_frac=0.5):
    """same computation as extract_bones.embed_bones (mean = extract.pool_stage; max over voxels >= max_frac)."""
    import torch
    import torch.nn.functional as F
    from extract import crop, pool_stage
    from patches import tiles_for
    from windowing import apply_window
    P, S, W = list(enc.patch_size), len(STAGES), len(WINDOWS)
    D = sum(enc.stage_channels(STAGES))
    emb = np.zeros((U, W, D), np.float32)
    emb_max = np.zeros((U, W, D), np.float32)
    nvox = np.zeros(U, np.int64)
    present = np.zeros(U, bool)
    nfb = np.zeros((U, S), np.int32)
    for k in range(U):
        idx = np.argwhere(units == k + 1)
        nvox[k] = len(idx)
        if len(idx) < min_vox:
            continue
        lo, hi, cen = idx.min(0), idx.max(0), idx.mean(0)
        tiles = []
        for s in tiles_for(lo, hi, cen, P, BASE_CFG["tile_overlap"]):
            sl = tuple(slice(max(s[i], 0), s[i] + P[i]) for i in range(3))
            n_in = int((units[sl] == k + 1).sum())
            if n_in:
                tiles.append((s, n_in))
        acc, wsum, raw, mx = 0.0, 0, [0.0] * S, [None] * S
        for start, n_in in tiles:
            h = torch.from_numpy(crop(hu, start, P, BASE_CFG["pad_hu"])).to(enc.device)
            m = torch.from_numpy(crop(units, start, P, 0) == k + 1).float().to(enc.device)[None, None]
            x = torch.stack([apply_window(h, w, enc.norm) for _, w in WINDOWS])[:, None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                feats = enc.stage_features(x, upto=max(STAGES))
            vs = []
            for j, s in enumerate(STAGES):
                v, fb, _ = pool_stage(feats[s], m, BASE_CFG["min_stage_weight"])
                nfb[k, j] += fb
                vs.append(F.normalize(v, dim=1))
                raw[j] = raw[j] + v * n_in
                t = max_stage(feats[s], m, max_frac)
                if t is not None:
                    mx[j] = t if mx[j] is None else torch.maximum(mx[j], t)
            acc = acc + torch.cat(vs, 1) * n_in
            wsum += n_in
        emb[k] = (acc / wsum).cpu().numpy()
        emb_max[k] = torch.cat([F.normalize(mx[j] if mx[j] is not None else raw[j] / wsum, dim=1)
                                for j in range(S)], 1).cpu().numpy()
        present[k] = True
    return dict(emb=emb.astype(np.float16), emb_max=emb_max.astype(np.float16), present=present, nvox=nvox, nfb=nfb)


def units_from(lab, enc):
    units = np.zeros(lab.shape, np.uint8)
    for k, lv in enumerate(LEVELS):
        units[lab == enc.labels[f"vertebrae_{lv}"]] = k + 1
    return units


def spine_roi(canon, img3, lab3, enc3):
    """world bbox of the 3 mm vertebra labels -> slice of the canonical CT (min 210 mm in x/y, +-100 mm in z)."""
    ids = [v for k, v in enc3.labels.items() if k.startswith("vertebrae_") or k == "sacrum"]
    idx = np.argwhere(np.isin(lab3, ids))  # (z,y,x) on the 3 mm grid
    if len(idx) == 0:
        return None
    lo, hi = idx.min(0)[::-1], idx.max(0)[::-1]  # -> (x,y,z) voxel
    corners = np.array([[a, b, c, 1] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], float)
    vox = (np.linalg.inv(canon.affine) @ img3.affine @ corners.T)[:3].T
    vlo, vhi = vox.min(0), vox.max(0)
    zooms = np.array(canon.header.get_zooms()[:3], float)
    sl = []
    for ax in range(3):
        cen, half = (vlo[ax] + vhi[ax]) / 2, (vhi[ax] - vlo[ax]) / 2
        half = max(half + 30 / zooms[ax], 105 / zooms[ax]) if ax < 2 else half + 100 / zooms[ax]
        a0, a1 = int(np.floor(cen - half)), int(np.ceil(cen + half)) + 1
        sl.append(slice(max(a0, 0), min(a1, canon.shape[ax])))
    return canon.slicer[tuple(sl)]


def cmd_extract(a):
    import nibabel as nib
    import pandas as pd
    import torch
    from encoder import NnUNetEncoder
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    from worker import seg_nnunet

    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = False
    ids = pd.read_csv(WORK / "pilot_ids.csv").study_id.tolist()
    ids = ids[a.wid::a.nw]
    (WORK / "emb").mkdir(exist_ok=True)
    (WORK / "ct").mkdir(exist_ok=True)
    todo = [c for c in ids if not (WORK / "emb" / f"{c}.npz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    enc3 = NnUNetEncoder(M3)
    enc15 = NnUNetEncoder(M15)
    print(f"w{a.wid}: {len(todo)} to do; 3mm ch {enc3.stage_channels(STAGES)} 1.5mm ch {enc15.stage_channels(STAGES)}",
          flush=True)
    pulled, pt = {}, {}

    def prefetch(cid):
        t = time.time()
        try:
            pull(cid, WORK / "ct" / f"{cid}.nii.gz")
            pulled[cid] = None
        except Exception as e:  # noqa: BLE001
            pulled[cid] = e
        pt[cid] = time.time() - t

    logf = WORK / f"timing_w{a.wid}.csv"
    new = not logf.exists()
    fh = open(logf, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["id", "pull", "load", "resample3", "seg3", "embed3", "roi_resample15", "seg15", "embed15",
                     "total_proc", "shape15", "n15", "n3", "status"])
    th = None
    if todo:
        th = threading.Thread(target=prefetch, args=(todo[0],)); th.start()
    for i, cid in enumerate(todo):
        th.join()
        if i + 1 < len(todo):
            th = threading.Thread(target=prefetch, args=(todo[i + 1],)); th.start()
        ct = WORK / "ct" / f"{cid}.nii.gz"
        T = {"pull": pt.get(cid, np.nan)}
        try:
            if pulled.get(cid) is not None:
                raise pulled[cid]
            t0 = t = time.time()
            raw = nib.load(str(ct))
            img = nib.Nifti1Image(np.asanyarray(raw.dataobj), raw.affine, raw.header)
            img.header.set_slope_inter(1, 0)
            canon = as_closest_canonical(img)
            T["load"] = time.time() - t; t = time.time()
            img3 = change_spacing(canon, list(enc3.spacing), order=1, dtype=np.int32, nr_cpus=a.threads)
            hu3 = np.asanyarray(img3.dataobj).astype(np.int16).transpose(2, 1, 0)
            T["resample3"] = time.time() - t; t = time.time()
            lab3 = seg_nnunet(enc3, hu3)
            T["seg3"] = time.time() - t; t = time.time()
            r3 = embed_units(enc3, hu3, units_from(lab3, enc3), len(LEVELS), MIN_VOX["r3"])
            T["embed3"] = time.time() - t; t = time.time()
            roi = spine_roi(canon, img3, lab3, enc3)
            if roi is None:
                raise RuntimeError("no vertebrae in 3 mm seg")
            img15 = change_spacing(roi, list(enc15.spacing), order=1, dtype=np.int32, nr_cpus=a.threads)
            hu15 = np.asanyarray(img15.dataobj).astype(np.int16).transpose(2, 1, 0)
            T["roi_resample15"] = time.time() - t; t = time.time()
            lab15 = seg_nnunet(enc15, hu15)
            T["seg15"] = time.time() - t; t = time.time()
            r15 = embed_units(enc15, hu15, units_from(lab15, enc15), len(LEVELS), MIN_VOX["r15"])
            T["embed15"] = time.time() - t
            T["total_proc"] = time.time() - t0
            out = {f"r3_{k}": v for k, v in r3.items()} | {f"r15_{k}": v for k, v in r15.items()}
            tmp = WORK / "emb" / f"{cid}.tmp.npz"
            np.savez(tmp, levels=np.array(LEVELS), windows=np.array([w for w, _ in WINDOWS]),
                     stages=np.array(STAGES), shape15=np.array(hu15.shape), **out)
            tmp.rename(WORK / "emb" / f"{cid}.npz")
            status, s15, n15, n3 = "ok", "x".join(map(str, hu15.shape)), int(r15["present"].sum()), int(r3["present"].sum())
        except Exception as e:  # noqa: BLE001
            status, s15, n15, n3 = f"FAIL {e!r}"[:200], "", 0, 0
        finally:
            ct.unlink(missing_ok=True)
            torch.cuda.empty_cache()
        wr.writerow([cid] + [f"{T.get(k, np.nan):.2f}" for k in ["pull", "load", "resample3", "seg3", "embed3",
                     "roi_resample15", "seg15", "embed15", "total_proc"]] + [s15, n15, n3, status])
        fh.flush()
        print(f"w{a.wid} [{i+1}/{len(todo)}] {cid} {status} n15={n15} n3={n3} " +
              " ".join(f"{k}={v:.1f}" for k, v in T.items()), flush=True)
    if th is not None:
        th.join()
    fh.close()


# ----------------------------------------------------------------------------------------------- probe
def cmd_probe(a):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from age_shortcut import matched_auc
    from probe import auroc

    df = pd.read_csv(WORK / "pilot_ids.csv")
    df = df[df.split.isin(["train", "val"])].reset_index(drop=True)          # never any test case
    meta = pd.read_csv(WORK / "data/metadata.csv").drop_duplicates("study id").set_index("study id")
    df["age"] = meta.reindex(df.study_id)["Age"].values.astype(float)
    Z = {}
    for cid in df.study_id:
        p = WORK / "emb" / f"{cid}.npz"
        if p.exists():
            Z[cid] = dict(np.load(p))
    keep = [c in Z and Z[c]["r15_present"].any() and Z[c]["r3_present"].any() for c in df.study_id]
    df = df[keep].reset_index(drop=True)
    print("cases with both embeddings:", df.groupby(["split", "group"]).size().to_dict(), flush=True)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    Cs = [1e-4, 1e-3, 1e-2, 1e-1]

    def bag(ids, res, pool, wins):
        X, g = [], []
        for i, c in enumerate(ids):
            z = Z[c]
            e = z[f"r{res}_{'emb' if pool == 'mean' else 'emb_max'}"].astype(np.float32)  # [U,W,D]
            pr = z[f"r{res}_present"]
            X.append(e[pr][:, wins].reshape(pr.sum(), -1))
            g += [i] * int(pr.sum())
        return np.concatenate(X), np.array(g)

    def fit(X, g, y_case, C):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        n_inst = np.bincount(g)
        cw = {1: 0.5 / y_case.mean(), 0: 0.5 / (1 - y_case.mean())}
        sw = np.array([cw[int(y_case[i])] / n_inst[i] for i in g])
        m = LogisticRegression(C=C, max_iter=3000, tol=1e-4)
        m.fit((X - mu) / sd, y_case[g], sample_weight=sw * len(y_case) / sw.sum())
        return lambda Xn, gn, n: np.array([s.max() for s in np.split(m.decision_function((Xn - mu) / sd),
                                                                     np.cumsum(np.bincount(gn, minlength=n))[:-1])])

    def run(target, res, pool, wins):
        if target == "compression":
            trs = tr[tr.group.isin(["comp", "neg"])]
            ytr = (trs.group == "comp").values.astype(float)
        else:
            trs = tr
            ytr = (trs.group != "neg").values.astype(float)
        X, g = bag(trs.study_id.tolist(), res, pool, wins)
        cv = {}
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for C in Cs:
            sc = np.zeros(len(ytr))
            for a_, b_ in skf.split(np.zeros(len(ytr)), ytr):
                ia = np.isin(g, a_)
                f = fit(X[ia], np.searchsorted(a_, g[ia]), ytr[a_], C)
                ib = np.isin(g, b_)
                sc[b_] = f(X[ib], np.searchsorted(b_, g[ib]), len(b_))
            cv[C] = auroc(ytr, sc)
        C = max(cv, key=cv.get)
        f = fit(X, g, ytr, C)
        Xv, gv = bag(va.study_id.tolist(), res, pool, wins)
        return C, cv[C], f(Xv, gv, len(va))

    yv = va.group.values
    agev = va.age.values
    rows, scores = [], {}
    for target in ["compression", "any_fracture"]:
        for res in ["15", "3"]:
            for pool in ["mean", "max"]:
                for wn, wins in [("baseline", [0]), ("3win_concat", [0, 1, 2])]:
                    C, cvauc, s = run(target, res, pool, wins)
                    scores[(target, res, pool, wn)] = s
                    r = {"target": target, "res": "1.5mm" if res == "15" else "3mm", "pool": pool, "windows": wn,
                         "C": C, "train_cv_auroc": cvauc}
                    for ev, pos in [("comp_vs_neg", ["comp"]), ("allfx_vs_neg", ["comp", "oth"])]:
                        sel = np.isin(yv, pos + ["neg"])
                        y = np.isin(yv[sel], pos).astype(float)
                        r[f"val_auroc_{ev}"] = auroc(y, s[sel])
                        ok = ~np.isnan(agev[sel])
                        r[f"val_agematched_{ev}"] = matched_auc(y[ok], s[sel][ok], agev[sel][ok], 2.0)
                    rows.append(r)
                    print(r, flush=True)
    for ev, pos in [("comp_vs_neg", ["comp"]), ("allfx_vs_neg", ["comp", "oth"])]:
        sel = np.isin(yv, pos + ["neg"]) & ~np.isnan(agev)
        y = np.isin(yv[sel], pos).astype(float)
        rows.append({"target": "age_only", "res": "-", "pool": "-", "windows": "-",
                     f"val_auroc_{ev}": auroc(y, agev[sel]),
                     f"val_agematched_{ev}": matched_auc(y, agev[sel], agev[sel], 2.0)})
    res_df = pd.DataFrame(rows).groupby(["target", "res", "pool", "windows"], sort=False).first().reset_index()
    res_df.to_csv(WORK / "results_val.csv", index=False)
    # bootstrap 1.5 mm - 3 mm, paired over val cases: compression head on comp vs neg, any-fracture head on
    # (comp+oth) vs neg
    bs_all = []
    for target, pos, fname in [("compression", ["comp"], "bootstrap_diff_compression_val.csv"),
                               ("any_fracture", ["comp", "oth"], "bootstrap_diff_anyfracture_val.csv")]:
        rng = np.random.default_rng(0)
        sel = np.isin(yv, pos + ["neg"])
        y = np.isin(yv[sel], pos).astype(float)
        bs = []
        for pool in ["mean", "max"]:
            for wn in ["baseline", "3win_concat"]:
                s15, s3 = scores[(target, "15", pool, wn)][sel], scores[(target, "3", pool, wn)][sel]
                d0 = auroc(y, s15) - auroc(y, s3)
                ds = []
                for _ in range(1000):
                    ii = rng.integers(0, len(y), len(y))
                    if 0 < y[ii].sum() < len(ii):
                        ds.append(auroc(y[ii], s15[ii]) - auroc(y[ii], s3[ii]))
                lo, hi = np.percentile(ds, [2.5, 97.5])
                bs.append({"target": target, "pool": pool, "windows": wn, "auroc_15": auroc(y, s15),
                           "auroc_3": auroc(y, s3), "diff": d0, "ci_lo": lo, "ci_hi": hi,
                           "p_le0": float(np.mean(np.array(ds) <= 0))})
        pd.DataFrame(bs).to_csv(WORK / fname, index=False)
        bs_all += bs
    bs = pd.DataFrame(bs_all)
    np.savez(WORK / "val_scores.npz", ids=va.study_id.values, group=yv,
             **{"|".join(k): v for k, v in scores.items()})
    pd.set_option("display.width", 250)
    print(res_df.round(3).to_string())
    print(bs.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("sample")
    e = sp.add_parser("extract")
    e.add_argument("--wid", type=int, default=0)
    e.add_argument("--nw", type=int, default=1)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--threads", type=int, default=4)
    sp.add_parser("probe")
    a = ap.parse_args()
    {"sample": cmd_sample, "extract": cmd_extract, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
