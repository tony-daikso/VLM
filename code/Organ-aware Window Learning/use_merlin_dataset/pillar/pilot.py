"""Pillar-0 pilot on 20 Merlin TRAIN cases: VRAM / timing, forward-mirror equivalence, mask alignment overlays,
per-unit z-crop coverage, feature sanity (global not NaN/constant, kidney L vs R)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract as ex  # noqa: E402
import pillar_common as pc  # noqa: E402

META = Path("/media/daikso/CTDATA/Merlin/merlinabdominalctdataset/metadata.csv")
CLASSMAP = Path(__file__).resolve().parent / "ts_total_classmap.json"


def pick_cases(n=20, seed=0):
    import SimpleITK as sitk
    man = {r["study id"]: r["manufacturer"] for r in csv.DictReader(open(META))}
    train = [c for c, _ in ex.case_list(["train"]) if ex.find_mask(c)]
    random.Random(seed).shuffle(train)
    pool = []
    for c in train[:240]:
        r = sitk.ImageFileReader()
        r.SetFileName(str(ex.CT_DIR / f"{c}.nii.gz"))
        r.ReadImageInformation()
        zmm = r.GetSize()[2] * r.GetSpacing()[2]
        pool.append((zmm, man.get(c, "?"), c))
    pool.sort()
    # 20 z-extent quantiles, prefer a manufacturer not yet used at each quantile
    picks, used = [], {}
    for k in range(n):
        lo, hi = int(k * len(pool) / n), int((k + 1) * len(pool) / n)
        cand = sorted(pool[lo:hi], key=lambda t: used.get(t[1], 0))
        picks.append(cand[0])
        used[cand[0][1]] = used.get(cand[0][1], 0) + 1
    return picks


def overlay(cid, out_png, lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vol, ref, info = pc.preprocess_ct(ex.CT_DIR / f"{cid}.nii.gz")
    m384, _ = pc.preprocess_mask(ex.find_mask(cid), ref, [tuple(p) for p in info["plan"]])
    d = np.load(Path(out_png).parent.parent / "feats" / f"{cid}.npz")
    g16 = d["grid16"].astype(np.float32)  # (4096,384) raster z,y,x
    # PCA-1 of the 16^3 tokens
    gc = g16 - g16.mean(0)
    _, _, vt = np.linalg.svd(gc, full_matrices=False)
    pc1 = (gc @ vt[0]).reshape(16, 16, 16)
    groups = {"liver": [lab["liver"]], "spleen": [lab["spleen"]], "kidney_R": [lab["kidney_right"]],
              "kidney_L": [lab["kidney_left"]],
              "vert T8-L5": [lab[f"vertebrae_{v}"] for v in ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]]}
    colors = {"liver": "orange", "spleen": "purple", "kidney_R": "cyan", "kidney_L": "lime", "vert T8-L5": "red"}
    kid = np.isin(m384, [lab["kidney_left"], lab["kidney_right"]])
    spine = np.isin(m384, groups["vert T8-L5"])
    yc = int(np.argwhere(kid)[:, 1].mean()) if kid.any() else 192
    xc = int(np.argwhere(spine)[:, 2].mean()) if spine.any() else 192
    ct = np.clip((vol.astype(np.float32) + 160) / 400, 0, 1)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for row, (name, sl_ct, sl_m, sl_pc, idx16) in enumerate([
        ("coronal y=%d" % yc, ct[:, yc, :], m384[:, yc, :], pc1[:, yc // 24, :], yc),
        ("sagittal x=%d" % xc, ct[:, :, xc], m384[:, :, xc], pc1[:, :, xc // 24], xc)]):
        ax = axs[row, 0]
        ax.imshow(sl_ct, cmap="gray", origin="lower")
        for g, ids in groups.items():
            mm = np.isin(sl_m, ids).astype(float)
            if mm.any():
                ax.contour(mm, levels=[0.5], colors=colors[g], linewidths=0.8, origin="lower")
        ax.set_title(f"{cid} {name} 384^3 + TS mask"); ax.axis("off")
        # 64^3 token-grid mask (majority label per 6^3 cell) overlay
        ax = axs[row, 1]
        ct64 = sl_ct.reshape(64, 6, 64, 6).mean((1, 3))
        ax.imshow(ct64, cmap="gray", origin="lower")
        for g, ids in groups.items():
            mm = np.isin(sl_m, ids).astype(float).reshape(64, 6, 64, 6).mean((1, 3))
            if mm.any():
                ax.contour(mm, levels=[0.5], colors=colors[g], linewidths=0.8, origin="lower")
        ax.set_title("64^3 token grid: CT + mask>50% of cell"); ax.axis("off")
        ax = axs[row, 2]
        ax.imshow(sl_pc, cmap="coolwarm", origin="lower")
        ax.set_title("16^3 feature PCA-1 (same slice)"); ax.axis("off")
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/daikso/Tony/pillar/pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = {int(k): v for k, v in json.load(open(CLASSMAP)).items()}
    lab = {v: k for k, v in cmap.items()}

    picks = pick_cases()
    print("[pilot] cases:", *[f"{c}:{m}:{z:.0f}mm" for z, m, c in picks], flush=True)
    cases = [(c, "train") for _, _, c in picks]

    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.workers, initializer=ex._init_worker, initargs=(a.threads,))
    import torch
    visual, miss, unexp = pc.load_model()
    print(f"[pilot] missing keys {len(miss)} {miss[:5]} | unexpected {len(unexp)} {unexp[:5]}", flush=True)
    equiv = {}

    def on_case(item, res, st):
        if not equiv:  # forward-mirror equivalence vs the model's own forward
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = pc.window_gpu(torch.from_numpy(item["vol"]).cuda())
                ref = visual({pc.MODALITY: x})[0].float().cpu().numpy()
            equiv["maxabs"] = float(np.abs(ref - res["glob"]).max())
            equiv["refnorm"] = float(np.abs(ref).max())
            del x
            torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = ex.run(cases, out, workers=a.workers, threads=a.threads, visual=visual, pool=pool, on_case=on_case)
    wall = time.time() - t0
    pool.close(); pool.join()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30

    # ---- timing summary
    S = {k: np.array([s[k] for s in stats]) for k in ["t_ct", "t_mask", "t_gpu"]}
    summ = dict(n=len(stats), wall_s=wall, e2e_s_per_case=wall / max(len(stats), 1),
                e2e_s_per_case_excl_first=None, gpu_s_median=float(np.median(S["t_gpu"])),
                gpu_s_mean_excl_first=float(S["t_gpu"][1:].mean()), cpu_ct_s_mean=float(S["t_ct"].mean()),
                cpu_mask_s_mean=float(S["t_mask"].mean()), cpu_total_s_mean=float((S["t_ct"] + S["t_mask"]).mean()),
                workers=a.workers, threads=a.threads, peak_alloc_gb=peak_alloc, peak_reserved_gb=peak_res,
                forward_equiv=equiv, missing_keys=miss, unexpected_keys=unexp)

    # ---- sanity + coverage
    feats = {c: np.load(out / "feats" / f"{c}.npz") for c, _ in cases if (out / "feats" / f"{c}.npz").exists()}
    G = np.stack([f["global"].astype(np.float32) for f in feats.values()])
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    cos = Gn @ Gn.T
    summ["global"] = dict(nan=int(np.isnan(G).sum()), std_over_cases_min=float(G.std(0).min()),
                          std_over_cases_mean=float(G.std(0).mean()), frac_dims_const=float((G.std(0) < 1e-4).mean()),
                          pair_cos_mean=float(cos[np.triu_indices(len(G), 1)].mean()),
                          pair_cos_min=float(cos[np.triu_indices(len(G), 1)].min()))
    kL, kR = lab["kidney_left"], lab["kidney_right"]
    dl = []
    cos_lr, cos_ll = [], []
    VL = {c: f["vec64"][kL].astype(np.float32) for c, f in feats.items() if f["w64"][kL] > 0 and f["w64"][kR] > 0}
    VR = {c: f["vec64"][kR].astype(np.float32) for c, f in feats.items() if f["w64"][kL] > 0 and f["w64"][kR] > 0}
    cs = list(VL)
    for c in cs:
        a_, b_ = VL[c], VR[c]
        cos_lr.append(float(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_)))
        dl.append(a_ - b_)
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a_, b_ = VL[cs[i]], VL[cs[j]]
            cos_ll.append(float(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_)))
    D = np.stack(dl)
    md = D.mean(0)
    cons = [float(d @ md / np.linalg.norm(d) / np.linalg.norm(md)) for d in D]
    # leave-one-out sign test: is L-R direction consistent across patients?
    loo = []
    for i in range(len(D)):
        m = np.delete(D, i, 0).mean(0)
        loo.append(float(np.sign((VL[cs[i]] - VR[cs[i]]) @ m)))
    summ["kidney_LR"] = dict(n=len(cs), cos_L_vs_R_same_case_mean=float(np.mean(cos_lr)),
                             cos_L_vs_L_across_cases_mean=float(np.mean(cos_ll)),
                             LminusR_consistency_cos_mean=float(np.mean(cons)),
                             loo_sign_correct=float(np.mean(np.array(loo) > 0)))
    units = ["vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12", "vertebrae_L1",
             "vertebrae_L5", "sacrum", "urinary_bladder", "lung_lower_lobe_left", "lung_lower_lobe_right", "liver",
             "spleen", "kidney_left", "kidney_right", "hip_left", "femur_left"]
    cov = {}
    for u in units:
        L = lab[u]
        v = np.array([f["coverage"][L] for f in feats.values()])
        pres = ~np.isnan(v)
        cov[u] = dict(present=int(pres.sum()), cut_any=int((v[pres] < 0.99).sum()), cut_half=int((v[pres] < 0.5).sum()),
                      lost=int((v[pres] == 0).sum()))
    summ["coverage"] = cov
    zinfo = {c: (json.loads(str(f["meta"]))["z_mm_total"]) for c, f in feats.items()}
    summ["z_mm_total"] = dict(min=min(zinfo.values()), max=max(zinfo.values()),
                              n_over_480=int(sum(z > 480 for z in zinfo.values())))
    json.dump(dict(summary=summ, per_case=stats, picks=[list(p) for p in picks]), open(out / "pilot_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)

    # ---- overlays: shortest, median, longest
    (out / "png").mkdir(exist_ok=True)
    for z, m, c in [picks[0], picks[len(picks) // 2], picks[-1]]:
        overlay(c, out / "png" / f"{c}_overlay.png", lab)
        print("[pilot] overlay", c, flush=True)


if __name__ == "__main__":
    main()
