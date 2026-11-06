"""Step 4: sanity checks (spec section 6). Results -> outputs/qc/sanity.json (+ PNGs).

1 normalization: baseline window vs nnU-Net DefaultPreprocessor on the same TS-resampled image (max abs err)
2 segmentation with baseline input through the full frozen network vs TS mask (Dice per unit, gate >0.9)
3 same with every non-baseline window (diagnostic)
4 axial slice through liver centroid per window with unit contours (5 random cases)
5 embedding health: NaN/Inf, mean raw (pre-L2) norm per window and stage
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from common import case_ids, load_cfg, log_progress, read_plans, unit_names, window_names
from windowing import apply_window


def check_norm(cfg, enc, cid):
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    img = as_closest_canonical(nib.load(str(Path(cfg["ct_dir"]) / f"{cid}.nii.gz")))
    img = change_spacing(img, list(enc.spacing), order=1, dtype=np.int32, nr_cpus=4)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "s01_0000.nii.gz"
        nib.save(img, f)  # exactly what TS hands to nnU-Net
        p = enc.predictor
        data, _, props = DefaultPreprocessor(verbose=False).run_case(
            [str(f)], None, p.plans_manager, p.configuration_manager, p.dataset_json)
    hu = np.load(cfg["out"] / "prep" / f"{cid}.npz")["hu"]
    bb = props["bbox_used_for_cropping"]
    ours = apply_window(hu[tuple(slice(a, b) for a, b in bb)], cfg["windows"]["baseline"], enc.norm)
    if ours.shape != data[0].shape:
        return {"id": cid, "ok": False, "err": f"shape {ours.shape} vs {data[0].shape}"}
    err = float(np.abs(ours - data[0]).max())
    return {"id": cid, "max_abs_err": err, "ok": err < 1e-4}


@torch.no_grad()
def seg_dice(cfg, enc, cid, win_name):
    prep = np.load(cfg["out"] / "prep" / f"{cid}.npz")
    hu, units = prep["hu"], prep["units"]
    x = torch.from_numpy(apply_window(hu, cfg["windows"][win_name], enc.norm))[None].float()
    logits = enc.predictor.predict_sliding_window_return_logits(x)
    pred = logits.argmax(0).cpu().numpy()
    from patches import label_lookup
    lookup = label_lookup(cfg)
    # nnU-Net label index == TS class id for the 'total' models
    out = {}
    for k, (u, classes) in enumerate(cfg["organ_map"].items()):
        ids = [c if isinstance(c, int) else lookup[c] for c in classes]
        a, b = np.isin(pred, ids), units == k + 1
        if b.sum() < cfg["min_organ_voxels"]:
            continue
        out[u] = round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 4)
    return out


def qc_png(cfg, enc, cid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prep = np.load(cfg["out"] / "prep" / f"{cid}.npz")
    hu, units = prep["hu"], prep["units"]
    names = unit_names(cfg)
    k = names.index("liver") + 1 if "liver" in names and (units == names.index("liver") + 1).any() else 1
    z = int(np.argwhere(units == k).mean(0)[0]) if (units == k).any() else hu.shape[0] // 2
    wins = window_names(cfg)
    fig, ax = plt.subplots(1, len(wins), figsize=(4 * len(wins), 4.4))
    for i, w in enumerate(wins):
        sl = apply_window(hu[z], cfg["windows"][w], enc.norm)
        ax[i].imshow(sl[::-1], cmap="gray")  # array axis1 = y (A), flip for radiological up=anterior
        lab = units[z][::-1]
        for u in np.unique(lab[lab > 0]):
            ax[i].contour(lab == u, levels=[0.5], linewidths=0.6, colors=[plt.cm.tab20(int(u) % 20)])
        ax[i].set_title(f"{w}  [{sl.min():.1f},{sl.max():.1f}]", fontsize=9)
        ax[i].axis("off")
    fig.suptitle(f"{cid} z={z}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def emb_health(cfg, max_cases=2000):
    from extract import window_key
    wk = {n: window_key(n, w) for n, w in cfg["windows"].items()}
    first = next(iter(wk.values()))
    ids = sorted(p.stem for p in (cfg["out"] / "emb" / first).glob("*.npz"))[:max_cases]
    bad, norms, fb, nt = [], [], [], []
    for cid in ids:
        zs = {n: np.load(cfg["out"] / "emb" / k / f"{cid}.npz") for n, k in wk.items()
              if (cfg["out"] / "emb" / k / f"{cid}.npz").exists()}
        if len(zs) < len(wk):
            continue
        if not all(np.isfinite(z["emb"].astype(np.float32)).all() for z in zs.values()):
            bad.append(cid)
        z0 = zs[next(iter(wk))]
        pr = z0["organ_present"]
        if pr.any():
            norms.append(np.stack([zs[n]["raw_norm"][pr].mean(0) for n in wk]))  # [W, S]
        fb.append(z0["n_fallback"])
        nt.append(z0["n_tiles"])
    z = z0
    norms = np.mean(norms, 0)
    fb = np.sum(fb, 0)
    return {"n_files": len(nt), "nonfinite": bad,
            "mean_raw_norm[window][stage]": {w: dict(zip(map(int, z["stages_used"]), map(float, norms[i].round(4))))
                                             for i, w in enumerate(wk)},
            "norm_ratio_max_over_min_per_stage": [float(r) for r in (norms.max(0) / norms.min(0)).round(3)],
            "fallback_counts[unit][stage]": {u: fb[k].tolist() for k, u in enumerate(z["organ_names"])},
            "mean_tiles_per_present_unit": {u: float(np.mean([t[k] for t in nt if t[k] > 0] or [0]))
                                            for k, u in enumerate(z["organ_names"])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-norm", type=int, default=3)
    ap.add_argument("--n-seg", type=int, default=3)
    ap.add_argument("--n-png", type=int, default=5)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    from encoder import build_encoder
    enc = build_encoder(cfg)
    ids = [i for i in case_ids(cfg) if (cfg["out"] / "prep" / f"{i}.npz").exists()]
    rng = random.Random(cfg["seed"])
    res = {}

    res["1_normalization"] = [check_norm(cfg, enc, c) for c in ids[:a.n_norm]]
    print("1", res["1_normalization"], flush=True)

    seg = {}
    for c in ids[:a.n_seg]:
        seg[c] = {w: seg_dice(cfg, enc, c, w) for w in window_names(cfg)}
        print("2/3", c, {w: round(float(np.mean(list(v.values()))), 3) for w, v in seg[c].items()}, flush=True)
    res["2_3_seg_dice"] = seg
    summ = {}
    for w in window_names(cfg):
        per_unit = {}
        for c in seg:
            for u, d in seg[c][w].items():
                per_unit.setdefault(u, []).append(d)
        summ[w] = {u: round(float(np.mean(v)), 4) for u, v in per_unit.items()}
        summ[w]["_mean"] = round(float(np.mean([np.mean(v) for v in per_unit.values()])), 4)
    res["2_3_seg_dice_summary"] = summ
    res["2_pass"] = all(d > 0.9 for u, d in summ["baseline"].items() if u != "_mean")

    for c in rng.sample(ids, min(a.n_png, len(ids))):
        qc_png(cfg, enc, c, cfg["out"] / "qc" / f"{c}.png")
    res["4_png_dir"] = str(cfg["out"] / "qc")

    res["5_embedding"] = emb_health(cfg) if any((cfg["out"] / "emb").rglob("*.npz")) else "no embeddings yet"
    (cfg["out"] / "qc" / "sanity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "2_3_seg_dice"}, indent=1))
    log_progress(f"qc.py: norm ok={all(r.get('ok') for r in res['1_normalization'])}, "
                 f"baseline Dice pass={res['2_pass']} (mean {summ['baseline']['_mean']}); "
                 f"-> {cfg['out']/'qc'/'sanity.json'}")


if __name__ == "__main__":
    main()
