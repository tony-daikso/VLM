#!/usr/bin/env python3
"""Option B (arm P_gt) unit tests. Three parts, each prints a JSON summary and exits non-zero on a failure.

  core   (anywhere, CPU, synthetic): exact composite merge vs direct recomputation from cell weights, z-valid cells,
         3-D position code, adapter shapes / no NaN / token order (global block first, unit k -> token n_global + k,
         blocks independent), z-padding cells really masked, missing units -> learned missing vector, ablation modes.
  npz    (CT-rate, numpy): packed store rows == the finished npz files (grid16 bitwise, units == merge_units, zvalid).
         python test_p_arm.py npz --store <store> --feats /media/daikso/Tony/pillar/merlin/feats
  data   (A6000, CPU): real tokenizer + real jsonl rows + store: shapes, finite, image-token count, placeholder order,
         z-valid flags, gt_dropout cue rebuilt for train rows, val rows untouched, adapter on a real batch.
         python test_p_arm.py data --store <store> --data <dir with train/val.jsonl> --tok <Lingshu-7B dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from pillar_units import (GRID, INFO_DIM, N_UNITS, UNIT_DIM, UNITS, check_classmap, merge_units,  # noqa: E402
                          z_valid_cells)


def core():
    import torch
    import p_arm
    res = {}
    rng = np.random.default_rng(0)
    # 1. exact merge: synthetic cells, label weights per cell, vec_L as extract.py computes it
    for scale in ("16", "64"):
        C, L = 300, 118
        f = rng.normal(size=(C, 384)).astype(np.float32)
        w = np.zeros((L, C))
        for lab in range(1, L):
            cells = rng.choice(C, size=rng.integers(0, 6), replace=False)
            w[lab, cells] = rng.random(len(cells))
        wl = w.sum(1)
        vec = np.zeros((L, 384), np.float32)
        nz = wl > 0
        vec[nz] = (w[nz] @ f) / wl[nz, None]
        vox = np.ceil(wl * 100).astype(np.int64)
        args = (vec, vec, wl, wl, vox, vox * 2)
        u, info = merge_units(*args)
        err = 0.0
        for k, (_, _, ids) in enumerate(UNITS):
            ws = w[ids].sum(0)
            if ws.sum() > 0:
                direct = (ws @ f) / ws.sum()
                err = max(err, float(np.abs(u[k, :384] - direct).max()), float(np.abs(u[k, 384:] - direct).max()))
                assert info[k, 0] == 1 and abs(info[k, 1] - 0.5) < 1e-6
            else:
                assert info[k, 0] == 0 and not u[k].any()
        res[f"merge_max_abs_err_{scale}"] = err
        assert err < 1e-4, err
    # 2. z-valid cells
    assert z_valid_cells([[0, 376, 4], [8, 392, 0], [8, 392, 0]]) == (0, 15)
    assert z_valid_cells([[0, 200, 92], [0, 384, 0], [0, 384, 0]]) == (3, 12)     # slices 92..291
    assert z_valid_cells([[10, 394, 0], [0, 384, 0], [0, 384, 0]]) == (0, 15)
    # 3. position code
    pe = p_arm.sincos_3d(GRID, 1024)
    assert pe.shape == (4096, 1024) and torch.isfinite(pe).all()
    assert len({tuple(r) for r in pe[:, :1020].round(decimals=4).tolist()}) == 4096, "position codes collide"
    # 4. adapter
    torch.manual_seed(0)
    H, B = 48, 3
    ad = p_arm.PillarAdapter(H, "both", n_global=16, dim=32, depth=2, heads=4).eval()
    grid = torch.randn(B, 4096, 385).half()
    grid[..., 384] = 1
    grid[1, :, 384] = (torch.arange(4096) // 256 >= 3).half()   # case 1: z-cells 0-2 padding
    units = torch.randn(B, N_UNITS, UNIT_DIM + INFO_DIM).half()
    units[..., UNIT_DIM] = 1
    units[2, 5, UNIT_DIM] = 0                                    # case 2 unit 5 missing
    with torch.no_grad():
        o = ad(grid, units)
        assert o.shape == (B, 16 + N_UNITS, H) and torch.isfinite(o).all()
        u2 = units.clone()
        u2[:, 7, :UNIT_DIM] += 1.0
        o2 = ad(grid, u2)
        d = (o2 - o).abs().amax(-1)
        assert d[:, 16 + 7].min() > 0 and d[:, :16 + 7].max() == 0 and d[:, 16 + 8:].max() == 0, "unit order"
        g2 = grid.clone()
        g2[..., :384] += 1.0
        d = (ad(g2, units) - o).abs().amax(-1)
        assert d[:, :16].min() > 0 and d[:, 16:].max() == 0, "global block order"
        g3 = grid.clone()
        g3[1, : 3 * 256, :384] = 50.0                            # padding cells of case 1 changed
        d = (ad(g3, units) - o).abs().amax(-1)
        assert d[1].max() == 0 and d[0].max() == 0, "z padding not masked"
        u3 = units.clone()
        u3[2, 5, :UNIT_DIM] = 99.0                               # features of a missing unit are ignored
        assert (ad(grid, u3) - o)[2, 16 + 5].abs().max() == 0
        gz = torch.zeros(1, 4096, 385).half()                    # all-missing image (no features at all)
        uz = torch.zeros(1, N_UNITS, UNIT_DIM + INFO_DIM).half()
        assert torch.isfinite(ad(gz, uz)).all()
    for mode, n in (("global", 16), ("unit", N_UNITS)):
        a2 = p_arm.PillarAdapter(H, mode, n_global=16, dim=32, depth=1, heads=4)
        gg = grid if mode == "global" else torch.zeros(B, 1, 385).half()
        assert a2(gg, units).shape == (B, n, H)
    # gradients reach every trainable tensor
    ad.train()
    ad(grid, units).float().pow(2).mean().backward()
    nog = [k for k, p in ad.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not nog, nog
    big = p_arm.PillarAdapter(3584, "both", 256)
    res["adapter_params_M_lingshu7b"] = round(sum(p.numel() for p in big.parameters()) / 1e6, 1)
    res["n_units"] = N_UNITS
    res["n_tokens_P_gt"] = 256 + N_UNITS
    cm = HERE.parent.parent.parent / "pillar" / "ts_total_classmap.json"
    for c in (cm, HERE / "ts_total_classmap.json"):
        if c.exists():
            res["classmap_ok"] = check_classmap(c)
            break
    return res


def npz_check(store, feats, n_max=50):
    s = Path(store)
    ids = (s / "ids.txt").read_text().split()
    a = {k: np.load(s / f"{k}.npy", mmap_mode="r") for k in ("grid16", "units", "uinfo", "zvalid", "done")}
    rows = [i for i in range(len(ids)) if a["done"][i]][:n_max]
    assert rows, "no packed rows"
    worst = 0.0
    for i in rows:
        z = np.load(Path(feats) / f"{ids[i]}.npz")
        assert np.array_equal(a["grid16"][i], z["grid16"]), ids[i]
        u, info = merge_units(z["vec16"], z["vec64"], z["w16"], z["w64"], z["vox_crop"], z["vox_total"])
        worst = max(worst, float(np.abs(a["units"][i].astype(np.float32) - u).max()))
        assert np.allclose(a["uinfo"][i], info)
        assert tuple(a["zvalid"][i]) == z_valid_cells(json.loads(str(z["meta"]))["plan"])
        assert np.isfinite(a["units"][i]).all()
        # single-label units reproduce the stored label vectors
        for k, (_, _, lab) in enumerate(UNITS):
            if len(lab) == 1 and z["w16"][lab[0]] > 0:
                assert np.allclose(a["units"][i][k, :384].astype(np.float32), z["vec16"][lab[0]].astype(np.float32),
                                   atol=2e-2)
    present = a["uinfo"][rows][..., 0]
    return {"rows_checked": len(rows), "units_fp16_max_abs_err": worst, "n_done": int(a["done"].sum()),
            "unit_present_rate": {UNITS[k][0]: round(float(present[:, k].mean()), 3) for k in range(N_UNITS)}}


def data_check(store, data, tok):
    import random
    import torch
    from transformers import AutoTokenizer
    import p_arm
    from m4_common import BACKBONES, parse_cue
    from m4_data import CueGT, collate
    p_arm.CFG.root, p_arm.CFG.mode, p_arm.CFG.n_global = store, "both", 256
    tk = AutoTokenizer.from_pretrained(tok)
    res = {}
    ds = p_arm.PDataset(Path(data) / "val.jsonl", tk, True, 1280, train=False)
    img_id = tk.convert_tokens_to_ids(BACKBONES["lingshu7b"]["img_token"])
    vs = tk.convert_tokens_to_ids(BACKBONES["lingshu7b"]["img_start"])
    ve = tk.convert_tokens_to_ids(BACKBONES["lingshu7b"]["img_end"])
    items = [ds[i] for i in range(min(4, len(ds)))]
    for it in items:
        p = it["prompt_ids"]
        pos = [k for k, x in enumerate(p) if x == img_id]
        assert len(pos) == 308 and pos == list(range(pos[0], pos[0] + 308)), "image placeholders not contiguous"
        assert p[pos[0] - 1] == vs and p[pos[-1] + 1] == ve
        assert it["tok"].shape == (4096, 385) and it["tok"].dtype == torch.float16
        assert it["mask"].shape == (N_UNITS, UNIT_DIM + INFO_DIM)
        assert torch.isfinite(it["tok"]).all() and torch.isfinite(it["mask"]).all()
        assert it["tok"][:, 384].sum() > 0
    b = collate(items, tk.pad_token_id, train=False)
    res["val_batch"] = {k: list(v.shape) for k, v in b.items() if hasattr(v, "shape")}
    res["val_zvalid_cells"] = [int(it["tok"][:, 384].sum()) // 256 for it in items]
    res["val_units_present"] = [int(it["mask"][:, UNIT_DIM].sum()) for it in items]
    ad = p_arm.PillarAdapter(64, "both", 256, dim=64, depth=1, heads=4)
    with torch.no_grad():
        o = ad(b["tok"], b["mask"])
    assert o.shape == (len(items), 308, 64) and torch.isfinite(o).all()
    # train rows: gt_dropout cue (sample, 0.3, unclear oof) as P_gt; val rows keep the classifier line
    cue = "/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/cue/cue_gt.json"
    if Path(cue).exists() and (Path(data) / "train.jsonl").exists():
        random.seed(0)
        dt = p_arm.PDataset(Path(data) / "train.jsonl", tk, True, 1280, train=True,
                            cue_gt=CueGT(cue, 0.3, "sample", unclear="oof"))
        it = dt[0]
        assert "cue_info" in it and len(it["input_ids"]) == len(it["labels"]) <= 1280
        res["train_item"] = {"len": len(it["input_ids"]), "n_target": sum(x != -100 for x in it["labels"]),
                             "cue_shown": len(it["cue_info"]["probs"]), "n_labelled": it["cue_info"]["n_labelled"]}
    dsv = p_arm.PDataset(Path(data) / "val.jsonl", tk, True, 1280, train=False)
    assert parse_cue(dsv.rows[0]["prompt"])[0], "val row has no classifier line"
    res["missing_in_store"] = {"val": ds.n_missing}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["core", "npz", "data"])
    ap.add_argument("--store", default="")
    ap.add_argument("--feats", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--tok", default="/home/advantech/Desktop/Tony/models/Lingshu-7B")
    a = ap.parse_args()
    if a.part == "core":
        r = core()
    elif a.part == "npz":
        r = npz_check(a.store, a.feats)
    else:
        r = data_check(a.store, a.data, a.tok)
    print(json.dumps({a.part: "PASS", **r}, indent=1))


if __name__ == "__main__":
    main()
