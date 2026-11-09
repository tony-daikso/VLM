#!/usr/bin/env python3
"""Pack the Merlin OWL unit features into one token file per study for method 4 (runs on A6000, CPU only).

Read-only on ~/Desktop/Tony/VLM/OWL/mw_exp/{outputs,outputs_final,outputs_vert15,outputs_ribs15}; writes
~/Desktop/Tony/VLM/OWL/report_gen/m4/features/
    main/<study_id>.npz   tok float16 [T, 448]  one OWL multi-scale pooled vector (stage 1-3 = 64+128+256) per token
                          mask bool   [T]       False = unit absent / not extracted (vector zeros)
    token_meta.csv        tok, group, unit, region, window, pool   (identical for every study -> type embedding)
    manifest.csv          study_id, split, path, n_valid
TRAIN + VAL ONLY: split.csv is filtered before anything else; test studies are never opened.

Token groups (layout checked on A6000 2026-11-09, shapes per study):
  organ3   12 organ units x 5 windows, mean                outputs_final/emb/<win>/<id>.npz        emb [12,448]
  bone3    17 bone units x 3 windows x {mean,max}           outputs/emb_bones/<win>/<id>.npz       emb, emb_max [17,448]
  rib3     18 rib units  x 3 windows x {mean,max}           outputs/emb_ribs/<win>/<id>.npz        [18,448] (train+val)
  vert15   T8-L5 1.5 mm  x 3 windows x {mean,max}           outputs_vert15/merged_final/emb15/      emb [10,3,448]
  rib15    18 rib units 1.5 mm x 3 windows x {mean,max}     outputs_ribs15/emb15/                   emb [18,3,448]
  ru_*     region-unit crops [K,R,W,896=mean|max]           outputs_final/region_units/<u>/emb/     see RU below
-> T = 60 + 102 + 108 + 60 + 108 + 644 = 1082 tokens (~0.97 MB fp16 per study, ~20 GB for 20,364 train+val).
Light on purpose: single process; launch with nice -n 19 ionice -c3; ~2 GB RAM.
usage: python pack_features.py [--limit N] [--ids file] [--out DIR]
"""
import argparse
import csv
import os

import numpy as np

MW = os.path.expanduser("~/Desktop/Tony/VLM/OWL/mw_exp/")
SPLIT = os.path.expanduser("~/Desktop/Tony/VLM/OWL/data/split.csv")
OUT = os.path.expanduser("~/Desktop/Tony/VLM/OWL/report_gen/m4/features/")
WIN5 = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
WIN3 = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
POOL = ["mean", "max"]
# region units: name -> path (per-study npz with emb [K,R,W,896] or [R,W,896], present [K] or ())
RU = {
    "hydronephrosis": "outputs_final/region_units/hydronephrosis/emb",          # [2,4,5]  renal sinus
    "ascites": "outputs_final/region_units/ascites/emb",                        # [3,4,5]  peritoneal pockets
    "atherosclerosis": "outputs_final/region_units/atherosclerosis/emb",        # [4,2,6]  vessel wall
    "free_air": "outputs_final/region_units/free_air/emb",                      # [2,4,6]  subdiaphragm / anterior
    "gallstones": "outputs_final/region_units/gallstones/emb",                  # [1,3,7]  gallbladder
    "bile_duct": "outputs_final/region_units/bile_duct_r2/emb",                 # [1,3,5]  hilum
    "bowel": "outputs_final/region_units/bowel_obstruction_r2/emb",             # [2,3,5]
    "pelvis_fx": "outputs_final/region_units/fracture_pelvis_r2/emb",           # [4,2,5]
    "rlq": "outputs_final/appendicitis_rlq_unit/emb_rlq",                       # [4,5]    right lower quadrant
}


def load(p):
    return np.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = a.out.rstrip("/") + "/"
    os.makedirs(out + "main", exist_ok=True)

    split = {}
    with open(SPLIT) as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train", "val"):          # test rows dropped before any feature is opened
                split[r["study_id"]] = r["split"]
    ids = sorted(split)
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        ids = [i for i in ids if i in keep]
    if a.limit:
        ids = ids[: a.limit]
    ref = ids[0]

    # ---- token meta from the first study
    meta = []
    z = np.load(MW + f"outputs_final/emb/baseline/{ref}.npz")
    organs = [str(x) for x in z["organ_names"]]
    meta += [("organ3", u, "", w, "mean") for u in organs for w in WIN5]
    z = np.load(MW + f"outputs/emb_bones/baseline/{ref}.npz")
    bones = [str(x) for x in z["organ_names"]]
    meta += [("bone3", u, "", w, p) for u in bones for w in WIN3 for p in POOL]
    z = np.load(MW + f"outputs/emb_ribs/baseline/{ref}.npz")
    ribs = [str(x) for x in z["organ_names"]]
    meta += [("rib3", u, "", w, p) for u in ribs for w in WIN3 for p in POOL]
    z = np.load(MW + f"outputs_vert15/merged_final/emb15/{ref}.npz")
    vlev, vwin = [str(x) for x in z["levels"]], [str(x) for x in z["windows"]]
    meta += [("vert15", u, "", w, p) for u in vlev for w in vwin for p in POOL]
    z = np.load(MW + f"outputs_ribs15/emb15/{ref}.npz")
    r15u, r15w = [str(x) for x in z["units"]], [str(x) for x in z["windows"]]
    meta += [("rib15", u, "", w, p) for u in r15u for w in r15w for p in POOL]
    ru_shape = {}
    for g, path in RU.items():
        e = np.load(MW + f"{path}/{ref}.npz")["emb"]
        sh = e.shape[:-1] if e.ndim == 4 else (1,) + e.shape[:-1]
        ru_shape[g] = sh
        K, R, W = sh
        meta += [(f"ru_{g}", f"k{k}", f"r{r}", f"w{w}", p) for k in range(K) for r in range(R) for w in range(W)
                 for p in POOL]
    T = len(meta)
    with open(out + "token_meta.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tok", "group", "unit", "region", "window", "pool"])
        wr.writerows([i, *m] for i, m in enumerate(meta))
    print("tokens per study", T, ru_shape, flush=True)

    man = []
    for n, cid in enumerate(ids):
        tok = np.zeros((T, 448), np.float16)
        mask = np.zeros(T, bool)
        j = 0
        blk = np.zeros((len(organs), len(WIN5), 448), np.float16)
        bm = np.zeros((len(organs), len(WIN5)), bool)
        for wi, w in enumerate(WIN5):
            z = load(MW + f"outputs_final/emb/{w}/{cid}.npz")
            if z is not None:
                blk[:, wi], bm[:, wi] = z["emb"], z["organ_present"]
        tok[j:j + blk[:, :, 0].size] = blk.reshape(-1, 448)
        mask[j:j + bm.size] = bm.reshape(-1)
        j += bm.size
        for grp, units in (("emb_bones", bones), ("emb_ribs", ribs)):   # bone3 / rib3: [U x 3 windows x 2 pools]
            blk = np.zeros((len(units), len(WIN3), 2, 448), np.float16)
            bm = np.zeros((len(units), len(WIN3), 2), bool)
            for wi, w in enumerate(WIN3):
                z = load(MW + f"outputs/{grp}/{w}/{cid}.npz")
                if z is not None:
                    blk[:, wi, 0], blk[:, wi, 1] = z["emb"], z["emb_max"]
                    bm[:, wi, :] = z["organ_present"][:, None]
            tok[j:j + bm.size] = blk.reshape(-1, 448)
            mask[j:j + bm.size] = bm.reshape(-1)
            j += bm.size
        for path, nu, nw in (("outputs_vert15/merged_final/emb15", len(vlev), len(vwin)),
                             ("outputs_ribs15/emb15", len(r15u), len(r15w))):
            nt = nu * nw * 2
            z = load(MW + f"{path}/{cid}.npz")
            if z is not None:
                tok[j:j + nt] = np.stack([z["emb"], z["emb_max"]], 2).reshape(-1, 448)
                mask[j:j + nt] = np.repeat(z["present"].astype(bool), nw * 2)
            j += nt
        for g, path in RU.items():
            K, R, W = ru_shape[g]
            nt = K * R * W * 2
            z = load(MW + f"{path}/{cid}.npz")
            if z is not None:
                e = z["emb"].reshape(K, R, W, 2, 448)
                pres = np.atleast_1d(z["present"]).astype(bool)
                pk = np.broadcast_to(pres.reshape(-1, 1, 1, 1), (K, R, W, 2)) if pres.size == K else \
                    np.full((K, R, W, 2), bool(pres.all()))
                tok[j:j + nt] = e.reshape(-1, 448)
                mask[j:j + nt] = pk.reshape(-1)
            j += nt
        assert j == T, (j, T)
        mask &= np.abs(tok.astype(np.float32)).sum(1) > 0
        fo = out + f"main/{cid}.npz"
        np.savez(fo + ".tmp.npz", tok=tok, mask=mask)
        os.replace(fo + ".tmp.npz", fo)
        man.append((cid, split[cid], f"main/{cid}.npz", int(mask.sum())))
        if n % 500 == 0:
            print(n, cid, int(mask.sum()), flush=True)
    with open(out + ("manifest.csv" if not (a.limit or a.ids) else "manifest_partial.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["study_id", "split", "path", "n_valid"])
        wr.writerows(man)
    print("done", len(man), "studies,", T, "tokens", flush=True)


if __name__ == "__main__":
    main()
