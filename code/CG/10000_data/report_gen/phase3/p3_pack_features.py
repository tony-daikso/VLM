#!/usr/bin/env python3
"""Pack the OWL features of the CG10k C+ main exams into one token file per exam for Phase 3 (runs on A6000).

Reads ONLY (never writes) ~/Desktop/Tony/VLM/OWL/cg10k_best/{mat,frac,bemb,feat}; writes
~/Desktop/Tony/VLM/report_gen_p3/features/main/<EXAMID>.npz  with
    tok  float16 [T, 448]   one OWL multi-scale pooled vector (stages 1-3 = 64+128+256) per token
    mask bool    [T]        False = unit absent / not extracted (vector is zeros)
and features/token_meta.csv (token index -> group, unit, sub-region, window, pool), identical for every exam,
features/manifest.csv (EXAMID, path, n_valid), features/handcrafted/*.csv (copy of feat/main, HU/geometry).

Token groups (T = 1062):
  organ3     12 organ units x 5 windows, mean pool                       (mat/cg_main.npz E[:, :12])
  bone3      17 bone units x 3 windows x {mean,max}                       (frac/cg_main_bones.npz)
  rib3       18 rib units  x 3 windows x {mean,max}                       (frac/cg_main_ribs3.npz)
  vert15     T8-L5 1.5 mm  x 3 windows x {mean,max}                       (frac/v15_main/<id>.npz)
  rib15      18 rib units 1.5 mm x 3 windows x {mean,max}                 (frac/r15_main/<id>.npz, 6935 exams)
  ru_<name>  specialised region units (renal sinus hn, ascites pocket asc, vessel wall ath, free-air fa,
             pelvis frp, gallbladder gs/gs5, hilum bd5, RLQ rlq): [K, R, W, 896=mean|max] split into 2 tokens
Light on purpose: single process, nice/ionice from the caller, ~3 GB RAM.
usage: python p3_pack_features.py [--limit N]
"""
import argparse
import csv
import os
import shutil

import numpy as np

H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/cg10k_best/")
O = os.path.expanduser("~/Desktop/Tony/VLM/report_gen_p3/features/")
RU = ["hn", "asc", "ath", "fa", "frp", "gs", "gs5", "bd5", "rlq"]
POOL = ["mean", "max"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(O + "main", exist_ok=True)
    m = np.load(H + "mat/cg_main.npz")
    ids = [str(x) for x in m["ids"]]
    units = [str(x) for x in m["units"]]
    E_org, P_org = m["E"][:, :12], m["P"][:, :12]
    WIN5 = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
    b = np.load(H + "frac/cg_main_bones.npz")
    assert [str(x) for x in b["ids"]] == ids
    E_b, P_b, bunits = b["E"], b["P"], [str(x) for x in b["units"]]
    r = np.load(H + "frac/cg_main_ribs3.npz")
    assert [str(x) for x in r["ids"]] == ids
    E_r, P_r, runits = r["E"], r["P"], [str(x) for x in r["units"]]
    WIN3 = ["baseline", "soft_tissue", "bone"]

    # ---- token meta (built from the first exam's per-case files)
    meta = []
    for u in range(12):
        for w in range(5):
            meta.append(("organ3", units[u], "", WIN5[w], "mean"))
    for u in bunits:
        for w in WIN3:
            for p in POOL:
                meta.append(("bone3", u, "", w, p))
    for u in runits:
        for w in WIN3:
            for p in POOL:
                meta.append(("rib3", u, "", w, p))
    z0 = np.load(H + f"frac/v15_main/{ids[0]}.npz")
    vlev, vwin = [str(x) for x in z0["levels"]], [str(x) for x in z0["windows"]]
    for u in vlev:
        for w in vwin:
            for p in POOL:
                meta.append(("vert15", u, "", w, p))
    z1 = np.load(H + f"frac/r15_main/{ids[0]}.npz")
    r15u, r15w = [str(x) for x in z1["units"]], [str(x) for x in z1["windows"]]
    for u in r15u:
        for w in r15w:
            for p in POOL:
                meta.append(("rib15", u, "", w, p))
    ru_shape = {}
    for g in RU:
        e = np.load(H + f"bemb/{g}/main/{ids[0]}.npz")["emb"]
        sh = e.shape[:-1]
        if len(sh) == 2:          # [R, W] (rlq, bd5, gs5): one sub-unit
            sh = (1,) + sh
        ru_shape[g] = sh
        K, R, W = sh
        for k in range(K):
            for rr in range(R):
                for w in range(W):
                    for p in POOL:
                        meta.append((f"ru_{g}", f"k{k}", f"r{rr}", f"w{w}", p))
    T = len(meta)
    with open(O + "token_meta.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["tok", "group", "unit", "region", "window", "pool"])
        for i, row in enumerate(meta):
            wr.writerow([i, *row])
    print("tokens per exam", T, {g: ru_shape[g] for g in RU}, flush=True)

    man = []
    n = len(ids) if not a.limit else a.limit
    for i in range(n):
        cid = ids[i]
        fo = O + f"main/{cid}.npz"
        tok = np.zeros((T, 448), np.float16)
        mask = np.zeros(T, bool)
        j = 0
        x = E_org[i].reshape(60, 448)
        tok[j:j + 60] = x
        mask[j:j + 60] = np.repeat(P_org[i], 5)
        j += 60
        x = E_b[i].reshape(-1, 448)
        tok[j:j + len(x)] = x
        mask[j:j + len(x)] = np.repeat(P_b[i], 6)
        j += len(x)
        x = E_r[i].reshape(-1, 448)
        tok[j:j + len(x)] = x
        mask[j:j + len(x)] = np.repeat(P_r[i], 6)
        j += len(x)
        nv = len(vlev) * len(vwin) * 2
        p = H + f"frac/v15_main/{cid}.npz"
        if os.path.exists(p):
            z = np.load(p)
            tok[j:j + nv] = np.stack([z["emb"], z["emb_max"]], 2).reshape(-1, 448)
            mask[j:j + nv] = np.repeat(z["present"], len(vwin) * 2)
        j += nv
        nr = len(r15u) * len(r15w) * 2
        p = H + f"frac/r15_main/{cid}.npz"
        if os.path.exists(p):
            z = np.load(p)
            tok[j:j + nr] = np.stack([z["emb"], z["emb_max"]], 2).reshape(-1, 448)
            mask[j:j + nr] = np.repeat(z["present"], len(r15w) * 2)
        j += nr
        for g in RU:
            K, R, W = ru_shape[g]
            nt = K * R * W * 2
            p = H + f"bemb/{g}/main/{cid}.npz"
            if os.path.exists(p):
                z = np.load(p)
                if "emb" in z.files:
                    e = z["emb"].reshape(K, R, W, 2, 448)
                    pres = np.atleast_1d(z["present"]).astype(bool)
                    tok[j:j + nt] = e.reshape(-1, 448)
                    pk = np.broadcast_to(pres.reshape(-1, 1, 1, 1), (K, R, W, 2)) if pres.size == K else \
                        np.full((K, R, W, 2), bool(pres.all()))
                    mask[j:j + nt] = pk.reshape(-1)
            j += nt
        assert j == T
        mask &= np.abs(tok).sum(1) > 0
        np.savez(fo + ".tmp.npz", tok=tok, mask=mask)
        os.replace(fo + ".tmp.npz", fo)
        man.append((cid, f"main/{cid}.npz", int(mask.sum())))
        if i % 500 == 0:
            print(i, cid, int(mask.sum()), flush=True)
    with open(O + "manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["EXAMID", "path", "n_valid"])
        wr.writerows(man)
    hc = O + "handcrafted/"
    os.makedirs(hc, exist_ok=True)
    for fn in os.listdir(H + "feat/main"):
        if fn.endswith(".csv"):
            shutil.copy2(H + "feat/main/" + fn, hc + fn)
    open(O + "PACK_DONE", "w").write(f"{len(man)} exams, {T} tokens\n")
    print("done", len(man))


if __name__ == "__main__":
    main()
