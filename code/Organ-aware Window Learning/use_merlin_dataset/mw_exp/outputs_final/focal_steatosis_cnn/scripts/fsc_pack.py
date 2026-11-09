"""Pack the Exp N fst crops (train/val ids in labels_trval.csv only) into memmaps for the CNN.
out: hu.i16 [N,3,48,48,48], msk.u8 [N,3,48,48,48] (bit0 r0 site-liver, bit1 r1 liver<=30mm of centre, bit2 r2 crop liver),
     meta.csv (study_id, split, fold, y, row, has_crop, present0..2, liver_med)   usage: fsc_pack.py WORK CROPS"""
import sys, os, numpy as np, pandas as pd
W, C = sys.argv[1], sys.argv[2]
L = pd.read_csv(f"{W}/labels_trval.csv"); assert set(L.split) <= {"train", "val"}
N, S = len(L), 48
hu = np.lib.format.open_memmap(f"{W}/hu.npy", "w+", np.int16, (N, 3, S, S, S))
mk = np.lib.format.open_memmap(f"{W}/msk.npy", "w+", np.uint8, (N, 3, S, S, S))
has = np.zeros(N, bool); pres = np.zeros((N, 3), bool); med = np.full(N, np.nan, np.float32)
for i, sid in enumerate(L.study_id):
    f = f"{C}/{sid}.npz"
    if not os.path.exists(f):
        hu[i] = -1024; continue
    z = np.load(f); H = z["hu"]; R = np.unpackbits(z["regions"])[: np.prod(z["rshape"])].reshape(z["rshape"]).astype(bool)
    hu[i] = H; mk[i] = R[:, 0] | (R[:, 1] << 1) | (R[:, 2] << 2)
    has[i] = True; pres[i] = z["present"] & R[:, 2].reshape(3, -1).any(1)
    v = H[R[:, 2] & pres[i][:, None, None, None]]
    if v.size > 100: med[i] = np.median(v)
    if i % 2000 == 0: print(i, N, flush=True)
hu.flush(); mk.flush()
L["row"] = np.arange(N); L["has_crop"] = has
for k in range(3): L[f"present{k}"] = pres[:, k]
L["liver_med"] = med
L.to_csv(f"{W}/meta.csv", index=False)
print("PACK_DONE", N, has.sum(), pres.sum(0), np.nanmedian(med), flush=True)
