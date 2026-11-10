"""Assemble CG10k per-case OWL embeddings into the same [N,29,5,448] float16 matrix layout as Merlin's
results/emb_matrix_<5 windows>+bones_emb_notest.npz (unit order, window order, bone-window fallback = soft_tissue,
bone pool = mean 'emb'), exactly like probe.load_data. usage: build_cg_matrix.py <owl set dir> <out.npz>"""
import sys, os, numpy as np
from concurrent.futures import ThreadPoolExecutor
S, OUT = sys.argv[1], sys.argv[2]
WK = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
BK = {"baseline": "baseline", "soft_tissue_L40_W400": "soft_tissue_L40_W400", "liver_L60_W150": "soft_tissue_L40_W400",
      "narrow_abd_L40_W250": "soft_tissue_L40_W400", "bone_L400_W1800": "bone_L400_W1800"}
MAIN = ["liver", "spleen", "pancreas", "kidney_left", "kidney_right", "gallbladder", "stomach", "adrenal", "aorta", "small_bowel", "colon", "lumbar_spine"]
BONE = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "ribs_left", "ribs_right", "sacrum", "hip_left", "hip_right", "femur_left", "femur_right"]
sets = [{f[:-4] for f in os.listdir(f"{S}/emb/{k}") if f.endswith(".npz") and not f.endswith(".tmp.npz")} for k in WK]
sets += [{f[:-4] for f in os.listdir(f"{S}/emb_bones/{k}") if f.endswith(".npz")} for k in set(BK.values())]
ids = sorted(set.intersection(*sets)); print("cases", len(ids), flush=True)
def one(cid):
    zs = [np.load(f"{S}/emb/{k}/{cid}.npz") for k in WK]
    mn = list(zs[0]["organ_names"]); assert mn == MAIN, mn
    e = np.stack([z["emb"] for z in zs], 1); pr = zs[0]["organ_present"]
    zb = {k: np.load(f"{S}/emb_bones/{k}/{cid}.npz") for k in set(BK.values())}
    bn = list(next(iter(zb.values()))["organ_names"]); bi = [bn.index(u) for u in BONE]
    eb = np.stack([zb[BK[k]]["emb"][bi] for k in WK], 1)
    return np.concatenate([e, eb.astype(e.dtype)], 0).astype(np.float16), np.concatenate([pr, next(iter(zb.values()))["organ_present"][bi]]).astype(bool)
with ThreadPoolExecutor(8) as ex: R = list(ex.map(one, ids))
E = np.stack([r[0] for r in R]); P = np.stack([r[1] for r in R])
np.savez(OUT, E=E, P=P, ids=np.array(ids), units=np.array(MAIN + BONE)); print(E.shape, P.mean(0).round(3))
