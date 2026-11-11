"""Option B (arm P_gt): Pillar-0 unit definitions + exact composite merge. numpy only (runs on CT-rate and A6000).

Unit set = the OWL units of m4/features/token_meta.csv in their token order, de-duplicated by anatomy (Pillar has one
vector per TotalSegmentator 'total' label and grid scale, so OWL's resolution duplicates collapse):
  organ3 (12) -> 12 units; bone3 (17) -> 17 units; rib3 (18) -> 18 units;
  vert15 (T8-L5) / rib15 = same anatomy as bone3 vertebrae / rib3 -> no extra unit;
  region units (ru_*) -> only those with a distinct TS composite are kept, appended at the end (5):
     ru_bowel k1 duodenum -> duodenum; ru_bile_duct -> portal/splenic vein (hilum proxy);
     ru_atherosclerosis k2 / k3 (common iliac R / L) -> iliac artery R / L; ru_ascites k2 (pelvic) -> urinary bladder.
     ru_hydronephrosis / ru_gallstones / ru_pelvis_fx / ru_bowel k0 / ru_atherosclerosis k0-k1 duplicate an organ or
     bone unit; ru_ascites k0-k1, ru_free_air, ru_rlq have no TS structure (covered by the 256 global grid tokens).
  -> 52 unit tokens.
TS ids: pillar/ts_total_classmap.json (1 spleen 2 kidney_right 3 kidney_left 4 gallbladder 5 liver 6 stomach 7 pancreas
8/9 adrenal R/L 18 small_bowel 19 duodenum 20 colon 21 urinary_bladder 25 sacrum 26 S1 27..36 L5..T8 52 aorta
64 portal_vein_and_splenic_vein 65/66 iliac_artery L/R 75/76 femur L/R 77/78 hip L/R 92..103 rib_left_1..12
104..115 rib_right_1..12).

Merge (exact): extract.py stores per label L the coverage-weighted mean vec_L = sum_c w_Lc f_c / w_L with
w_L = sum_c w_Lc (w64 / w16 = label volume in 64^3 / 16^3 cell units). A composite of labels S is therefore exactly
sum_{L in S} w_L vec_L / sum_{L in S} w_L, separately for vec16 (weights w16) and vec64 (weights w64).
Unit info (3 numbers): present (sum w16 > 0), crop coverage (sum vox_crop / sum vox_total, 0 when absent),
log1p(volume in crop, mL) / 5 (64^3 cell = 7.5 mm cube = 0.421875 mL).
"""
from __future__ import annotations

import json

import numpy as np

_V = {"T8": 36, "T9": 35, "T10": 34, "T11": 33, "T12": 32, "L1": 31, "L2": 30, "L3": 29, "L4": 28, "L5": 27}
_RL = {"left": 92, "right": 104}          # rib_<side>_1 id; rib n = base + n - 1

UNITS = [  # (unit name, OWL source, TS ids)
    ("liver", "organ3/liver", [5]), ("spleen", "organ3/spleen", [1]), ("pancreas", "organ3/pancreas", [7]),
    ("kidney_left", "organ3/kidney_left", [3]), ("kidney_right", "organ3/kidney_right", [2]),
    ("gallbladder", "organ3/gallbladder", [4]), ("stomach", "organ3/stomach", [6]),
    ("adrenal", "organ3/adrenal", [8, 9]), ("aorta", "organ3/aorta", [52]),
    ("small_bowel", "organ3/small_bowel", [18]), ("colon", "organ3/colon", [20]),
    ("lumbar_spine", "organ3/lumbar_spine", [31, 30, 29, 28, 27]),
] + [(v, f"bone3/{v}", [i]) for v, i in _V.items()] + [
    ("ribs_left", "bone3/ribs_left", list(range(92, 104))), ("ribs_right", "bone3/ribs_right", list(range(104, 116))),
    ("sacrum", "bone3/sacrum", [25, 26]), ("hip_left", "bone3/hip_left", [77]), ("hip_right", "bone3/hip_right", [78]),
    ("femur_left", "bone3/femur_left", [75]), ("femur_right", "bone3/femur_right", [76]),
] + [(f"rib_{s}_{n}", f"rib3/rib_{s}_{n}", [_RL[s] + i - 1 for i in ((1, 2, 3, 4) if n == "1_4" else (int(n),))])
     for s in ("left", "right") for n in ["1_4", "5", "6", "7", "8", "9", "10", "11", "12"]] + [
    ("duodenum", "ru_bowel/k1", [19]), ("portal_splenic_vein", "ru_bile_duct/k0", [64]),
    ("iliac_artery_right", "ru_atherosclerosis/k2", [66]), ("iliac_artery_left", "ru_atherosclerosis/k3", [65]),
    ("urinary_bladder", "ru_ascites/k2", [21]),
]
N_UNITS = len(UNITS)
UNIT_NAMES = [u[0] for u in UNITS]
UNIT_DIM = 768            # vec16 (384) | vec64 (384)
INFO_DIM = 3              # present, coverage, log1p(mL)/5
GRID = 16                 # grid16 = 16^3 cells of 24 voxels (30 mm), raster (z, y, x) of the 384^3 input
CELL_VOX = 24
ML_PER_CELL64 = 0.421875
assert N_UNITS == 52 and len(set(UNIT_NAMES)) == N_UNITS


def check_classmap(path):
    """sanity: the TS ids above carry the names this file assumes."""
    cm = {int(k): v for k, v in json.load(open(path)).items()}
    want = {5: "liver", 1: "spleen", 2: "kidney_right", 3: "kidney_left", 19: "duodenum", 21: "urinary_bladder",
            25: "sacrum", 26: "vertebrae_S1", 52: "aorta", 64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left",
            66: "iliac_artery_right", 75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
            92: "rib_left_1", 103: "rib_left_12", 104: "rib_right_1", 115: "rib_right_12"}
    want.update({i: f"vertebrae_{v}" for v, i in _V.items()})
    bad = {k: (v, cm.get(k)) for k, v in want.items() if cm.get(k) != v}
    if bad:
        raise ValueError(f"TS class map differs: {bad}")
    for name, _, ids in UNITS:
        for i in ids:
            assert i in cm, (name, i)
    return True


def merge_units(vec16, vec64, w16, w64, vox_crop, vox_total):
    """one case -> (units [52, 768] float32, info [52, 3] float32). Inputs as stored by extract.py."""
    vec16 = np.asarray(vec16, np.float32)
    vec64 = np.asarray(vec64, np.float32)
    w16 = np.nan_to_num(np.asarray(w16, np.float64))
    w64 = np.nan_to_num(np.asarray(w64, np.float64))
    out = np.zeros((N_UNITS, UNIT_DIM), np.float32)
    info = np.zeros((N_UNITS, INFO_DIM), np.float32)
    for u, (_, _, ids) in enumerate(UNITS):
        ids = np.asarray(ids)
        s16, s64 = w16[ids].sum(), w64[ids].sum()
        if s16 > 0:
            out[u, :384] = (w16[ids, None] * vec16[ids]).sum(0) / s16
        if s64 > 0:
            out[u, 384:] = (w64[ids, None] * vec64[ids]).sum(0) / s64
        tot = float(np.asarray(vox_total)[ids].sum())
        info[u, 0] = float(s16 > 0 and s64 > 0)
        info[u, 1] = float(np.asarray(vox_crop)[ids].sum()) / tot if tot > 0 else 0.0
        info[u, 2] = np.log1p(s64 * ML_PER_CELL64) / 5.0
    return out, info


def z_valid_cells(plan):
    """meta['plan'] = [[z0, z1, zd], ...] (1.25 mm slices) -> (first, last) grid16 z-cell that holds image data
    (cells entirely inside the -1024 z padding are masked out of the global pooling)."""
    z0, z1, zd = plan[0]
    lo, hi = zd, zd + (z1 - z0)                  # valid slices [lo, hi)
    cells = [k for k in range(GRID) if k * CELL_VOX < hi and (k + 1) * CELL_VOX > lo]
    return (cells[0], cells[-1]) if cells else (0, GRID - 1)
