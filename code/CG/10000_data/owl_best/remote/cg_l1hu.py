"""CG10k image-derived osteoporosis label (user definition 2026-11-06):
  "L1 trabecular ROI HU, continuous; <=110 HU positive (high specificity); <=160 screening positive.
   L1 fractured or sclerotic -> use L2; exclude bone cement."
ROI / HU = UNCHANGED Merlin implementation: mw_exp/src/geom15.py (Exp J) level_geom 'hu_body' = median HU of the
vertebral body in the mid-sagittal plane (slab x = median x +-2 vox, 1.5 mm Dataset292 seg15), body component eroded
2 px (3 mm), voxels > -1500 HU.  (Same value as osteopenia_v15 'hu15_L1' and osteopenia_labels 'L1 HU'.)
NEW rules (not in the Merlin code; parameters fixed here a priori, flagged for the user):
  fracture : Qwen fracture quotes (category yes/probably_yes) mention that level (regex \bL1\b / \bL2\b), OR
             [report has a vertebral fracture (sites contains 'vertebra' or vertebral_compression=yes) AND geom15 'loss'
             (max a/m/p height loss vs neighbours) >= 0.40 at that level].  Geometry alone is NOT used: on Merlin
             (labels not looked at) 20 % of all L1 have loss >= 0.20 and 10 % >= 0.375 -> segmentation noise.
  sclerosis: hu_body - hu_body(other level) > 100 HU, or hu_body > 500 HU (an absolute 300 HU cut-off would flag
             normal young bone: Merlin L1 hu_body 95th percentile = 310 HU).
  cement   : >= 5 % of the same ROI voxels > 1000 HU  -> level unusable
  choose L1 if usable, else L2 if usable, else excluded (reason logged).
Output: <out>/l1hu_levels.csv (geom15 levels + cement frac) and <out>/osteo_hu_label.csv
        (study_id, level_used, hu, y_hu110, y_hu160, excl_reason, L1/L2 diagnostics).
usage: cg_l1hu.py <seg15_dir> <qwen_fracture_csv> <out_dir> [nproc]"""
import sys, os, re, json, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import ndimage as ndi
G15 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geom15.py")   # verbatim copy of mw_exp/src/geom15.py
spec = importlib.util.spec_from_file_location("geom15", G15); g15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g15)
LOSS_T, SCL_ABS, SCL_REL, CEM_HU, CEM_FRAC = 0.40, 500.0, 100.0, 1000.0, 0.05


def cement_frac(path, level):
    """fraction of ROI voxels > 1000 HU, with exactly the ROI of geom15.level_geom (re-derived, same steps)."""
    z = np.load(path, allow_pickle=True); lab, labels = z["lab"], json.loads(str(z["labels"]))
    key = f"vertebrae_{level}"
    if key not in labels or "hu_slab" not in z: return np.nan
    V = lab == int(labels[key]); hu = z["hu_slab"]; x0 = int(z["slab_x0"])
    if V.sum() < 800: return np.nan
    xx = np.nonzero(V)[2]; xc = int(round(np.median(xx))); xs = slice(max(xc - 2, 0), xc + 3)
    occ = V[:, :, xs].mean(2); cc, k = ndi.label(occ > 0)
    comps = [(np.nonzero(cc == i)[1].max(), occ[cc == i].sum(), i) for i in range(1, k + 1) if occ[cc == i].sum() >= 20]
    if not comps: return np.nan
    big = max(c[1] for c in comps); ib = max([c for c in comps if c[1] >= 0.25 * big], key=lambda c: c[0])[2]
    core = ndi.binary_erosion(cc == ib, iterations=2)
    vals = [hu[:, :, x - x0][core & V[:, :, x]] for x in range(xs.start, min(xs.stop, V.shape[2])) if 0 <= x - x0 < hu.shape[2]]
    if not vals: return np.nan
    v = np.concatenate(vals); v = v[v > -1500]
    return float((v > CEM_HU).mean()) if len(v) > 20 else np.nan


def one(path):
    try:
        rows = g15.case_levels(path)
        for r in rows:
            if r["level"] in ("L1", "L2"): r["cement_frac"] = cement_frac(path, r["level"])
        return rows
    except Exception as e:
        return [{"id": os.path.basename(path)[:-4], "level": "ERR", "err": str(e)[:200]}]


if __name__ == "__main__":
    sd, qf, out = sys.argv[1], sys.argv[2], sys.argv[3]; npr = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.makedirs(out, exist_ok=True)
    files = sorted(os.path.join(sd, f) for f in os.listdir(sd) if f.endswith(".npz"))
    with Pool(npr) as p: R = [r for rr in p.imap_unordered(one, files, chunksize=8) for r in rr]
    df = pd.DataFrame(R); err = df[df.level == "ERR"]; df = df[df.level != "ERR"]
    lv = g15.add_ratios(df.drop(columns=[c for c in ("err",) if c in df]))
    lv.to_csv(os.path.join(out, "l1hu_levels.csv"), index=False)
    q = pd.read_csv(qf).set_index("EXAMID")       # EXAMID, fx_cat, fx_quote
    rows = []
    for cid in sorted(set(df.id) | set(err.id if len(err) else [])):
        g = lv[(lv.id == cid) & lv.level.isin(["L1", "L2"])].set_index("level")
        qq = q.loc[cid] if cid in q.index else None
        fxpos = qq is not None and qq.fx_cat in ("yes", "probably_yes")
        fxq = str(qq.fx_quote) if fxpos else ""
        fxvert = fxpos and ("vertebra" in str(qq.fx_sites) or str(qq.fx_comp) == "yes")
        d = {"study_id": cid}; usable = {}
        for L, O in (("L1", "L2"), ("L2", "L1")):
            if L not in g.index or np.isnan(g.at[L, "hu_body"]):
                usable[L] = "no_seg_or_hu"; continue
            h = g.at[L, "hu_body"]; lo = g.at[L, "loss"]; cf = g.at[L, "cement_frac"]
            ho = g.at[O, "hu_body"] if O in g.index else np.nan
            d.update({f"{L}_hu": h, f"{L}_loss": lo, f"{L}_cement_frac": cf})
            if cf >= CEM_FRAC: usable[L] = "cement"
            elif re.search(rf"\b{L}\b", fxq) or (fxvert and lo >= LOSS_T): usable[L] = "fracture"
            elif h > SCL_ABS or (not np.isnan(ho) and h - ho > SCL_REL): usable[L] = "sclerosis"
            else: usable[L] = "ok"
        d["L1_status"], d["L2_status"] = usable.get("L1"), usable.get("L2")
        if usable.get("L1") == "ok": d["level_used"], d["hu"] = "L1", d["L1_hu"]
        elif usable.get("L2") == "ok": d["level_used"], d["hu"] = "L2", d["L2_hu"]
        else: d["level_used"], d["hu"] = "", np.nan
        d["excl_reason"] = "" if d["level_used"] else f"L1:{usable.get('L1')};L2:{usable.get('L2')}"
        d["y_hu110"] = float(d["hu"] <= 110) if d["level_used"] else np.nan
        d["y_hu160"] = float(d["hu"] <= 160) if d["level_used"] else np.nan
        rows.append(d)
    o = pd.DataFrame(rows); o.to_csv(os.path.join(out, "osteo_hu_label.csv"), index=False)
    print("cases", len(o), "errors", len(err), o.level_used.value_counts().to_dict(), "y110", o.y_hu110.sum(), "y160", o.y_hu160.sum())
    print(o.excl_reason.value_counts().head(10).to_dict())
