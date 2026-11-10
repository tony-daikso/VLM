"""Small check of cam_ext outside the masks.
ascites       : mean cam_ext in the perihepatic ring (10-30 mm from the liver mask, voxels nearer the liver than the
                spleen, not inside any TS main unit) + fluid fraction there (HU -10..35); report ascites (massive /
                increase) vs report-negative (ITK false positives + low-probability negative controls)
hydronephrosis: perirenal ring (10-30 mm from each kidney, nearer that kidney) mean cam_ext, reported side vs other side
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu, spearmanr

R = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
O, C = R / "owl_viz_cam", R / "owl_viz_cam/_cache"
lab = {json.loads(l)["study_id"]: json.loads(l)["answer"] for l in open(R / "disease14_qwen.jsonl")}


def load(d, cid):
    ext = np.asanyarray(nib.load(str(d / "cam_ext.nii.gz")).dataobj).transpose(2, 1, 0)
    p = np.load(C / "prep" / f"{cid}.npz")
    return ext, p["hu"], p["units"], list(p["unit_names"])


def ring(m, other, units, a=10, b=30):
    dm = distance_transform_edt(~m, sampling=3.0)
    do = distance_transform_edt(~other, sampling=3.0)
    return (dm > a) & (dm <= b) & (dm < do) & (units == 0)


print("== 腹水: perihepatic ring 10-30 mm")
rows = []
for d in sorted(list((O / "腹水").glob("EX*")) + list((O / "_neg_controls/腹水").glob("EX*"))):
    cid = d.name
    ext, hu, units, nm = load(d, cid)
    liv, spl = units == nm.index("liver") + 1, units == nm.index("spleen") + 1
    r = ring(liv, spl, units)
    a = lab[cid]["ascites"]
    pos = a["category"] in ("yes", "probably_yes")
    v = ext[r]
    rows.append((cid, pos, a["detail"]["amount"], float(v.mean()), float(ext[liv].mean()),
                 float(((hu[r] > -10) & (hu[r] < 35)).mean()), " | ".join(a["quotes"][:1])[:50]))
for x in rows:
    print(f"{x[0]} 報告腹水={'是' if x[1] else '否'} amount={x[2]:<10} ring={x[3]:6.2f} liver_in={x[4]:6.2f} fluidfrac={x[5]:.2f}  {x[6]}")
p = [x[3] for x in rows if x[1]]
n = [x[3] for x in rows if not x[1]]
print(f"ring mean: ascites+ {np.mean(p):.2f} (n={len(p)})  vs  report-neg {np.mean(n):.2f} (n={len(n)}); "
      f"Mann-Whitney p={mannwhitneyu(p, n).pvalue:.3g}")
print(f"Spearman(ring response, ring fluid fraction) = {spearmanr([x[3] for x in rows], [x[5] for x in rows]).correlation:.2f}")

print("\n== 腎水腫: perirenal ring 10-30 mm, reported side vs other")
for d in sorted((O / "腎水腫").glob("EX*")):
    cid = d.name
    side = lab[cid]["hydronephrosis"]["detail"]["side"]
    ext, hu, units, nm = load(d, cid)
    kl, kr = units == nm.index("kidney_left") + 1, units == nm.index("kidney_right") + 1
    if kl.sum() < 100 or kr.sum() < 100:
        print(cid, side, "single kidney - skipped")
        continue
    rl, rr = ext[ring(kl, kr, units)].mean(), ext[ring(kr, kl, units)].mean()
    il, ir = ext[kl].mean(), ext[kr].mean()
    hit = "" if side not in ("left", "right") else ("一致" if (rl > rr) == (side == "left") else "不一致")
    print(f"{cid} side={side:<10} ring L={rl:6.2f} R={rr:6.2f}  inside L={il:6.2f} R={ir:6.2f}  ring 高側與報告{hit}")
