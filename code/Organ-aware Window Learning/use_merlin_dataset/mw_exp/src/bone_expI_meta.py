"""Exp I: per-case metadata for the bone_matrix ids (train/val only; built from the analysis tables, which were themselves
built after filtering split.csv to train/val -- no label / report line of a test study is read here).
-> outputs_fracture/bones/expI/meta.csv  (+ levels column: reported vertebral levels within T8..L5, ';'-joined)
Columns: split, age, y, comp, mild, site_vert/rib/pelv/hip, grp(acute/old/not_stated/neg), cs (臨床顯著),
         hn_age75, hn_osteo, hn_degen, hn_scolio_kyph, hn_hardware, fov_out, levels"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
FR = MW / "outputs_fracture"
sys.path.insert(0, str(FR / "analysis_compression"))
from textparse import compression_levels, flags, msk_section, COV  # noqa: E402

OUT = FR / "bones/expI"; OUT.mkdir(parents=True, exist_ok=True)
z = np.load(FR / "bones/bone_matrix.npz", allow_pickle=True, mmap_mode="r")
ids = [str(i) for i in z["ids"]]; S = np.asarray(z["split"]).astype(str)
assert set(S) <= {"train", "val"}
t = pd.read_pickle(FR / "analysis_compression/table_trval.pkl")
assert set(t.split) <= {"train", "val"}
o = pd.read_pickle(FR / "analysis_old/table_old_flags.pkl")
assert set(o.split) <= {"train", "val"}
t = t.reindex(ids)
assert (t.split.values == S).all()
POSC, NEGC = ["yes", "probably_yes"], ["no", "probably_no", "not_mentioned"]
d = pd.DataFrame(index=pd.Index(ids, name="id"))
d["split"] = S
d["age"] = t.Age.astype(float)
d["y"] = np.where(t.cat.isin(POSC), 1.0, np.where(t.cat.isin(NEGC), 0.0, np.nan))
pos = d.y == 1
sites = t.sites.fillna("").str.split("|")
d["comp"] = pos & (t.comp == "yes")
q = t.quotes.fillna("")
qf = q.map(flags).apply(pd.Series)
d["mild"] = d.comp & qf["mild"] & ~qf["moderate_severe"]
d["site_vert"] = pos & sites.map(lambda L: "vertebra" in L)
d["site_rib"] = pos & sites.map(lambda L: "rib" in L)
d["site_pelv"] = pos & sites.map(lambda L: any(x in L for x in ("pelvis", "sacrum", "hip_femur")))
d["site_hip"] = pos & sites.map(lambda L: "hip_femur" in L)
d["grp"] = np.where(~pos, "neg", t.acute.map({"no": "old", "yes": "acute", "mixed": "acute"}).fillna("not_stated"))
# 臨床顯著 (analysis_old/analyze.py): acute OR (compression & moderate/severe words in quotes) OR (hip/femur & no hardware words in quotes)
ql = q.str.lower()
ms_kw = ql.str.contains(r"moderate|severe|marked|significant|burst|retropuls|plana|>\s?(?:40|50)", regex=True)
q_hw = ql.str.contains(r"fixation|\borif\b|screws?\b|\brods?\b|\bplate\b|hardware|(?<!ef)fusion|arthroplast|prosthe|intramedullary|\bnail|instrumentation|laminectomy", regex=True)
d["cs"] = pos & ((d.grp == "acute") | (d.comp & ms_kw) | (d.site_hip & ~q_hw))
# hard-negative flags (report text; analysis_old definitions + kyphosis)
rep = t.report.fillna("")
msk = rep.map(msk_section)
o = o.reindex(ids)
d["hn_age75"] = d.age >= 75
d["hn_osteo"] = rep.str.contains(r"osteopeni|osteopor|demineraliz", case=False, regex=True)
d["hn_degen"] = msk.str.contains(r"degenerat|spondylo|osteophyt|disc (?:space|height|bulg)|facet arthro", case=False, regex=True)
d["hn_scolio_kyph"] = rep.str.contains(r"scolio|kyphos|kyphotic|curvature", case=False, regex=True)
d["hn_hardware"] = o.r_hardware.fillna(False).astype(bool)
# metal / implant / arthroplasty anywhere in the report (analysis_rib pelv_info p_hardware OR analysis_old r_hardware)
d["hn_metal"] = d.hn_hardware | rep.str.contains(r"arthroplast|prosthe(?!tic (?:aortic |mitral |heart )?valve)|hardware|screw|fixation|\borif\b|\brods?\b|\bnail|\bplates?\b(?!-like)|joint replacement", case=False, regex=True)
# FOV / mask: reported vertebral levels all outside T8-L5, or rib-only fracture whose reported ribs are not in the masks
d["fov_out"] = pos & (o.vert_out_fov.fillna(False).astype(bool) |
                      (d.site_rib & ~d.site_vert & ~d.site_pelv & (o.rib_fov_frac == 0)))
# reported vertebral levels (T8..L5) for vertebral / compression positives; empty for everything else
clv = q.map(compression_levels)
d["levels"] = [";".join(l for l in L if l in COV) if (p_ and (sv or c)) else "" for L, p_, sv, c in zip(clv, pos, d.site_vert, d.comp)]
d.to_csv(OUT / "meta.csv")
T = d[d.split == "train"]
P = T[T.y == 1]; N = T[T.y == 0]
print("train pos", len(P), "neg", len(N), "| comp", int(P.comp.sum()), "mild", int(P.mild.sum()), "| levels named (vert/comp pos):",
      int((P.levels != "").sum()), "/", int((P.site_vert | P.comp).sum()), "| fov_out", int(P.fov_out.sum()), f"({P.fov_out.mean():.3f})")
hn = N[["hn_age75", "hn_osteo", "hn_degen", "hn_scolio_kyph", "hn_hardware"]]
print("metal neg rate", round(N.hn_metal.mean(), 3), "| rib-only pos", int((P.site_rib & ~P.site_vert & ~P.site_pelv & ~P.comp).sum()), "pelv-only", int((P.site_pelv & ~P.site_vert & ~P.site_rib & ~P.comp).sum()))
print("train neg flag rates:", hn.mean().round(3).to_dict(), "any:", round(hn.any(axis=1).mean(), 3),
      "narrow(any w/o degen):", round(hn.drop(columns="hn_degen").any(axis=1).mean(), 3))
print("cs train", int(T.cs.sum()), "val", int(d[d.split == "val"].cs.sum()))
