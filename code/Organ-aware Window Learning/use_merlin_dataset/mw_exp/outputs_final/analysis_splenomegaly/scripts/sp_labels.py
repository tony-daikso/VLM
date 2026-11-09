"""Per-study splenomegaly label / report-keyword table for train+val ONLY.
Split is filtered (split.csv + record split) BEFORE any label or report text is read. Test is never touched.
No report text is written out; only categories, the Qwen length_cm field and keyword flags."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_splenomegaly/data/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}   # 'unclear' -> NaN (as probe labels)
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
okids = set(spl[spl.isin(OK)].index)


def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan


rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or r.get("study_id") not in okids:
        continue
    g = r["answer"]["splenomegaly"]
    L = (g.get("detail") or {}).get("length_cm")
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], sp_cat=g["category"], y_sp=b(g["category"]),
                               len_cm_label=float(L) if L is not None else np.nan, _q=" ".join(g.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
SPL = re.compile(r"spleen|splen", re.I)
pat = {
    "kw_splenomegaly": r"splenomegal",
    "kw_enlarged": r"enlarged spleen|spleen (?:is |appears |remains |was )?(?:\w+ )?enlarged|enlargement of the spleen|splenic enlargement",
    "kw_mild": r"\bmild|minimal|slight|borderline|upper limits? of normal|top (?:limits? )?(?:of )?normal|upper normal|at the upper",
    "kw_borderline": r"borderline|upper limits? of normal|top (?:limits? )?(?:of )?normal|upper normal|at the upper limit",
    "kw_moderate": r"moderate",
    "kw_massive": r"massive|marked|severe|significant",
    "kw_normal_size": r"(?:spleen|splenic)[^.]{0,40}normal (?:in )?size|normal[- ]sized spleen|spleen is (?:unremarkable|normal)|spleen:? (?:unremarkable|normal|within normal)|no splenomegaly",
    "kw_measure": r"\d+(?:\.\d+)? ?(?:cm|mm)",
    "kw_splenectomy": r"splenectomy|spleen (?:is )?(?:surgically )?absent|absent spleen|status post .{0,20}splen|asplen",
    "kw_accessory": r"splenule|accessory spleen|splenosis|polysplen",
    "kw_lesion": r"infarct|cyst|lesion|hemangioma|mass|calcifi|granulom|hypodens|laceration|hematoma|nodule",
    "kw_uncertain": r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|likely|probable|suggest|could reflect|may reflect|nonspecific",
    "kw_history": r"history of|known|clinical|indication|reason for",
}
SYS = {   # whole-report systemic / confounder flags
    "rp_cirrhosis": r"cirrho|nodular (?:contour|liver|hepatic)|fibrosis",
    "rp_portal_htn": r"portal hypertension|varic|recanaliz|portosystemic|splenorenal|portal venous hypertension",
    "rp_ascites": r"ascites",
    "rp_heme": r"lymphoma|leukemi|myelofib|myeloprolif|polycythemia|sickle|thalass|hemoly|mononucleosis|\bebv\b|gaucher|sarcoid",
    "rp_lymphadenopathy": r"lymphadenopathy|enlarged (?:lymph )?nodes",
    "rp_transplant": r"transplant",
    "rp_hepatomegaly": r"hepatomegal|enlarged liver|liver is enlarged",
}
for sid, d in rows.items():
    q = d.pop("_q")
    txt = (rep.get(sid) or {}).get("report", "") if sid in okids else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if SPL.search(s)]
    stxt = " ".join(sents) + " " + q
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, stxt, re.I)))
        d["q_" + k] = int(bool(re.search(p, q, re.I)))
    for k, p in SYS.items():
        d[k] = int(bool(re.search(p, txt, re.I)))
    # any spleen length measurement in spleen sentences (cm); max value
    ms = []
    for s in sents:
        for m in re.finditer(r"(\d+(?:\.\d+)?) ?(cm|mm)", s, re.I):
            v = float(m.group(1)) / (10 if m.group(2).lower() == "mm" else 1)
            if 5 <= v <= 35:
                ms.append(v)
    d["len_cm_report"] = max(ms) if ms else np.nan
    d["n_spleen_sent"] = len(sents)
    d["spleen_mentioned"] = int(len(sents) > 0)
    d["has_report"] = int(bool(txt))
    d["quote_empty"] = int(not q.strip())
df = pd.DataFrame(rows.values())
df.to_csv(OUT + "labels_trval.csv", index=False)
t = df[df.split == "train"]
print(df.groupby("split").size().to_dict())
print("train cats:", t.sp_cat.value_counts().to_dict())
print("train pos", int(t.y_sp.sum()), "neg", int((t.y_sp == 0).sum()), "nan", int(t.y_sp.isna().sum()))
kc = [c for c in df.columns if c.startswith(("kw_", "rp_", "q_kw_"))]
print("pos rate:", t[t.y_sp == 1][kc].mean().round(3).to_dict())
print("neg rate:", t[t.y_sp == 0][kc].mean().round(3).to_dict())
print("len_cm_label pos describe:", t[t.y_sp == 1].len_cm_label.describe().round(2).to_dict())
print("len_cm_label neg describe:", t[t.y_sp == 0].len_cm_label.describe().round(2).to_dict())
