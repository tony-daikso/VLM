#!/usr/bin/env python3
"""Shared helpers for the std14 standardisation: plain-text renderer and programmatic preservation checks."""
import re

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
# fixed section order of the rendered report
SECTIONS = [("Liver", ["hepatic_steatosis"]),
            ("Gallbladder / biliary", ["surgically_absent_gallbladder", "gallstones", "biliary_ductal_dilation"]),
            ("Spleen", ["splenomegaly"]),
            ("Kidneys", ["hydronephrosis"]),
            ("Vessels", ["abdominal_aortic_aneurysm", "atherosclerosis"]),
            ("Bowel", ["bowel_obstruction", "appendicitis"]),
            ("Peritoneum", ["ascites", "free_air"]),
            ("Bones", ["osteopenia", "fracture"])]
assert sorted(sum((d for _, d in SECTIONS), [])) == sorted(DISEASES)

# disease14 category -> std14 status
CAT2STATUS = {"yes": "positive", "probably_yes": "positive", "unclear": "uncertain", "probably_no": "negative",
              "no": "negative", "not_mentioned": "not_mentioned"}


def render(ans):
    """JSON answer -> plain-text unified report (only findings the report mentions; fixed section order)."""
    out = []
    if ans.get("clinical_info"):
        out.append(f"CLINICAL INFORMATION: {ans['clinical_info'].strip()}")
    if ans.get("comparison_ref"):
        out.append(f"COMPARISON: {ans['comparison_ref'].strip()}")
    out.append("FINDINGS:")
    for sec, ds in SECTIONS:
        sents = [ans["diseases"][d]["normalized_sentence"].strip() for d in ds
                 if ans["diseases"][d]["status"] != "not_mentioned" and ans["diseases"][d]["normalized_sentence"].strip()]
        sents = list(dict.fromkeys(sents))  # same sentence for two findings (e.g. blanket normal) once
        if sents:
            out.append(f"{sec}: " + " ".join(s if s.endswith(".") else s + "." for s in sents))
    comp = [f"{d}: {ans['diseases'][d]['comparison']}" for d in DISEASES
            if ans["diseases"][d]["comparison"] != "none" and ans["diseases"][d]["status"] != "not_mentioned"]
    if comp:
        out.append("CHANGE FROM PRIOR: " + "; ".join(comp))
    return "\n".join(out)


def render_findings_only(ans):
    """Findings text only (no clinical info / comparison) - what the disease14 re-extraction reads."""
    lines = []
    for sec, ds in SECTIONS:
        sents = [ans["diseases"][d]["normalized_sentence"].strip() for d in ds
                 if ans["diseases"][d]["status"] != "not_mentioned" and ans["diseases"][d]["normalized_sentence"].strip()]
        sents = list(dict.fromkeys(sents))
        if sents:
            lines.append(f"{sec}: " + " ".join(s if s.endswith(".") else s + "." for s in sents))
    return "Findings:\n" + ("\n".join(lines) if lines else "(none of the target findings mentioned)")


# ---------------------------------------------------------------- checks
WS = re.compile(r"\s+")
BULLET = re.compile(r"^\s*(?:[\*\-•#?]+|\(?\d{1,2}[\.\)])\s*")


def norm_ws(s):
    return WS.sub(" ", s).strip()


def loose(s):
    """lower-case, alphanumerics only - tolerant verbatim match (spacing / punctuation differences)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


NUMUNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?)\s*)?(?:x\s*(\d+(?:\.\d+)?)\s*)?(mm|cm)\b", re.I)
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")


def sizes_mm(text):
    """every size with a unit, converted to mm (rounded to 0.1)."""
    out = []
    for m in NUMUNIT.finditer(text):
        f = 10.0 if m.group(4).lower() == "cm" else 1.0
        for g in m.groups()[:3]:
            if g:
                out.append(round(float(g) * f, 1))
    return out


def all_numbers(text):
    return {round(float(x), 1) for x in re.findall(r"\d+(?:\.\d+)?", text)}


SEIM = re.compile(r"(?:se(?:ries)?\s*[:#.]?\s*\d+[\s,/\-]*im(?:age)?\s*[:#.]?\s*\d+(?:\s*[,~\-]\s*\d+)*)", re.I)


def seim_refs(text):
    return [loose(m.group(0)) for m in SEIM.finditer(text)]


TEMPORAL = re.compile(r"\b(stable|stationary|unchanged|new|newly|progressi\w*|progress\w*|increas\w*|decreas\w*|"
                      r"improv\w*|resolv\w*|persist\w*|again|compar\w*|interval|regress\w*|"
                      r"enlarging|shrink\w*)\b|->", re.I)


def check_one(report, ans):
    """Programmatic checks for one report. Returns a dict of counters and a list of issue strings."""
    rep_loose = loose(report)
    rep_sizes = set(sizes_mm(report))
    rep_nums = all_numbers(report)
    rep_seim = set(seim_refs(report))
    c = dict(src_total=0, src_verbatim=0, size_src=0, size_kept=0, size_norm=0, size_norm_ok=0,
             seim_src=0, seim_kept=0, seim_norm=0, seim_norm_ok=0, temporal_left=0, no_source=0, empty_norm=0, nonempty_nm=0)
    issues = []
    for d in DISEASES:
        x = ans["diseases"][d]
        st, ns = x["status"], x["normalized_sentence"] or ""
        if st == "not_mentioned":
            if (x["source_sentences"] or ns.strip()) and not x.get("postproc"):
                c["nonempty_nm"] += 1
            continue
        if not x["source_sentences"]:
            c["no_source"] = c.get("no_source", 0) + 1
            issues.append(f"{d}: {st} without any source sentence: {ns[:80]!r}")
        if not ns.strip():
            c["empty_norm"] += 1
            issues.append(f"{d}: empty normalized_sentence")
        for s in x["source_sentences"]:
            c["src_total"] += 1
            if loose(BULLET.sub("", s)) in rep_loose:
                c["src_verbatim"] += 1
            else:
                issues.append(f"{d}: source not verbatim: {s[:90]!r}")
        src = " ".join(x["source_sentences"])
        attr_txt = " ".join(str(v) for v in x["attributes"].values())
        out_sizes = set(sizes_mm(ns)) | {round(float(v), 1) for v in x["attributes"].values()
                                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for v in set(sizes_mm(src)):            # every size stated in the source -> in normalized (mm)
            c["size_src"] += 1
            if v in out_sizes or v in all_numbers(ns):
                c["size_kept"] += 1
            else:
                issues.append(f"{d}: size {v} mm from source missing in normalized")
        for v in set(sizes_mm(ns)):             # every size in normalized -> stated in the report
            c["size_norm"] += 1
            if v in rep_sizes or v in rep_nums:
                c["size_norm_ok"] += 1
            else:
                issues.append(f"{d}: size {v} mm in normalized not in report")
        ns_seim = seim_refs(ns)
        for r in set(seim_refs(src)):
            c["seim_src"] += 1
            if r in set(ns_seim) or r in loose(ns):
                c["seim_kept"] += 1
            else:
                issues.append(f"{d}: se/im {r} dropped")
        for r in set(ns_seim):
            c["seim_norm"] += 1
            if r in rep_seim or r in rep_loose:
                c["seim_norm_ok"] += 1
            else:
                issues.append(f"{d}: se/im {r} in normalized not in report")
        if TEMPORAL.search(ns):
            c["temporal_left"] += 1
            issues.append(f"{d}: temporal wording left: {ns[:90]!r}")
        _ = attr_txt
    return c, issues


# wording that legitimately supports a negative for each finding (finding-specific words + blanket normals)
BLANKET = r"unremarkable|unremakable|unremarkble|normal|remarkable|\(-\)|\[-\]|\[ *# *unremarkable|grossly|negative|intact|patent"
OK = {
    "splenomegaly": r"splenomegaly|splenogemaly|enlarg|size|cm|mm|volume",
    "hepatic_steatosis": r"fat|steato|density|attenuation|hu\b",
    "hydronephrosis": r"hydro|uropathy|pelvi|calyc|calic|dilat|obstruct|ureter",
    "gallstones": r"stone|lith|calcul|sludge",
    "biliary_ductal_dilation": r"dilat|ectasia|ihd|cbd|chd|duct|biliary",
    "bowel_obstruction": r"obstruct|ileus|dilat|distend",
    "abdominal_aortic_aneurysm": r"aneurysm|aaa|dilat|diameter|calib|ectatic|evar",
    "fracture": r"fractur|destruct|bony pathology|osseous|bone|bony|lytic",
    "ascites": r"ascit|fluid",
    "free_air": r"air|pneumoperitoneum|gas",
    "appendicitis": r"append",
    "osteopenia": r"osteop|density|demineral",
    "atherosclerosis": r"ather|calcif|plaque",
    "surgically_absent_gallbladder": r"gall|gb\b|cholecyst",
}
LESION = r"lesion|mass|nodul|tumou?r|sol\b|space.occupying|cyst|metasta|hypodens|hyperdens|enhanc"


LESION_FIX = {"splenomegaly", "hepatic_steatosis", "hydronephrosis", "gallstones", "bowel_obstruction", "fracture"}


def lesion_only(d, x):
    if x["status"] != "negative":
        return False
    src = " ".join(x["source_sentences"]).lower()
    return bool(re.search(LESION, src)) and not re.search(OK[d] + "|" + BLANKET, src)




NEG_START = re.compile(r"^\s*(no|normal|unremarkable|without)\b", re.I)


def postproc(ans):
    """Deterministic fix-ups after the model (recorded in x['postproc']), applied in check + finalize:
    - atherosclerosis whose only named vessel is coronary was sometimes returned as 'negative'; by the target
      definition (aorta / non-coronary branches) it is not mentioned -> status not_mentioned, record kept."""
    x = ans["diseases"]["atherosclerosis"]
    src = " ".join(x["source_sentences"]).lower()
    if (x["status"] == "negative" and "coron" in src and not NEG_START.match(x["normalized_sentence"])
            and not re.search(r"aort|iliac|arteries(?! *(at|may))|visceral|renal|carotid|vascul", src.replace("coronary arteries", ""))):
        x["status"] = "not_mentioned"
        x["postproc"] = "coronary_only_to_not_mentioned"
    # - negatives supported only by mass / lesion / nodule statements ("No definite lesions in the spleen")
    #   say nothing about the target finding -> not_mentioned (record and sentence kept).
    for d in LESION_FIX:
        y = ans["diseases"][d]
        if not y.get("postproc") and lesion_only(d, y):
            y["status"] = "not_mentioned"
            y["postproc"] = "lesion_only_to_not_mentioned"
    return ans
