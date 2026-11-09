#!/usr/bin/env python3
"""S1: sentence bank per LOC x organ slot x polarity from all 10k CG reports (rule / keyword only, no Qwen).

Findings-section sentences only (text before the first IMP/IMPRESSION header; indication / technique lines
dropped). Each sentence gets the set of slots whose keyword regex matches and a polarity (neg = negation cue).
Canonical sentence per LOC x slot x polarity = most frequent "clean" sentence (only that slot matched, no
numbers / se-im / comparison words / grading words, <= 90 chars); manual fallbacks when a site has none.

Outputs (data/.../report_gen/): sentence_bank.csv (top 25 per LOC x slot x polarity, all reports counted
once per sentence), canonical_sentences.csv (the sentence S5 uses), s1_coverage.csv.
"""
import collections
import re

import pandas as pd

from common import OUT, read_reports

# slot -> (type, regex). type "model" = rendered from OWL decisions; "na" = [NOT ASSESSED] normal sentence.
SLOTS = {
    "hepatic_steatosis": ("model", r"fatty (liver|change of (the )?liver|metamor|infiltration)|steatosis"),
    "surgically_absent_gallbladder": ("model", r"cholecystectomy"),
    "gallstones": ("model", r"gall ?stone|gb stone|gall ?bladder stone|cholelith"),
    "gallbladder_normal": ("model", r"^(unremarkable|normal)( study of)?( the)? gall ?bladder$|^gb: ?normal$"
                                    r"|^normal gall ?bladder without stone$|^unremarkable gall ?bladder without stone$"),
    "biliary_ductal_dilation": ("model", r"biliary|bile duct|\bihds?\b|\bcbd\b"),
    "splenomegaly": ("model", r"splenomegaly"),
    "hydronephrosis": ("model", r"hydroneph|obstructive uropathy|hydroureter"),
    "abdominal_aortic_aneurysm": ("model", r"aneurysm|diameter of the aorta|abdominal aorta (is )?normal|^aorta normal"),
    "atherosclerosis": ("model", r"atheros|athersc|calcified (atherosclerotic )?plaque|vascular calcification"),
    "bowel_obstruction": ("model", r"(bowel|intestinal|gi tract) obstruction|bowel patent|patent gi tract|"
                                   r"bowel (loop )?dilatation|dilatation of bowel|ileus"),
    "appendicitis": ("model", r"appendi(x|citis|cities|colith)"),
    "ascites": ("model", r"ascites|fluid collection in (the )?(abdomen|peritoneal)|peritoneal fluid|free fluid"),
    "free_air": ("model", r"free air|pneumoperiton|free gas"),
    "osteopenia": ("model", r"osteopen|osteopor|bone density|deminerali"),
    "fracture": ("model", r"fracture"),
    # not assessed by the model
    "liver_lesion": ("na", r"(space occupying lesion|tumor|mass|nodule|focal lesion).* liver|liver (tumor|mass)"),
    "portal_vein": ("na", r"portal (vein|trunk|venous)|hepatic vasculature|hepatic veins"),
    "pancreas": ("na", r"pancrea|p-duct"),
    "adrenal": ("na", r"adrenal"),
    "lymph_nodes": ("na", r"lymph|\blns?\b|l/ns?\b|\blap\b|nodes|lymphadenopathy"),
    "urinary_bladder": ("na", r"(?<!gall )(?<!gall)bladder"),
    "lower_chest": ("na", r"pleural effusion|lung|pneumothorax"),
    "mesentery": ("na", r"mesenter|omentum|retroperitoneum"),
    "bone_lesion": ("na", r"destructive|bony destruction|bone destruction|osteolytic|bony lesion|bony structure lesion"),
    "kidney_other": ("na", r"\bkidneys?\b|\brenal"),
}
# a pancreas sentence that also names spleen / liver / kidney / GB is a blanket normal that would also speak for
# model-covered findings; these organ words disqualify a sentence from being canonical for an NA slot.
COVERED_ORGANS = {"liver": r"liver", "spleen": r"spleen|splen", "gb": r"gall ?bladder|\bgb\b",
                  "kidney": r"kidney|\brenal", "aorta": r"aort", "bowel": r"bowel|gi tract",
                  "bone": r"\bbone|bony", "ascites": r"ascites"}
NA_OWN = {"liver_lesion": "liver", "bone_lesion": "bone", "kidney_other": "kidney"}

NEG = re.compile(r"^(no|not|without|negative|unremarkable|normal|clear|patent|patency|grossly unremarkable|"
                 r"grossly normal|there (is|are) no|grossly, no|nor)\b|\bno\b|\bwithout\b|unremarkable|\(-\)|"
                 r"\bnormal\b|\bnot\b|\bpatent\b|patency", re.I)
HEDGE = re.compile(r"suspici|possib|probab|likely|r/o|ddx|cannot|could|may |favor|consider|suggest|\?|versus|vs\.", re.I)
DIRTY = re.compile(r"\d|\bse\b|\bim\b|series|image|stationary|stable|compar|previous|progress|regress|interval|"
                   r"resolv|decrease|increase|unchanged|again|mild|moderate|severe|marked|heavy|minimal|massive|"
                   r"small|large|tiny|multiple|bilateral|bil\.|left|right|but|ct number|imp|little|amount|:|;|>|\*|\[", re.I)
STOP_LINE = re.compile(r"^\s*(imp\b|imp\.|impression|conclusion|diagnosis)", re.I)
SKIP_LINE = re.compile(r"^\s*(indications?|clinical information|brief hx|history|hx|techniques?|technique of|"
                       r"contrast|from (lower|upper)|coronal|axial|scanning|computed tomography|ct (scan|of|study)|"
                       r"exam:|a \d+ ?y/?o)", re.I)

# manual fallbacks (used when a site has no clean canonical sentence of its own); generic CG-style wording
FALLBACK = {
    ("hepatic_steatosis", "neg"): "No fatty liver",
    ("hepatic_steatosis", "pos"): "Fatty liver",
    ("surgically_absent_gallbladder", "pos"): "S/P cholecystectomy",
    ("gallstones", "neg"): "No gallbladder stone",
    ("gallstones", "pos"): "Gallstones",
    ("gallbladder_normal", "neg"): "Unremarkable gallbladder",
    ("biliary_ductal_dilation", "neg"): "No biliary tree dilatation",
    ("biliary_ductal_dilation", "pos"): "Biliary tree dilatation",
    ("splenomegaly", "neg"): "No splenomegaly",
    ("splenomegaly", "pos"): "Splenomegaly",
    ("hydronephrosis", "neg"): "No hydronephrosis",
    ("hydronephrosis", "pos"): "Hydronephrosis",
    ("abdominal_aortic_aneurysm", "neg"): "No abdominal aortic aneurysm",
    ("abdominal_aortic_aneurysm", "pos"): "Abdominal aortic aneurysm",
    ("atherosclerosis", "neg"): "No atherosclerotic calcification of the aorta",
    ("atherosclerosis", "pos"): "Atherosclerosis of aorta and its branches",
    ("bowel_obstruction", "neg"): "No bowel obstruction",
    ("bowel_obstruction", "pos"): "Bowel obstruction",
    ("appendicitis", "neg"): "No dilated appendix",
    ("appendicitis", "pos"): "Acute appendicitis",
    ("ascites", "neg"): "No ascites",
    ("ascites", "pos"): "Ascites",
    ("free_air", "neg"): "No pneumoperitoneum",
    ("free_air", "pos"): "Pneumoperitoneum",
    ("osteopenia", "neg"): "No abnormal bone density",
    ("osteopenia", "pos"): "Osteopenia",
    ("fracture", "neg"): "No definite bony fracture",
    ("fracture", "pos"): "Fracture",
    ("liver_lesion", "neg"): "No definite space occupying lesion in the liver",
    ("portal_vein", "neg"): "Patency of portal vein",
    ("pancreas", "neg"): "Unremarkable pancreas",
    ("adrenal", "neg"): "No adrenal mass",
    ("lymph_nodes", "neg"): "No enlarged lymph nodes in the abdomen and pelvis",
    ("urinary_bladder", "neg"): "Unremarkable urinary bladder",
    ("lower_chest", "neg"): "No pleural effusion",
    ("mesentery", "neg"): "Clear mesentery",
    ("bone_lesion", "neg"): "No destructive bony lesion",
}
# canonical positive sentences must name the finding itself (not e.g. "liver cirrhosis with splenomegaly")
POS_MUST = {"surgically_absent_gallbladder": r"^s/p cholecystectomy$", "abdominal_aortic_aneurysm": r"aneurysm",
            "appendicitis": r"appendicitis", "osteopenia": r"osteopen|osteopor", "hydronephrosis": r"hydroneph",
            "biliary_ductal_dilation": r"dilat", "bowel_obstruction": r"obstruct|ileus", "free_air": r"air|pneumoperit",
            "gallstones": r"stone", "ascites": r"ascites"}
NEG_MUST = {"abdominal_aortic_aneurysm": r"aneurysm|diameter", "appendicitis": r"appendi",
            "osteopenia": r"bone density|osteopen|osteopor", "fracture": r"fracture",
            "atherosclerosis": r"aort", "gallstones": r"^(no|without)\b", "lower_chest": r"pleural effusion|lung base|lower lung"}
NEG_NOT = {"free_air": r"fluid|ascites", "fracture": r"skull|orbit|nasal|facial|mandib", "ascites": r"free air|pneumoperit"}
# a canonical negative sentence must read as a negative statement from its start (or end in unremarkable/(-))
NEG_FORM = re.compile(r"^(no|not|without|normal|unremarkable|grossly unremarkable|patent|patency|clear|there (is|are) no)\b|"
                      r"(unremarkable|normal|patent|\(-\))$", re.I)


def findings_sentences(report):
    out = []
    for line in str(report).replace("\r", "\n").split("\n"):
        if STOP_LINE.match(line):
            break
        line = line.strip()
        if not line or SKIP_LINE.match(line):
            continue
        for s in re.split(r"(?<=[a-z\)])\.\s+(?=[A-Z])", line):
            s = re.sub(r"^\s*(\(?\d+[\.\)]|[-\*•>#]+)\s*", "", s).strip().rstrip(".").strip()
            s = re.sub(r"\s+", " ", s)
            if len(s) >= 4:
                out.append(s)
    return out


COMPILED = {k: re.compile(v, re.I) for k, (_, v) in SLOTS.items()}


def slots_of(s):
    low = s.lower()
    hit = [k for k, p in COMPILED.items() if p.search(low)]
    if "gallbladder_normal" in hit:
        hit = [h for h in hit if h not in ("gallstones",)]
    return hit


def polarity(s):
    return "neg" if NEG.search(s) else "pos"


def main():
    df = read_reports()
    cnt = collections.defaultdict(collections.Counter)  # (loc, slot, pol) -> lowered sentence -> n reports
    single = collections.defaultdict(set)  # lowered sentences that match exactly one slot
    casing = collections.defaultdict(collections.Counter)
    cov = collections.Counter()
    for loc, rep in zip(df.LOC, df.Report):
        seen = set()
        for s in findings_sentences(rep):
            low = s.lower()
            hs = slots_of(low)
            cov[(loc, "n_sent")] += 1
            cov[(loc, "n_slotted")] += bool(hs)
            casing[low][s] += 1
            if len(hs) == 1:
                single[hs[0]].add(low)
            for h in hs:
                key = (loc, h, polarity(low), low)
                if key in seen:
                    continue
                seen.add(key)
                cnt[(loc, h, polarity(low))][low] += 1

    rows, canon = [], []
    for (loc, slot, pol), c in sorted(cnt.items()):
        for rank, (low, n) in enumerate(c.most_common(25), 1):
            rows.append(dict(LOC=loc, slot=slot, slot_type=SLOTS[slot][0], polarity=pol, rank=rank,
                             sentence=next(v for v, _ in casing[low].most_common() + [(low, 0)] if not v.isupper()), n_reports=n,
                             single_slot=low in single[slot], hedged=bool(HEDGE.search(low))))
    bank = pd.DataFrame(rows)
    bank.to_csv(OUT / "sentence_bank.csv", index=False)

    for loc in sorted(df.LOC.unique()):
        for slot, (typ, _) in SLOTS.items():
            for pol in ("neg", "pos"):
                if (slot, pol) not in FALLBACK:
                    continue
                c = cnt.get((loc, slot, pol), collections.Counter())
                pick, n = None, 0
                for low, k in c.most_common():
                    if low not in single[slot] or DIRTY.search(low) or HEDGE.search(low) or len(low) > 90:
                        continue
                    if pol == "pos" and slot in POS_MUST and not re.search(POS_MUST[slot], low):
                        continue
                    if pol == "neg" and slot in NEG_MUST and not re.search(NEG_MUST[slot], low):
                        continue
                    if typ == "na" and any(re.search(r, low) for o, r in COVERED_ORGANS.items() if NA_OWN.get(slot) != o):
                        continue
                    if pol == "neg" and (not NEG_FORM.search(low) or (slot in NEG_NOT and re.search(NEG_NOT[slot], low))):
                        continue
                    if k < 3:
                        break
                    pick, n = next(v for v, _ in casing[low].most_common() + [(low, 0)] if not v.isupper()), k
                    break
                src = "bank"
                if pick is None:
                    pick, src = FALLBACK[(slot, pol)], "fallback"
                pick = pick[0].upper() + pick[1:]
                canon.append(dict(LOC=loc, slot=slot, slot_type=typ, polarity=pol, sentence=pick, n_reports=n,
                                  source=src))
    can = pd.DataFrame(canon)
    can.to_csv(OUT / "canonical_sentences.csv", index=False)
    cv = pd.DataFrame([dict(LOC=loc, n_findings_sentences=cov[(loc, "n_sent")],
                            n_slotted=cov[(loc, "n_slotted")],
                            frac_slotted=round(cov[(loc, "n_slotted")] / max(cov[(loc, "n_sent")], 1), 3))
                       for loc in sorted(df.LOC.unique())])
    cv.to_csv(OUT / "s1_coverage.csv", index=False)
    print(cv.to_string(index=False))
    pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500)
    print(can.to_string(index=False))


if __name__ == "__main__":
    main()
