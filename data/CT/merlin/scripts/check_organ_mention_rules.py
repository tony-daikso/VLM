"""
Rule-based replacement for check_organ_mention.py (Step 1: organ mention detection).

Same input (merlin_report.json) and output format (merlin_report_mention.json) as the
official LLM script, but decisions are made by keyword / section-header rules so the
whole train split can be processed locally without an LLM API.

Two kinds of rules:
  - EXPLICIT: the organ or one of its sub-structures is named in the report
    (follows the "Supplementary anatomical knowledge" in the official prompts).
  - LENIENT: generic Merlin sections that do not name the organ are still mapped to
    an organ. Each lenient rule can be switched off in LENIENT_RULES
    (see RADAR_report_preprocess.md section 7 for the examples behind each rule).

Usage:
    python check_organ_mention_rules.py \
        --input  /path/to/merlin_report.json \
        --output /path/to/merlin_report_mention.json \
        [--strict]            # disable all lenient rules
        [--split train]
"""
import argparse
import json
import re
from collections import Counter

ORGANS = [
    "adrenal gland", "aorta", "large bowel", "duodenum", "esophagus", "gallbladder",
    "heart", "iliac artery", "iliac vena", "inferior vena cava", "kidney", "liver",
    "lung", "pancreas", "portal vein", "pulmonary artery", "rib", "sacrum",
    "small bowel", "spleen", "stomach", "trachea", "bladder", "cervical vertebrae",
    "thoracic vertebrae", "lumbar vertebrae",
]

# --> EXPLICIT: organ or sub-structure is named (case-insensitive regex).
# Vertebral / sacral levels (L3, T12, S1) are matched case-sensitively via (?-i:...).
# Rules were tuned against 300 LLM-labelled reports (RADAR_report_preprocess.md section 8).
EXPLICIT = {
    "adrenal gland": r"adrenal",
    # para-/peri-aortic and aortocaval are lymph node locations; aortic valve/annulus belong to the heart
    "aorta": r"(?<!para-)(?<!peri-)\baort(a|ic)\b(?! valv| annul)|\baorto(?!caval)",
    "large bowel": r"\bcolon|colonic|colect|colost|colorectal|\bcec(um|al)\b|(?<!peri-)(?<!peri)\brect(um|al)\b|sigmoid|append|anal canal"
                   r"|large (and|or|&) small bowel|large bowel|large intestin|hepatic flexure|splenic flexure"
                   r"|diverticulosis",
    "duodenum": r"duoden",
    "esophagus": r"(?<!para)(?<!peri)(?<!azygo)esophag",  # para-/peri-esophageal nodes/varices are locations
    "gallbladder": r"gall ?bladder|cholecyst|cholelith|gallstone",
    "heart": r"\bheart\b|cardiac|cardiomegaly|pericardi|myocard|coronary|\batri(um|al)\b|ventric(?!ulo-?peritoneal)",
    "iliac artery": r"iliac arter|iliac vessels|aorto-?(bi-?)?iliac",
    # "iliac vessels" covers veins too, except in atherosclerosis/calcification context ("aorta and iliac vessels")
    "iliac vena": r"iliac vein|iliac vena"
                  r"|(?<!aorta and )(?<!aortic and )(?<!calcified )(?<!calcification of the )(?<!atherosclerotic )"
                  r"(?<!atherosclerosis of the )(?<!atherosclerosis of )iliac vessels",
    "inferior vena cava": r"\bivc\b|vena cava",
    "kidney": r"kidney|\brenal\b|nephr|ureter|pyel",
    "liver": r"\bliver\b|hepat(?!ic flexure|ic arter|icojejun)|biliary",
    "lung": r"\blungs?\b|pulmonary(?! arter)|pleura|(upper|middle|lower) lobes?\b|lingula|atelecta|bronch"
            r"|emphysem(?!atous (cyst|chole|pyelo|gastr))|bibasilar|lung bases?",
    "pancreas": r"pancrea|whipple(?! disease)",  # Whipple disease is an infection, not the Whipple procedure
    # "portal venous (and delayed) phase(s)" is a CT phase
    "portal vein": r"\bportal\b(?! (venous )?(and \w+ )?phases?)|splenic vein|\bsmv\b|superior mesenteric vein",
    "pulmonary artery": r"pulmonary arter|pulmonary embol",
    "rib": r"\bribs?\b",
    # S2:/S4: are Stanford report summary codes, not sacral levels; pre-sacral is a location
    "sacrum": r"(?<!pre)(?<!pre-)sacr(um|al)\b|sacro-?iliac|lumbosacral|(?-i:\bS[1-5]\b(?!:))",
    # hepatico-/pancreatico-/choledocho-jejunostomy are biliary/pancreatic anastomoses; ileocolic nodes/vessels are locations
    "small bowel": r"small (and|or|&) large bowel|small bowel|small intestin|(?<!hepatico)(?<!pancreatico)(?<!choledocho)jejun"
                   r"|\bile(um|al)\b|ileocolic(?! (lymph|node|arter|vein|vessel))"
                   r"|ileocolonic|\bsbo\b",
    # splenic flexure = colon; splenic artery/vein and the portal-splenic confluence are vessels
    "spleen": r"spleen|splen(?!ic (flexure|arter|vein|confluence)|orenal)",
    # hiatal hernia / GE junction count as stomach; peri-/epi-gastric are locations
    "stomach": r"stomach|(?<!peri)(?<!epi)(?<!left )gastric(?! arter)|gastrectomy|gastrostomy|gastrojejun|gastro-?esophageal"
               r"|pylor|antrum|hiatal hernia",
    "trachea": r"trache",
    "bladder": r"(?<!gall)(?<!gall )\bbladder",
    "cervical vertebrae": r"cervical (spine|vertebra)|(?-i:\bC[1-7]\b)",
    "thoracic vertebrae": r"thoracic (spine|vertebra)|thoracolumbar|(?-i:\bT([1-9]|1[0-2])\b)",
    "lumbar vertebrae": r"lumbar|lumbosacral|thoracolumbar|(?-i:\bL[1-5]\b)",
}

# --> LENIENT: generic wording mapped to organs. Key = rule id in RADAR_report_preprocess.md section 7.
LENIENT_RULES = {
    # 1/7: any Vasculature section counts as describing the aorta
    "vasculature_to_aorta": (r"Vasculature:", ["aorta"]),
    # 2: femoral vessels are attributed to iliac vessels
    "femoral_to_iliac_artery": (r"femoral (vessel|arter)", ["iliac artery"]),
    "femoral_to_iliac_vena": (r"femoral (vessel|vein)", ["iliac vena"]),
    # 3: generic Musculoskeletal section counts as describing the vertebrae in the field of view
    "musculoskeletal_to_vertebrae": (r"Musculoskeletal:", ["thoracic vertebrae", "lumbar vertebrae"]),
    # 4: Lower thorax section counts as describing the lung
    "lower_thorax_to_lung": (r"Lower thorax:", ["lung"]),
    # 5: unspecified "spine"/"vertebra" counts as thoracic + lumbar
    "spine_to_vertebrae": (r"\bspine\b|spinal|vertebra", ["thoracic vertebrae", "lumbar vertebrae"]),
    # 6: unspecified "bowel" / GI tract section counts as small + large bowel
    "bowel_to_small_large": (r"\bbowel\b|Gastrointestinal tract:", ["small bowel", "large bowel"]),
    # 8: para-/peri-aortic location words count as aorta (already matched by EXPLICIT via "aortic")
    "paraaortic_to_aorta": (r"(para|peri)-?aortic", ["aorta"]),
    # extra: sub-structures implied by common procedures / terms
    "whipple_to_duodenum": (r"whipple", ["duodenum"]),
    "presacral_to_sacrum": (r"pre-?sacral", ["sacrum"]),
    "hiatal_hernia_to_stomach_esophagus": (r"hiatal hernia", ["stomach", "esophagus"]),
}


def normalize(report):
    report = report.strip().replace("\n", " ")
    return re.sub(r"\s{2,}", " ", report)


def detect(report, lenient=True):
    found = {}
    for organ, pattern in EXPLICIT.items():
        if re.search(pattern, report, re.IGNORECASE):
            found[organ] = "explicit"
    if lenient:
        for rule, (pattern, organs) in LENIENT_RULES.items():
            if re.search(pattern, report, re.IGNORECASE):
                for organ in organs:
                    found.setdefault(organ, f"lenient:{rule}")
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../ckpt/merlin_report.json")
    parser.add_argument("--output", default="./merlin_report_mention.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--strict", action="store_true", help="disable all lenient rules")
    args = parser.parse_args()

    data = json.load(open(args.input))
    new_info = {}
    source_counter = Counter()
    for patient_id, v in data.items():
        desc, conc = v["report"], v["impression"]
        if v["split"] != args.split:
            continue
        if not isinstance(desc, str) or not isinstance(conc, str):
            continue
        report = normalize(desc)
        found = detect(report, lenient=not args.strict)
        new_info[patient_id] = {
            "report": report,
            "mention": {organ: ("yes" if organ in found else "no") for organ in ORGANS},
            # extra field (not in the official format): why each organ was marked yes
            "mention_source": found,
        }
        source_counter.update(src.split(":")[0] for src in found.values())

    json.dump(new_info, open(args.output, "w"), ensure_ascii=False, indent=4)

    n = len(new_info)
    yes = Counter(o for v in new_info.values() for o, s in v["mention"].items() if s == "yes")
    print(f"processed {n} reports ({'strict' if args.strict else 'lenient'}), "
          f"avg yes/report = {sum(yes.values()) / max(n, 1):.1f}, sources = {dict(source_counter)}")
    for organ in ORGANS:
        print(f"  {organ:20s} {yes[organ]:6d} ({yes[organ] / max(n, 1) * 100:5.1f}%)")
    print(f"saved to {args.output}")
