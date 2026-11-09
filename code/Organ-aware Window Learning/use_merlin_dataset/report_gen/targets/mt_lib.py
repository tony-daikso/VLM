#!/usr/bin/env python3
"""Merlin report_gen step 2: rule-based per-disease sentence extraction, cleaning and attributes.

Report format: a findings part organised by organ-section headers (separated by runs of spaces), then a numbered
impression (with or without an impression header). Report text is DUA-restricted and never stored in this folder.

Pipeline per report:
  parse_report  -> list of sentence items {i, part (findings|impression), sec, grp, text, pos}
  extract       -> per disease: matched sentence items + per-sentence rule status
  clean_sentence-> remove comparison wording (clause level), se/im refs, unmeasured sizes, typos
  attributes    -> per disease attribute dict from the matched (raw) sentences
Rules were developed on train/val only (never on test).
"""
import re

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}

# ------------------------------------------------------------------ parsing
WS = re.compile(r"\s+")


def norm_ws(s):
    return WS.sub(" ", s).strip()


def header_groups(h):
    """section header text -> set of organ groups (a combined header such as 'Gastrointestinal tract/peritoneal cavity'
    belongs to several)."""
    x = h.lower()
    g = set()
    if re.search(r"impression|conclusion|summary", x):
        return {"impression"}
    if re.search(r"gallbladder", x):
        g.add("gb")
        if re.search(r"bil|bile|duct", x):
            g.add("biliary")
    if re.search(r"liver|hepat(?!ic\s+vascul)", x):
        g.add("liver")
        if re.search(r"bil|bile", x):
            g.add("biliary")
    if re.search(r"bil(e|iary)", x):
        g.add("biliary")
    if re.search(r"spleen", x):
        g.add("spleen")
    if re.search(r"kidney|\brenal|urinary\s+tract|genitourinary|ureter|\bgu\b", x):
        g.add("kidney")
    if re.search(r"gastrointestinal|bowel|appendix|stomach|colon|\bgi\b|mesentery", x):
        g.add("gi")
    if re.search(r"peritone", x):
        g.add("peritoneum")
    if re.search(r"vascul|vessel|abdominal\s+aorta|cardiovascular", x):
        g.add("vessels")
    if re.search(r"musculoskeletal|bone|osseous|skelet|spine", x):
        g.add("msk")
    return g or {"other"}


# header candidates: capitalised phrase (<= 6 words) followed by ':'; accepted if in the header vocabulary
HDR_CAND = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=\.))([A-Z][A-Za-z/,&\-()]*(?: [A-Za-z/,&\-()]+){0,5}):(?=\s|$)")
HEADER_VOCAB = None  # set of lower-cased accepted headers (built from train/val by build_header_vocab)
FIXED_HEADERS = {"findings", "impression", "impressions", "conclusion", "summary"}


def build_header_vocab(reports, min_count=20):
    import collections
    c = collections.Counter()
    for r in reports:
        for m in HDR_CAND.finditer(r):
            c[m.group(1).strip().lower()] += 1
    voc = {h for h, n in c.items() if n >= min_count}
    return voc | FIXED_HEADERS


SENT_SPLIT = re.compile(r"(?<!\bvs\.)(?<!\bDr\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bapprox\.)(?<!\bSt\.)(?<=[.!?])\s+(?=[A-Z0-9(<\[\"'])")
NUM_PREFIX = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s*")
BOILER = re.compile(r"discussed with|<HCW>|preliminary (?:report|results)|radiology consult line|SUMMARY:\s*\d|"
                    r"ATTENDING RADIOLOGIST|I have (?:personally )?reviewed|This report|&lt;DELETED|Addendum", re.I)


def split_sentences(body):
    out = []
    for p in SENT_SPLIT.split(body):
        p = norm_ws(p)
        if p and re.search(r"[A-Za-z]", p):
            out.append(p)
    return out


def parse_report(report, vocab):
    """-> list of dict(i, part, sec, grp, text, first) in report order. `first` = first sentence of its section."""
    rep = report.replace("\r", "")
    # impression start: 'IMPRESSION:' header or the first newline (numbered impression list)
    m = re.search(r"(?:^|\s)(?:IMPRESSION|Impression|CONCLUSION|Conclusion)S?\s*:", rep)
    m1 = re.search(r"\n\s*1\s*[.)]", rep)
    cands = [x.start() for x in (m, m1) if x]
    if cands:
        imp_start = min(cands)
    else:
        nl = rep.find("\n")
        imp_start = nl if nl >= 0 else len(rep)
    findings, imp = rep[:imp_start], rep[imp_start:]
    items = []
    # ---- findings: split on accepted headers
    hdrs = [mm for mm in HDR_CAND.finditer(findings) if mm.group(1).strip().lower() in vocab]
    segs = []
    if not hdrs:
        segs.append(("", findings))
    else:
        if findings[:hdrs[0].start()].strip():
            segs.append(("", findings[:hdrs[0].start()]))
        for k, mm in enumerate(hdrs):
            end = hdrs[k + 1].start() if k + 1 < len(hdrs) else len(findings)
            segs.append((mm.group(1).strip(), findings[mm.end():end]))
    for sec, body in segs:
        if sec.lower() == "findings" and not body.strip():
            continue
        grps = header_groups(sec) if sec else {"other"}
        if sec.lower() == "findings":
            sec, grps = "", {"other"}
        for j, s in enumerate(split_sentences(body.replace("\n", " "))):
            items.append(dict(part="findings", sec=sec, grps=grps, text=s, first=(j == 0)))
    # ---- impression: numbered lines, then sentences
    imp = re.sub(r"^\s*(?:IMPRESSION|Impression|CONCLUSION|Conclusion)S?\s*:", "", imp.strip())
    findings = re.sub(r"[\s:]+$", " ", findings)
    for line in re.split(r"\n+", imp):
        line = NUM_PREFIX.sub("", line.strip())
        if not line:
            continue
        for s in split_sentences(line):
            if s.isupper() and len(s) > 20:   # all-caps impressions are still the doctor's text; keep as is
                pass
            items.append(dict(part="impression", sec="IMPRESSION", grps={"impression"}, text=s, first=False))
    items = [it for it in items if not BOILER.search(it["text"])]
    for i, it in enumerate(items):
        it["i"] = i
    return items


# ------------------------------------------------------------------ disease keyword rules
def R(p):
    return re.compile(p, re.I)


NEG_TRIG = R(r"\b(?:no|not|without|negative\s+for|free\s+of|absence\s+of|absent|nor|rather\s+than|"
             r"resolved|resolution\s+of|no\s+longer|neither|exclude[sd]?|ruled\s+out|unlikely|lack\s+of)\b")
# clause terminators for negation scope
SCOPE_END = R(r"\b(?:but|however|although|though|except|which|whereas|while|with\s+(?:a|an|the|mild|moderate|severe|small|large|marked))\b|[;:]|\.\s")
HEDGE_UNCERTAIN = R(r"\b(?:possibl[ey]|may\s+(?:be|represent|reflect|have)|might|could|cannot\s+be\s+(?:excluded|ruled\s+out)|"
                    r"can\s*not\s+be\s+excluded|not\s+(?:entirely\s+)?excluded|questionable|equivocal|indeterminate|"
                    r"versus|vs\.?|differential|borderline|query|\?|suspected|concern\s+for|r/o)\b|\?")
HEDGE_PROBABLE = R(r"\b(?:likely|probabl[ey]|suspicious\s+for|suggestive\s+of|suggest(?:s|ing)?|concerning\s+for|"
                   r"worrisome\s+for|favor(?:ed|s)?|presumed|presumably|compatible\s+with|consistent\s+with)\b")

KW = {
    "hepatic_steatosis": R(r"steato\w*|fatty\s+(?:liver|infiltration|change|deposition|sparing|infiltrat\w*)|hepatic\s+fat\b|"
                           r"focal\s+fat\b|fat(?:ty)?\s+sparing|fat\s+deposition|"
                           r"(?:hepatic|liver|parenchymal)\s+(?:hypo)?attenuation|attenuation\s+(?:of|throughout|in|within)\s+(?:the\s+)?(?:liver|hepatic)|"
                           r"(?:hypo|low|decreased|diminished)[\s-]*attenuat\w*\s+(?:of\s+)?(?:the\s+)?(?:liver|hepatic\s+parenchyma)|"
                           r"liver\s+(?:is|appears)\s+(?:diffusely\s+)?(?:hypo\w+|low\s+in\s+(?:attenuation|density))|"
                           r"(?:contour|size)\s+and\s+attenuation|normal\s+(?:hepatic\s+)?attenuation"),
    "surgically_absent_gallbladder": R(r"cholecystectom\w*|gallbladder\s+(?:is\s+|has\s+been\s+)?(?:surgically\s+)?(?:absent|removed|resected)|"
                                       r"(?:surgically\s+)?absent\s+gallbladder|gallbladder\s+(?:is\s+)?(?:not\s+(?:seen|visualized)|surgically)"),
    "splenomegaly": R(r"splenomegaly|spleen\s+(?:is\s+|appears\s+|remains\s+)?(?:mildly\s+|moderately\s+|markedly\s+|massively\s+|slightly\s+|borderline\s+|top\s+normal\s+|upper\s+normal\s+)?"
                      r"(?:enlarged|prominent|large|normal\s+in\s+size|of\s+normal\s+size|not\s+enlarged|upper\s+limits?\s+of\s+normal|top\s+normal|measur\w*)|"
                      r"(?:enlarged|large|prominent)\s+spleen|splenic\s+(?:size|length|enlargement|tip)|spleen\s+size|size\s+of\s+the\s+spleen|"
                      r"spleen\s+(?:is\s+)?(?:normal|unremarkable)|normal\s+(?:size\s+)?(?:of\s+the\s+)?spleen|spleen\s+measures|"
                      r"spleen\s+(?:is\s+)?(?:surgically\s+)?absent|splenectomy"),
    "hydronephrosis": R(r"hydro(?:uretero)?nephros\w*|hydroureter\w*|ureterohydronephros\w*|pelvi?o?caliectasis|pelvocaliectasis|"
                        r"pyelocaliectasis|caliectasis|calyectasis|pyelectasis|collecting\s+system\s+(?:dilat\w*|distension|fullness)|"
                        r"(?:dilat\w*|distension|fullness|prominence)\s+of\s+the\s+(?:\w+\s+)?(?:renal\s+)?(?:collecting\s+system|renal\s+pelvis|calyces|pelvicalyceal)|"
                        r"obstructive\s+uropathy|extrarenal\s+pelvis|renal\s+pelvis\s+(?:is\s+)?(?:dilat\w*|distended|prominent)|"
                        r"(?:dilat\w*|distended|prominent|full)\s+(?:\w+\s+)?(?:renal\s+pelvis|collecting\s+system|calyces)|pelviectasis"),
    "abdominal_aortic_aneurysm": R(r"\baneurysm\w*|\bAAA\b|\bectatic|\bectasia|endograft|\bEVAR\b|endovascular\s+(?:aortic\s+)?(?:stent|repair|graft)|"
                                   r"aort\w*\s+(?:stent|graft)|aorta\b[^.;]{0,45}?\b(?:caliber|calibre|dilated|dilatation|enlarged|measur\w*|diameter)|normal[- ]appearing\s+(?:abdominal\s+)?aorta|"
                                   r"aortic\s+(?:diameter|caliber|dilat\w*)|(?:dilated|dilatation\s+of\s+the)\s+(?:abdominal\s+)?aorta|normal\s+(?:caliber|calibre)\s+(?:abdominal\s+)?aorta"),
    "atherosclerosis": R(r"atheroscl\w*|atherom\w*|arterioscl\w*|"
                         r"(?:vascular|arterial|aortic|aortoiliac|aorto-iliac|iliac|mural|aorta|vessel)\s+(?:wall\s+)?calcifi\w*|"
                         r"calcifi\w*\s+(?:atherosclerotic\s+)?(?:plaque|of\s+the\s+(?:abdominal\s+)?(?:aorta|iliac|arter\w*|vessel\w*|vascul\w*))|"
                         r"(?:calcified|noncalcified|mixed|soft|atheromatous)\s+plaque\w*|\bplaque\w*"),
    "osteopenia": R(r"osteopeni\w*|osteoporo\w*|demineraliz\w*|demineralis\w*|(?:low|decreased|diminished|reduced|normal)\s+bone\s+(?:mineral\s+)?density|bone\s+(?:mineral\s+)?density"),
    "bowel_obstruction": R(r"obstruct\w*|\bSBO\b|\bLBO\b|transition\s+point|ileus|(?:small|large)\s+bowel\s+(?:dilat\w*|distension)|"
                           r"(?:dilated|distended)\s+(?:loops?\s+of\s+)?(?:small|large)?\s*bowel|(?:dilated|distended)\s+(?:small\s+bowel\s+)?loops|"
                           r"bowel\s+(?:is\s+)?(?:normal\s+in\s+caliber|of\s+normal\s+caliber|nondilated|non-dilated)|normal\s+(?:caliber|calibre)\s+(?:small\s+|large\s+)?bowel|"
                           r"pseudo-?obstruction|volvulus"),
    "appendicitis": R(r"append\w*"),
    "ascites": R(r"ascit\w*|free\s+fluid|(?:peritoneal|intraperitoneal|intra-abdominal|abdominopelvic|pelvic|perihepatic|perisplenic|paracolic|"
                 r"morison\w*|subhepatic|cul-de-sac|dependent|abdominal|mesenteric)\s+(?:free\s+)?fluid|fluid\s+in\s+the\s+(?:pelvis|cul-de-sac|abdomen|peritoneal|paracolic|"
                 r"dependent|right|left|perihepatic|perisplenic|morison)|hemoperitoneum|haemoperitoneum|fluid\s+(?:within|throughout)\s+the\s+(?:abdomen|pelvis|peritoneal)"),
    "free_air": R(r"free\s+(?:intraperitoneal\s+|intra-abdominal\s+|peritoneal\s+)?(?:fluid\s+(?:or|and|/|nor)\s+)?(?:air|gas)|pneumoperitoneum|(?:extraluminal|intraperitoneal|extraintestinal)\s+(?:free\s+)?(?:air|gas)"),
    "gallstones": R(r"cholelith\w*|gall\s*stones?|gallbladder\s+(?:stone|calcul\w*|sludge)|(?:stones?|calcul\w*|sludge|lith\w*)\s+(?:\w+\s+){0,3}(?:in|within)\s+the\s+gallbladder|"
                    r"cholecystolith\w*"),
    "biliary_ductal_dilation": R(r"(?:biliary|bile\s+duct\w*|ductal|duct|\bIHD|\bCBD\b|\bCHD\b|intrahepatic|extrahepatic|choledoch\w*|common\s+(?:bile\s+|hepatic\s+)?duct)\b[^.;]{0,40}?(?:dilat\w*|distended|distension|prominen\w*|ectasia|enlarged|caliber|calibre|measur\w*)|"
                                 r"(?:dilat\w*|prominen\w*|distended|ectatic)\s+(?:\w+\s+){0,3}(?:biliary|bile\s+duct\w*|ducts?|\bCBD\b|common\s+(?:bile\s+)?duct)|"
                                 r"biliary\s+(?:tree|ductal|ducts?)\s+(?:is\s+|are\s+)?(?:normal|unremarkable|not\s+dilated)|pneumobilia\s+without|"
                                 r"(?:common\s+bile\s+duct|\bCBD\b)\s+(?:is\s+)?(?:normal|unremarkable)|biliary\s+dilat\w*|ductal\s+dilat\w*"),
    "fracture": R(r"fractur\w*|\bfx\b|compression\s+deformit\w*|vertebroplasty|kyphoplasty|(?:anterior\s+)?wedg\w*\s+(?:compression\s+)?(?:deformit\w*)?|"
                  r"(?:loss\s+of\s+(?:\w+\s+)?(?:vertebral\s+)?(?:body\s+)?height|height\s+loss)|"
                  r"(?:acute\s+)?(?:osseous|bony|skeletal)\s+(?:abnormalit\w*|injur\w*|trauma)|"
                  r"cement\s+augment\w*|(?:vertebral|endplate)\s+(?:body\s+)?(?:compression|collapse|depression|deformit\w*)|"
                  r"schmorl\w*\s+node\w*\s+fractur"),
}
# sentence-level exclusions (keyword match does not count when this matches the same sentence/region)
EXCL = {
    "bowel_obstruction": R(r"(?:obstruct\w*\s+(?:renal|ureteral|urinary|biliary|ureteric|uropathy|stone|calcul\w*|\w+\s+calcul\w*)|"
                           r"(?:renal|ureteral|urinary|biliary|ureteric|ureteropelvic|UPJ|UVJ|ureterovesic\w*)\s+(?:\w+\s+){0,2}obstruct\w*|"
                           r"(?:non|un)?-?obstructing\s+(?:\w+\s+){0,3}(?:stone|calcul\w*|calculi)|obstructive\s+uropathy|"
                           r"airway|bronch\w*|gastric\s+outlet|outlet\s+obstruct\w*\s+(?:of\s+the\s+)?bladder|bladder\s+outlet|biliary|common\s+bile|vascular|venous|"
                           r"obstruct\w*\s+(?:of\s+the\s+)?(?:\w+\s+)?(?:duct|ureter|kidney|vein|artery|stent|catheter|tube))"),
    "atherosclerosis": R(r"^(?!.*(?:aort|iliac|arteries|vascul|mesenteric|renal\s+arter|celiac|femoral|splenic\s+arter|branch|vessel)).*coronar"),
    "hepatic_steatosis": R(r"(?:hypo|low)[\s-]*attenuat\w*\s+(?:lesion|mass|focus|foci|nodule|cyst|area|region|structure|collection|lesions)"),
    "appendicitis": R(r"\bappendic(?:es|ular)\b|epiploic|appendages?\b"),
    "ascites": R(r"pleural|pericardial|joint|bladder|collection\s+of\s+fluid|fluid\s+collection|fluid-filled|fluid\s+filled|bowel\s+(?:loops?\s+)?fluid"),
    "splenomegaly": R(r"splenule|accessory\s+spleen"),
    "abdominal_aortic_aneurysm": R(r"(?:intracranial|cerebral|cardiac|ventricular|coronary|pulmonary|pseudoaneurysm\s+of\s+the\s+(?:femoral|radial))"
                                   r"|venous\s+aneurysm|vein|ectatic\s+(?:duct|bowel|pancreatic|ureter|vein)|ductal\s+ectasia|tubular\s+ectasia|ectasia\s+of\s+the\s+(?:duct|pancreatic|rete)"),
    "fracture": R(r"(?:no|without)\s+(?:\w+\s+){0,3}(?:fracture)\s+(?:of|in)\s+(?:the\s+)?(?:catheter|stent|lead)"),
}
# negative / normal predicates after the keyword (organ is normal in size / ducts are not dilated)
POST_NEG = R(r"^[^.;]{0,40}?\b(?:(?:is|are|appears?|remains?)\s+(?:normal|unremarkable|not\s+(?:dilated|enlarged|seen|identified|present|visualized)|"
             r"within\s+normal\s+limits|nondilated|non-dilated|of\s+normal)|normal\s+in\s+(?:size|caliber|calibre|appearance))")
POST_WITHOUT = {"biliary_ductal_dilation": R(r"^[^.;]{0,30}?\bwithout\s+(?:evidence\s+of\s+)?dilat"),
                "splenomegaly": R(r"^[^.;]{0,30}?\bwithout\s+(?:evidence\s+of\s+)?enlarg"),
                "abdominal_aortic_aneurysm": R(r"^[^.;]{0,30}?\bwithout\s+(?:evidence\s+of\s+)?(?:aneurysm|dilat|enlarg)")}
BLANKET_SENT = R(r"^(?:(?:is|are|appears?)\s+)?(?:grossly\s+|otherwise\s+|essentially\s+|largely\s+)?(?:normal|unremarkable|within\s+normal\s+limits|"
                 r"no\s+(?:acute\s+|significant\s+|focal\s+)?(?:abnormalit\w*|findings?|pathology)|negative|clear|intact|wnl|not\s+enlarged|"
                 r"normal\s+(?:in\s+)?size)\b(?![^.;]{0,30}(?:stomach|esophag|appendix|duoden|diverticul|pancrea|adrenal))[^.;]{0,30}\.?$")
NORMAL_PRED = R(r"\b(?:is|are|appears?|appear|otherwise|remain)\s+(?:grossly\s+|otherwise\s+|essentially\s+)?(?:normal|unremarkable)\b|\bunremarkable\b|^normal\b|\bwithin\s+normal\s+limits")

# section groups whose blanket normal covers a disease (Merlin label prompt: liver/biliary normal covers steatosis + biliary;
# gallbladder normal -> GB present + no gallstones; peritoneal normal -> ascites + free air; kidneys normal -> hydronephrosis;
# bowel normal -> obstruction; musculoskeletal normal -> fracture)
BLANKET_GROUPS = {
    "hepatic_steatosis": {"liver"},
    "biliary_ductal_dilation": {"biliary"},
    "surgically_absent_gallbladder": {"gb"},
    "gallstones": {"gb"},
    "splenomegaly": {"spleen"},
    "hydronephrosis": {"kidney"},
    "bowel_obstruction": {"gi"},
    "appendicitis": set(),
    "ascites": {"peritoneum"},
    "free_air": {"peritoneum"},
    "abdominal_aortic_aneurysm": {"vessels"},
    "atherosclerosis": {"vessels"},
    "fracture": {"msk"},
    "osteopenia": set(),
}
# organ words for list-style blanket normals in free-text reports (a list of organs + 'appear normal')
ORGAN_WORD = {
    "hepatic_steatosis": R(r"\bliver\b"), "splenomegaly": R(r"\bspleen\b"), "hydronephrosis": R(r"\bkidneys?\b"),
    "surgically_absent_gallbladder": R(r"\bgallbladder\b"), "gallstones": R(r"\bgallbladder\b"),
    "biliary_ductal_dilation": R(r"\bbiliary\s+(?:tree|system)\b|\bbile\s+ducts\b"),
    "bowel_obstruction": R(r"\b(?:bowel|small\s+and\s+large\s+bowel)\b"), "fracture": R(r"\b(?:bones|osseous\s+structures)\b"),
}
# section groups where the keyword must sit (None = anywhere). Restricting avoids e.g. 'fluid' in the thorax section.
ALLOWED_GRP = {
    "ascites": {"peritoneum", "gi", "liver", "other", "impression", "gb", "spleen", "kidney", "biliary"},
}
GB_DESCRIBED = R(r"\bgallbladder\b|\bGB\b")
GB_ABSENT_START = R(r"^(?:the\s+gallbladder\s+(?:is|has\s+been)\s+)?(?:surgically\s+|been\s+)?(?:absent|removed|resected)|^(?:status\s+post|s/p|post)[\s-]+cholecystectomy|^cholecystectomy|^(?:prior|previous)\s+cholecystectomy")
GB_ABSENT_ANY = R(r"gallbladder\s+(?:is\s+|has\s+been\s+)?(?:surgically\s+)?(?:absent|removed|resected)|(?:status\s+post|s/p|post[\s-]?)\s*cholecystectomy|surgically\s+absent\s+gallbladder")
SPLEEN_SMALL = R(r"^(?:the\s+spleen\s+is\s+)?(?:(?:surgically\s+)?absent|atrophic|small|status\s+post\s+splenectomy|splenectomy|not\s+(?:seen|visualized|enlarged)|post-?splenectomy|normal|unremarkable)")


def kw_spans(d, text):
    return [(m.start(), m.end()) for m in KW[d].finditer(text)]


def negated(text, start):
    """NegEx-style: negation trigger before `start` inside the same clause."""
    pre = text[:start]
    cut = 0
    for m in SCOPE_END.finditer(pre):
        cut = m.end()
    pre = pre[cut:]
    toks = pre.split()
    window = " ".join(toks[-13:])
    return bool(NEG_TRIG.search(window))


BELOW_CUT = {
    "biliary_ductal_dilation": R(r"(?:post-?\s*cholecystectomy|following\s+cholecystectomy|after\s+cholecystectomy|cholecystectomy\s+state|"
                                 r"within\s+normal\s+limits|normal\s+for\s+(?:age|the\s+patient)|upper\s+limits?\s+of\s+normal|\btrace\b|tapers?\s+(?:normally|distally|smoothly|gradually)|"
                                 r"age[- ]related|not\s+unexpected|borderline)"),
    "ascites": R(r"^(?!.*\b(?:ascites|moderate|large)\b).*\b(?:small|mild|minimal|trace|tiny|scant|trivial)\b[^.;]{0,40}\b(?:fluid)\b[^.;]{0,40}\b(?:pelvi\w*|cul-de-sac)|"
                 r"^(?!.*\b(?:ascites|moderate|large)\b).*\b(?:small|mild|minimal|trace|tiny|scant|trivial)\b[^.;]{0,25}\b(?:pelvi\w*|cul-de-sac)\b[^.;]{0,25}fluid|"
                 r"\b(?:trace|traces|minimal|tiny|scant|trivial|physiologic\w*|small\s+amount\s+of\s+(?:physiologic|simple\s+)?(?:pelvic\s+)?free\s+fluid\s+in\s+the\s+(?:dependent\s+)?pelvis|"
                 r"small\s+(?:volume|amount)\s+(?:of\s+)?(?:free\s+)?(?:fluid\s+)?(?:in\s+the\s+)?(?:dependent\s+)?pelvi\w*\s+(?:free\s+)?fluid|"
                 r"(?:small|minimal|trace)\s+(?:amount\s+of\s+)?(?:free\s+)?fluid\s+in\s+the\s+(?:dependent\s+)?pelvis|small\s+pelvic\s+free\s+fluid|"
                 r"likely\s+physiologic)\b"),
    "abdominal_aortic_aneurysm": R(r"\b(?:ectatic|ectasia)\b(?![^.]{0,30}aneurysm(?!al\s+dilat))"),
    "bowel_obstruction": R(r"\b(?:ileus|functional|pseudo-?obstruction|less\s+likely|unlikely|not\s+favored|without\s+(?:evidence\s+of\s+)?(?:a\s+)?(?:transition|obstruction))\b"),
    "gallstones": R(r"^(?!.*(?:stone|lith|calcul)).*sludge"),
}


def sent_status(d, it):
    """rule status of one sentence for disease d: pos | uncertain | neg | None (not about d)."""
    t = it["text"]
    if d in EXCL and EXCL[d].search(t) and d in ("atherosclerosis", "appendicitis"):
        # coronary-only atherosclerosis / appendices epiploicae
        if d == "atherosclerosis":
            return None
        if not re.search(r"\bappendi(?:x|citis|ceal|colith|ectomy)", t, re.I):
            return None
    if d in ALLOWED_GRP and not (it["grps"] & ALLOWED_GRP[d]):
        return None
    spans = kw_spans(d, t)
    if d == "ascites" and EXCL["ascites"].search(t):
        spans = [s for s in spans if not re.search(r"pleural|pericardial", t[max(0, s[0] - 40):s[1] + 20], re.I)
                 and not re.search(r"collection", t[s[0]:s[1] + 15], re.I)]
    if d == "bowel_obstruction" and spans:
        keep = []
        for s0, s1 in spans:
            w = t[max(0, s0 - 45):s1 + 45]
            word = t[s0:s1].lower()
            if word.startswith("obstruct") and EXCL["bowel_obstruction"].search(w):
                if not re.search(r"bowel|\bSBO\b|loops?|transition|colon|small|gastric|intestin|duoden|gastrointestinal", w, re.I):
                    continue
                # urinary / biliary obstruction wording right next to the keyword -> skip this span
                near = t[max(0, s0 - 25):s1 + 25]
                if re.search(r"renal|ureter|urinary|biliary|stone|calcul|uropathy|duct", near, re.I) and \
                        not re.search(r"bowel|\bSBO\b|loops|transition", near, re.I):
                    continue
            if word.startswith("obstruct") and not (it["grps"] & {"gi", "impression", "other", "peritoneum"}):
                continue
            if word.startswith("obstruct") and (it["grps"] & {"impression", "other"}) and \
                    not re.search(r"bowel|\bSBO\b|loops?|transition|colon|small|gastric|intestin|duoden|gastrointestinal|appendicitis|diverticulitis|colitis|hernia", t, re.I):
                continue
            keep.append((s0, s1))
        spans = keep
    if d == "abdominal_aortic_aneurysm" and spans:
        if EXCL[d].search(t) and not re.search(r"aort|\bAAA\b|EVAR|endograft", t, re.I):
            return None
        # aneurysm of a named non-aortic vessel alone still copied (status neg by definition)
    if d == "hepatic_steatosis" and spans and EXCL[d].search(t):
        spans = [s for s in spans if not re.search(r"lesion|mass|nodule|cyst|focus|foci", t[s[0]:s[1] + 25], re.I)]
    if d == "hepatic_steatosis" and spans and not ("liver" in it["grps"]) and not re.search(r"liver|hepat|steato", t, re.I):
        spans = []
    if d == "fracture" and spans:
        spans = [s for s in spans if not re.search(r"(?:catheter|stent|lead|wire)", t[s[0]:s[1] + 25], re.I)]
    if d == "atherosclerosis" and spans:
        spans = [s for s in spans if not (t[s[0]:s[1]].lower().startswith("plaque") and not re.search(r"aort|iliac|arter|vascul|vessel|calcif|atheros", t, re.I))]
    if d == "abdominal_aortic_aneurysm" and spans and not re.search(r"aort|\bAAA\b|aneurysm|EVAR|endograft", t, re.I):
        spans = []
    if d == "abdominal_aortic_aneurysm" and spans and re.search(r"\bnode|nodal|conglomerate|encas", t, re.I) and not re.search(r"aneurysm|\bAAA\b", t, re.I):
        spans = []
    if d == "biliary_ductal_dilation" and spans and re.search(r"pancrea", t, re.I) and \
            not re.search(r"\bbil|bile|\bCBD|common\s+duct|hepatic\s+duct|intrahepatic", t, re.I):
        spans = []
    if d == "biliary_ductal_dilation" and spans:
        spans = [s for s in spans if not re.search(r"pancreatic\s+duct|main\s+pancreatic|\bPD\b|duct\s+of\s+wirsung|ureter|thoracic\s+duct|lactiferous",
                                                  t[max(0, s[0] - 25):s[1] + 5], re.I)]
    if not spans:
        if d in ("surgically_absent_gallbladder", "gallstones") and "gb" in it["grps"] and it["first"] and GB_ABSENT_START.search(t):
            return "pos" if d == "surgically_absent_gallbladder" else "neg_absent"
        if d == "gallstones" and GB_ABSENT_ANY.search(t):
            return "neg_absent"
        if d == "splenomegaly" and "spleen" in it["grps"] and it["first"] and SPLEEN_SMALL.search(t):
            return "neg"
        if d == "hepatic_steatosis" and "liver" in it["grps"] and re.search(r"attenuation|\bdensity|echogenicity", t, re.I) and \
                not re.search(r"(?:attenuat\w*|dens\w*)\s+(?:\w+\s+){0,2}(?:lesion|mass|focus|foci|nodule|cyst|structure|area)|(?:lesion|mass|cyst|nodule)s?\b[^.;]{0,30}attenuat", t, re.I):
            if re.search(r"decreas|\blow\b|hypo|diminish|reduced", t, re.I):
                return "uncertain" if HEDGE_UNCERTAIN.search(t) else "pos"
            if re.search(r"normal|homogen|unremarkable", t, re.I):
                return "neg"
        # blanket normal sentence of a covering section
        if it["first"] and (it["grps"] & BLANKET_GROUPS.get(d, set())) and BLANKET_SENT.search(t):
            return "neg_blanket"
        if d in ORGAN_WORD and ORGAN_WORD[d].search(t) and NORMAL_PRED.search(t) and len(t) < 160 and \
                not re.search(r"\b(?:lesion|mass|nodule|cyst)s?\b", t, re.I) or \
                (d in ORGAN_WORD and ORGAN_WORD[d].search(t) and re.search(r"\b(?:appear|are)\s+(?:grossly\s+)?(?:normal|unremarkable)", t, re.I) and len(t) < 200):
            if d in ("hepatic_steatosis", "splenomegaly", "hydronephrosis", "bowel_obstruction", "fracture",
                     "surgically_absent_gallbladder", "gallstones", "biliary_ductal_dilation"):
                return "neg_blanket"
        # gallbladder described (present) -> no cholecystectomy
        if d == "surgically_absent_gallbladder" and ("gb" in it["grps"]) and it["first"] and \
                not re.search(r"absent|cholecystectom|removed|resected|not\s+(?:seen|visualized)", t, re.I):
            return "neg_present"
        if d == "surgically_absent_gallbladder" and GB_DESCRIBED.search(t) and \
                re.search(r"gallbladder\s+(?:is\s+|appears\s+)?(?:\w+\s+){0,2}(?:distended|contracted|collapsed|decompressed|wall|normal|unremarkable|sludge|stones?)", t, re.I) and \
                not re.search(r"absent|cholecystectom|removed|resected|fossa", t, re.I):
            return "neg_present"
        if d == "gallstones" and ("gb" in it["grps"]) and it["first"] and \
                re.search(r"otherwise\s+(?:normal|unremarkable)|^normal\b|^unremarkable\b", t, re.I) and not re.search(r"stone|lith|calcul|sludge", t, re.I):
            return "neg_blanket"
        return None
    # status from the keyword occurrences
    stats = []
    for s0, s1 in spans:
        word = t[s0:s1]
        if d == "surgically_absent_gallbladder":
            if negated(t, s0) and not re.search(r"absent|cholecystectom", word, re.I):
                stats.append("neg")
            elif re.search(r"not\s+(?:seen|visualized)", word, re.I):
                stats.append("uncertain")
            else:
                stats.append("pos")
            continue
        if d == "splenomegaly" and re.search(r"absent|splenectomy", word, re.I):
            stats.append("neg")
            continue
        if negated(t, s0):
            stats.append("neg")
            continue
        if HYPOTHETICAL.search(t[max(0, s0 - 60):s0]):
            continue
        if d != "appendicitis" and (re.search(r"\b(?:without|no|not)\b|non-?dilated|non-?obstruct", word, re.I) or
                                    re.search(r"(?:\bnon-?|\bun)$", t[max(0, s0 - 4):s0], re.I)):
            stats.append("neg")
            continue
        if d == "abdominal_aortic_aneurysm" and re.search(r"graft|stent|repair|reconstruct", word, re.I) and \
                not re.search(r"aneurysm|\bAAA\b|endograft|\bEVAR\b|endovascular", t, re.I):
            stats.append("other")
            continue
        if (d not in ("atherosclerosis", "fracture", "appendicitis", "surgically_absent_gallbladder") and POST_NEG.search(t[s1:])) or \
                (d in POST_WITHOUT and POST_WITHOUT[d].search(t[s1:])) or re.search(r"^(?:normal|unremarkable)|normal\s+(?:caliber|calibre|size|attenuation)|not\s+(?:dilated|enlarged)|nondilated|non-dilated",
                                              word, re.I):
            stats.append("neg")
            continue
        if d in ("hepatic_steatosis",) and re.search(r"normal\s+(?:contour\s+(?:and|,)\s+)?(?:hepatic\s+)?(?:size\s+and\s+)?attenuation|contour\s+and\s+attenuation", t[max(0, s0 - 30):s1 + 5], re.I) \
                and not re.search(r"decreas|low|hypo|diminish|steato|fat", t, re.I):
            stats.append("neg")
            continue
        if d == "hepatic_steatosis" and re.search(r"focal\s+fat|fat(?:ty)?\s+sparing|focal\s+(?:fatty\s+)?(?:deposition|infiltration)|fat\s+deposition", t, re.I) and \
                not re.search(r"steatosis|diffuse", t, re.I):
            stats.append("other")   # focal fat: labels treat it inconsistently -> resolved by the label
            continue
        if d in ("hepatic_steatosis",) and re.search(r"attenuation", word, re.I) and not re.search(r"decreas|low|hypo|diminish|reduced", t, re.I):
            stats.append("neg" if re.search(r"normal", t, re.I) else "other")
            continue
        if d == "splenomegaly" and re.search(r"measur|size|length|tip", word, re.I) and not re.search(r"enlarg|splenomegaly|prominent|large", t, re.I):
            stats.append("neg" if re.search(r"normal|upper\s+limits?|not\s+enlarged|within", t, re.I) else "other")
            continue
        if d == "abdominal_aortic_aneurysm" and not re.search(r"aneurysm|AAA|ectat|ectasia|dilat|enlarg|endograft|EVAR|stent|graft|repair", t[max(0, s0 - 10):s1 + 40], re.I):
            stats.append("neg" if re.search(r"normal", t, re.I) else "other")
            continue
        if d == "biliary_ductal_dilation" and re.search(r"normal|unremarkable", word, re.I):
            stats.append("neg")
            continue
        if d == "biliary_ductal_dilation" and re.search(r"pancrea", t, re.I) and not re.search(r"bil|bile|\bCBD|common\s+duct|hepatic\s+duct|intrahepatic", t, re.I):
            continue
        if re.search(r"^[^.;]{0,40}\b(?:is|are|has|have)\s+(?:\w+\s+)?(?:resolved|no\s+longer\s+(?:seen|present|visualized))", t[s1:], re.I):
            stats.append("neg")
            continue
        if d == "biliary_ductal_dilation" and re.search(r"measur|caliber|calibre", word, re.I) and not re.search(r"dilat|distend|prominen|ectasia|enlarg", t, re.I):
            stats.append("neg" if re.search(r"normal|within|upper\s+limits", t, re.I) else "other")
            continue
        if d == "atherosclerosis" and re.search(r"normal", t[max(0, s0 - 20):s1 + 20], re.I) and re.search(r"\bno\b|without", t, re.I):
            stats.append("neg")
            continue
        # below the definition's cut -> neg
        if d in BELOW_CUT and BELOW_CUT[d].search(t):
            if d == "abdominal_aortic_aneurysm" and re.search(r"aneurysm", t, re.I) and not negated(t, t.lower().find("aneurysm")):
                pass
            else:
                stats.append("neg")
                continue
        if d == "bowel_obstruction" and re.search(r"(?:dilat|distend|caliber)", word, re.I) and not re.search(r"obstruct|SBO|LBO|transition", t, re.I):
            stats.append("neg")   # dilated loops without obstruction -> below cut
            continue
        if d == "appendicitis":
            stats.append(appendicitis_status(t))
            continue
        if d == "fracture" and re.search(r"(?:osseous|bony|skeletal)\s+(?:abnormal|injur|trauma)", word, re.I):
            stats.append("other")
            continue
        if d == "hydronephrosis" and re.search(r"extrarenal\s+pelvis", word, re.I):
            stats.append("neg")
            continue
        if d == "free_air" and re.search(r"pneumomediastinum|pneumothorax", word, re.I):
            continue
        if HEDGE_UNCERTAIN.search(t):
            stats.append("uncertain")
        else:
            stats.append("pos")
    stats = [s for s in stats if s]
    if not stats:
        return None
    if "pos" in stats:
        return "pos"
    if "uncertain" in stats:
        return "uncertain"
    if "neg" in stats:
        return "neg"
    return "other"   # about the disease but no clear status (e.g. a bare duct measurement)


APP_POS = R(r"appendicitis|periappendiceal\s+(?:inflammat\w*|stranding|fluid|abscess|fat\s+stranding|phlegmon)|"
            r"appendiceal\s+(?:abscess|perforation|rupture|inflammation|phlegmon|wall\s+thickening)|"
            r"(?:dilated|thick\w*|enlarged|inflamed|hyperemic|fluid-filled|distended)\s+(?:\w+\s+){0,2}appendix|"
            r"appendix\s+(?:is\s+|appears\s+)?(?:\w+\s+)?(?:dilated|thickened|enlarged|inflamed|distended)")


HYPOTHETICAL = R(r"risk\s+(?:for|of)|predispos\w*|symptoms\s+of|evaluat\w*\s+for|assess\w*\s+for|history\s+of|clinical\w*\s+(?:concern|suspicion|question)|"
                 r"as\s+queried|indication|to\s+exclude")


def appendicitis_status(t):
    """status of one appendix sentence: each inflammatory mention is checked for negation / low likelihood."""
    res = []
    for m in APP_POS.finditer(t):
        pre = t[max(0, m.start() - 12):m.start()]
        post = t[m.end():m.end() + 40]
        if negated(t, m.start()) or re.search(r"non-?$|noninflamed", pre, re.I) or \
                re.search(r"^[^.;]{0,25}\b(?:less\s+likely|unlikely|not\s+favored|resolved|excluded)", post, re.I):
            res.append("neg")
        elif re.search(r"appendectomy|s/p|status\s+post", t, re.I) and not re.search(r"stump", t, re.I):
            res.append("neg")
        elif HEDGE_UNCERTAIN.search(t):
            res.append("uncertain")
        else:
            res.append("pos")
    if "pos" in res:
        return "pos"
    if "uncertain" in res:
        return "uncertain"
    return "neg"   # normal / not seen / appendectomy / every inflammatory mention negated


def combine(statuses):
    """per-sentence statuses -> disease status (pos > uncertain > neg > other)."""
    s = set(statuses)
    if "pos" in s:
        return "pos"
    if "uncertain" in s:
        return "uncertain"
    if any(x.startswith("neg") for x in s):
        return "neg"
    if "other" in s:
        return "other"
    return "not_mentioned"


def extract(items):
    """-> {d: [(item_index, sentence_status), ...]}"""
    out = {d: [] for d in DISEASES}
    for it in items:
        for d in DISEASES:
            st = sent_status(d, it)
            if st:
                out[d].append((it["i"], st))
    # gallbladder: if cholecystectomy is stated, 'gallbladder present' sentences are spurious
    return out


CAT2STATUS = {"yes": "pos", "probably_yes": "pos", "unclear": "uncertain", "probably_no": "neg", "no": "neg",
              "not_mentioned": "not_mentioned"}

# ------------------------------------------------------------------ cleaning
KEEP_STATUS = (r"(?:stable|stably|stability|stationary|unchanged|unchanging|persistent(?:ly)?|persists?|persisted|persisting|"
               r"redemonstrat\w*|re-demonstrat\w*|again\s+(?:seen|noted|shown|demonstrated|visualized|identified)|"
               r"again|still|known|previously\s+(?:seen|noted|described|demonstrated|identified)|"
               r"re-?identified|grossly\s+unchanged)")
DATE = r"(?:<DATE>|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:19|20)\d{2})"
PRIOR = r"(?:the\s+)?(?:most\s+recent\s+|recent\s+|patient's\s+)?(?:prior|previous|preceding|last|earlier|outside|comparison|old|initial|baseline)"
STUDY = r"(?:\w+\s+){0,3}?(?:study|studies|exam\w*|CT|CTs|MRI?|imaging|scan\w*|ultrasound|US|PET(?:/CT)?|radiograph\w*|film\w*|images?|CTA|evaluation)"
CMP_CLAUSE = re.compile(
    r"[,;]?\s*\(?\s*(?:when\s+|as\s+|which\s+(?:is|are)\s+)?(?:compared|comparing|in\s+comparison|comparison)\s+(?:with|to)\b[^,;.)]*\)?"
    r"|[,;]?\s*\b(?:than|from|since|versus|vs\.?)\s+(?:on\s+)?" + PRIOR + r"\b[^,;.]*"
    r"|[,;]?\s*\b(?:than|from|since|on|in|at|of)\s+(?:the\s+)?(?:CT\s+(?:of|from|dated)\s+)?" + DATE + r"[^,;.]*"
    r"|[,;]?\s*\b(?:on|in|at|of|from|to)\s+" + PRIOR + r"\s+" + STUDY + r"\b(?:\s+(?:of|from|dated)\s+" + DATE + r")?"
    r"|[,;]?\s*\b(?:than|as)\s+(?:before|previously)\b"
    r"|[,;]?\s*\b(?:previously|formerly|prior)\s+(?:measur\w*\s+)?(?:up\s+to\s+)?\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)*(?:mm|cm)\b"
    r"|\s*\(\s*(?:previously|prior|formerly|was|previous)[^)]*\)"
    r"|[,;]?\s*\b(?:since|from)\s+(?:the\s+)?(?:prior|previous)\b[^,;.]*"
    r"|[,;]?\s*\b(?:in\s+the\s+)?interval(?:ly)?\b(?!\s+(?:change|increase|decrease|development|progression|resolution|improvement|placement|worsening|enlargement|growth|decrease|removal|development))"
    r"|[,;]?\s*\b(?:with(?:out)?\s+)?(?:no\s+(?:significant\s+|substantial\s+|appreciable\s+|definite\s+|obvious\s+)?(?:interval\s+)?change)\b(?:\s+(?:in|of)\s+(?:size|appearance|extent))?"
    r"|[,;]?\s*\bnot\s+(?:significantly\s+)?changed\b"
    r"|[,;]?\s*\b(?:status|condition)\s+quo\b", re.I)
# change-describing wording -> clause removed (not the whole sentence when other clauses remain)
DESCR = (r"(?!\s+(?:\w+\s+)?(?:attenuation|density|densities|height|enhancement|opacit\w*|signal|uptake|lucency|"
         r"stranding|vascularity|fat|wall|thickness|calibe?r|bone|mineral|perfusion|excretion|echogenicity|distensi\w*|conspicuity|lumen))")
CHANGE_WORD = re.compile(r"\b(?:(?:interval(?:ly)?\s+)?(?:increas\w*|decreas\w*|reduc\w*)" + DESCR + r"|progress\w*|regress\w*|improv\w*|"
                         r"resolv\w*|resolution|disappear\w*|new|newly|enlarging|enlarged\s+since|growing|grown|shrink\w*|"
                         r"(?:smaller|larger|bigger|more|less)\s+(?:than|compared)|worse\w*|worsen\w*|aggravat\w*|subsid\w*|"
                         r"no\s+longer|now|developing|evolving|evolution|evolved|interval\s+(?:development|placement|removal|growth|enlargement|change))\b|->|→", re.I)
KEEP_RE = re.compile(r"\b" + KEEP_STATUS + r"\b\s*", re.I)
COPULA_KEEP = re.compile(r"\s+(?:is|are|was|were)\s+(?:(?:again|still|re-?)\s*(?:seen|noted|demonstrated|visualized|identified|redemonstrated|present)|"
                         r"unchanged|stable|stationary|persistent|redemonstrated|similar(?:\s+in\s+(?:size|appearance))?)\b", re.I)

# se/im references: parenthesised series/image numbers in several notations
SEIM = re.compile(r"\s*\(\s*(?:(?:series|se|image|images|im|img|axial|coronal|sagittal|cor|sag|ax)\b\.?\s*:?\s*)?\d+\s*[/:]\s*\d+[^()]*\)"
                  r"|\s*\(\s*(?:on\s+)?(?:series|se)\s*\.?\s*:?\s*\d+[^()]*\)"
                  r"|\s*\(\s*(?:image|images|im|img)s?\s*\.?\s*:?\s*\d+[^()]*\)"
                  r"|,?\s*\b(?:on\s+|at\s+|see\s+)?(?:series|se)\s*\.?\s*:?\s*\d+\s*,?\s*(?:image|images|im|img)s?\s*\.?\s*:?\s*\d+(?:\s*(?:-|through|to|and|,)\s*\d+)*"
                  r"|,?\s*\b(?:on\s+|at\s+)?(?:image|images)\s+\d+(?:\s*(?:-|through|to|and|,)\s*\d+)*\s+(?:of|on)\s+series\s+\d+"
                  r"|\s*\(\s*\d+\s*-\s*\d+\s*\)"
                  r"|\s*\(\s*(?:<TIME>)?\d+[\d\s,;:/\-]*\)|\s*\(\s*<TIME>\s*\d*\s*\)", re.I)
SIZE_NUM = r"\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?"
SIZE = re.compile(SIZE_NUM + r"(?:\s*x\s*" + SIZE_NUM + r")*\s*-?\s*(?:mm|cm|millimeters?|centimeters?)\b", re.I)
# size wording to remove when unmeasured (order matters)
SIZE_PHRASES = [
    re.compile(r"\s*\b(?:to|of|at)\s+(?:approximately\s+|about\s+|up\s+to\s+)?" + SIZE.pattern + r"(?:\s+in\s+(?:diameter|size|caliber|length))?", re.I),
    re.compile(r"\s*\(\s*(?:up\s+to\s+|approximately\s+|about\s+|measuring\s+)?" + SIZE.pattern + r"[^()]{0,30}\)", re.I),
    re.compile(r",?\s*(?:which\s+|that\s+)?(?:measuring|measures?|measured)\s+(?:up\s+to\s+|approximately\s+|about\s+|at\s+least\s+|roughly\s+|nearly\s+|"
               r"at\s+most\s+|less\s+than\s+|greater\s+than\s+|over\s+)?" + SIZE.pattern + r"(?:\s+(?:in\s+)?(?:greatest\s+|maximal\s+|maximum\s+|short\s+axis\s+|long\s+axis\s+)?"
               r"(?:diameter|dimension|caliber|calibre|size|length|width|thickness|transverse|axial|AP|craniocaudal)(?:\s+dimension)?)?", re.I),
    re.compile(r"\b(?:up\s+to\s+|approximately\s+|about\s+|nearly\s+)" + SIZE.pattern + r"(?:\s+in\s+(?:diameter|size|caliber|length))?\s*", re.I),
    re.compile(r"(?<=\b[aA]\s)" + SIZE.pattern + r"\s*", re.I),
    re.compile(r"\b" + SIZE.pattern + r"(?:\s+in\s+(?:diameter|size|caliber|length))?\s*", re.I),
]
SIZE_KEEP_CTX = re.compile(r"aort|aneurysm|\bAAA\b|spleen|splen", re.I)
SIZE_KEEP_DISEASES = {"abdominal_aortic_aneurysm", "splenomegaly"}

TEMPORAL_LEFT = re.compile(r"\b(?:stable|stationary|unchanged|interval|progress\w*|regress\w*|improv\w*|resolv\w*|"
                           r"compar\w*|previous(?:ly)?|prior\s+(?:study|exam|CT|imaging|scan)|again|redemonstrat\w*|persistent|new|newly|now)\b|->", re.I)


def _tidy(t):
    t = re.sub(r"[\s:]+$", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+([.,;:)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"([,;])\s*([,;.])", r"\2", t)
    t = re.sub(r"^[\s,;:]+", "", t)
    t = re.sub(r"\b(a)\s+(?=[aeiouAEIOU])(?!(?:uni|eu|one|use|usu))", lambda m: m.group(1) + "n ", t)
    t = re.sub(r"\b(an)\s+(?=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])(?!(?:hour|honest))", lambda m: m.group(1)[0], t)
    t = re.sub(r"\b(and|or|with|of|in|the|a|an)\s*([.;,])", r"\2", t) if re.search(r"\b(?:and|or|with|of|in|the|a|an)\s*[.;]$", t) else t
    t = t.strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and not re.search(r"[.!?]$", t):
        t = t + "."
    t = re.sub(r"\.{2,}$", ".", t)
    return t


def split_clauses(t):
    """split a sentence into clauses at ', ' / '; ' / ' and ' boundaries that are safe to drop."""
    return re.split(r"(?<=[,;])\s+|\s+(?=\b(?:and|which|with|but)\b\s)", t)


def strip_temporal(s, log):
    """remove comparison wording. Returns (text or None if nothing usable is left)."""
    t = s
    t2 = CMP_CLAUSE.sub("", t)
    if t2 != t:
        log["cmp_clause"] += 1
        t = t2
    # redemonstration / again-seen openers
    t2 = re.sub(r"^(?:there\s+(?:is|are)\s+)?(?:redemonstration|re-demonstration|again\s+(?:seen|noted|demonstrated)\s+(?:is|are))\s+(?:of\s+)?(?:the\s+)?", "", t, flags=re.I)
    if t2 != t:
        log["keep_word"] += 1
        t = t2
    t2 = COPULA_KEEP.sub("", t)
    t2 = re.sub(r"^(?:there\s+(?:has|have)\s+been\s+)?(?:interval\s+)?(?:development|appearance)\s+of\s+(?:the\s+)?", "", t2, flags=re.I)
    t2 = KEEP_RE.sub("", t2)
    if t2 != t:
        log["keep_word"] += 1
        t = t2
    t = re.sub(r"\b(is|are|was|were)\s+(?=[.,;]|$)", "", t)          # dangling copula after a removed status word
    # change wording: strip adjective-use modifiers (new / increased ...), else drop the clause
    if CHANGE_WORD.search(t):
        t2 = re.sub(r"^(?:(?:interval(?:ly)?|slight(?:ly)?|mild(?:ly)?|minimal(?:ly)?|significant(?:ly)?|marked(?:ly)?)\s+)?"
                    r"(?:new(?:ly)?|increas\w*|decreas\w*|improv\w*|worsen\w*|enlarging|progressive|progressing|developing|evolving)\s+"
                    r"(?:(?:in|of)\s+(?:the\s+)?(?:(?:size|amount|degree|extent|volume|number)\s+(?:of\s+)?(?:the\s+)?)?)?", "", t, flags=re.I)
        t2 = re.sub(r"\b(?:(?:interval|slight|mild|minimal|significant|marked|further|minor)\s+)*(?:increase|decrease|improvement|worsening|progression|regression|reduction)\s+"
                    r"(?:in|of)\s+(?:the\s+)?(?:(?:size|amount|degree|extent|volume|number|caliber)\s+)?(?:of\s+)?(?:the\s+)?", "", t2, flags=re.I)
        t2 = re.sub(r",?\s*\b(?:now|currently)\s+(?=measur)", " ", t2, flags=re.I)
        t2 = re.sub(r"\b(?:new|newly)\s+(?=\w)", "", t2, flags=re.I)
        t2 = re.sub(r"\b(?:has|have)\s+(?:slightly\s+|mildly\s+|minimally\s+)?(?:increased|decreased|improved|progressed|worsened|enlarged|regressed)\s+(?:in\s+(?:size|extent|amount))?", "", t2, flags=re.I)
        t2 = re.sub(r"\b(?:is|are)\s+(?:slightly\s+|mildly\s+|minimally\s+|markedly\s+|significantly\s+)?(?:increased|decreased|improved|progressed|worsened|enlarged|regressed|larger|smaller)\s+"
                    r"(?:in\s+(?:size|extent|amount|caliber))?", "", t2, flags=re.I)
        if t2 != t:
            log["change_word_stripped"] += 1
            t = t2
        if CHANGE_WORD.search(t):
            parts = split_clauses(t)
            kept = [p for p in parts if not CHANGE_WORD.search(p)]
            if kept and len(kept) < len(parts):
                log["change_clause_dropped"] += 1
                t = " ".join(kept)
            elif CHANGE_WORD.search(t):
                log["change_sentence_dropped"] += 1
                return None
    t = _tidy(t)
    if len(re.sub(r"[^a-z]", "", t.lower())) < 4:
        return None
    return t


def strip_seim(t, log):
    t2 = SEIM.sub("", t)
    t2 = re.sub(r"\s*\([^()]*<DATE>[^()]*\)", "", t2)
    if t2 != t:
        log["seim"] += 1
    return t2


def strip_sizes(t, diseases, log):
    """remove sizes without a measurement basis (everything except aorta and spleen sizes)."""
    if not SIZE.search(t):
        return t
    keep_ctx = bool(set(diseases) & SIZE_KEEP_DISEASES) and SIZE_KEEP_CTX.search(t)
    if keep_ctx:
        log["size_kept"] += 1
        return t
    t2 = t
    for p in SIZE_PHRASES:
        t2 = p.sub(" ", t2)
    if t2 != t:
        log["size_removed"] += 1
    return t2


DEGRADED = re.compile(r"^(?:and|or|but|which|with|distal\s+to\s+which|having|than|while)\b|\b(?:to|of|at|measuring)\s*[,;]|\bto\s+with\b|\b(?:the|a|an)\s+(?:and|or|with|in)\b", re.I)


def degraded(raw, clean):
    """rule cleaning produced an unusable/meaning-losing fragment (candidate for a Qwen rewrite)."""
    if clean is None:
        return True
    body = re.sub(r"^[A-Z][A-Za-z/,&\- ]{1,50}:\s*", "", clean)
    if DEGRADED.search(body):
        return True
    lr = len(re.sub(r"\s", "", raw))
    return lr > 40 and len(re.sub(r"\s", "", clean)) < 0.4 * lr


def clean_sentence(s, diseases, typo_fix, log):
    t = strip_seim(s, log)
    t = strip_temporal(t, log)
    if t is None:
        return None
    t = strip_sizes(t, diseases, log)
    t, fixes = fix_typos(t, typo_fix)
    for a, b in fixes:
        log["typo"] += 1
        log["typo_list"].append((a, b))
    t = _tidy(t)
    # removal left a dangling fragment -> unusable
    if re.search(r",\s*\.$|\b(?:measures?|measuring|measured|of|to|and|with|the|a|an|in|at|from|than|by)\s*\.$", t, re.I):
        log["broken_dropped"] += 1
        return None
    # predicate removed entirely (only the organ noun left) -> unusable
    if re.match(r"^(?:the\s+)?(?:common\s+bile\s+duct|cbd|spleen|aorta|abdominal\s+aorta)\.?$", t, re.I):
        return None
    if len(re.sub(r"[^a-z]", "", t.lower())) < 4:
        return None
    return t


def fix_typos(s, fix):
    if not fix:
        return s, []
    fixed = []

    def rep(m):
        w = m.group(0)
        f = fix.get(w.lower())
        if f is None:
            return w
        fixed.append((w, f))
        return f.capitalize() if w[0].isupper() else f
    return re.sub(r"[A-Za-z][A-Za-z\-]+", rep, s), fixed


# ------------------------------------------------------------------ attributes
def num_cm_to_mm(v, unit):
    v = float(v)
    return round(v * 10, 1) if unit.lower().startswith("c") else round(v, 1)


def sizes_mm(text):
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*(?:x|-|to)\s*(\d+(?:\.\d+)?))?(?:\s*x\s*(\d+(?:\.\d+)?))?\s*-?\s*(mm|cm)\b", text, re.I):
        for g in m.groups()[:3]:
            if g:
                out.append(num_cm_to_mm(g, m.group(4)))
    return out


GRADE = [("severe", r"severe|marked|massive|extensive|large|gross|dense|heavy|significant|advanced"),
         ("moderate", r"moderate"), ("mild", r"mild|minimal|slight|minor|small|trace|subtle|borderline")]
GRADE_ORDER = {"mild": 1, "moderate": 2, "severe": 3}


def grade_of(text, near=None):
    """worst grade word in text ('mild to moderate' -> moderate)."""
    found = []
    for g, p in GRADE:
        if re.search(r"\b(?:" + p + r")\w*", text, re.I):
            found.append(g)
    if not found:
        return "not_stated"
    return max(found, key=lambda g: GRADE_ORDER[g])


def side_of(text):
    t = text.lower()
    if re.search(r"bilateral|both\s+(?:kidneys|sides)|bilaterally|right\s+(?:and|greater\s+than)\s+left|left\s+(?:and|greater\s+than)\s+right", t):
        return "bilateral"
    if re.search(r"transplant", t):
        return "transplant"
    r_, l_ = bool(re.search(r"\bright\b|\brt\b", t)), bool(re.search(r"\bleft\b|\blt\b", t))
    if r_ and l_:
        return "bilateral"
    if r_:
        return "right"
    if l_:
        return "left"
    return "not_stated"


VERT = re.compile(r"\b([CTLS])\s?-?(\d{1,2})\b(?:\s*(?:-|through|to|and)\s*([CTLS])?\s?-?(\d{1,2})\b)?")
RIB_N = re.compile(r"(?:(right|left|bilateral)\s+)?((?:\d{1,2}(?:st|nd|rd|th)?(?:\s*(?:,|-|and|through|to)\s*)?)+)\s*(?:anterior\s+|posterior\s+|lateral\s+)?ribs?\b|"
                   r"(?:(right|left)\s+)?ribs?\s+((?:\d{1,2}(?:\s*(?:,|-|and|through|to)\s*)?)+)", re.I)


def vertebral_levels(text):
    out = []
    for m in VERT.finditer(text):
        a, n1, b, n2 = m.group(1).upper(), int(m.group(2)), (m.group(3) or m.group(1)).upper(), m.group(4)
        lim = {"C": 7, "T": 12, "L": 5, "S": 5}
        if n1 < 1 or n1 > lim[a]:
            continue
        if n2 and b == a and int(n2) >= n1 and int(n2) <= lim[a] and int(n2) - n1 <= 6 and re.search(r"-|through|to", m.group(0)):
            out += [f"{a}{k}" for k in range(n1, int(n2) + 1)]
        else:
            out.append(f"{a}{n1}")
            if n2 and 1 <= int(n2) <= lim[b]:
                out.append(f"{b}{int(n2)}")
    return list(dict.fromkeys(out))


def attributes(d, texts, status):
    """attribute dict from the disease's raw sentences (joined). Only words in the text; never inferred."""
    t = " ".join(texts)
    a = {}
    if d == "hepatic_steatosis":
        g = grade_of(re.sub(r"\b(?:mild|moderate|severe|marked)\w*\s+(?:hepatomegaly|enlarge\w*)", "", t)) if re.search(r"steato|fat|attenuat", t, re.I) else "not_stated"
        # 'small'/'large' are not steatosis grades
        m = re.search(r"\b(mild|moderate|severe|marked|minimal|diffuse\s+severe|significant)\w*(?:\s+(?:to|-)\s+(moderate|severe|marked))?\s+(?:\w+\s+){0,2}(?:hepatic\s+)?(?:steato|fatty|fat\b|decreas|hypoattenuat|low\s+attenuat)", t, re.I)
        if m:
            g = {"marked": "severe", "minimal": "mild", "significant": "severe"}.get((m.group(2) or m.group(1)).lower().split()[-1], (m.group(2) or m.group(1)).lower().split()[-1])
            if g not in GRADE_ORDER:
                g = "not_stated"
        else:
            g = "not_stated"
        a["grade"] = g
        a["distribution"] = ("focal" if re.search(r"focal|geographic|patchy|regional", t, re.I) and not re.search(r"diffuse", t, re.I)
                             else "diffuse" if re.search(r"diffuse", t, re.I) else "not_stated")
    elif d == "splenomegaly":
        m = re.search(r"\b(mild|moderate|severe|marked|massive|slight|minimal)\w*(?:\s+(?:to|-)\s+(moderate|severe|marked|massive))?\s+(?:\w+\s+){0,1}(?:splenomegaly|enlarge\w*|prominen\w*)", t, re.I)
        g = (m.group(2) or m.group(1)).lower() if m else "not_stated"
        a["grade"] = {"marked": "severe", "massive": "severe", "slight": "mild", "minimal": "mild"}.get(g, g)
        L = [v for mm in re.finditer(r"(?:spleen|splenic|splenomegaly)[^.;]{0,60}?(\d+(?:\.\d+)?)\s*(cm|mm)|(\d+(?:\.\d+)?)\s*(cm|mm)[^.;]{0,30}?(?:spleen|splenic|splenomegaly)", t, re.I)
             for v in [num_cm_to_mm(mm.group(1) or mm.group(3), mm.group(2) or mm.group(4))]]
        a["length_mm"] = max(L) if L else None
    elif d == "hydronephrosis":
        sents = [x for x in texts if re.search(KW[d].pattern, x, re.I)]
        tt = " ".join(sents) if sents else t
        a["side"] = side_of(" ".join(x for x in sents if not negated(x, (KW[d].search(x) or re.search('.', x)).start()))) if sents else "not_stated"
        m = re.findall(r"\b(mild|moderate|severe|marked|minimal|slight|gross)\w*(?:\s+(?:to|-)\s+(moderate|severe|marked))?\s+(?:\w+\s+){0,3}?(?:hydro|pelvi|calie|caly|collecting|pyelo|fullness|dilat)", tt, re.I)
        gs = [{"marked": "severe", "gross": "severe", "minimal": "mild", "slight": "mild"}.get((b or x).lower(), (b or x).lower()) for x, b in m]
        a["grade"] = max(gs, key=lambda g: GRADE_ORDER.get(g, 0)) if gs else "not_stated"
        c = re.search(r"(?:secondary\s+to|due\s+to|caused\s+by|from|related\s+to|upstream\s+(?:of|from)|proximal\s+to)\s+(?:an?\s+|the\s+)?([^.;,]{3,60})", tt, re.I)
        if not c:
            c = re.search(r"((?:obstructing|obstructive)\s+[^.;,]{3,50}?(?:stone|calcul\w*|mass|lesion))", tt, re.I)
        a["cause"] = norm_ws(SIZE.sub("", c.group(1))).strip() if c else ""
    elif d == "abdominal_aortic_aneurysm":
        sz = []
        for seg in re.split(r"(?<=[.;])\s+", t):
            if not re.search(r"aort|aneurysm|\bAAA\b", seg, re.I) or \
                    re.search(r"\bnode|nodal|mass|conglomerate|tumou?r|lesion|encas", seg, re.I):
                continue
            seg = SEIM.sub("", seg)
            if re.search(r"thoracic|ascending|descending|arch|iliac|splenic|renal\s+artery|femoral|popliteal|hepatic\s+artery", seg, re.I) and \
                    not re.search(r"abdominal|infrarenal|suprarenal|juxtarenal|\bAAA\b", seg, re.I):
                continue
            sz += sizes_mm(seg)
        a["diameter_mm"] = max(sz) if sz else None
        loc = re.findall(r"infrarenal|suprarenal|juxtarenal|pararenal|thoracoabdominal|distal\s+abdominal|abdominal", t, re.I)
        a["location"] = loc[0].lower() if loc else ""
        a["repaired"] = "yes" if re.search(r"endograft|\bEVAR\b|stent\s*-?graft|aort\w*\s+(?:stent|graft|repair)|endovascular|repair|bypass\s+graft|bifemoral\s+graft", t, re.I) else "not_stated"
    elif d == "atherosclerosis":
        g = re.search(r"\b(mild|moderate|severe|extensive|dense|heavy|marked|scattered|minimal|advanced|diffuse)\w*(?:\s+(?:to|-)\s+(moderate|severe|extensive|dense|heavy|marked))?\s+(?:\w+\s+){0,3}?(?:atheroscl|calcif|plaque|atherom)", t, re.I)
        w = (g.group(2) or g.group(1)).lower() if g else ""
        a["grade"] = {"extensive": "severe", "dense": "severe", "heavy": "severe", "marked": "severe", "advanced": "severe",
                      "minimal": "mild", "scattered": "mild"}.get(w, w if w in GRADE_ORDER else "not_stated")
        if status not in ("pos", "uncertain"):
            a["grade"] = "not_stated"
        vs = []
        for v, p in [("aorta", r"aort(?!o-?iliac)"), ("aortoiliac", r"aorto-?iliac"), ("iliac", r"(?<!aorto)(?<!aorto-)iliac"),
                     ("coronary", r"coronar"), ("mesenteric", r"mesenteric|SMA\b|celiac|visceral"), ("renal", r"renal\s+arter"),
                     ("femoral", r"femoral"), ("splenic", r"splenic\s+arter"), ("general", r"vascul|arteries|vessels")]:
            if re.search(p, t, re.I):
                vs.append(v)
        a["location"] = vs
    elif d == "osteopenia":
        m = re.search(r"\b(mild|moderate|severe|marked|diffuse|profound|minimal)\w*\s+(?:\w+\s+){0,1}(?:osteop|demineral)", t, re.I)
        g = m.group(1).lower() if m else ""
        a["grade"] = {"marked": "severe", "profound": "severe", "minimal": "mild"}.get(g, g if g in GRADE_ORDER else "not_stated")
    elif d == "bowel_obstruction":
        typ = []
        if re.search(r"small\s+bowel\s+obstruct|\bSBO\b|small[- ]bowel|ileal|jejun|duoden", t, re.I):
            typ.append("SBO")
        if re.search(r"large\s+bowel\s+obstruct|\bLBO\b|colonic\s+obstruct|colon\w*|sigmoid\s+volvulus|cecal\s+volvulus", t, re.I):
            typ.append("LBO")
        if status not in ("pos", "uncertain"):
            a["type"] = "not_stated"
        else:
            a["type"] = "both" if len(typ) == 2 else (typ[0] if typ else "not_stated")
        a["partial"] = ("yes" if re.search(r"partial|incomplete|low[- ]grade|early", t, re.I) else
                        "no" if re.search(r"complete|high[- ]grade", t, re.I) else "not_stated")
        a["high_grade"] = "yes" if re.search(r"high[- ]grade", t, re.I) else "not_stated"
        a["transition_point"] = ("yes" if re.search(r"transition\s+point", t, re.I) and not re.search(r"no\s+(?:\w+\s+){0,2}transition", t, re.I)
                                 else "no" if re.search(r"no\s+(?:\w+\s+){0,2}transition|without\s+(?:a\s+)?(?:discrete\s+|definite\s+)?transition", t, re.I) else "not_stated")
    elif d == "appendicitis":
        a["perforated"] = ("no" if re.search(r"(?:no|without)\s+(?:\w+\s+){0,4}(?:perforat|abscess)", t, re.I) else
                           "yes" if re.search(r"perforat|rupture", t, re.I) else "not_stated")
        a["appendicolith"] = ("no" if re.search(r"(?:no|without)\s+(?:\w+\s+){0,3}appendicolith", t, re.I) else
                              "yes" if re.search(r"appendicolith|fecalith", t, re.I) else "not_stated")
        a["abscess"] = ("no" if re.search(r"(?:no|without)\s+(?:\w+\s+){0,4}abscess", t, re.I) else
                        "yes" if re.search(r"abscess|phlegmon", t, re.I) else "not_stated")
    elif d == "ascites":
        m = re.search(r"\b(trace|traces|minimal|tiny|scant|small|moderate|large|massive|mild|marked|extensive|trivial)\b(?:\s+(?:to|-)\s+(moderate|large|massive))?[^.;]{0,25}?(?:ascit|fluid|hemoperitoneum)", t, re.I)
        w = (m.group(2) or m.group(1)).lower() if m else ""
        a["amount"] = {"traces": "trace", "minimal": "trace", "tiny": "trace", "scant": "trace", "trivial": "trace", "mild": "small",
                       "massive": "large", "marked": "large", "extensive": "large"}.get(w, w if w in ("trace", "small", "moderate", "large") else "not_stated")
        locs = re.findall(r"\b(pelvi[cs]|perihepatic|perisplenic|paracolic|morison\w*|subhepatic|cul-de-sac|diffuse|throughout\s+the\s+abdomen|"
                          r"upper\s+abdom\w*|lower\s+abdom\w*|mesenter\w*|right\s+lower\s+quadrant|left\s+lower\s+quadrant|abdomen\s+and\s+pelvis)", t, re.I)
        a["location"] = list(dict.fromkeys(x.lower().replace("pelvis", "pelvic") for x in locs))
        a["hemorrhagic"] = "yes" if re.search(r"hemoperitoneum|haemoperitoneum|hemorrhag\w*|high[- ]density|hyperdense|blood", t, re.I) else "not_stated"
    elif d == "free_air":
        a["postoperative"] = "yes" if re.search(r"post-?operative|post-?surgical|postprocedur\w*|recent\s+(?:surgery|procedure|laparotomy|laparoscop)|expected|surgical", t, re.I) else "not_stated"
        a["amount"] = ("small" if re.search(r"\b(?:small|tiny|trace|minimal|few|scattered)\b[^.;]{0,30}(?:air|gas|pneumoperitoneum)", t, re.I) else
                       "large" if re.search(r"\b(?:large|extensive|moderate|significant)\b[^.;]{0,30}(?:air|gas|pneumoperitoneum)", t, re.I) else "not_stated")
    elif d == "gallstones":
        if status in ("pos", "uncertain"):
            a["number"] = ("multiple" if re.search(r"multiple|several|numerous|gallstones|stones|calculi|innumerable|few", t, re.I)
                           else "single" if re.search(r"\b(?:a|single|solitary|one)\s+(?:\w+\s+){0,3}(?:gallstone|stone|calcul)|gallstone\b|\bstone\b", t, re.I) else "not_stated")
        else:
            a["number"] = "not_stated"
        a["calcified"] = "yes" if re.search(r"calcified|radiopaque|radio-opaque|dense|hyperdense", t, re.I) else "not_stated"
        a["size_mm"] = max(sizes_mm(t)) if status in ("pos", "uncertain") and sizes_mm(t) else None
        a["sludge"] = "yes" if re.search(r"sludge", t, re.I) and not re.search(r"no\s+(?:\w+\s+){0,3}sludge", t, re.I) else "not_stated"
    elif d == "biliary_ductal_dilation":
        intra = bool(re.search(r"intrahepatic|\bIHD|intra-hepatic|intra-\s+and|intra-\s+or", t, re.I))
        extra = bool(re.search(r"extrahepatic|common\s+(?:bile\s+|hepatic\s+)?duct|\bCBD\b|\bCHD\b|choledoch", t, re.I))
        a["site"] = ("both" if intra and extra else "intrahepatic" if intra else "extrahepatic" if extra else "not_stated") if status in ("pos", "uncertain") else "not_stated"
        cbd = [v for mm in re.finditer(r"(?:common\s+(?:bile\s+)?duct|\bCBD\b|common\s+duct|extrahepatic\s+(?:bile\s+)?duct)[^.;]{0,60}?(\d+(?:\.\d+)?)\s*(mm|cm)", t, re.I)
               for v in [num_cm_to_mm(mm.group(1), mm.group(2))]]
        a["cbd_mm"] = max(cbd) if cbd else None
        a["grade"] = grade_of(" ".join(re.findall(r"\b(?:mild|moderate|severe|marked|minimal|slight)\w*(?=\s+(?:\w+\s+){0,4}(?:dilat|prominen|ectasia))", t, re.I))) if status in ("pos", "uncertain") else "not_stated"
        c = re.search(r"(?:secondary\s+to|due\s+to|caused\s+by|related\s+to)\s+(?:an?\s+|the\s+)?([^.;,]{3,60})", t, re.I)
        a["cause"] = norm_ws(SIZE.sub("", c.group(1))) if c and status in ("pos", "uncertain") else ""
    elif d == "fracture":
        pos_txt = " ".join(x for x in texts if re.search(r"fractur|compress|wedg|height|plasty|deformit", x, re.I))
        a["vertebral_levels"] = vertebral_levels(pos_txt) if status in ("pos", "uncertain") else []
        sites = []
        if a["vertebral_levels"] or re.search(r"vertebra|spine|spinous|transverse\s+process|endplate|compression", pos_txt, re.I):
            sites.append("vertebra")
        if re.search(r"\bribs?\b", pos_txt, re.I):
            sites.append("rib")
        if re.search(r"sacr\w*|coccy\w*", pos_txt, re.I):
            sites.append("sacrum")
        if re.search(r"pelvi\w*|pubic|ramus|rami|acetabul\w*|iliac\s+(?:wing|bone|crest)|ischi\w*|ilium", pos_txt, re.I):
            sites.append("pelvis")
        if re.search(r"femur|femoral\s+(?:neck|head|shaft)|intertrochanter\w*|hip\b|trochanter", pos_txt, re.I):
            sites.append("hip_femur")
        a["sites"] = sites if status in ("pos", "uncertain") else []
        ribs = []
        for m in RIB_N.finditer(pos_txt):
            ribs.append(norm_ws(m.group(0)))
        a["rib_desc"] = ribs if status in ("pos", "uncertain") else []
        a["vertebral_compression"] = ("yes" if re.search(r"compression|wedg|height\s+loss|loss\s+of\s+(?:\w+\s+)?height|vertebroplasty|kyphoplasty|"
                                                         r"vertebral\s+body\s+(?:collapse|fracture)|endplate\s+fracture|burst", pos_txt, re.I)
                                      and status in ("pos", "uncertain") else "not_stated")
        acute = bool(re.search(r"\bacute\b|subacute|nondisplaced\s+fracture|displaced", pos_txt, re.I)) and not re.search(r"no\s+(?:\w+\s+){0,2}acute", pos_txt, re.I)
        old = bool(re.search(r"\bold\b|chronic|healed|healing|remote|age-indeterminate|age\s+indeterminate|sclerotic|callus|prior", pos_txt, re.I))
        a["acuity"] = ("mixed" if acute and old else "acute" if acute else "old" if old else "not_stated") if status in ("pos", "uncertain") else "not_stated"
    return a


# attributes that count toward "coverage" (stated) per disease
ATTR_KEYS = {
    "hepatic_steatosis": ["grade", "distribution"], "splenomegaly": ["grade", "length_mm"],
    "hydronephrosis": ["side", "grade", "cause"], "abdominal_aortic_aneurysm": ["diameter_mm", "location", "repaired"],
    "atherosclerosis": ["grade", "location"], "osteopenia": ["grade"],
    "bowel_obstruction": ["type", "partial", "transition_point"], "appendicitis": ["perforated", "appendicolith", "abscess"],
    "ascites": ["amount", "location", "hemorrhagic"], "free_air": ["postoperative", "amount"],
    "gallstones": ["number", "size_mm", "calcified"], "biliary_ductal_dilation": ["site", "cbd_mm", "grade", "cause"],
    "fracture": ["sites", "vertebral_levels", "vertebral_compression", "acuity"], "surgically_absent_gallbladder": [],
}


def stated(v):
    return v not in (None, "", "not_stated", [], {})
