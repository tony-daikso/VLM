#!/usr/bin/env python3
"""Flag normalized sentences that use a disease-specific term absent from their source sentences
(e.g. source "No bony destruction" -> normalized "No fracture.")."""
import json
import re
import sys

TERMS = {  # disease -> (regex for the specific term in normalized, regex of acceptable source wording)
    "fracture": (r"fractur", r"fractur|fx|compression|vertebroplasty|kyphoplasty|height|collapse"),
    "gallstones": (r"gallstone|stone", r"stone|lith|calcul"),
    "hydronephrosis": (r"hydronephrosis|hydroureter|pelviectasis|caliectasis", r"hydro|pelviectasis|caliectasis|uropathy|pelvicalyceal dilat|dilat"),
    "bowel_obstruction": (r"obstruct", r"obstruct|ileus|patent|dilat|sbo|lbo"),
    "ascites": (r"ascites", r"ascit|fluid"),
    "free_air": (r"free air|pneumoperitoneum", r"air|pneumoperitoneum|gas"),
    "splenomegaly": (r"splenomegaly|enlarg", r"splenomegaly|enlarg|size|spleen.{0,20}(-)|cm|mm"),
    "hepatic_steatosis": (r"steatosis|fatty", r"steato|fat"),
    "abdominal_aortic_aneurysm": (r"aneurysm", r"aneurysm|aaa|evar|dilat|ectatic|endograft|stent"),
    "biliary_ductal_dilation": (r"dilat", r"dilat|ectasia|prominen|biliary|ihd|cbd|duct"),
    "appendicitis": (r"appendicitis", r"appendicitis|append"),
    "osteopenia": (r"osteopeni|osteoporo", r"osteopeni|osteoporo|density|demineral"),
    "atherosclerosis": (r"atherosclero", r"ather|athro|calcif|plaque"),
    "surgically_absent_gallbladder": (r"cholecystectomy", r"cholecystectomy|absent|absence|clip|removed"),
}


def flags(ans):
    out = []
    for d, x in ans["diseases"].items():
        if x["status"] == "not_mentioned":
            continue
        ns, src = x["normalized_sentence"].lower(), " ".join(x["source_sentences"]).lower()
        t, ok = TERMS[d]
        if re.search(t, ns) and not re.search(ok, src):
            out.append((d, x["status"], x["source_sentences"], x["normalized_sentence"]))
    return out


if __name__ == "__main__":
    n = tot = 0
    for p in sys.argv[1:]:
        for l in open(p):
            j = json.loads(l)
            if not j.get("parse_ok"):
                continue
            n += 1
            for f in flags(j["answer"]):
                tot += 1
                print(j["exam_id"], *f, sep=" | ")
    print(f"{tot} flagged in {n} reports")
