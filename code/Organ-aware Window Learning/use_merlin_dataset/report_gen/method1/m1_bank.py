#!/usr/bin/env python3
"""Method 1, step S1: sentence bank from TRAIN physician sentences only.

Source of physician sentences (per study x disease, with polarity):
  * preferred: agent B's cleaned targets (report_gen/targets/, adapter `load_targets`), once they exist;
  * fallback (provisional): the `quotes` of labels_disease14.jsonl (verbatim report sentences chosen by the same Qwen
    run that produced the labels), whitespace-normalised. Test rows are never parsed.

Merlin reports are organ-sectioned and radiologists usually state negatives at organ level (a "<organ header>: normal"
style sentence), often one sentence covering two of the 14 diseases (e.g. one disease present, its partner absent).
So the bank is organised by organ group and by the polarity combination of the group's diseases:
  joint[group][combo] = most frequent train sentence that is simultaneously a quote of every disease in the group with
                        the required polarity (pos for pos/verify, neg for neg), count >= MIN_JOINT;
  single[disease][pol] = most frequent sentence for that disease and polarity that names the disease (regex) and
                        does not assert another disease, used when no joint sentence exists.
Filters: no digits, no comparison wording (stable/prior/compared/interval/new/unchanged/again ...), no se/im, no
laterality / grade words in positive sentences (method 1 cannot know them).
Diseases whose negatives are almost never written (train def-B negatives < 20 % of def-A negatives: osteopenia,
atherosclerosis) get no negative sentence ("silent negative"), matching how Merlin radiologists write.
Outputs: method1/sentence_bank.json, method1/sentence_bank.csv (readable)
"""
import collections
import json
import re
import sys

import numpy as np
import pandas as pd

from m1_common import D14, LABELS, NEG, OUT, POS, ROOT, ZH, defa, defb, load_labels

GROUPS = [("liver", ["hepatic_steatosis", "biliary_ductal_dilation"]),
          ("gallbladder", ["surgically_absent_gallbladder", "gallstones"]),
          ("spleen", ["splenomegaly"]),
          ("kidneys", ["hydronephrosis"]),
          ("gi", ["bowel_obstruction", "appendicitis"]),
          ("peritoneum", ["ascites", "free_air"]),
          ("vasculature", ["abdominal_aortic_aneurysm", "atherosclerosis"]),
          ("msk", ["osteopenia", "fracture"])]
KW = {"hepatic_steatosis": r"steatosis|fatty", "surgically_absent_gallbladder": r"surgically absent|cholecystectomy",
      "splenomegaly": r"splenomegaly|spleen", "hydronephrosis": r"hydro", "abdominal_aortic_aneurysm": r"aneurysm|aorta|aortic",
      "atherosclerosis": r"atheroscl|calcifi", "osteopenia": r"osteopeni|demineral|osteopor", "bowel_obstruction": r"obstruct",
      "appendicitis": r"append", "ascites": r"ascites|free fluid", "free_air": r"free air|pneumoperitoneum|intraperitoneal air|free gas",
      "gallstones": r"gallstone|cholelith", "biliary_ductal_dilation": r"biliary|bile duct|ductal dilat|\bcbd\b",
      "fracture": r"fractur"}
BAD = re.compile(r"\d|\bstable\b|\bprior\b|previous|compared|comparison|interval|\bnew\b|newly|unchanged|again|\bknown\b|"
                 r"redemonstrat|persist|increas|decreas|improv|resolv|progress|\bse\b|\bim\b|series|image|\bsee\b", re.I)
SIDE_GRADE = re.compile(r"\bleft\b|\bright\b|bilateral|\bmild|moderate|severe|marked|\bsmall\b|\blarge\b|trace|minimal|"
                        r"\bslight|massive|\bL\d|\bT\d|\brib|vertebra|sacr|pelvi|\bhip\b|femur|focal|diffuse", re.I)
NEGWORD = re.compile(r"\bno\b|\bnot\b|without|normal|unremarkable|patent", re.I)
HEADER_NORMAL = re.compile(r"^[A-Za-z ,/]+: (normal|patent|unremarkable)\.?$", re.I)
MIN_JOINT, MIN_SINGLE, SILENT_RATE = 20, 20, 0.20
# generic positive sentences (no side/grade/site) used when train has no attribute-free positive sentence seen >= 20x
FALLBACK_POS = {"hydronephrosis": "Hydronephrosis.", "abdominal_aortic_aneurysm": "Abdominal aortic aneurysm.",
                "bowel_obstruction": "Bowel obstruction.", "ascites": "Ascites.", "free_air": "Pneumoperitoneum.",
                "biliary_ductal_dilation": "Biliary ductal dilatation.", "fracture": "Fracture.",
                "splenomegaly": "Splenomegaly.", "osteopenia": "Osteopenia.", "gallstones": "Cholelithiasis.",
                "appendicitis": "Acute appendicitis.", "hepatic_steatosis": "Hepatic steatosis.",
                "atherosclerosis": "Atherosclerosis.", "surgically_absent_gallbladder": "Gallbladder: Surgically absent."}


GOF = {d: g for g, ds in GROUPS for d in ds}


def norm(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if s.endswith(".") else s + "."


def key(s):
    return norm(s).lower().rstrip(".")


def load_quotes():
    """{study_id: {disease: (polarity, [sentences])}} for train only. polarity: pos / neg / None."""
    tgt = ROOT / "targets"
    if (tgt / "USE_FOR_BANK").exists():                     # adapter switched on once B's format is checked
        return load_targets(tgt)
    out = {}
    for line in open(LABELS):
        sp = line[line.index('"split": "') + 10:].split('"', 1)[0]
        if sp != "train":
            continue
        r = json.loads(line); a = r.get("answer") or {}
        out[r["study_id"]] = {d: ("pos" if (a.get(d) or {}).get("category") in POS else
                                  "neg" if (a.get(d) or {}).get("category") in NEG else None,
                                  [norm(q) for q in ((a.get(d) or {}).get("quotes") or []) if str(q).strip()])
                              for d in D14}
    return out


def load_targets(tgt):
    raise NotImplementedError("fill in once targets/ format is known")


def main():
    Q = load_quotes()
    L = load_labels(("train",))
    silent = {}
    for d in D14:
        a, b = defa(L[d]), defb(L[d])
        rate = float((b == 0).sum() / max((a == 0).sum(), 1)); silent[d] = rate < SILENT_RATE
    # single-disease sentences
    single = {}
    for d in D14:
        for pol in ("pos", "neg"):
            c = collections.Counter(); surf = {}
            for sid, q in Q.items():
                p, ss = q[d]
                if p != pol:
                    continue
                for s in ss:
                    if BAD.search(s) or not re.search(KW[d], s, re.I):
                        continue
                    if pol == "pos" and (SIDE_GRADE.search(s) or NEGWORD.search(s)):
                        continue
                    if pol == "neg" and not NEGWORD.search(s):
                        continue
                    if pol == "neg" and HEADER_NORMAL.match(s) and len(dict(GROUPS)[GOF[d]]) > 1:
                        continue                          # an organ-level "normal" sentence would also deny the partner disease
                    # must not assert another disease (another disease's keyword without negation context)
                    others = [o for o in D14 if o != d and re.search(KW[o], s, re.I)]
                    if pol == "pos" and others:
                        continue
                    if pol == "neg" and any(q[o][0] == "pos" for o in others):
                        continue
                    k = key(s); c[k] += 1; surf.setdefault(k, collections.Counter())[s] += 1
            top = [(surf[k].most_common(1)[0][0], n) for k, n in c.most_common(5)]
            if pol == "pos" and (not top or top[0][1] < MIN_SINGLE):
                top = [(FALLBACK_POS[d], 0)] + top          # n = 0 marks the generic fallback
            single[(d, pol)] = top
    # joint (group x combo) sentences
    joint = {}
    for g, ds in GROUPS:
        combos = collections.defaultdict(collections.Counter); surf = {}
        for sid, q in Q.items():
            # silent-negative diseases enter the key only when positive
            act = tuple(d for d in ds if not (silent[d] and q[d][0] != "pos"))
            if not act or any(q[d][0] is None for d in act):
                continue
            pols = tuple(f"{d}:{q[d][0]}" for d in act)
            sets = [set(map(key, q[d][1])) for d in act]
            common = set.intersection(*sets) if sets else set()
            for k in common:
                s = next(x for x in q[act[0]][1] if key(x) == k)
                if BAD.search(s):
                    continue
                if any(p.endswith(":pos") for p in pols) and SIDE_GRADE.search(s):
                    continue
                combos[pols][k] += 1; surf.setdefault(k, collections.Counter())[s] += 1
        for pols, c in combos.items():
            top = [(surf[k].most_common(1)[0][0], n) for k, n in c.most_common(5)]
            if top and top[0][1] >= MIN_JOINT:
                joint[(g, pols)] = top
    bank = dict(groups=GROUPS, silent_negative={d: bool(v) for d, v in silent.items()},
                single={f"{d}|{p}": v for (d, p), v in single.items()},
                joint={f"{g}|{','.join(p)}": v for (g, p), v in joint.items()},
                source="labels_disease14 quotes (provisional)" if not (ROOT / "targets" / "USE_FOR_BANK").exists() else "targets/")
    json.dump(bank, open(OUT / "sentence_bank.json", "w"), indent=1, ensure_ascii=False)
    rows = []
    for (d, p), v in single.items():
        rows.append(dict(kind="single", key=f"{ZH[d]}|{p}", sentence=v[0][0] if v else "", n=v[0][1] if v else 0,
                         alt=" || ".join(f"{s} ({n})" for s, n in v[1:3])))
    for (g, p), v in joint.items():
        rows.append(dict(kind="joint", key=f"{g}|{'+'.join(ZH[x.split(':')[0]] + ':' + x.split(':')[1] for x in p)}",
                         sentence=v[0][0], n=v[0][1], alt=" || ".join(f"{s} ({n})" for s, n in v[1:3])))
    pd.DataFrame(rows).to_csv(OUT / "sentence_bank.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(pd.DataFrame(rows)[["kind", "key", "sentence", "n"]].to_string(index=False))
    print("silent negatives:", [ZH[d] for d, v in silent.items() if v])


if __name__ == "__main__":
    main()
