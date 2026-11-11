#!/usr/bin/env python3
"""Cells where the prompt cue was POSITIVE (value >= t_f1, from infer.py <out>.cues.jsonl) but the Qwen re-extraction of
the generated report is not positive (yes / probably_yes), split into
  漏寫          : the generated output has no sentence tagged with that finding
  有句但判未提及 : a tagged sentence exists but Qwen calls the finding not_mentioned
  寫成不確定     : Qwen category unclear
  措辭被判反     : Qwen no / probably_no, but the tagged sentence has no negation wording (rule-based: likely a wording /
                   extraction problem, the sentence reads positive)
  真的寫反       : Qwen no / probably_no and the tagged sentence carries negation wording ("no", "without", "normal", ...)
The negation rule is a heuristic (a sentence tagged with two findings can negate the other one); --show N prints N
example sentences per class to the terminal for a manual check (never written to a file).
Aggregate counts only are printed / written (no ids, no text) unless --show.
usage: python cue_pos_breakdown.py --gen gen.jsonl --cues gen.cues.jsonl --thr cue_gt.json --cats per_case_categories.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import DISEASES, ZH  # noqa: E402

NEG = re.compile(r"(?i)\b(no|not|without|negative|unremarkable|normal|normally|resolved|none|absence|free of|"
                 r"nondilated|non-dilated|patent|intact)\b")
POS_SAG = re.compile(r"(?i)(cholecystectomy|surgically absent|absent|removed|post.?surgical|not (seen|visualized))")
CLASSES = ["漏寫", "有句但判未提及", "寫成不確定", "措辭被判反", "真的寫反"]


def classify(d, gen_cat, sentence):
    if gen_cat == "not_mentioned":
        return "漏寫" if not sentence else "有句但判未提及"
    if gen_cat == "unclear":
        return "寫成不確定"
    if d == "surgically_absent_gallbladder":
        return "措辭被判反" if POS_SAG.search(sentence or "") else "真的寫反"
    return "真的寫反" if NEG.search(sentence or "") else "措辭被判反"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--cues", required=True)
    ap.add_argument("--thr", required=True, help="cue_gt.json (t_f1 thresholds)")
    ap.add_argument("--cats", required=True, help="evaluate.py per_case_categories.jsonl ([reference, generated])")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    thr = json.load(open(a.thr))["thresholds"]
    gen = {json.loads(l)["case_id"]: json.loads(l)["findings"] for l in open(a.gen)}
    cue = {json.loads(l)["case_id"]: json.loads(l)["cue"] for l in open(a.cues)}
    cats = {json.loads(l)["case_id"]: json.loads(l) for l in open(a.cats)}
    per = defaultdict(Counter)
    n_pos = Counter()
    ex = defaultdict(list)
    for cid, c in cue.items():
        if cid not in cats:
            continue
        for d, v in c.items():
            if v < thr[d]:
                continue
            n_pos[d] += 1
            g = cats[cid][d][1]
            if g in ("yes", "probably_yes"):
                continue
            s = gen.get(cid, {}).get(d)
            k = classify(d, g, s)
            per[d][k] += 1
            if a.show and len(ex[k]) < a.show * 3:
                ex[k].append((ZH[d], s))
    tot = Counter()
    for d in DISEASES:
        tot.update(per[d])
    res = {"n_cue_pos": sum(n_pos.values()), "not_pos": sum(tot.values()), "classes": dict(tot),
           "per_disease": {ZH[d]: {"n_cue_pos": n_pos[d], **per[d]} for d in DISEASES}}
    print(f"提示陽性 {res['n_cue_pos']} 格；生成沒寫成陽性 {res['not_pos']} 格："
          + "、".join(f"{k} {tot[k]}" for k in CLASSES))
    print("| 疾病 | 提示陽性 | " + " | ".join(CLASSES) + " |")
    print("|---|---|" + "---|" * len(CLASSES))
    for d in DISEASES:
        print(f"| {ZH[d]} | {n_pos[d]} | " + " | ".join(str(per[d][k]) for k in CLASSES) + " |")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if a.show:
        import random
        for k, L in ex.items():
            print(f"\n== {k} ==")
            for z, s in random.Random(0).sample(L, min(a.show, len(L))):
                print(f"  [{z}] {s}")


if __name__ == "__main__":
    main()
