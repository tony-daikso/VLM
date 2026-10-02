import json, glob, os, re
from collections import Counter, defaultdict

S = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(S)  # data/CT/merlin
ORGANS = ["adrenal gland", "aorta", "large bowel", "duodenum", "esophagus", "gallbladder", "heart",
          "iliac artery", "iliac vena", "inferior vena cava", "kidney", "liver", "lung", "pancreas",
          "portal vein", "pulmonary artery", "rib", "sacrum", "small bowel", "spleen", "stomach",
          "trachea", "bladder", "cervical vertebrae", "thoracic vertebrae", "lumbar vertebrae"]

llm = {}
for f in sorted(glob.glob(f"{S}/llm_[0-9].json")):
    llm.update(json.load(open(f)))
bad = {o for v in llm.values() for o in v if o not in ORGANS}
assert not bad, bad
pids = json.load(open(f"{S}/sample_pids.json"))
missing = [p for p in pids if p not in llm]
print(f"LLM labels: {len(llm)} reports, missing {len(missing)}")
pids = [p for p in pids if p in llm]

rules = {"lenient": json.load(open(f"{D}/merlin_report_mention.json")),
         "strict": json.load(open(f"{D}/merlin_report_mention_strict.json"))}

out = {}
for name, R in rules.items():
    tp = fp = fn = tn = 0
    per = defaultdict(lambda: [0, 0, 0])  # organ -> [rule_only(FP), llm_only(FN), both]
    exact = 0
    fp_src = Counter()
    examples = defaultdict(list)
    for p in pids:
        L = set(llm[p])
        Ry = {o for o, s in R[p]["mention"].items() if s == "yes"}
        exact += L == Ry
        for o in ORGANS:
            a, b = o in Ry, o in L
            if a and b: tp += 1; per[o][2] += 1
            elif a and not b:
                fp += 1; per[o][0] += 1
                fp_src[R[p]["mention_source"][o]] += 1
                examples[(o, "rule_only")].append(p)
            elif b and not a:
                fn += 1; per[o][1] += 1
                examples[(o, "llm_only")].append(p)
            else: tn += 1
    n = len(pids) * len(ORGANS)
    print(f"\n=== {name} vs LLM ({len(pids)} reports x 26 = {n} cells) ===")
    print(f"agreement {(tp + tn) / n * 100:.2f}%  | rule-only yes {fp}  | LLM-only yes {fn}  | "
          f"precision {tp / (tp + fp) * 100:.1f}%  recall {tp / (tp + fn) * 100:.1f}%  | "
          f"reports identical {exact}/{len(pids)}")
    print("rule-only yes by source:", dict(fp_src.most_common()))
    print(f"{'organ':20s} rule_only llm_only both")
    for o in ORGANS:
        r = per[o]
        if r[0] or r[1]:
            print(f"{o:20s} {r[0]:9d} {r[1]:8d} {r[2]:4d}")
    out[name] = {k[0] + "|" + k[1]: v for k, v in examples.items()}
json.dump(out, open(f"{S}/disagreements.json", "w"), indent=1)
