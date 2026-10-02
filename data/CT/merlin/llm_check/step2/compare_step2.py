# Step 2 比對：LLM 抽取的器官描述 vs 規則版（merlin_report_organ_report_strict.json）
import json, glob, os, re
from collections import Counter, defaultdict

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(os.path.dirname(H))  # data/CT/merlin

llm = {}
for f in sorted(glob.glob(f"{H}/out/llm_[0-9].json")):
    llm.update(json.load(open(f)))
inp = {}
for f in sorted(glob.glob(f"{H}/in/batch_[0-9].json")):
    inp.update(json.load(open(f)))
rules = json.load(open(f"{D}/merlin_report_organ_report_strict.json"))
missing = [p for p in inp if p not in llm]
print(f"LLM reports: {len(llm)} / {len(inp)} (missing {len(missing)})")


def norm(s):
    s = re.sub(r"^[a-z ]+:\s*", "", s.strip(), flags=re.I) if s else ""  # drop "organ:" prefix if kept
    s = re.sub(r"\((?:series\s*)?[\d<>A-Z/;,\- ]*\d[\d<>A-Z/;,\- ]*\)", " ", s)  # image refs like (3/216)
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())


def f1(a, b):
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    if not a and not b:
        return 1.0, 1.0, 1.0
    p = inter / max(sum(cb.values()), 1)   # rule precision (how much of rule text is in LLM)
    r = inter / max(sum(ca.values()), 1)   # rule recall   (how much of LLM text rule covers)
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


rows = []
for pid, v in inp.items():
    if pid not in llm:
        continue
    for organ in v["organs"]:
        L = llm[pid].get(organ, "") or ""
        R = rules[pid].get(organ, "")
        a, b = norm(L), norm(R)
        F, P, Rc = f1(a, b)
        rows.append(dict(pid=pid, organ=organ, llm=L, rule=R, exact=a == b, f1=F, p=P, r=Rc,
                         llm_empty=not a, rule_empty=not b))

n = len(rows)
print(f"pairs: {n}")
print(f"exact match (normalized): {sum(x['exact'] for x in rows) / n * 100:.1f}%")
print(f"mean token F1: {sum(x['f1'] for x in rows) / n * 100:.1f}%   "
      f"mean rule precision {sum(x['p'] for x in rows) / n * 100:.1f}%   mean rule recall {sum(x['r'] for x in rows) / n * 100:.1f}%")
buckets = Counter("F1=1" if x["f1"] == 1 else "F1>=0.8" if x["f1"] >= .8 else "0.5<=F1<0.8" if x["f1"] >= .5 else "F1<0.5" for x in rows)
print("F1 distribution:", {k: f"{v} ({v / n * 100:.1f}%)" for k, v in buckets.items()})
print(f"LLM empty but rule not: {sum(x['llm_empty'] and not x['rule_empty'] for x in rows)} | "
      f"rule empty but LLM not: {sum(x['rule_empty'] and not x['llm_empty'] for x in rows)}")
print(f"rule longer (rule has extra text, recall>=0.95 & precision<0.8): "
      f"{sum(x['r'] >= .95 and x['p'] < .8 for x in rows)} | "
      f"rule misses LLM text (recall<0.8): {sum(x['r'] < .8 for x in rows)}")

per = defaultdict(list)
for x in rows:
    per[x["organ"]].append(x)
print(f"\n{'organ':20s} {'pairs':>5s} {'exact':>6s} {'F1':>6s} {'recall':>6s} {'prec':>6s}")
for o, xs in sorted(per.items(), key=lambda kv: sum(x['f1'] for x in kv[1]) / len(kv[1])):
    k = len(xs)
    print(f"{o:20s} {k:5d} {sum(x['exact'] for x in xs) / k * 100:5.1f}% {sum(x['f1'] for x in xs) / k * 100:5.1f}% "
          f"{sum(x['r'] for x in xs) / k * 100:5.1f}% {sum(x['p'] for x in xs) / k * 100:5.1f}%")

rows.sort(key=lambda x: x["f1"])
json.dump(rows, open(f"{H}/step2_pairs_sorted_by_f1.json", "w"), ensure_ascii=False, indent=1)
print(f"\nall pairs (worst first) -> {H}/step2_pairs_sorted_by_f1.json")
