"""Compare Claude blind annotations with Qwen combined outputs.

    python3 compare_claude.py                    # out/        vs Qwen _v3 (official knowledge)
    python3 compare_claude.py out_strict _v4strict  # strict rules on both sides
"""

import collections
import glob
import json
import re
import sys

sys.path.insert(0, "/datadrive/VLM/data/CT/CG/radar_preprocess/scripts")
from radar_llm_preprocess import ORGANS  # noqa: E402


def toks(s):
    s = re.sub(r"\(\s*\d+\s*/\s*\d+\s*\)", " ", s.lower())
    return [t.strip(".") for t in re.findall(r"[a-z0-9.]+", s) if t.strip(".")]


def f1(x, y):
    x, y = collections.Counter(toks(x)), collections.Counter(toks(y))
    common = sum((x & y).values())
    if not x and not y:
        return 1.0
    if common == 0:
        return 0.0
    p, r = common / sum(y.values()), common / sum(x.values())
    return 2 * p * r / (p + r)

d = "/datadrive/VLM/data/CT/CG/radar_preprocess"
cdir, qtag = (sys.argv[1:3] if len(sys.argv) > 2 else ("out", "_v3"))
data = json.load(open(f"{d}/cg_report.json"))
qm = json.load(open(f"{d}/cg_report_mention{qtag}.json"))
qp = json.load(open(f"{d}/cg_report_organ_report{qtag}.json"))
qn = json.load(open(f"{d}/cg_report_organ_normal{qtag}.json"))
claude = {}
for f in sorted(glob.glob(f"{d}/claude_check/{cdir}/claude_*.json")):
    claude.update(json.load(open(f)))

pids = [p for p in json.load(open(f"{d}/claude_check/sample_pids.json")) if p in claude]
print(f"reports: {len(pids)}  "
      f"({collections.Counter(data[p]['type'] for p in pids)})")

# Step 1
cells = same = 0
tp = fp = fn = 0
dis = collections.Counter()
by_type = collections.defaultdict(lambda: [0, 0])
examples = collections.defaultdict(list)
for p in pids:
    for o in ORGANS:
        c = "yes" if o in claude[p] else "no"
        q = qm[p]["mention"][o]
        cells += 1
        same += c == q
        by_type[data[p]["type"]][0] += c == q
        by_type[data[p]["type"]][1] += 1
        tp += c == q == "yes"
        fp += q == "yes" and c == "no"
        fn += q == "no" and c == "yes"
        if c != q:
            key = (o, f"Qwen {q} / Claude {c}")
            dis[key] += 1
            desc = claude[p][o]["description"] if c == "yes" else qp[p][o]
            examples[key].append(f"{p}: {desc[:140]}")
print(f"\n== Step 1 mention: cell agreement {same}/{cells} = {same / cells:.2%}")
for t, (a, n) in by_type.items():
    print(f"   {t}: {a / n:.2%}")
print(f"   Qwen precision / recall (Claude as reference): {tp / (tp + fp):.2%} / {tp / (tp + fn):.2%}"
      f"   (Qwen extra yes {fp}, Qwen missed {fn})")
full = sum(all(("yes" if o in claude[p] else "no") == qm[p]["mention"][o] for o in ORGANS) for p in pids)
print(f"   reports with all 26 identical: {full}/{len(pids)}")
for k, v in dis.most_common(12):
    print(f"   {v:3d}  {k[0]:20s} {k[1]}")
    for e in examples[k][:2]:
        print(f"          {e}")

# Step 3
both = [(p, o) for p in pids for o in ORGANS if o in claude[p] and o in qn.get(p, {})]
agree = sum(claude[p][o]["status"] == qn[p][o] for p, o in both)
conf = collections.Counter((qn[p][o], claude[p][o]["status"]) for p, o in both)
print(f"\n== Step 3 normal/abnormal (organs both mention): {agree}/{len(both)} = {agree / len(both):.2%}")
print("   (Qwen, Claude):", dict(conf))
for p, o in [(p, o) for p, o in both if claude[p][o]["status"] != qn[p][o]][:12]:
    print(f"   {p} {o}: Qwen {qn[p][o]}, Claude {claude[p][o]['status']} | {claude[p][o]['description'][:130]}")

# Step 2
fs = [f1(claude[p][o]["description"], qp[p][o]) for p, o in both]
exact = sum(toks(claude[p][o]["description"]) == toks(qp[p][o]) for p, o in both)
print(f"\n== Step 2 description: exact {exact / len(both):.2%}, mean token F1 {sum(fs) / len(fs):.2%}, "
      f"F1<0.5: {sum(f < 0.5 for f in fs)}")
for f, (p, o) in sorted(zip(fs, both))[:6]:
    print(f"   F1={f:.2f} {p} {o}\n      Qwen  : {qp[p][o][:200]}\n      Claude: {claude[p][o]['description'][:200]}")
