"""Compare the combined single-call outputs with the per-organ (official-style) outputs."""

import collections
import json
import re
import sys

from radar_llm_preprocess import ORGANS

d = "/datadrive/VLM/data/CT/CG/radar_preprocess"
a_tag, b_tag = (sys.argv[1:3] if len(sys.argv) > 2 else ("_pilot", "_pilot_combined"))
load = lambda name, tag: json.load(open(f"{d}/cg_report_{name}{tag}.json"))
ma, mb = load("mention", a_tag), load("mention", b_tag)
pa, pb = load("organ_report", a_tag), load("organ_report", b_tag)
na, nb = load("organ_normal", a_tag), load("organ_normal", b_tag)


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


pids = sorted(set(ma) & set(mb))
cells = same = 0
dis = collections.Counter()
for pid in pids:
    for o in ORGANS:
        x, y = ma[pid]["mention"][o], mb[pid]["mention"][o]
        cells += 1
        same += x == y
        if x != y:
            dis[(o, f"per-organ {x} / combined {y}")] += 1
print(f"reports: {len(pids)}")
print(f"Step 1 cell agreement: {same}/{cells} = {same / cells:.2%}")
for k, v in dis.most_common(15):
    print(f"   {v:4d}  {k[0]:20s} {k[1]}")

both = [(p, o) for p in pids for o in ORGANS
        if ma[p]["mention"][o] == "yes" and mb[p]["mention"][o] == "yes"
        and o in na.get(p, {}) and o in nb.get(p, {})]
agree = sum(na[p][o] == nb[p][o] for p, o in both)
print(f"\nStep 3 agreement (organs both say mentioned): {agree}/{len(both)} = {agree / max(len(both), 1):.2%}")
conf = collections.Counter((na[p][o], nb[p][o]) for p, o in both)
print("   (per-organ, combined):", dict(conf))

fs = [f1(pa[p][o], pb[p][o]) for p, o in both]
exact = sum(toks(pa[p][o]) == toks(pb[p][o]) for p, o in both)
print(f"\nStep 2 description: exact {exact / len(both):.2%}, mean token F1 {sum(fs) / len(fs):.2%}, "
      f"F1<0.5: {sum(f < 0.5 for f in fs)}")

low = sorted(zip(fs, both))[:8]
for f, (p, o) in low:
    print(f"   F1={f:.2f} {p} {o}\n      per-organ: {pa[p][o]}\n      combined : {pb[p][o]}")
diff3 = [(p, o) for p, o in both if na[p][o] != nb[p][o]][:8]
print("\nStep 3 disagreements:")
for p, o in diff3:
    print(f"   {p} {o}: per-organ {na[p][o]}, combined {nb[p][o]} | {pb[p][o][:150]}")
