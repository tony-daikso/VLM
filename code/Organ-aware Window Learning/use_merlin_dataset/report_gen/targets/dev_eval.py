#!/usr/bin/env python3
"""Rule development on train/val only: rule status vs labels_disease14 category, plus disagreement samples.
usage: python dev_eval.py [--split train|val|trval] [--disease d] [--show N] [--kind fn|fp|conf]"""
import argparse, collections, json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mt_lib as L

BASE = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/disease14"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--disease", default=None)
ap.add_argument("--show", type=int, default=0)
ap.add_argument("--kind", default="conf")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

reps = json.load(open(f"{BASE}/reports.json"))
labs = {}
for l in open(f"{BASE}/labels_disease14.jsonl"):
    x = json.loads(l)
    labs[x["study_id"]] = x["answer"]
trval = [k for k, v in reps.items() if v["split"] in ("train", "val")]
vocab = L.build_header_vocab([reps[k]["report"] for k in trval])
splits = ("train", "val") if a.split == "trval" else (a.split,)
ids = [k for k, v in reps.items() if v["split"] in splits]
assert all(reps[k]["split"] != "test" for k in ids)
random.seed(a.seed)
ids = random.sample(ids, min(a.n, len(ids)))
conf = {d: collections.Counter() for d in L.DISEASES}
ex = collections.defaultdict(list)
LB = {"yes": "pos", "probably_yes": "pos", "unclear": "unc", "probably_no": "neg", "no": "neg", "not_mentioned": "nm"}
for k in ids:
    items = L.parse_report(reps[k]["report"], vocab)
    got = L.extract(items)
    for d in L.DISEASES:
        st = L.combine([s for _, s in got[d]])
        st = {"pos": "pos", "uncertain": "unc", "neg": "neg", "other": "oth", "not_mentioned": "nm"}[st]
        lb = LB[labs[k][d]["category"]]
        conf[d][(st, lb)] += 1
        if st != lb:
            ex[(d, st, lb)].append((k, [items[i]["text"] for i, _ in got[d]], labs[k][d]["quotes"], labs[k][d]["category"]))
for d in L.DISEASES:
    if a.disease and d != a.disease:
        continue
    c = conf[d]
    n = sum(c.values())
    agree = sum(v for (s, l), v in c.items() if s == l or (s == "oth" and l in ("neg", "unc")))
    mention_agree = sum(v for (s, l), v in c.items() if (s == "nm") == (l == "nm"))
    print(f"{d:30s} n={n} status_agree={agree/n:.3f} mention_agree={mention_agree/n:.3f}  " +
          " ".join(f"{s}>{l}:{v}" for (s, l), v in sorted(c.items(), key=lambda kv: -kv[1]) if s != l))
if a.show and a.disease:
    for (d, s, l), lst in sorted(ex.items(), key=lambda kv: -len(kv[1])):
        if d != a.disease:
            continue
        print(f"\n===== {d} rule={s} label={l}  n={len(lst)}")
        for k, mine, q, cat in random.sample(lst, min(a.show, len(lst))):
            print(f"  [{k}] RULE: {mine}\n           LABEL({cat}): {q}")
