#!/usr/bin/env python3
"""Compare two targets versions (e.g. v2 vs v3) case by case. Run where both jsonl live (A6000).
Prints JSON: per split / per disease changed class counts and sentence-count deltas, target_text changes, and a
random sample of changed val pairs (for manual review; contains report sentences -> store data side only).
usage: python compare_versions.py <dir> <old_suffix> <new_suffix> [--sample 20]"""
import collections, json, random, sys

d0, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
nsample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 20
res = {}
changed_val = []
for sp in ("train", "val", "test"):
    o = {}
    for l in open(f"{d0}/targets_{sp}{old}.jsonl"):
        r = json.loads(l)
        o[r["case_id"]] = r
    per = collections.defaultdict(collections.Counter)
    tt = 0
    for l in open(f"{d0}/targets_{sp}{new}.jsonl"):
        r = json.loads(l)
        p = o[r["case_id"]]
        tt += r["target_text"] != p["target_text"]
        for d, x in r["diseases"].items():
            y = p["diseases"][d]
            c = per[d]
            ns, os_ = len(x["sentences"]), len(y["sentences"])
            c["sent_old"] += os_
            c["sent_new"] += ns
            if [s["i"] for s in x["sentences"]] != [s["i"] for s in y["sentences"]]:
                c["changed"] += 1
                c["became_empty"] += int(ns == 0 and os_ > 0)
                c["became_nonempty"] += int(ns > 0 and os_ == 0)
                if sp == "val":
                    changed_val.append(dict(case_id=r["case_id"], disease=d, status=x["status"], label=x["label_category"],
                                            old=[s["raw"] for s in y["sentences"]], new=[s["raw"] for s in x["sentences"]],
                                            flags=x["flags"]))
            for f in x["flags"]:
                if f.startswith("qwen_select"):
                    c[f] += 1
    res[sp] = dict(target_text_changed=tt, n=len(o), per={d: dict(c) for d, c in per.items()})
random.seed(20261109)
res["val_sample"] = random.sample(changed_val, min(nsample, len(changed_val)))
print(json.dumps(res, ensure_ascii=False))
