#!/usr/bin/env python3
"""Find negatives whose source sentences only talk about masses / lesions / nodules (no statement about the
target finding itself, nor a blanket 'normal/unremarkable'). By the prompt these should be not_mentioned.
Usage: python lesion_only_check.py shards/*.jsonl [--show N]"""
import collections
import json
import sys

from std14_lib import OK, lesion_only  # noqa: E402  (same rule as postproc)


if __name__ == "__main__":
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    paths = [p for p in sys.argv[1:] if p.endswith(".jsonl")]
    n, neg, hit, ex = 0, collections.Counter(), collections.Counter(), collections.defaultdict(list)
    seen = set()
    for p in paths:
        for l in open(p):
            j = json.loads(l)
            if not j.get("parse_ok") or j["exam_id"] in seen:
                continue
            seen.add(j["exam_id"])
            n += 1
            for d, x in j["answer"]["diseases"].items():
                if x is None:
                    continue
                neg[d] += x["status"] == "negative"
                if lesion_only(d, x):
                    hit[d] += 1
                    ex[d].append((j["exam_id"], x["source_sentences"], x["normalized_sentence"]))
    print(f"{n} reports")
    for d in OK:
        print(f"{d:30s} negatives {neg[d]:5d}  lesion-only negatives {hit[d]:4d} ({hit[d] / max(neg[d], 1):.1%})")
    for d in OK:
        for e in ex[d][:show]:
            print(d, "|", e[0], "|", " || ".join(s[:120] for s in e[1]), "=>", e[2])
