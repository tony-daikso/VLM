#!/usr/bin/env python3
"""Run on A6000: compact agent B's TRAIN targets for the method-1 sentence bank.
in : ~/Desktop/Tony/VLM/OWL/report_gen/targets/targets_train.jsonl
out: <out> jsonl, one row per train case: {"case_id", "d": {disease: [status, [clean sentences]]}}
status = pos | uncertain | neg | not_mentioned (agent B's final status). Only split == train rows are written."""
import json
import os
import sys

src = os.path.expanduser("~/Desktop/Tony/VLM/OWL/report_gen/targets/targets_train.jsonl")
out = sys.argv[1]
n = 0
with open(out, "w") as f:
    for line in open(src):
        r = json.loads(line)
        if r.get("split") != "train":
            continue
        d = {}
        for k, v in r["diseases"].items():
            sents = [s["clean"] for s in v.get("sentences", []) if s.get("clean") and s.get("part") == "findings"]
            if not sents and v.get("sentence_clean"):
                sents = [v["sentence_clean"]]
            d[k] = [v.get("status"), sents]
        f.write(json.dumps({"case_id": r["case_id"], "d": d}) + "\n"); n += 1
print(n, "train cases ->", out)
