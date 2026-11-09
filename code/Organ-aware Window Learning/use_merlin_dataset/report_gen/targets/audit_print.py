#!/usr/bin/env python3
"""Print a targets jsonl (e.g. sample_val50.jsonl) for manual audit: raw vs clean per disease. stdout only.
usage: python audit_print.py <jsonl> [--changed-only]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_lib as L

only = "--changed-only" in sys.argv
for l in open(sys.argv[1]):
    r = json.loads(l)
    print(f"#### {r['case_id']}")
    for d in L.DISEASES:
        x = r["diseases"][d]
        if x["status"] == "not_mentioned" and not x["label_only_neg"]:
            continue
        if only and x["sentence_raw"] == x["sentence_clean"]:
            continue
        print(f"- {d} [{x['status']} / rule {x['status_rule']} / label {x['label_category']} / {x['source']}] {x['flags']}")
        print(f"    RAW  : {x['sentence_raw']}")
        if x["sentence_clean"] != x["sentence_raw"]:
            print(f"    CLEAN: {x['sentence_clean']}")
        if x["label_only_neg"]:
            print(f"    LABEL-ONLY NEG: {x['label_only_neg']}")
        attrs = {k: v for k, v in x["attributes"].items() if L.stated(v)}
        if attrs or x["attributes_raw_only"]:
            print(f"    ATTR : {attrs}  raw_only={x['attributes_raw_only']}")
