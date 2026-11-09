#!/usr/bin/env python3
"""Merge std14 shard outputs into reports_std.jsonl (+ rendered plain-text report) and a wide CSV of statuses.

Usage: python finalize_std14.py --jsonl shards/std14_a6000.jsonl shards/std14_spark.jsonl
Writes into the Mac output folder: reports_std.jsonl, std14_status.csv, missing_ids.txt
"""
import argparse
import json
import os

import pandas as pd

from std14_lib import DISEASES, postproc, render

OUT = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_gen/standardize"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    a = ap.parse_args()
    rep = json.load(open(os.path.join(OUT, "reports_std_in.json")))
    best = {}
    for p in a.jsonl:
        for l in open(p):
            try:
                j = json.loads(l)
            except Exception:  # noqa: BLE001
                continue
            if j.get("parse_ok") or j["exam_id"] not in best:
                best[j["exam_id"]] = j
    rows = []
    with open(os.path.join(OUT, "reports_std.jsonl"), "w") as f:
        for eid in sorted(rep, key=lambda k: rep[k]["row"]):
            j = best.get(eid)
            if not j or not j.get("parse_ok"):
                continue
            ans = postproc(j["answer"])
            f.write(json.dumps({"exam_id": eid, "pid": rep[eid]["PID"], "loc": rep[eid]["LOC"],
                                "edate": rep[eid]["EDATE"], "clinical_info": ans["clinical_info"],
                                "comparison_ref": ans["comparison_ref"], "diseases": ans["diseases"],
                                "rendered": render(ans)}, ensure_ascii=False) + "\n")
            rows.append({"EXAMID": eid, "PID": rep[eid]["PID"], "LOC": rep[eid]["LOC"],
                         **{d: ans["diseases"][d]["status"] for d in DISEASES},
                         **{d + "_cmp": ans["diseases"][d]["comparison"] for d in DISEASES}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "std14_status.csv"), index=False)
    miss = [e for e in rep if e not in {r["EXAMID"] for r in rows}]
    open(os.path.join(OUT, "missing_ids.txt"), "w").write("\n".join(miss) + ("\n" if miss else ""))
    print(f"{len(rows)} written, {len(miss)} missing")


if __name__ == "__main__":
    main()
