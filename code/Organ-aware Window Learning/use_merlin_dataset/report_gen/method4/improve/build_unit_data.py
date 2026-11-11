#!/usr/bin/env python3
"""Improvement 2 (agent F): add the unit-level probe line to method-4 jsonl rows.
Each row gets the line "Unit-level probe scores (0-1): right kidney hydronephrosis x; left kidney hydronephrosis y;
vertebral fracture T8 a, ..., L5 b." right after the classifier line, from improve/unit_probe.py probe_scores.csv
(train = 5-fold OOF, val = full-train refit), and "unit_scores" = the same numbers (2 decimals) so train.py
--unit-cue-gt can rebuild the line from labels. No report text printed; outputs stay on A6000.
usage: python build_unit_data.py --scores probe_scores.csv --in data/train.jsonl --out data_unit/train.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import UNIT_LEVELS, UNIT_PREFIX, render_unit_line, replace_unit_line  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = {r["case_id"]: r for r in csv.DictReader(open(a.scores))}
    v = lambda x: None if x in ("", None) else round(float(x), 2)   # noqa: E731
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = miss = 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            s = sc.get(r["case_id"])
            if s is None:
                miss += 1
                kid, vert = [None, None], [None] * 10
            else:
                assert s["split"] == r["split"], r["case_id"]
                kid, vert = [v(s["kid_right"]), v(s["kid_left"])], [v(s[lv]) for lv in UNIT_LEVELS]
            r["unit_scores"] = {"kid": kid, "vert": vert}
            r["prompt"] = replace_unit_line(r["prompt"], render_unit_line(kid, vert))
            assert sum(ln.startswith(UNIT_PREFIX) for ln in r["prompt"].split("\n")) == 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows written, {miss} without probe scores")


if __name__ == "__main__":
    main()
