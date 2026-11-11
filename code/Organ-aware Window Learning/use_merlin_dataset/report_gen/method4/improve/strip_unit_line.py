#!/usr/bin/env python3
"""Agent G, U_gt_aux data: data_unit_v2 rows (completed classifier line) WITHOUT the unit-level probe line, so the
image tokens (+ --aux-unit-loss) are the only per-unit source. Everything else is kept character for character.
usage: python strip_unit_line.py --in improve/data_unit_v2/val.jsonl --out improve/data_v2_nounit/val.jsonl"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m4_common import UNIT_PREFIX, replace_unit_line  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
n = 0
with open(a.inp) as fi, open(a.out, "w") as fo:
    for line in fi:
        r = json.loads(line)
        r["prompt"] = replace_unit_line(r["prompt"], "")
        assert not any(ln.startswith(UNIT_PREFIX) for ln in r["prompt"].split("\n"))
        r.pop("unit_scores", None)
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(f"{n} rows -> {a.out}")
