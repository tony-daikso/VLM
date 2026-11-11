#!/usr/bin/env python3
"""11-11 data fix: rebuild the classifier line of method-4 jsonl rows from the COMPLETED score file
(report_gen/oof/out/scores_trval_prob_complete.csv: the old scores plus the cells that were missing because the label was
unclear -- the omission had marked "unclear"). Same string format as build_jsonl.make_prompt (m4_common.render_cue_line,
2 decimals). Values already in the old prompt are kept as they are (the prompt was built from full-precision probabilities; the
5-decimal csv re-rounds a few .xx5 values differently), so only the missing cells are added; counts the added cells. No report text printed.
usage: python rebuild_cue.py --scores scores_trval_prob_complete.csv --in data_unit/train.jsonl --out data_unit_v2/train.jsonl
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import DISEASES, parse_cue, replace_cue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = {r["study_id"]: r for r in csv.DictReader(open(a.scores))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    added, still, changed = Counter(), Counter(), 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            s = sc[r["case_id"]]
            assert s["split"] == r["split"]
            old, _ = parse_cue(r["prompt"])
            new = {d: float(s[d]) for d in DISEASES if s[d] != ""}
            for d in DISEASES:
                if d in old:
                    if f"{old[d]:.2f}" != f"{new[d]:.2f}":
                        changed += 1      # 5-decimal csv vs full-precision value at a .xx5 boundary
                    new[d] = old[d]       # values already in the prompt are kept as they were
                elif d in new:
                    added[d] += 1
                else:
                    still[d] += 1
            r["prompt"] = replace_cue(r["prompt"], new)
            assert parse_cue(r["prompt"])[0].keys() == new.keys()
            r["cue_scores"] = "scores_trval_prob_complete"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows; old values kept ({changed} would differ by 0.01 if re-rounded from the 5-decimal csv); added {sum(added.values())} cells {dict(added)}; "
          f"still missing {sum(still.values())} {dict(still)}")


if __name__ == "__main__":
    main()
