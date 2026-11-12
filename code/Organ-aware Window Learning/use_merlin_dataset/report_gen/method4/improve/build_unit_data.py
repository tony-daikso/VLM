#!/usr/bin/env python3
"""Improvement 2 (agent F): add the unit-level probe line to method-4 jsonl rows.
Each row gets the line "Unit-level probe scores (0-1): right kidney hydronephrosis x; left kidney hydronephrosis y;
vertebral fracture T8 a, ..., L5 b." right after the classifier line, from improve/unit_probe.py probe_scores.csv
(train = 5-fold OOF, val = full-train refit), and "unit_scores" = the same numbers (2 decimals) so train.py
--unit-cue-gt can rebuild the line from labels. No report text printed; outputs stay on A6000.
usage: python build_unit_data.py --scores probe_scores.csv --in data/train.jsonl --out data_unit/train.jsonl

A+ (agent H, 11-12; optional, defaults = the behaviour above): --grade-cue-gt cue/cue_gt.json also writes
"grade_scores" ({hs_grade, asc_amount, ath_grade}, 2 decimals, from the extra --scores columns of
pillar/optionA/build_aplus.py probe_scores_pillar.csv) and the grade line right after the unit line, holding only the
keys whose disease cue in the row's own classifier line is >= the method-1 threshold of cue_gt.json (train: OOF cue;
train.py --unit-cue-grades rebuilds it from the gt cue). --stats-out writes coverage / shown counts (no ids).
Test rows are refused.

Hydronephrosis grade (agent I, 11-12; optional): --hy-grade-scores probe_scores_hygrade.csv (improve/hygrade_probe.py,
column hy_grade; needs --grade-cue-gt) adds "hy_grade" to grade_scores and the item "hydronephrosis moderate or severe x"
to the grade line when the row's hydronephrosis cue is >= its method-1 threshold. Without it the output is unchanged.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import (GRADE_KEYS, GRADE_KEYS_HY, GRADE_PREFIX, HY_GRADE_KEY, UNIT_LEVELS, UNIT_PREFIX,  # noqa: E402
                       grade_line_from_scores, render_unit_line, replace_grade_line, replace_unit_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grade-cue-gt", default="", help="A+: cue_gt.json (method-1 thresholds) -> grade line + grade_scores")
    ap.add_argument("--stats-out", default="", help="optional json: coverage / grade-shown counts (aggregates only)")
    ap.add_argument("--hy-grade-scores", default="",
                    help="optional: improve/hygrade_probe.py probe_scores_hygrade.csv -> hydronephrosis grade item")
    a = ap.parse_args()
    if a.hy_grade_scores and not a.grade_cue_gt:
        raise SystemExit("--hy-grade-scores needs --grade-cue-gt")
    KEYS = GRADE_KEYS_HY if a.hy_grade_scores else GRADE_KEYS
    hy = {r["case_id"]: r for r in csv.DictReader(open(a.hy_grade_scores))} if a.hy_grade_scores else None
    sc = {r["case_id"]: r for r in csv.DictReader(open(a.scores))}
    cue_thr = json.load(open(a.grade_cue_gt))["thresholds"] if a.grade_cue_gt else None
    st = {"rows": 0, "no_scores": 0, "splits": {}, "kid_scored": [0, 0], "vert_any": 0, "vert_units": 0}
    if cue_thr:
        st.update(grade_scored={k: 0 for k, _, _ in KEYS}, grade_shown={k: 0 for k, _, _ in KEYS},
                  grade_line_rows=0)
    v = lambda x: None if x in ("", None) else round(float(x), 2)   # noqa: E731
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = miss = 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            r = json.loads(line)
            if r["split"] == "test":
                raise SystemExit("test split is locked")
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
            st["rows"] += 1
            st["no_scores"] += s is None
            st["splits"][r["split"]] = st["splits"].get(r["split"], 0) + 1
            st["kid_scored"] = [n_ + (x is not None) for n_, x in zip(st["kid_scored"], kid)]
            st["vert_any"] += any(x is not None for x in vert)
            st["vert_units"] += sum(x is not None for x in vert)
            if cue_thr:
                gs = {k: (None if s is None else v(s.get(k))) for k, _, _ in GRADE_KEYS}
                if hy is not None:
                    h = hy.get(r["case_id"])
                    assert h is None or h["split"] == r["split"], r["case_id"]
                    gs[HY_GRADE_KEY[0]] = None if h is None else v(h[HY_GRADE_KEY[0]])
                r["grade_scores"] = gs
                gl = grade_line_from_scores(r["prompt"], gs, cue_thr, KEYS)
                r["prompt"] = replace_grade_line(r["prompt"], gl)
                assert sum(ln.startswith(GRADE_PREFIX) for ln in r["prompt"].split("\n")) == (1 if gl else 0)
                st["grade_line_rows"] += bool(gl)
                for k, _, txt in KEYS:
                    st["grade_scored"][k] += gs[k] is not None
                    st["grade_shown"][k] += f" {txt} " in f" {gl[len(GRADE_PREFIX):]}"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows written, {miss} without probe scores")
    if a.stats_out:
        Path(a.stats_out).write_text(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
