#!/usr/bin/env python3
"""Agent H (11-12), arm T_unit_gt_pil (A+): train-time cue file + sanity checks for the Pillar-0 option A+ unit / grade
cues (improve/probe_pillar, written by pillar/optionA/build_aplus.py; read-only here).

1. writes <data>/unit_cue_pil.json = probe_pillar/unit_cue_pillar.json (labels_train, thresholds kidney .29 /
   vertebra .24 / grades, pools) + "grade_cue_thr" = the method-1 cue thresholds of the three graded diseases
   (cue/cue_gt.json), read by train.py --unit-cue-gt ... --unit-cue-grades.
2. checks (aggregates only, no ids / text, -> <data>/pil_checks.json; any failed assert stops the build):
   - no test row in any input / output; labels_train holds train cases only;
   - coverage of the unit / grade scores in the built data (train / val / val600);
   - threshold-positive rate vs train labels (train OOF scores: rate at the threshold, label prevalence, precision,
     recall, F1) for kidney / vertebra / the three grades; the OWL probe (.65 / .63) alongside for scale;
   - val600 rows = the matching val.jsonl rows; prompts equal the base (data_unit_v2) prompts except the unit line and
     the grade line; grade line shown <=> disease cue >= threshold;
   - train-time gt rebuild (CueGT + UnitCueGT, the train.py recipe, fixed seed, all train rows): grade line shown rate
     per key vs the stored OOF-cue prompts, and the side of the sampled grade value vs the grade label.
usage (A6000): python pil_cue.py --probe-pillar improve/probe_pillar --probe improve/probe --cue-gt cue/cue_gt.json \
   --data improve/data_unit_pil --base improve/data_unit_v2 --val600 improve/data/val600_unit_pil.jsonl \
   --base-val600 improve/data/val600_unit_v2.jsonl

Arm T_unit_gt_pil_hy (agent I, 11-12; optional switches, defaults = the behaviour above): --hy-grade-scores
probe_hygrade/probe_scores_hygrade.csv --cue-json probe_hygrade/unit_cue_hygrade.json --out-name unit_cue_pil_hy.json
--checks-name pil_hy_checks.json adds the hydronephrosis grade key (improve/hygrade_probe.py) to every check, and
--aplus-data improve/data_unit_pil --aplus-val600 improve/data/val600_unit_pil.jsonl asserts that every row equals the
A+ row once the hydronephrosis grade item / score is taken out.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from m4_common import (GRADE_KEYS, GRADE_KEYS_HY, GRADE_PREFIX, HY_GRADE_KEY, UNIT_LEVELS, UNIT_PREFIX,  # noqa: E402
                       grade_keys_shown, grade_line_from_scores, parse_cue, replace_cue, replace_grade_line)

EXPECTED_THR = {"kidney": 0.29, "vertebra": 0.24, "hs_grade": 0.32, "asc_amount": 0.37, "ath_grade": 0.45}


def fnum(x):
    return None if x in ("", None) else float(x)


def rate_vs_labels(pairs, t):
    """pairs: [(score, label 1/0)] -> threshold-positive rate vs label prevalence + P/R/F1 at t."""
    n = len(pairs)
    pos = sum(s >= t for s, _ in pairs)
    yp = sum(y for _, y in pairs)
    tp = sum(s >= t and y == 1 for s, y in pairs)
    return {"n": n, "thr": t, "pred_pos_rate": pos / max(n, 1), "label_pos_rate": yp / max(n, 1),
            "precision": tp / max(pos, 1), "recall": tp / max(yp, 1), "f1": 2 * tp / max(pos + yp, 1)}


def read_rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-pillar", required=True)
    ap.add_argument("--probe", required=True, help="OWL unit probe dir (probe_scores.csv, unit_cue.json), for scale only")
    ap.add_argument("--cue-gt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--val600", required=True)
    ap.add_argument("--base-val600", required=True)
    ap.add_argument("--sim-seed", type=int, default=20261112)
    ap.add_argument("--hy-grade-scores", default="", help="probe_scores_hygrade.csv -> also the hydronephrosis grade key")
    ap.add_argument("--cue-json", default="", help="unit cue json to start from (default <probe-pillar>/unit_cue_pillar.json)")
    ap.add_argument("--out-name", default="unit_cue_pil.json")
    ap.add_argument("--checks-name", default="pil_checks.json")
    ap.add_argument("--aplus-data", default="", help="A+ data dir: rows must equal it without the hydronephrosis grade item")
    ap.add_argument("--aplus-val600", default="")
    a = ap.parse_args()
    pp, data = Path(a.probe_pillar), Path(a.data)
    chk = {}
    KEYS = GRADE_KEYS_HY if a.hy_grade_scores else GRADE_KEYS      # GRADE_KEYS below = the keys of this run
    HYK = HY_GRADE_KEY[0]

    # ------------------------------------------------------------ 1. cue file
    uc = json.load(open(a.cue_json or pp / "unit_cue_pillar.json"))
    cg = json.load(open(a.cue_gt))
    for k, t in EXPECTED_THR.items():
        assert abs(uc["thresholds"][k] - t) < 1e-9, (k, uc["thresholds"][k])
    thr_keys = list(EXPECTED_THR) + ([HYK] if a.hy_grade_scores else [])
    # pools are 2-decimal OOF scores; a raw score just below the threshold can round onto it, so neg <= thr (counted)
    chk["neg_pool_values_at_thr"] = {}
    for k in thr_keys:
        t, pl = uc["thresholds"][k], uc["pools"][k]
        assert len(pl["pos"]) and len(pl["neg"]), k
        assert min(pl["pos"]) >= t >= max(pl["neg"]), k
        chk["neg_pool_values_at_thr"][k] = [sum(x == t for x in pl["neg"]), len(pl["neg"])]
    uc["grade_cue_thr"] = {d: cg["thresholds"][d] for _, d, _ in KEYS}
    uc["grade_cue_note"] = "agent H 11-12: grade item shown iff the disease cue >= grade_cue_thr (method-1, cue_gt.json)"
    (data / a.out_name).write_text(json.dumps(uc))
    chk["grade_cue_thr"] = uc["grade_cue_thr"]
    chk["unit_grade_thresholds"] = uc["thresholds"]

    # ------------------------------------------------------------ 2. scores: splits, threshold-positive rate vs labels
    sc = list(csv.DictReader(open(pp / "probe_scores_pillar.csv")))
    split = {r["case_id"]: r["split"] for r in sc}
    chk["scores_rows"] = {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}
    assert chk["scores_rows"]["test"] == 0 and set(split.values()) <= {"train", "val"}
    lab = uc["labels_train"]
    assert all(split.get(c) == "train" for c in lab), "labels_train must hold train cases only"
    chk["labels_train_cases"] = len(lab)
    by = {r["case_id"]: r for r in sc}
    if a.hy_grade_scores:                                        # merge the hydronephrosis grade column (train / val only)
        hs = list(csv.DictReader(open(a.hy_grade_scores)))
        assert {r["split"] for r in hs} <= {"train", "val"} and all(split.get(r["case_id"]) == r["split"] for r in hs)
        assert len(hs) == len(sc)
        for r in hs:
            by[r["case_id"]][HYK] = r[HYK]
    rates = {}
    for nm, key, cols, idx in (("kidney", "kid", ["kid_right", "kid_left"], None),
                               ("vertebra", "vert", UNIT_LEVELS, None)):
        pairs = []
        for cid, l_ in lab.items():
            for c, y in zip(cols, l_[key]):
                s = fnum(by[cid][c])
                if s is not None and y is not None:
                    pairs.append((s, y))
        rates[nm] = rate_vs_labels(pairs, uc["thresholds"][nm])
        rates[nm]["val_pred_pos_rate"] = (lambda v: sum(x >= uc["thresholds"][nm] for x in v) / max(len(v), 1))(
            [fnum(r[c]) for r in sc if r["split"] == "val" for c in cols if fnum(r[c]) is not None])
    for k, _, _ in KEYS:
        pairs = [(fnum(by[cid][k]), l_[k]) for cid, l_ in lab.items() if l_.get(k) is not None and fnum(by[cid][k]) is not None]
        rates[k] = rate_vs_labels(pairs, uc["thresholds"][k])
    # OWL probe at its own thresholds (.65 / .63), same labels, for scale
    uo = json.load(open(Path(a.probe) / "unit_cue.json"))
    so = {r["case_id"]: r for r in csv.DictReader(open(Path(a.probe) / "probe_scores.csv"))}
    for nm, key, cols in (("kidney", "kid", ["kid_right", "kid_left"]), ("vertebra", "vert", UNIT_LEVELS)):
        pairs = [(fnum(so[cid][c]), y) for cid, l_ in uo["labels_train"].items() if cid in so
                 for c, y in zip(cols, l_[key]) if y is not None and fnum(so[cid][c]) is not None]
        rates[nm + "_owl_probe"] = rate_vs_labels(pairs, uo["thresholds"][nm])
    chk["train_oof_rate_vs_labels"] = rates

    # ------------------------------------------------------------ 3. built data
    cue_thr = uc["grade_cue_thr"]
    out = {}
    rows = {}
    for sp in ("train", "val"):
        rows[sp] = read_rows(data / f"{sp}.jsonl")
        base = {r["case_id"]: r for r in read_rows(Path(a.base) / f"{sp}.jsonl")}
        assert len(base) == len(rows[sp])
        c = {"rows": len(rows[sp]), "kid_scored": [0, 0], "vert_any": 0, "vert_units": 0,
             "grade_scored": {k: 0 for k, _, _ in KEYS}, "grade_shown": {k: 0 for k, _, _ in KEYS},
             "cue_pos": {d: 0 for _, d, _ in KEYS}, "grade_line_rows": 0}
        for r in rows[sp]:
            assert r["split"] == sp, "split mismatch / test row"
            b = base[r["case_id"]]
            assert r["target"] == b["target"]
            nl = [ln for ln in r["prompt"].split("\n") if not ln.startswith((UNIT_PREFIX, GRADE_PREFIX))]
            ol = [ln for ln in b["prompt"].split("\n") if not ln.startswith(UNIT_PREFIX)]
            assert nl == ol, "prompt differs outside the unit / grade lines"
            us, gs = r["unit_scores"], r["grade_scores"]
            c["kid_scored"] = [n + (x is not None) for n, x in zip(c["kid_scored"], us["kid"])]
            c["vert_any"] += any(x is not None for x in us["vert"])
            c["vert_units"] += sum(x is not None for x in us["vert"])
            shown = grade_keys_shown(r["prompt"], cue_thr, KEYS)
            gl = [ln for ln in r["prompt"].split("\n") if ln.startswith(GRADE_PREFIX)]
            assert len(gl) == (1 if any(gs[k] is not None for k in shown) else 0)
            cue, _ = parse_cue(r["prompt"])
            for k, d, txt in KEYS:
                c["grade_scored"][k] += gs[k] is not None
                c["cue_pos"][d] += cue.get(d) is not None and cue[d] >= cue_thr[d]
                inl = bool(gl) and f" {txt} " in " " + gl[0][len(GRADE_PREFIX):]
                assert inl == (k in shown and gs[k] is not None)
                c["grade_shown"][k] += inl
            c["grade_line_rows"] += bool(gl)
        out[sp] = c
    v600 = read_rows(a.val600)
    b600 = read_rows(a.base_val600)
    vby = {r["case_id"]: r for r in rows["val"]}
    assert len(v600) == len(b600) and [r["case_id"] for r in v600] == [r["case_id"] for r in b600]
    assert all(r["split"] == "val" and r == vby[r["case_id"]] for r in v600), "val600 rows != val.jsonl rows"
    out["val600"] = {"rows": len(v600), "grade_line_rows": sum(any(ln.startswith(GRADE_PREFIX) for ln in r["prompt"].split("\n")) for r in v600),
                     "identical_to_val_rows": True}
    chk["coverage"] = out
    if a.aplus_data:                                             # rows = the A+ rows + the hydronephrosis grade item only
        def strip_hy(r):
            q = dict(r)
            q["grade_scores"] = {k: v for k, v in r["grade_scores"].items() if k != HYK}
            q["prompt"] = replace_grade_line(r["prompt"], grade_line_from_scores(r["prompt"], q["grade_scores"], cue_thr))
            return q
        same = {}
        for nm, new, old in [(sp, rows[sp], read_rows(Path(a.aplus_data) / f"{sp}.jsonl")) for sp in ("train", "val")] + \
                            [("val600", v600, read_rows(a.aplus_val600))]:
            assert len(new) == len(old) and all(strip_hy(x) == y for x, y in zip(new, old)), f"{nm}: differs from A+"
            same[nm] = len(new)
        chk["equal_to_aplus_without_hy_item_rows"] = same

    # ------------------------------------------------------------ 4. train-time gt rebuild (train.py recipe)
    from m4_data import CueGT, UnitCueGT          # noqa: E402  (torch import; CPU only)
    cgt = CueGT(a.cue_gt, 0.3, "sample", unclear="oof")
    ugt = UnitCueGT(str(data / a.out_name), 0.3, grades=True)
    rng = random.Random(a.sim_seed)
    sim = {k: {"shown": 0, "labelled_shown": 0, "side_ok": 0, "null_shown_own_oof": 0} for k, _, _ in KEYS}
    unit_drop = {"kid": 0, "vert": 0}
    for r in rows["train"]:
        p = r["prompt"]
        probs, _, _ = cgt.probs(r["case_id"], rng, parse_cue(p)[0])
        p = replace_cue(p, probs)
        ul = ugt.line(r, rng)
        unit_drop["kid"] += "kidney" not in ul
        unit_drop["vert"] += "vertebral" not in ul
        shown = grade_keys_shown(p, cue_thr, KEYS)
        gl = ugt.grade_line(r, p, rng)
        l_ = lab[r["case_id"]]
        for k, d, txt in KEYS:
            if k not in shown:
                assert f" {txt} " not in " " + gl
                continue
            if f" {txt} " not in " " + gl:                       # cue positive, no grade label and no own score
                assert l_.get(k) is None and r["grade_scores"][k] is None
                sim[k]["no_value"] = sim[k].get("no_value", 0) + 1
                continue
            sim[k]["shown"] += 1
            val = float(gl.split(txt + " ")[1][:4])
            if l_.get(k) is None:
                sim[k]["null_shown_own_oof"] += abs(val - r["grade_scores"][k]) < 1e-9
            else:
                sim[k]["labelled_shown"] += 1
                t = uc["thresholds"][k]
                sim[k]["side_ok"] += (val >= t) if l_[k] else (val <= t)
    for k, d, _ in KEYS:
        s = sim[k]
        s["shown_rate_gt_rebuild"] = s["shown"] / len(rows["train"])
        s["shown_rate_stored_oof_prompt"] = out["train"]["grade_shown"][k] / len(rows["train"])
        s["shown_rate_val_classifier"] = out["val"]["grade_shown"][k] / len(rows["val"])
        assert s["side_ok"] == s["labelled_shown"], k
        assert s["null_shown_own_oof"] == s["shown"] - s["labelled_shown"], k
    chk["train_gt_rebuild_sim"] = {"seed": a.sim_seed, "grades": sim,
                                   "unit_part_dropped_rate": {k: v / len(rows["train"]) for k, v in unit_drop.items()}}
    chk["note"] = "aggregates only; all thresholds / pools from train OOF; val rates are reported, never used"
    (data / a.checks_name).write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
