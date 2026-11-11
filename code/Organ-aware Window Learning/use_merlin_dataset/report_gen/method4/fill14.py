#!/usr/bin/env python3
"""Method 4, improvement 1 ("fill all 14"): post-process a method-4 generation so that every finding the report left
out gets the method-1 template sentence that matches the cue the model was given.

A finding counts as left out ("omitted") when the Qwen re-extraction of the ORIGINAL generation says not_mentioned
(evaluate.py per_case_categories.jsonl, [reference, generated]; --omit-rule qwen, default). --omit-rule dict uses the
generation's own findings dict instead (no sentence under that key); it over-fills appendicitis, which the models
usually mention inside the bowel sentence.
Fill decision per omitted finding, from the cue fed to the model (<gen>.cues.jsonl: cue + flips; flips applied):
  cue >= method-1 threshold (cues.meta.json, = t_f1 of method 1 variant B) -> positive template sentence
  cue <  threshold -> negative template sentence, except the silent-negative findings of method 1 (osteopenia,
                      atherosclerosis: method 1 writes nothing when negative) -> left as is
  no cue -> left as is
Sentences come from the method-1 bank exactly as m1_assemble.py picks them (top single sentence per finding|polarity;
the joint sentence of an organ group only when every active finding of that group is being filled and the bank has the
joint key; AAA / splenomegaly positive sentences carry the segmentation measurement, linearly calibrated on train).

Outputs:
  <out>.jsonl              {case_id, findings: filled dict, filled: {disease: "pos"|"neg"}}   (evaluate.py --gen)
  <out>.merge_ext.jsonl    {case_id, overrides: {disease: {category, detail, quotes}}}          (evaluate.py
                           --ext-override, with --extract-from <original gen>: "merge" scoring = original extraction,
                           filled findings take the template state; validated against a real Qwen re-extraction)
  <out>.summary.json       counts only (no text)
Val only; the test split is never read (rows of other splits in the label file are skipped before parsing).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def text_of(v):
    if isinstance(v, list):
        v = " ".join(x for x in v if x)
    return (v or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="method-4 generation jsonl (infer.py output)")
    ap.add_argument("--cues", required=True, help="<gen>.cues.jsonl (cues.meta.json next to it holds the thresholds)")
    ap.add_argument("--per-case", default="", help="evaluate.py per_case_categories.jsonl of the original generation")
    ap.add_argument("--omit-rule", choices=("qwen", "dict"), default="qwen")
    ap.add_argument("--bank", required=True, help="method-1 sentence_bank.json")
    ap.add_argument("--measure", required=True, help="oof/out/measure_trval.csv")
    ap.add_argument("--calib", required=True, help="method-1 measure_calibration.json")
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    a = ap.parse_args()

    bank = json.load(open(a.bank))
    groups, silent = bank["groups"], bank["silent_negative"]
    single = {k: v[0][0] for k, v in bank["single"].items() if v}
    joint = {k: v[0][0] for k, v in bank["joint"].items()}
    thr = json.loads(Path(a.cues[: -len(".jsonl")] + ".meta.json").read_text())["thresholds"]
    mc = json.load(open(a.calib))
    meas = {}
    with open(a.measure) as f:
        for r in csv.DictReader(f):
            if r.get("split") == "val":
                meas[r["study_id"]] = r

    cues = {}
    for r in read_jsonl(a.cues):
        c = dict(r["cue"])
        for d, (old, new, _dir) in (r.get("flips") or {}).items():
            c[d] = new
        cues[r["case_id"]] = c
    pc = {}
    if a.omit_rule == "qwen":
        assert a.per_case, "--omit-rule qwen needs --per-case"
        pc = {r["case_id"]: r for r in read_jsonl(a.per_case)}

    def measured(cid, key):
        m = meas.get(cid)
        col = mc[key]["col"]
        if not m or m.get(col) in (None, "", "nan"):
            return None
        x = float(m[col]) / (10.0 if key == "aaa_cm" else 1.0)
        return mc[key]["a"] * x + mc[key]["b"]

    def pos_sentence(cid, d):
        if d == "abdominal_aortic_aneurysm":
            v = measured(cid, "aaa_cm")
            if v is not None:
                return f"Abdominal aortic aneurysm measuring up to {v:.1f} cm.", {"diameter_cm": round(v, 1)}
        if d == "splenomegaly":
            v = measured(cid, "spleen_cm")
            if v is not None:
                return f"Splenomegaly, measuring {v:.1f} cm in craniocaudal length.", {"length_cm": round(v, 1)}
        return single.get(f"{d}|pos"), {}

    cnt, cnt_d, n_cases_filled, n_skip = Counter(), Counter(), 0, Counter()
    out_rows, merge_rows = [], []
    for r in read_jsonl(a.gen):
        cid = r["case_id"]
        f = {d: text_of(v) for d, v in (r.get("findings") or {}).items()}
        cue = cues.get(cid)
        if cue is None:
            raise SystemExit(f"no cue row for a generated case (row {len(out_rows)})")
        if a.omit_rule == "qwen":
            if cid not in pc:          # extraction failed for this case: nothing known -> leave it unchanged
                n_skip["no_extraction"] += 1
                omitted = set()
            else:
                omitted = {d for d in DISEASES if pc[cid][d][1] == "not_mentioned"}
        else:
            omitted = {d for d in DISEASES if not f.get(d)}
        pol = {}
        for d in omitted:
            v = cue.get(d)
            if v is None:
                n_skip["no_cue"] += 1
                continue
            p = "pos" if v >= thr[d] else "neg"
            if p == "neg" and silent[d]:
                n_skip["silent_negative"] += 1
                continue
            pol[d] = p
        new = dict(f)
        filled, over = {}, {}
        for g, ds in groups:
            fill_here = [d for d in ds if d in pol]
            if not fill_here:
                continue
            # method-1 active set of the group under the cue decisions
            act = [d for d in ds if cue.get(d) is not None and not (silent[d] and cue[d] < thr[d])]
            gpol = {d: ("pos" if cue[d] >= thr[d] else "neg") for d in act}
            k = g + "|" + ",".join(f"{d}:{gpol[d]}" for d in act)
            use_joint = (set(act) == set(fill_here) and k in joint and
                         not any(d in ("abdominal_aortic_aneurysm", "splenomegaly") and gpol[d] == "pos" for d in act))
            for d in fill_here:
                det = {}
                if use_joint:
                    s = joint[k]
                elif pol[d] == "pos":
                    s, det = pos_sentence(cid, d)
                else:
                    s = single.get(f"{d}|neg")
                if not s:
                    n_skip["no_bank_sentence"] += 1
                    continue
                new[d] = s
                filled[d] = pol[d]
                over[d] = {"category": "yes" if pol[d] == "pos" else "no", "detail": det, "quotes": [s],
                           "why": "fill14 template"}
                cnt[pol[d]] += 1
                cnt_d[(d, pol[d])] += 1
        n_cases_filled += bool(filled)
        out_rows.append({"case_id": cid, "findings": new, "filled": filled})
        merge_rows.append({"case_id": cid, "overrides": over})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for suf, rows in ((".jsonl", out_rows), (".merge_ext.jsonl", merge_rows)):
        with open(a.out + suf, "w") as fo:
            for x in rows:
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    summ = {"gen": a.gen, "cues": a.cues, "omit_rule": a.omit_rule, "n_cases": len(out_rows),
            "n_cases_filled": n_cases_filled, "n_filled_pos": cnt["pos"], "n_filled_neg": cnt["neg"],
            "skipped": dict(n_skip),
            "per_disease": {d: {"pos": cnt_d[(d, "pos")], "neg": cnt_d[(d, "neg")]} for d in DISEASES}}
    Path(a.out + ".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_disease"}))


if __name__ == "__main__":
    main()
