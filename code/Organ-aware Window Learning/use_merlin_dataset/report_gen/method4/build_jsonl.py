#!/usr/bin/env python3
"""Method 4 dataset builder: agent B targets + agent A leak-free OWL scores + measurements + metadata -> jsonl.

Inputs (train + val only; test rows are refused):
  --targets      report_gen/targets/targets_{train,val}.jsonl (agent B): case_id, split, per disease
                 {status, sentence_raw, sentence_clean, attributes, source[, pos]}
  --scores       report_gen/oof/out/scores_trval_prob.csv (agent A): study_id, split, 14 calibrated probabilities;
                 train = 5-fold OOF, val = full-train model (no leakage)
  --measure      report_gen/oof/out/measure_trval.csv (aorta / spleen geometry) and
  --calib        report_gen/method1/measure_calibration.json (linear map image measure -> report cm, fit on train)
  --meta         disease14/metadata.csv (phase, scanner)
  --manifest     A6000 report_gen/m4/features/manifest.csv (pack_features.py)
Output row:
  case_id, split, feature_path, n_valid,
  prompt   text with one "<image>" marker (the dataset expands it to 32 image tokens, or drops it for text-only)
  target   tagged lines "[disease(,disease)] doctor sentence", report order (B's `pos` if given, else organ order)
  target_diseases  diseases with a target sentence
The rows contain the doctor's sentences (DUA): build into a scratch dir or directly on A6000, never commit.
usage: python build_jsonl.py --targets targets_train.jsonl targets_val.jsonl --out <dir>   (writes <dir>/{train,val}.jsonl)
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m4_common import DISEASES, GRADE_SYN, NAME, RENDER_ORDER  # noqa: E402

RG = Path(os.environ.get("MERLIN_DATA", "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin")) / "report_gen"
FEAT_ROOT = "/home/advantech/Desktop/Tony/VLM/OWL/report_gen/m4/features/"


def per_disease(r):
    for k in ("diseases", "findings", "per_disease", "targets"):
        if isinstance(r.get(k), dict):
            return r[k]
    return {d: r[d] for d in DISEASES if d in r}


def pos_of(v):
    p = v.get("pos", v.get("order")) if isinstance(v, dict) else None
    if isinstance(p, list):
        p = min(p) if p else None
    return p


def attr_terms(row):
    """words to up-weight with --attr-token-weight: B's stated attribute values (side, grade, amount, type, site,
    location, rib descriptions, vertebral levels) + synonyms; numbers with cm/mm and vertebral levels are matched by
    regex in the dataset. Lower-cased, de-duplicated."""
    terms = set()
    for d, v in per_disease(row).items():
        if not isinstance(v, dict) or v.get("status") not in ("pos", "uncertain"):
            continue
        for k, val in (v.get("attributes") or {}).items():
            vals = val if isinstance(val, list) else [val]
            for x in vals:
                if not isinstance(x, str) or x in ("", "yes", "no", "not_stated") or k in ("cause",):
                    continue
                terms.update(t.lower() for t in GRADE_SYN.get(x, [x]))
    return sorted(terms)


def make_target(row):
    """B's target_lines (report order, cleaned doctor sentences, diseases per line) -> "[tags] sentence" lines.
    Fallback without target_lines: per-disease sentence_clean, identical sentences merged into one line."""
    if isinstance(row.get("target_lines"), list):
        lines, ds = [], set()
        for ln in row["target_lines"]:
            txt = " ".join((ln.get("clean") or "").split())
            tags = sorted({d for d in ln.get("diseases", []) if d in DISEASES}, key=RENDER_ORDER.index)
            if txt and tags:
                lines.append(f"[{','.join(tags)}] {txt}")
                ds.update(tags)
        return "\n".join(lines), sorted(ds, key=RENDER_ORDER.index)
    per = per_disease(row)
    sent2tags, sent2pos = {}, {}
    for d in DISEASES:
        v = per.get(d)
        s = (v.get("sentence_clean") if isinstance(v, dict) else v) or ""
        if isinstance(s, list):
            s = " ".join(x for x in s if x)
        s = " ".join(s.split())
        if not s:
            continue
        sent2tags.setdefault(s, []).append(d)
        p = pos_of(v)
        key = p if p is not None else 10_000 + RENDER_ORDER.index(d)
        sent2pos[s] = min(sent2pos.get(s, key), key)
    lines = []
    for s in sorted(sent2tags, key=lambda x: sent2pos[x]):
        tags = sorted(sent2tags[s], key=RENDER_ORDER.index)
        lines.append(f"[{','.join(tags)}] {s}")
    return "\n".join(lines), sorted({d for t in sent2tags.values() for d in t}, key=RENDER_ORDER.index)


def fnum(x, nd=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(v, nd)


def _int(x):
    v = fnum(x, 0)
    return None if v is None else int(round(v))


def make_prompt(meta, probs, meas, calib, with_scores=True, hu=None, demographics=False):
    """hu (prompt v1, 11-09): {'liver_hu', 'spleen_hu', 'wall_max_hu', 'wall_n300'} from OWL outputs; demographics adds
    age / sex from Merlin metadata (skipped when missing). Measurements are always listed, 'unknown' when missing."""
    parts = []
    if meta:
        parts.append(f"CT abdomen/pelvis, {meta.get('phase', 'unknown').replace('_', ' ')} phase; scanner "
                     f"{meta.get('manufacturer', '?')} {meta.get('manufacturermodelname', '')}".strip() + ".")
    else:
        parts.append("CT abdomen/pelvis.")
    if demographics and meta:
        age, sex = _int(meta.get("Age")), (meta.get("Gender") or "").strip().lower()
        demo = " ".join(x for x in ((f"{age}-year-old" if age is not None else ""),
                                    (sex if sex in ("male", "female") else "")) if x)
        if demo:
            parts.append(f"Patient: {demo}.")
    parts.append("<image>")
    if with_scores and probs:
        parts.append("Image classifier probabilities (0-1): " +
                     "; ".join(f"{NAME[d]} {probs[d]:.2f}" for d in DISEASES if probs.get(d) is not None) + ".")
    m = []
    meas = meas or {}
    unk = "unknown"
    if calib:
        c = calib["aaa_cm"]                                       # image measure (mm) -> report cm, fit on train
        v = fnum(meas.get(c["col"]), 3)
        txt = f"{c['a'] * v / 10 + c['b']:.1f} cm" if v is not None else unk
        m.append(f"abdominal aortic max diameter {txt}")
        c = calib["spleen_cm"]
        v = fnum(meas.get(c["col"]), 3)
        txt = f"{c['a'] * v + c['b']:.1f} cm" if v is not None else unk
        m.append(f"spleen craniocaudal length {txt}")
    v = fnum(meas.get("sp_ml"), 0)
    m.append(f"spleen volume {f'{v:.0f} mL' if v is not None else unk}")
    if hu is not None:
        lv, sv = _int(hu.get("liver_hu")), _int(hu.get("spleen_hu"))
        m.append(f"liver {f'{lv} HU' if lv is not None else unk}")
        m.append(f"spleen {f'{sv} HU' if sv is not None else unk}")
        wm, wn = _int(hu.get("wall_max_hu")), _int(hu.get("wall_n300"))
        m.append("aortoiliac wall calcification " + (f"max {wm} HU, {wn} voxels >=300 HU" if wm is not None else unk))
    parts.append("Image measurements: " + "; ".join(m) + ".")
    parts.append("Write the report sentences about these 14 findings (hepatic steatosis, gallbladder, spleen size, "
                 "hydronephrosis, abdominal aorta, atherosclerosis, bone density, bowel obstruction, appendix, "
                 "ascites, free air, gallstones, bile ducts, fracture) in the department's wording, positive and "
                 "negative, one sentence per line, each line starting with the finding tag(s) it covers in brackets; "
                 "leave out findings the radiologist would not mention.")
    return "\n".join(parts)


def load_hu(liver_spleen_csv, wall_csv):
    """OWL HU values: liver_hu / spleen_hu (outputs_final/analysis_biliary/hu_all_final.csv) and the aortoiliac
    vessel-wall shell HU (outputs_final/region_units/atherosclerosis/hu_feats.csv: max over the 4 crops of nh_*_max,
    sum of nh_*_n300)."""
    hu = {}
    if liver_spleen_csv and os.path.exists(liver_spleen_csv):
        for r in csv.DictReader(open(liver_spleen_csv)):
            hu.setdefault(r["study_id"], {}).update(liver_hu=r.get("liver_hu"), spleen_hu=r.get("spleen_hu"))
    if wall_csv and os.path.exists(wall_csv):
        segs = ["infrarenal", "bifurcation", "cia_right", "cia_left"]
        for r in csv.DictReader(open(wall_csv)):
            mx = [fnum(r.get(f"nh_{s}_max"), 1) for s in segs]
            nn = [fnum(r.get(f"nh_{s}_n300"), 1) for s in segs]
            mx = [x for x in mx if x is not None]
            nn = [x for x in nn if x is not None]
            hu.setdefault(r["study_id"], {}).update(wall_max_hu=max(mx) if mx else None,
                                                    wall_n300=sum(nn) if nn else None)
    return hu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--scores", default=str(RG / "oof/out/scores_trval_prob.csv"))
    ap.add_argument("--measure", default=str(RG / "oof/out/measure_trval.csv"))
    ap.add_argument("--calib", default=str(RG / "method1/measure_calibration.json"))
    ap.add_argument("--meta", default=str(RG.parent / "disease14/metadata.csv"))
    ap.add_argument("--manifest", default="", help="features manifest.csv (else feature_path is assumed)")
    ap.add_argument("--feat-root", default=FEAT_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hu-liver-spleen", default="", help="OWL analysis_biliary/hu_all_final.csv (prompt v1)")
    ap.add_argument("--hu-wall", default="", help="OWL region_units/atherosclerosis/hu_feats.csv (prompt v1)")
    ap.add_argument("--demographics", action="store_true", help="age / sex from metadata in the prompt (prompt v1)")
    ap.add_argument("--drop-empty", action="store_true", help="leave out cases whose target is empty")
    a = ap.parse_args()
    use_hu = bool(a.hu_liver_spleen or a.hu_wall)
    hu = load_hu(a.hu_liver_spleen, a.hu_wall) if use_hu else {}

    probs = {}
    with open(a.scores) as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                continue
            probs[r["study_id"]] = {d: fnum(r.get(d), 4) for d in DISEASES}
    meas = {}
    if a.measure and os.path.exists(a.measure):
        with open(a.measure) as f:
            meas = {r["study_id"]: r for r in csv.DictReader(f) if r.get("split") != "test"}
    calib = json.load(open(a.calib)) if a.calib and os.path.exists(a.calib) else None
    meta = {r["study id"]: r for r in csv.DictReader(open(a.meta))}
    man = {}
    if a.manifest:
        with open(a.manifest) as f:
            man = {r["study_id"]: r for r in csv.DictReader(f)}

    os.makedirs(a.out, exist_ok=True)
    fh, stats = {}, {}
    for tf in a.targets:
        for n, line in enumerate(open(tf)):
            if a.limit and n >= a.limit:
                break
            r = json.loads(line)
            cid = str(r.get("case_id", r.get("study_id")))
            sp = r.get("split")
            if sp not in ("train", "val"):
                raise SystemExit(f"{cid}: split {sp!r} refused (train/val only; test is locked)")
            st = stats.setdefault(sp, {"rows": 0, "written": 0, "no_scores": 0, "no_features": 0, "empty_target": 0,
                                       "has_age": 0, "has_sex": 0, "has_liver_hu": 0, "has_spleen_hu": 0,
                                       "has_wall_hu": 0})
            st["rows"] += 1
            if cid not in probs:
                st["no_scores"] += 1
            if man and cid not in man:
                st["no_features"] += 1
                continue
            tgt, tds = make_target(r)
            st["empty_target"] += int(not tgt)
            if a.drop_empty and not tgt:
                continue
            m_, h_ = meta.get(cid) or {}, hu.get(cid) or {}
            st["has_age"] += int(_int(m_.get("Age")) is not None)
            st["has_sex"] += int((m_.get("Gender") or "").strip().lower() in ("male", "female"))
            st["has_liver_hu"] += int(_int(h_.get("liver_hu")) is not None)
            st["has_spleen_hu"] += int(_int(h_.get("spleen_hu")) is not None)
            st["has_wall_hu"] += int(_int(h_.get("wall_max_hu")) is not None)
            st["written"] += 1
            row = {"case_id": cid, "split": sp,
                   "feature_path": a.feat_root + (man[cid]["path"] if man else f"main/{cid}.npz"),
                   "n_valid": int(man[cid]["n_valid"]) if man else None,
                   "prompt": make_prompt(meta.get(cid), probs.get(cid), meas.get(cid), calib,
                                         hu=h_ if use_hu else None, demographics=a.demographics),
                   "target": tgt, "target_diseases": tds, "attr_terms": attr_terms(r)}
            if sp not in fh:
                fh[sp] = open(Path(a.out) / f"{sp}.jsonl", "w")
            fh[sp].write(json.dumps(row, ensure_ascii=False) + "\n")
    for f in fh.values():
        f.close()
    import hashlib
    stats["_inputs_md5"] = {str(x): hashlib.md5(Path(x).read_bytes()).hexdigest()
                            for x in [*a.targets, a.scores, a.measure, a.calib, a.meta, a.manifest, a.hu_liver_spleen,
                                      a.hu_wall] if x and os.path.exists(x)}
    stats["_args"] = vars(a)
    json.dump(stats, open(Path(a.out) / "build_stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
