#!/usr/bin/env python3
"""Phase 3 dataset builder: std14 output + split + OWL outputs + measurements + feature paths -> train/val/test JSONL.

usage:
  python p3_build_jsonl.py                                   # final: standardize/reports_std.jsonl
  python p3_build_jsonl.py --std a.jsonl b.jsonl --dry-run   # partial shards (raw std14 rows) -> jsonl_dryrun/
Each output row (one exam):
  exam_id, pid, loc, site, scanner, split, p1_split
  feature_path      A6000 path of the packed token file (tok [1062,448] fp16, mask [1062]); token_meta.csv
  feature_n_valid
  owl               {disease: {logit, score=sigmoid(logit), call in neg/verify/pos/omit (Phase 1 thresholds)}}
  meas              image measurements (measurements_p3.csv + Phase 1 L1 HU / TS spleen & aorta)
  prompt            text part of the input (site, scanner, OWL calls, measurements); image tokens are prepended
                    by the model code at <image>
  target            doctor's original sentences about the 14 diseases, report order, one per line (p3_targets_lib)
  target_lines      [{text, diseases, sec, action}]
  ref               evaluation references only (never model input): per disease std14 status / attributes /
                    normalized_sentence / comparison and the disease14 (Qwen) category
  flags             counters from target construction
Exams without a parsed std14 row are skipped (listed in build_summary.json).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "standardize"))
from p3_target_stats import load_std  # noqa: E402
from p3_targets_lib import DISEASES, build_target, load_typo_fix  # noqa: E402

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
RG = ROOT / "report_gen"
P3 = RG / "phase3"
FEAT_ROOT = "/home/advantech/Desktop/Tony/VLM/report_gen_p3/features/"
NAME = {"hepatic_steatosis": "fatty liver", "surgically_absent_gallbladder": "post-cholecystectomy (absent gallbladder)",
        "splenomegaly": "splenomegaly", "hydronephrosis": "hydronephrosis", "abdominal_aortic_aneurysm": "abdominal aortic aneurysm",
        "atherosclerosis": "atherosclerosis", "osteopenia": "osteopenia", "bowel_obstruction": "bowel obstruction",
        "appendicitis": "appendicitis", "ascites": "ascites", "free_air": "free air (pneumoperitoneum)",
        "gallstones": "gallstones", "biliary_ductal_dilation": "biliary ductal dilatation", "fracture": "fracture"}
CALL_TXT = {"neg": "unlikely", "verify": "uncertain", "pos": "likely", "omit": "rare finding, low score"}


def sig(x):
    return 1 / (1 + math.exp(-x)) if x == x else float("nan")


def fmt(v, nd=0, unit=""):
    if v is None or v != v:
        return "n/a"
    return f"{v:.{nd}f}{unit}"


def make_prompt(site, scanner, owl, m):
    o = "; ".join(f"{NAME[d]} {owl[d]['score']:.2f} ({CALL_TXT[owl[d]['call']]})" for d in DISEASES)
    meas = (f"liver {fmt(m.get('liver_hu'))} HU, spleen {fmt(m.get('spleen_hu'))} HU (liver-spleen {fmt(m.get('liver_minus_spleen_hu'))}); "
            f"spleen volume {fmt(m.get('spleen_ml'))} mL, craniocaudal {fmt(m.get('spleen_cc_cm'), 1)} cm; "
            f"infrarenal aortic max diameter {fmt(m.get('aorta_infra_max_mm'))} mm; "
            f"L1 trabecular {fmt(m.get('L1_hu'))} HU; small-bowel max diameter {fmt(m.get('sb_diam_max_mm'))} mm; "
            f"renal sinus fluid L/R {fmt(m.get('kid_l_sinus_fluid'))}/{fmt(m.get('kid_r_sinus_fluid'))} voxels; "
            f"gallbladder voxels {fmt(m.get('gb_vox'))}, dense (>130 HU) {fmt(m.get('gb_dense_n130'))}; "
            f"extraluminal air {fmt(m.get('extra_air_deep'))} voxels")
    return (f"Site: {site}. Scanner: {scanner}.\n<image>\n"
            f"OWL classifier scores (0-1, call): {o}.\n"
            f"Measurements: {meas}.\n"
            f"Write the findings sentences about the 14 target findings (fatty liver, gallbladder status, spleen size, "
            f"hydronephrosis, aorta, atherosclerosis, bone density, bowel obstruction, appendix, ascites, free air, "
            f"gallstones, biliary ducts, fracture) in this site's reporting style, one sentence per line.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", nargs="+", default=[str(RG / "standardize/reports_std.jsonl")])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="output dir (default data/.../phase3/jsonl[_dryrun]); the JSONL are "
                    "~45 MB in total -> build into a scratch dir and rsync to A6000 report_gen_p3/data/ (no large files on the Mac)")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else P3 / ("jsonl_dryrun" if a.dry_run else "jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = json.load(open(RG / "standardize/reports_std_in.json"))
    std = load_std(a.std)
    fix = load_typo_fix(P3 / "typo_dictionary.csv")
    split = pd.read_csv(P3 / "split_p3.csv").set_index("EXAMID")
    probs = pd.read_csv(ROOT / "owl_infer/best/probs_best_main.csv").set_index("study_id")
    thr = pd.read_csv(RG / "thresholds.csv").set_index("disease")
    raw = pd.read_csv(ROOT / "deliver_10k_reports_v1(in).csv", encoding="cp1252")
    raw.columns = ["PID", "EXAMID", "EDATE", "LOC", "scanner", "Report"]
    scanner = raw.set_index("EXAMID").scanner
    meas = pd.read_csv(P3 / "measurements_p3.csv").set_index("EXAMID")
    m1 = pd.read_csv(RG / "measurements.csv").set_index("EXAMID")
    man = pd.read_csv(P3 / "manifest.csv").set_index("EXAMID")
    lab = pd.read_csv(ROOT / "disease14_qwen.csv", encoding="utf-8-sig").set_index("EXAMID")

    outs = {s: open(out_dir / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    summ = {"n_split": len(split), "written": {s: 0 for s in outs}, "missing_std": 0, "empty_target": {s: 0 for s in outs},
            "fallback_norm": 0, "temporal_stripped": 0, "temporal_drop": 0, "typo_fixed": 0, "imp_only": 0,
            "target_chars_mean": {}, "target_lines_mean": {}}
    miss, chars, nlines = [], {s: [] for s in outs}, {s: [] for s in outs}
    for eid, sp in split.iterrows():
        if eid not in std:
            miss.append(eid)
            continue
        ds = std[eid]
        t = build_target(rep[eid]["report"], ds, fix)
        owl = {}
        for d in DISEASES:
            lg = float(probs.at[eid, d])
            tn, tp = thr.at[d, "t_neg"], thr.at[d, "t_pos"]
            rule = thr.at[d, "rule"]
            call = "neg" if lg < tn else ("pos" if lg >= tp else ("omit" if rule == "rare_silent" else "verify"))
            owl[d] = {"logit": round(lg, 4), "score": round(sig(lg), 4), "call": call}
        m = {k: (None if pd.isna(v) else float(v)) for k, v in meas.loc[eid].items()}
        m["L1_hu"] = None if pd.isna(m1.at[eid, "L1_hu"]) else float(m1.at[eid, "L1_hu"])
        m["L1_level"] = None if pd.isna(m1.at[eid, "hu_level"]) else str(m1.at[eid, "hu_level"])
        site = "site" + str(sp.LOC).split("(")[-1].rstrip(")")
        sc = str(scanner.get(eid, ""))
        row = dict(exam_id=eid, pid=sp.PID, loc=sp.LOC, site=site, scanner=sc, split=sp.split, p1_split=sp.p1_split,
                   feature_path=FEAT_ROOT + man.at[eid, "path"], feature_n_valid=int(man.at[eid, "n_valid"]),
                   owl=owl, meas=m, prompt=make_prompt(site, sc, owl, m), target=t["target"],
                   target_lines=[{k: ln[k] for k in ("text", "diseases", "sec", "action")} for ln in t["lines"]],
                   ref={d: {"status": ds[d]["status"], "attributes": ds[d]["attributes"],
                            "normalized_sentence": ds[d]["normalized_sentence"], "comparison": ds[d]["comparison"],
                            "d14_cat": str(lab.at[eid, d + "_cat"])} for d in DISEASES},
                   flags=t["counters"])
        outs[sp.split].write(json.dumps(row, ensure_ascii=False) + "\n")
        summ["written"][sp.split] += 1
        summ["empty_target"][sp.split] += int(not t["lines"])
        for k in ("fallback_norm", "temporal_stripped", "temporal_drop", "typo_fixed", "imp_only"):
            summ[k] += t["counters"][k]
        chars[sp.split].append(len(t["target"]))
        nlines[sp.split].append(len(t["lines"]))
    for f in outs.values():
        f.close()
    summ["missing_std"] = len(miss)
    summ["target_chars_mean"] = {s: round(sum(v) / max(len(v), 1), 1) for s, v in chars.items()}
    summ["target_lines_mean"] = {s: round(sum(v) / max(len(v), 1), 2) for s, v in nlines.items()}
    summ["missing_std_ids_head"] = miss[:20]
    json.dump(summ, open(out_dir / "build_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
