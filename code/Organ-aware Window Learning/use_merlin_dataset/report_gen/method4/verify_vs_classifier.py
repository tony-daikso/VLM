#!/usr/bin/env python3
"""Hypothesis check: does the method-4 generated report score close to (and not above) the classifier itself at t_f1
on the 14-class presence task?

Same 600 val cases (or full val), same reference definition as evaluate.py ("owl": reference yes|probably_yes = 1,
no|probably_no|not_mentioned = 0, unclear excluded; generated strict: yes|probably_yes = positive).

Compared methods
  classifier      : OWL score (report_gen/oof/out/scores_trval_prob.csv) >= method-1 t_f1 (method1/thresholds.csv);
                    a missing score counts as "not called positive". No Qwen involved.
  classifier_cue  : same, on the 2-decimal cue value actually printed in the prompt (<gen>.cues.jsonl).
  m1B             : method 1 variant B templates, Qwen re-extracted (eval/results/m1B_sub600/per_case_categories.jsonl).
  U_oof/U_gt/T_oof/T_gt : method 4 generations, Qwen re-extracted (m4/eval/results_cue/<g>_val600/).
  <g>+fill14      : the "merge" scoring of fill14.py (original extraction, filled findings take the template state;
                    fill14/gen/<g>/val600.merge_ext.jsonl), recomputed here for every group; checked against
                    m4/eval/results_fill14/<g>_val600_fill_merge when that exists.

Outputs (numbers only, no report text, no ids): <out>/verify_vs_classifier.json and verify_vs_classifier.md.
Gap decomposition per cell (case x finding, reference not unclear): agree (both right / both wrong), deviation that
fixes the classifier, deviation that breaks it; breaks are split by the cue_pos_breakdown.py rule (tagged sentence
missing / present but not_mentioned / unclear / negation wording or not) -- a heuristic on the generated sentence.
Paired case bootstrap (1,000 resamples) for macro / micro F1 differences vs the classifier.
The test split is never read: label-file lines of other splits are skipped before parsing.

usage (A6000, CPU):
  nice -n 19 ~/miniconda3/envs/radar/bin/python verify_vs_classifier.py --m4 ~/Desktop/Tony/VLM/OWL/report_gen/m4 \
      --inputs <dir with thresholds.csv, scores_trval_prob.csv, val_decisions_B.csv, m1B_sub600_per_case.jsonl> \
      --labels ~/Desktop/Tony/VLM/OWL/data/labels_disease14_all.jsonl --out <dir>
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DISEASES = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis",
            "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis",
            "ascites", "free_air", "gallstones", "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
GROUPS = ["U_oof", "U_gt", "T_oof", "T_gt"]
POS = {"yes", "probably_yes"}
NEG_EXPLICIT = {"no", "probably_no"}
# same rule as improve/cue_pos_breakdown.py
NEG = re.compile(r"(?i)\b(no|not|without|negative|unremarkable|normal|normally|resolved|none|absence|free of|"
                 r"nondilated|non-dilated|patent|intact)\b")
POS_SAG = re.compile(r"(?i)(cholecystectomy|surgically absent|absent|removed|post.?surgical|not (seen|visualized))")
BREAK_FN = ["漏寫", "有句但判未提及", "寫成不確定", "措辭被判反", "真的寫反"]
BREAK_FP = ["他句被判陽", "陰性措辭被判陽", "真的寫成陽性"]
B = 1000
SEED = 20261111


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def ref_owl(cat):
    return 1.0 if cat in POS else 0.0 if (cat in NEG_EXPLICIT or cat == "not_mentioned") else np.nan


def sentence_negative(d, s):
    """True when the tagged sentence reads negative for finding d (heuristic)."""
    if d == "surgically_absent_gallbladder":
        return not POS_SAG.search(s or "")
    return bool(NEG.search(s or ""))


def classify_fn(d, gcat, s):
    """classifier positive and right, generation not positive."""
    if gcat == "not_mentioned":
        return "漏寫" if not s else "有句但判未提及"
    if gcat == "unclear":
        return "寫成不確定"
    return "真的寫反" if sentence_negative(d, s) else "措辭被判反"


def classify_fp(d, s):
    """classifier negative and right, generation positive."""
    if not s:
        return "他句被判陽"
    return "陰性措辭被判陽" if sentence_negative(d, s) else "真的寫成陽性"


def counts(pred, y):
    """pred, y: (N, 14) bool / float(nan) -> per-cell indicator arrays (N, 14) for TP, FP, FN, TN."""
    ok = ~np.isnan(y)
    yy = (y == 1) & ok
    yn = (y == 0) & ok
    return dict(TP=(pred & yy).astype(float), FP=(pred & yn).astype(float), FN=(~pred & yy).astype(float),
                TN=(~pred & yn).astype(float))


def metr(C, W=None):
    """C: indicator arrays (N, 14); W: (B, N) bootstrap weights or None. Returns dict of arrays (.., 14) + macro/micro."""
    agg = {k: (v.sum(0) if W is None else W @ v) for k, v in C.items()}
    TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, spec = TP / (TP + FN), TN / (TN + FP)
        f1 = 2 * TP / (2 * TP + FP + FN)
        sT, sP, sN, sTN = TP.sum(-1), FP.sum(-1), FN.sum(-1), TN.sum(-1)
        out = dict(sens=sens, spec=spec, F1=f1,
                   macro_sens=np.nanmean(sens, -1), macro_spec=np.nanmean(spec, -1), macro_F1=np.nanmean(f1, -1),
                   micro_sens=sT / (sT + sN), micro_spec=sTN / (sTN + sP), micro_F1=2 * sT / (2 * sT + sP + sN))
    return out


def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fs(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def load_labels(path, split):
    out = {}
    with open(path) as f:
        for line in f:
            if f'"split": "{split}"' not in line and f'"split":"{split}"' not in line:
                continue                                   # other splits (incl. test) skipped before parsing
            r = json.loads(line)
            if r.get("split") != split:
                continue
            ans = r["answer"] if isinstance(r["answer"], dict) else json.loads(r["answer"])
            out[str(r["study_id"])] = {d: ans[d]["category"] for d in DISEASES}
    return out


def analyse(tag, ids, ref, preds, gen_sent, clf_name="classifier"):
    """preds: {method: (N,14) bool}; gen_sent: {method: {(i,k): sentence or None}} (only for decomposition)."""
    y = np.array([[ref_owl(ref[c][d]) for d in DISEASES] for c in ids])
    C = {m: counts(p, y) for m, p in preds.items()}
    point = {m: metr(c) for m, c in C.items()}
    rng = np.random.default_rng(SEED)
    N = len(ids)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)
    boot = {m: metr(c, W) for m, c in C.items()}
    res = {"n_cases": N, "methods": {}}
    for m in preds:
        P, Bm, Bc = point[m], boot[m], boot[clf_name]
        r = {k: float(P[k]) for k in ("macro_sens", "macro_spec", "macro_F1", "micro_sens", "micro_spec", "micro_F1")}
        r["macro_F1_ci"] = ci(Bm["macro_F1"])
        r["micro_F1_ci"] = ci(Bm["micro_F1"])
        r["per_class"] = {d: {k: float(P[k][j]) for k in ("sens", "spec", "F1")} for j, d in enumerate(DISEASES)}
        if m != clf_name:
            for k in ("macro_F1", "micro_F1", "macro_sens", "macro_spec"):
                dlt = Bm[k] - Bc[k]
                r[f"d_{k}"] = float(P[k] - point[clf_name][k])
                r[f"d_{k}_ci"] = ci(dlt)
                r[f"d_{k}_p_gt0"] = float(np.mean(dlt > 0))
            dd = Bm["F1"] - Bc["F1"]
            r["d_F1_class"] = {d: {"d": float(P["F1"][j] - point[clf_name]["F1"][j]), "ci": ci(dd[:, j])}
                               for j, d in enumerate(DISEASES)}
        res["methods"][m] = r
    # gap decomposition vs the classifier
    pc = preds[clf_name]
    ok = ~np.isnan(y)
    yb = y == 1
    for m in preds:
        if m.startswith("classifier"):
            continue
        pg = preds[m]
        dec = Counter()
        per = defaultdict(Counter)
        sents = gen_sent.get(m)
        for i in range(N):
            for k, d in enumerate(DISEASES):
                if not ok[i, k]:
                    continue
                c_right = pc[i, k] == yb[i, k]
                g_right = pg[i, k] == yb[i, k]
                if pc[i, k] == pg[i, k]:
                    key = "一致_都對" if c_right else "一致_都錯"
                elif g_right:
                    key = "改對_補上陽性" if pg[i, k] else "改對_去掉假陽"
                else:
                    if pc[i, k]:   # classifier TP, generation FN
                        sub = classify_fn(d, ref_gen_cat[m][i][k], sents.get((i, k))) if sents is not None else "未細分"
                        key = "改錯_漏陽:" + sub
                    else:          # classifier TN, generation FP
                        sub = classify_fp(d, sents.get((i, k))) if sents is not None else "未細分"
                        key = "改錯_多陽:" + sub
                dec[key] += 1
                per[d][key.split(":")[0]] += 1
                if ":" in key:
                    per[d][key] += 1
        res["methods"][m]["decomp"] = dict(dec)
        res["methods"][m]["decomp_class"] = {d: dict(per[d]) for d in DISEASES}
    return res


ref_gen_cat = {}   # method -> list over cases of 14 generated categories (used by classify_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M4, INP, OUT = Path(a.m4).expanduser(), Path(a.inputs).expanduser(), Path(a.out).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    thr = {r["disease"]: float(r["t_f1"]) for r in csv.DictReader(open(INP / "thresholds.csv"))}
    scores, split_of = {}, {}
    for r in csv.DictReader(open(INP / "scores_trval_prob.csv")):
        if r["split"] == "test":
            continue
        split_of[r["study_id"]] = r["split"]
        scores[r["study_id"]] = {d: (float(r[d]) if r[d] not in ("", "nan", "NaN") else None) for d in DISEASES}
    lab_val = load_labels(a.labels, "val")
    lab_train = load_labels(a.labels, "train")
    report = {"thresholds_t_f1": thr}

    # ---------------------------------------------------------------- point 5: missing scores / cues
    miss = {"val_full": {}, "train": {}}
    for sp, lab in (("val_full", lab_val), ("train", lab_train)):
        tab = {}
        for d in DISEASES:
            t = Counter()
            for c, L in lab.items():
                s = scores.get(c, {}).get(d) if c in scores else None
                t[("missing" if s is None else "scored") + "|" + ("unclear" if L[d] == "unclear" else "labelled")] += 1
            tab[d] = dict(t)
        miss[sp] = tab
    report["missing_vs_unclear"] = miss

    # ---------------------------------------------------------------- val600 analysis
    m4cats = {g: {r["case_id"]: r for r in read_jsonl(M4 / f"eval/results_cue/{g}_val600/per_case_categories.jsonl")}
              for g in GROUPS}
    m1b = {r["case_id"]: r for r in read_jsonl(INP / "m1B_sub600_per_case.jsonl")}
    decB = {r["study_id"]: r for r in csv.DictReader(open(INP / "val_decisions_B.csv"))}
    cues = {g: {r["case_id"]: r for r in read_jsonl(M4 / f"gen_cue/{g}/val600.cues.jsonl")} for g in GROUPS}
    gens = {g: {r["case_id"]: r["findings"] for r in read_jsonl(M4 / f"gen_cue/{g}/val600.jsonl")} for g in GROUPS}
    fill = {}
    for g in GROUPS:
        p = M4 / f"fill14/gen/{g}/val600.merge_ext.jsonl"
        if p.exists():
            fill[g] = ({r["case_id"]: r.get("overrides", {}) for r in read_jsonl(p)},
                       {r["case_id"]: r["findings"] for r in read_jsonl(M4 / f"fill14/gen/{g}/val600.jsonl")})
    ids = sorted(set(m1b).intersection(*[set(m4cats[g]) for g in GROUPS]))
    report["val600"] = {"n_common": len(ids), "n_m1b": len(m1b), **{f"n_{g}": len(m4cats[g]) for g in GROUPS}}
    # reference consistency
    ref = {c: {d: m4cats["U_oof"][c][d][0] for d in DISEASES} for c in ids}
    bad_ref = sum(1 for c in ids for d in DISEASES
                  if any(m4cats[g][c][d][0] != ref[c][d] for g in GROUPS) or m1b[c][d][0] != ref[c][d]
                  or lab_val.get(c, {}).get(d) != ref[c][d])
    report["val600"]["ref_mismatch_cells"] = bad_ref
    # cue vs score consistency
    cue_ne_score, cue_dec_ne = 0, 0
    for c in ids:
        for d in DISEASES:
            s, v = scores[c][d], cues["U_oof"][c]["cue"].get(d)
            if (s is None) != (v is None) or (s is not None and abs(round(s, 2) - v) > 0.0051):
                cue_ne_score += 1
            if s is not None and v is not None and (s >= thr[d]) != (v >= thr[d]):
                cue_dec_ne += 1
    same_cue = all(cues[g][c]["cue"] == cues["U_oof"][c]["cue"] for g in GROUPS for c in ids)
    report["val600"].update(cue_vs_score_mismatch=cue_ne_score, cue_rounding_decision_changes=cue_dec_ne,
                            cue_identical_across_groups=same_cue)
    # m1B decisions vs t_f1
    dB = Counter()
    for c in ids:
        for d in DISEASES:
            s, x = scores[c][d], decB[c][d]
            clf = None if s is None else ("pos" if s >= thr[d] else "neg")
            dB[f"{clf}->{x}"] += 1
    report["val600"]["m1B_decision_vs_tf1"] = dict(dB)

    # predictions
    preds = {}
    preds["classifier"] = np.array([[scores[c][d] is not None and scores[c][d] >= thr[d] for d in DISEASES] for c in ids])
    preds["classifier_cue"] = np.array([[cues["U_oof"][c]["cue"].get(d) is not None
                                         and cues["U_oof"][c]["cue"][d] >= thr[d] for d in DISEASES] for c in ids])
    preds["m1B"] = np.array([[m1b[c][d][1] in POS for d in DISEASES] for c in ids])
    ref_gen_cat["m1B"] = [[m1b[c][d][1] for d in DISEASES] for c in ids]
    gen_sent = {}
    # m1B "sentence": a template is written exactly when decision B is pos/neg; give the polarity word as the sentence
    gen_sent["m1B"] = {(i, k): ({"pos": "x", "neg": "no"}.get(decB[c][d]) or None)
                       for i, c in enumerate(ids) for k, d in enumerate(DISEASES)}
    fill_check = {}
    for g in GROUPS:
        preds[g] = np.array([[m4cats[g][c][d][1] in POS for d in DISEASES] for c in ids])
        ref_gen_cat[g] = [[m4cats[g][c][d][1] for d in DISEASES] for c in ids]
        gen_sent[g] = {(i, k): gens[g].get(c, {}).get(d) for i, c in enumerate(ids) for k, d in enumerate(DISEASES)}
        if g in fill:
            ov, ff = fill[g]
            cat = [[(ov.get(c, {}).get(d, {}) or {}).get("category", m4cats[g][c][d][1]) for d in DISEASES] for c in ids]
            preds[g + "+fill14"] = np.array([[x in POS for x in row] for row in cat])
            ref_gen_cat[g + "+fill14"] = cat
            gen_sent[g + "+fill14"] = {(i, k): ff.get(c, {}).get(d) for i, c in enumerate(ids)
                                       for k, d in enumerate(DISEASES)}
            pe = M4 / f"eval/results_fill14/{g}_val600_fill_merge/per_case_categories.jsonl"
            if pe.exists():
                ev = {r["case_id"]: r for r in read_jsonl(pe)}
                fill_check[g] = sum(1 for i, c in enumerate(ids) for k, d in enumerate(DISEASES)
                                    if c in ev and ev[c][d][1] != cat[i][k])
    report["val600"]["fill14_vs_official_merge_mismatch_cells"] = fill_check
    report["val600"]["fill14_groups"] = sorted(fill)
    report["val600"]["res"] = analyse("val600", ids, ref, preds, gen_sent)
    report["implicit"] = implicit_thresholds(ids, ref, scores, cues, m4cats, lab_train, thr, preds,
                                             M4 / "cue/cue_gt.json")

    # point 5 on val600: missing cue cells and how the generation treated them
    mc = {"n_cases_with_missing": 0, "n_cells": 0, "by_class": Counter(), "ref_cat": Counter(),
          "n_missing_per_case": Counter(), "gen_cat": {g: Counter() for g in GROUPS},
          "gen_cat_scored_unclear_ref": {g: Counter() for g in GROUPS}, "gen_has_sentence": Counter()}
    for c in ids:
        miss_d = [d for d in DISEASES if cues["U_oof"][c]["cue"].get(d) is None]
        mc["n_missing_per_case"][len(miss_d)] += 1
        if miss_d:
            mc["n_cases_with_missing"] += 1
        for d in miss_d:
            mc["n_cells"] += 1
            mc["by_class"][d] += 1
            mc["ref_cat"][ref[c][d]] += 1
            for g in GROUPS:
                mc["gen_cat"][g][m4cats[g][c][d][1]] += 1
                mc["gen_has_sentence"][g] += int(bool(gens[g].get(c, {}).get(d)))
    # baseline: all scored cells -> share of "unclear" generated
    base = {g: Counter(m4cats[g][c][d][1] for c in ids for d in DISEASES if cues["U_oof"][c]["cue"].get(d) is not None)
            for g in GROUPS}
    mc = {k: (dict(v) if isinstance(v, Counter) else {g: dict(x) for g, x in v.items()} if isinstance(v, dict) else v)
          for k, v in mc.items()}
    mc["gen_cat_scored_cells"] = {g: dict(v) for g, v in base.items()}
    report["val600"]["missing_cue"] = mc

    # ---------------------------------------------------------------- full val (if any group finished)
    report["valfull"] = {}
    for g in GROUPS:
        pe = M4 / f"eval/results_cue/{g}_valfull/per_case_categories.jsonl"
        if not pe.exists():
            continue
        cats = {r["case_id"]: r for r in read_jsonl(pe)}
        gg = {r["case_id"]: r["findings"] for r in read_jsonl(M4 / f"gen_cue/{g}/val_full.jsonl")}
        fid = sorted(cats)
        fref = {c: {d: cats[c][d][0] for d in DISEASES} for c in fid}
        fp = {"classifier": np.array([[scores[c][d] is not None and scores[c][d] >= thr[d] for d in DISEASES]
                                      for c in fid]),
              g: np.array([[cats[c][d][1] in POS for d in DISEASES] for c in fid])}
        ref_gen_cat[g] = [[cats[c][d][1] for d in DISEASES] for c in fid]
        gs = {g: {(i, k): gg.get(c, {}).get(d) for i, c in enumerate(fid) for k, d in enumerate(DISEASES)}}
        report["valfull"][g] = analyse("valfull", fid, fref, fp, gs)

    (OUT / "verify_vs_classifier.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (OUT / "verify_vs_classifier.md").write_text(render_md(report))
    print(f"wrote {OUT}/verify_vs_classifier.json, .md")


NAME = {"classifier": "分類器（分數 ≥ t_f1）", "classifier_cue": "分類器（提示值兩位小數 ≥ t_f1）",
        "m1B": "方法一 1B 樣板", "U_oof": "U-oof", "U_gt": "U-gt", "T_oof": "T-oof", "T_gt": "T-gt"}


def nm(m):
    return NAME.get(m, NAME.get(m.split("+")[0], m) + "＋補滿14" if m.endswith("+fill14") else m)


def table_main(res):
    L = ["| 方法 | macro 敏感度 | macro 特異度 | macro F1 [95% CI] | micro 敏感度 | micro 特異度 | micro F1 [95% CI] | "
         "Δmacro F1 vs 分類器 [95% CI] | P(Δ>0) | Δmicro F1 [95% CI] |", "|---|" + "---|" * 9]
    for m, r in res["methods"].items():
        d = (f"{fs(r['d_macro_F1'])} [{fs(r['d_macro_F1_ci'][0])}, {fs(r['d_macro_F1_ci'][1])}] | "
             f"{r['d_macro_F1_p_gt0']:.3f} | {fs(r['d_micro_F1'])} [{fs(r['d_micro_F1_ci'][0])}, {fs(r['d_micro_F1_ci'][1])}]"
             if "d_macro_F1" in r else "— | — | —")
        L.append(f"| {nm(m)} | {f3(r['macro_sens'])} | {f3(r['macro_spec'])} | {f3(r['macro_F1'])} "
                 f"[{f3(r['macro_F1_ci'][0])}, {f3(r['macro_F1_ci'][1])}] | {f3(r['micro_sens'])} | {f3(r['micro_spec'])} | "
                 f"{f3(r['micro_F1'])} [{f3(r['micro_F1_ci'][0])}, {f3(r['micro_F1_ci'][1])}] | {d} |")
    return L


def table_class(res):
    ms = [m for m in res["methods"] if m != "classifier"]
    L = ["| 疾病 | 分類器 F1 | " + " | ".join(f"Δ {nm(m)}" for m in ms) + " |", "|---|---|" + "---|" * len(ms)]
    for d in DISEASES:
        row = [ZH[d], f3(res["methods"]["classifier"]["per_class"][d]["F1"])]
        for m in ms:
            x = res["methods"][m]["d_F1_class"][d]
            star = "↑" if x["ci"][0] > 0 else "↓" if x["ci"][1] < 0 else ""
            row.append(f"{fs(x['d'])}{star}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("↑／↓：該類 F1 差距的配對 bootstrap 95% CI 整段在 0 之上／之下。")
    return L


def table_decomp(res):
    ms = [m for m in res["methods"] if "decomp" in res["methods"][m]]
    keys = (["一致_都對", "一致_都錯", "改對_補上陽性", "改對_去掉假陽"] + ["改錯_漏陽:" + k for k in BREAK_FN]
            + ["改錯_多陽:" + k for k in BREAK_FP])
    lab = {"一致_都對": "一致（都對）", "一致_都錯": "一致（都錯）", "改對_補上陽性": "改對：補上分類器漏的陽性",
           "改對_去掉假陽": "改對：去掉分類器假陽性"}
    L = ["| 格子類型 | " + " | ".join(nm(m) for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for k in keys:
        name = lab.get(k) or (("改錯（分類器陽性對、生成非陽）：" if "漏陽" in k else "改錯（分類器陰性對、生成陽）：")
                              + k.split(":")[1])
        L.append(f"| {name} | " + " | ".join(str(res["methods"][m]["decomp"].get(k, 0)) for m in ms) + " |")
    tot = lambda m, p: sum(v for k, v in res["methods"][m]["decomp"].items() if k.startswith(p))  # noqa: E731
    L.append("| **改對合計** | " + " | ".join(str(tot(m, "改對")) for m in ms) + " |")
    L.append("| **改錯合計** | " + " | ".join(str(tot(m, "改錯")) for m in ms) + " |")
    L.append("| **淨（改對−改錯）** | " + " | ".join(str(tot(m, "改對") - tot(m, "改錯")) for m in ms) + " |")
    return L


def table_decomp_class(res):
    ms = [m for m in res["methods"] if "decomp" in res["methods"][m]]
    L = ["| 疾病 | " + " | ".join(nm(m) for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for d in DISEASES:
        cells = []
        for m in ms:
            x = res["methods"][m]["decomp_class"][d]
            fix = x.get("改對_補上陽性", 0) + x.get("改對_去掉假陽", 0)
            brk = x.get("改錯_漏陽", 0) + x.get("改錯_多陽", 0)
            cells.append(f"{fix}／{brk}")
        L.append(f"| {ZH[d]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("格式：改對格數／改錯格數（相對分類器）。")
    return L


SHORT = {"漏寫": "漏", "有句但判未提及": "未提", "寫成不確定": "不確", "措辭被判反": "措反", "真的寫反": "寫反",
         "他句被判陽": "他句", "陰性措辭被判陽": "陰措", "真的寫成陽性": "寫陽"}


def table_break_class(res):
    ms = [m for m in res["methods"] if "decomp" in res["methods"][m] and m != "m1B"]
    L = ["| 疾病 | " + " | ".join(nm(m) for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for d in DISEASES:
        cells = []
        for m in ms:
            x = res["methods"][m]["decomp_class"][d]
            parts = [f"{SHORT[k.split(':')[1]]}{v}" for k, v in x.items() if ":" in k and v]
            cells.append(" ".join(parts) or "0")
        L.append(f"| {ZH[d]} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("縮寫：漏＝漏寫、未提＝有句但判未提及、不確＝寫成不確定、措反＝措辭被判反、寫反＝真的寫反（以上為分類器陽性對、生成非陽）；"
             "他句／陰措／寫陽＝分類器陰性對、生成被判陽的三種。")
    return L


def logit(v):
    v = np.clip(np.asarray(v, float), 0.005, 0.995)
    return np.log(v / (1 - v))


def half_point(x, z):
    """logistic P(z=1 | logit(x)) -> x where the fit crosses 0.5 (nan when degenerate / decreasing)."""
    from sklearn.linear_model import LogisticRegression
    x, z = np.asarray(x, float), np.asarray(z, int)
    if len(set(z.tolist())) < 2:
        return float("nan")
    m = LogisticRegression(C=1e4, max_iter=10000).fit(logit(x)[:, None], z)
    a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
    if a <= 0:
        return float("nan")
    return float(1 / (1 + np.exp(b / a)))


def implicit_thresholds(ids, ref, scores, cues, m4cats, lab_train, thr, preds, cue_gt_path):
    """Hypothesis: greedy decoding = implicit cut at P(doctor writes positive) 0.5, above t_f1.
    implicit  : logistic P(generated positive | cue value printed in the prompt) on val600 cells with a cue, 0.5 point.
    calib05   : logistic P(train label positive | train OOF score) on TRAIN only (owl binary, unclear excluded), 0.5 point.
    mid_rate  : share written positive among val600 cells with t_f1 <= cue < 0.5.
    boundary  : share of cells (score not missing) with |score - t_f1| <= 0.1 among the gt / oof break and fix cells vs
                among all scored, reference-labelled cells.
    pools     : gt_dropout --cue-gt-map sample draws (cue_gt.json pools, train OOF): share of the pos pool in
                [t, t+0.1) and of the neg pool in [t-0.1, t)."""
    out = {"implicit": {}, "calib05": {}, "mid_rate": {}, "boundary": {}, "pools": {}}
    for d in DISEASES:
        xs, zs = [], []
        for c, L in lab_train.items():
            s = scores.get(c, {}).get(d)
            y = 1 if L[d] in POS else 0 if (L[d] in NEG_EXPLICIT or L[d] == "not_mentioned") else None
            if s is None or y is None:
                continue
            xs.append(s)
            zs.append(y)
        out["calib05"][d] = {"t": half_point(xs, zs), "n": len(xs), "n_pos": int(sum(zs))}
    for g in GROUPS:
        out["implicit"][g], out["mid_rate"][g] = {}, {}
        for d in DISEASES:
            xs, zs = [], []
            for c in ids:
                v = cues[g][c]["cue"].get(d)
                if v is None:
                    continue
                xs.append(v)
                zs.append(int(m4cats[g][c][d][1] in POS))
            xs, zs = np.array(xs), np.array(zs)
            out["implicit"][g][d] = {"t": half_point(xs, zs), "n_gen_pos": int(zs.sum())}
            mk = (xs >= thr[d]) & (xs < 0.5)
            out["mid_rate"][g][d] = {"n": int(mk.sum()), "pos": int(zs[mk].sum())}
    y = np.array([[ref_owl(ref[c][d]) for d in DISEASES] for c in ids])
    S = np.array([[np.nan if scores[c][d] is None else scores[c][d] for d in DISEASES] for c in ids])
    T = np.array([thr[d] for d in DISEASES])[None, :]
    ok = ~np.isnan(y)
    near = np.abs(S - T) <= 0.1
    pc = preds["classifier"]
    base = ok & ~np.isnan(S)
    out["boundary"]["all_cells"] = {"n": int(base.sum()), "near": int((near & base).sum())}
    out["boundary"]["clf_pos_cells"] = {"n": int((base & pc).sum()), "near": int((near & base & pc).sum())}
    for g in GROUPS:
        pg = preds[g]
        for kind, mk in (("break", ok & (pc != pg) & (pg != (y == 1))), ("fix", ok & (pc != pg) & (pg == (y == 1)))):
            sc = mk & ~np.isnan(S)
            out["boundary"][f"{g}_{kind}"] = {"n": int(mk.sum()), "n_scored": int(sc.sum()), "near": int((near & sc).sum()),
                                             "by_class": {d: [int(sc[:, k].sum()), int((near & sc)[:, k].sum())]
                                                          for k, d in enumerate(DISEASES)}}
    # classifier-positive & right cells the generation did not write positive: where do their scores sit?
    for g in GROUPS:
        mk = ok & pc & ~preds[g] & (y == 1) & ~np.isnan(S)
        v = S[mk]
        t = np.broadcast_to(T, S.shape)[mk]
        out["boundary"][f"{g}_fnbands"] = {"n": int(mk.sum()), "t_to_t+0.1": int(((v >= t) & (v < t + 0.1)).sum()),
                                          "t+0.1_to_0.5": int(((v >= t + 0.1) & (v < 0.5)).sum()),
                                          "ge_0.5": int((v >= np.maximum(0.5, t + 0.1)).sum())}
    if Path(cue_gt_path).exists():
        pools = json.load(open(cue_gt_path))["pools"]          # train OOF values only (labels / ids not used)
        for d in DISEASES:
            p, n = np.array(pools[d]["pos"]), np.array(pools[d]["neg"])
            out["pools"][d] = {"n_pos": len(p), "pos_near": int(((p >= thr[d]) & (p < thr[d] + 0.1)).sum()),
                               "pos_median": float(np.median(p)), "n_neg": len(n),
                               "neg_near": int(((n >= thr[d] - 0.1) & (n < thr[d])).sum()),
                               "neg_median": float(np.median(n))}
    return out


def render_implicit(I, thr):
    L = ["## 追加驗證：greedy 解碼的隱含門檻 vs t_f1", "",
         "- 隱含門檻：val600 有提示值的格子，logistic 擬合 P(生成被 Qwen 判陽 | 提示值)（自變數取 logit），取 0.5 交叉點；"
         "「—」表示該組該類生成陽性太少或擬合退化。",
         "- 校準 0.5 點：只用 train OOF 分數與 train 標籤（owl 定義，unclear 不計）擬合 P(標籤陽 | 分數) 的 0.5 交叉點。",
         "- 中間帶寫陽：提示值落在 [t_f1, 0.5) 的格子中被寫成陽性的比例（陽/格數）。", "",
         "| 疾病 | t_f1 | 校準 0.5 點（train） | 隱含 U-oof | 隱含 T-oof | 隱含 U-gt | 隱含 T-gt | 中間帶寫陽 U-oof | T-oof | U-gt | T-gt |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in DISEASES:
        mr = lambda g: (f"{I['mid_rate'][g][d]['pos']}/{I['mid_rate'][g][d]['n']}"  # noqa: E731
                        if I["mid_rate"][g][d]["n"] else "—")
        L.append(f"| {ZH[d]} | {thr[d]:.3f} | {f3(I['calib05'][d]['t'])} | "
                 + " | ".join(f3(I["implicit"][g][d]["t"]) for g in ("U_oof", "T_oof", "U_gt", "T_gt")) + " | "
                 + " | ".join(mr(g) for g in ("U_oof", "T_oof", "U_gt", "T_gt")) + " |")
    med = lambda v: float(np.nanmedian(v))  # noqa: E731
    L.append(f"| 中位數 | {med([thr[d] for d in DISEASES]):.3f} | {f3(med([I['calib05'][d]['t'] for d in DISEASES]))} | "
             + " | ".join(f3(med([I['implicit'][g][d]['t'] for d in DISEASES])) for g in ("U_oof", "T_oof", "U_gt", "T_gt"))
             + " | " + " | ".join(
                 f"{sum(I['mid_rate'][g][d]['pos'] for d in DISEASES)}/{sum(I['mid_rate'][g][d]['n'] for d in DISEASES)}"
                 for g in ("U_oof", "T_oof", "U_gt", "T_gt")) + " |")
    B_ = I["boundary"]
    pct = lambda a, b: f"{a}/{b}（{100 * a / b:.0f}%）" if b else "—"  # noqa: E731
    L += ["", "改錯／改對格（相對分類器）中，分數落在 t_f1 ±0.1 的比例（分數缺漏的格子不計）：", "",
          "| 對照 | 格數 | 在 t_f1 ±0.1 內 |", "|---|---|---|",
          f"| 所有有分數、參考非 unclear 的格子 | {B_['all_cells']['n']} | {pct(B_['all_cells']['near'], B_['all_cells']['n'])} |",
          f"| 分類器判陽的格子 | {B_['clf_pos_cells']['n']} | {pct(B_['clf_pos_cells']['near'], B_['clf_pos_cells']['n'])} |"]
    for g in ("U_gt", "T_gt", "U_oof", "T_oof"):
        for kind, zh in (("break", "改錯"), ("fix", "改對")):
            x = B_[f"{g}_{kind}"]
            L.append(f"| {nm(g)} {zh} | {x['n_scored']} | {pct(x['near'], x['n_scored'])} |")
    L += ["", "分類器判陽且正確、生成沒寫成陽性的格子（改錯中「漏陽」那一類），依分數所在區間：", "",
          "| 組別 | 格數 | [t_f1, t_f1+0.1) | [t_f1+0.1, 0.5) | ≥ 0.5 |", "|---|---|---|---|---|"]
    for g in ("U_oof", "T_oof", "U_gt", "T_gt"):
        x = B_[f"{g}_fnbands"]
        L.append(f"| {nm(g)} | {x['n']} | {pct(x['t_to_t+0.1'], x['n'])} | {pct(x['t+0.1_to_0.5'], x['n'])} | "
                 f"{pct(x['ge_0.5'], x['n'])} |")
    L += ["", "gt 組改錯格在 ±0.1 內的類別分布（在帶內格數/改錯格數）："]
    for g in ("U_gt", "T_gt"):
        bc = B_[f"{g}_break"]["by_class"]
        L.append(f"- {nm(g)}：" + "、".join(f"{ZH[d]} {bc[d][1]}/{bc[d][0]}" for d in DISEASES if bc[d][0]))
    if I["pools"]:
        L += ["", "gt_dropout `--cue-gt-map sample` 的抽樣池（cue_gt.json，train OOF 分數，兩位小數）：", "",
              "| 疾病 | t_f1 | 陽性池 n | 陽性池中位數 | 陽性池在 [t, t+0.1) | 陰性池 n | 陰性池中位數 | 陰性池在 [t−0.1, t) |",
              "|---|---|---|---|---|---|---|---|"]
        for d in DISEASES:
            p = I["pools"][d]
            L.append(f"| {ZH[d]} | {thr[d]:.3f} | {p['n_pos']} | {p['pos_median']:.2f} | {pct(p['pos_near'], p['n_pos'])} | "
                     f"{p['n_neg']} | {p['neg_median']:.2f} | {pct(p['neg_near'], p['n_neg'])} |")
    return L


def render_md(R):
    v = R["val600"]
    res = v["res"]
    L = ["# 方法四生成 vs 分類器本身（t_f1）：假設驗證", "",
         "由 `code/.../report_gen/method4/verify_vs_classifier.py` 產生；只有數字，無報告原文、無病例編號。",
         "標籤定義同 evaluate.py（owl：原報告 yes／probably_yes＝陽；no／probably_no／not_mentioned＝陰；unclear 不計）；"
         "生成端 strict（Qwen 反抽 yes／probably_yes＝陽）。分類器端不經 Qwen：OWL 分數 ≥ 方法一 t_f1 判陽，分數缺漏視為未判陽。",
         f"配對 bootstrap：以病例為單位重抽 {B} 次（seed {SEED}），各方法與分類器用同一組重抽。", "",
         f"## val600（共同病例 {v['n_common']} 例）", "",
         f"- 參考標籤一致性：四組、方法一 1B 的 per_case 參考類別與標籤檔逐格比對，不一致 {v['ref_mismatch_cells']} 格。",
         f"- 提示值與分數檔：四組提示完全相同＝{v['cue_identical_across_groups']}；提示值（兩位小數）與分數檔不符 "
         f"{v['cue_vs_score_mismatch']} 格；四捨五入造成判陰陽改變 {v['cue_rounding_decision_changes']} 格。",
         f"- 方法一 1B 決策 vs 分數 ≥ t_f1（分類器→樣板）：{v['m1B_decision_vs_tf1']}。",
         f"- 補滿 14 類（fill14 merge 計分）：本腳本對 {', '.join(v['fill14_groups']) or '（無）'} 用 merge_ext 重算；"
         f"與正式 fill_merge 評估逐格比對不一致數：{v['fill14_vs_official_merge_mismatch_cells'] or '（尚無正式評估）'}。", "",
         "### 1. 主表（macro／micro 敏感度、特異度、F1）", ""]
    L += table_main(res)
    L += ["", "### 2. 各類 F1 差距（相對分類器 ≥ t_f1）", ""]
    L += table_class(res)
    L += ["", "### 3. 差距拆解（逐格：病例×疾病，參考非 unclear）", "",
          "改錯細分規則同 improve/cue_pos_breakdown.py（對生成句子的否定詞規則，屬啟發式）："
          "漏寫＝沒有該疾病標籤的句子；有句但判未提及＝有句但 Qwen 判 not_mentioned；寫成不確定＝Qwen 判 unclear；"
          "措辭被判反＝Qwen 判 no／probably_no 但句子無否定詞；真的寫反＝句子含否定詞。"
          "反方向（分類器陰性對、生成被判陽）：他句被判陽＝該疾病沒有自己的句子；陰性措辭被判陽＝句子含否定詞；真的寫成陽性＝句子無否定詞。"
          "方法一 1B 的「句子」就是樣板決策（pos／neg／omit）。", ""]
    L += table_decomp(res)
    L += ["", "各類改對／改錯：", ""]
    L += table_decomp_class(res)
    L += ["", "改錯細分 × 疾病：", ""]
    L += table_break_class(res)
    mc = v["missing_cue"]
    L += ["", "### 5. 提示中缺分數的格子", "",
          f"- val600 有缺的病例 {mc['n_cases_with_missing']} 例；缺的格子 {mc['n_cells']} 格；每例缺幾類：{mc['n_missing_per_case']}。",
          "- 各類缺格數：" + "、".join(f"{ZH[d]} {mc['by_class'].get(d, 0)}" for d in DISEASES),
          f"- 缺格的參考類別：{mc['ref_cat']}",
          "- 缺格的生成類別（Qwen 反抽）：" + "；".join(f"{nm(g)} {mc['gen_cat'][g]}" for g in GROUPS),
          "- 有分數格的生成類別（對照）：" + "；".join(f"{nm(g)} {mc['gen_cat_scored_cells'][g]}" for g in GROUPS),
          "- 缺格中生成有該疾病句子的格數：" + "；".join(f"{nm(g)} {mc['gen_has_sentence'].get(g, 0)}" for g in GROUPS), "",
          "完整 val／train：分數缺漏 vs 參考 unclear 交叉表（每類：missing|unclear、missing|labelled、scored|unclear、scored|labelled）：", "",
          "| 疾病 | val 缺分且 unclear | val 缺分但有標籤 | val 有分但 unclear | train 缺分且 unclear | train 缺分但有標籤 | train 有分但 unclear |",
          "|---|---|---|---|---|---|---|"]
    for d in DISEASES:
        a, b = R["missing_vs_unclear"]["val_full"][d], R["missing_vs_unclear"]["train"][d]
        L.append(f"| {ZH[d]} | {a.get('missing|unclear', 0)} | {a.get('missing|labelled', 0)} | {a.get('scored|unclear', 0)} | "
                 f"{b.get('missing|unclear', 0)} | {b.get('missing|labelled', 0)} | {b.get('scored|unclear', 0)} |")
    L += ["", "## 完整 val（5,055 例）", ""]
    if not R["valfull"]:
        L.append("尚無任何一組完成完整 val 評估（eval/results_cue/*_valfull 無 per_case_categories），待補。")
    for g, r in R["valfull"].items():
        L += [f"### {nm(g)}（{r['n_cases']} 例）", ""] + table_main(r) + [""] + table_class(r) + [""] + table_decomp(r) + [""]
    if "implicit" in R:
        L += [""] + render_implicit(R["implicit"], R["thresholds_t_f1"])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
