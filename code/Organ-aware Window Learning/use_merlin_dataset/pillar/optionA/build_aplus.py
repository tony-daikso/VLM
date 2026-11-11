"""Option A+ (A6000, CPU): OWL+Pillar score table for the method-4 prompt, in the format of the peer's unit probe
(report_gen/m4/improve/probe/{probe_scores.csv, unit_cue.json}); nothing in the peer's dirs is modified.

  kid_right / kid_left : OWL+Pillar stack (option-A 'st_owl_p': unit LR on logit(peer OWL unit probe) + logit(Pillar
                         vec64+vec16 unit probe) + unit one-hot), P(hydronephrosis in that kidney)
  T8 .. L5             : OWL+Pillar+M4 stack (option-A 'st_owl_p_m4': + M4 per-vertebra geometry), P(fracture at level)
  hs_grade             : P(steatosis moderate/severe | steatosis reported), stack of logit(OWL steatosis prob) + Pillar
                         liver/spleen probe + liver-spleen HU (option-A 2c 'st'), sigmoid of the LR decision function
  asc_amount           : P(ascites moderate/large | ascites reported), OWL + Pillar organ-unit probe stack
  ath_grade            : P(atherosclerosis moderate/severe | atherosclerosis reported), OWL + Pillar aorta/iliac stack
Train = OOF: unit scores with unit_probe.py folds (fh / ff); grade scores with the class's official OWL stack fold
(cases.csv fold_<disease>) - labelled train rows are scored by the model of their own fold (exactly option A), and
unlabelled train rows (disease negative / grade not written) are scored by the same own-fold model (never trained on
them). Val = full-train refit, every val case. Test: no features exist, nothing is read or written.
Writes <out>/probe_scores_pillar.csv, <out>/unit_cue_pillar.json, <out>/check.json (val metrics vs option A).
usage: optA-env python build_aplus.py --out ~/Desktop/Tony/VLM/OWL/report_gen/m4/improve/probe_pillar
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optA as oa  # noqa: E402

W = oa.W
GRADES = {"hs_grade": "hepatic_steatosis", "asc_amount": "ascites", "ath_grade": "atherosclerosis"}


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def grade_scores():
    """re-run option A 2c 'st' for the three grade attributes, scoring every train (own-fold OOF) and val case."""
    pf = oa.PF()
    C = pd.read_csv(W + "in/cases.csv")
    A = pd.read_csv(W + "in/attrs.csv", low_memory=False).set_index("study_id").reindex(C.study_id).reset_index()
    r, have = pf.rows(C.study_id.values)
    assert have.all()
    out, check = {}, {}
    ref = json.load(open(W + "out/cheap_metrics.json"))
    for t, d in GRADES.items():
        y = A[t].values.astype(float)
        fold = C[f"fold_{d}"].values
        tr_all = (C.split == "train").values
        va_all = (C.split == "val").values
        ow = oa.logit(C[f"owl_{d}"].values)
        pu, _ = oa.nested(pf.unit_X(r, d), np.where(tr_all, y, np.nan), fold, tr_all, va_all, oa.CS, "lr")
        Z = np.c_[ow, pu]
        if t == "hs_grade":
            Z = np.c_[Z, -(A["liver_hu_med"].values - A["spleen_hu_med"].values)]
        st = oa.stack_lr(Z, np.where(tr_all, y, np.nan), fold, tr_all, va_all)
        out[t] = sig(st)
        lv = va_all & ~np.isnan(y)
        lt = tr_all & ~np.isnan(y)
        check[t] = {"val_auroc": oa.auc(y[lv], st[lv]), "option_a_val_auroc": ref[t]["val"]["st"]["auroc"],
                    "train_oof_auroc": oa.auc(y[lt], st[lt]), "n_val_labelled": int(lv.sum()),
                    "n_train_labelled": int(lt.sum())}
        print(t, check[t], flush=True)
    return C.study_id.values, C.split.values, out, check, A


def pools_thr(y, s):
    t = oa.f1_thr(y, s)
    ok = ~np.isnan(y) & ~np.isnan(s)
    pos = sorted(round(float(v), 2) for v in s[ok & (y == 1) & (s >= t)])
    neg = sorted(round(float(v), 2) for v in s[ok & (y == 0) & (s < t)])
    return t, {"pos": pos, "neg": neg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    K = np.load(W + "out/kidney_unit_scores.npz", allow_pickle=True)
    V = np.load(W + "out/vert_unit_scores.npz", allow_pickle=True)
    ids, split = K["ids"], K["split"].astype(str)
    assert (V["ids"] == ids).all() and set(split) <= {"train", "val"}
    sk, sv = K["s_st_owl_p"], V["s_st_owl_p_m4"]
    UK, UV = K["Y"], V["Y"]
    tr, va = split == "train", split == "val"
    gid, gsplit, G, gcheck, A = grade_scores()
    gi = {c: i for i, c in enumerate(gid)}
    gix = np.array([gi[c] for c in ids])
    assert (gsplit[gix] == split).all()
    Gs = {t: v[gix] for t, v in G.items()}
    Gy = {t: A[t].values.astype(float)[gix] for t in GRADES}

    # ---- check: val metrics reproduce option A
    km = json.load(open(W + "out/kidney_unit_metrics.json"))
    vm = json.load(open(W + "out/vert_unit_metrics.json"))
    check = {"kidney_val_unit_auroc": oa.auc(UK[va][K["present"][va]], sk[va][K["present"][va]]),
             "kidney_option_a": km["val"]["auc_st_owl_p"]["value"],
             "vertebra_val_unit_auroc": oa.auc(UV[va][V["present"][va]], sv[va][V["present"][va]]),
             "vertebra_option_a": vm["val"]["auc_st_owl_p_m4"]["value"],
             "kidney_side_exact_option_a": km["val"]["side_exact_st_owl_p"]["value"],
             "vertebra_top1_option_a": vm["val"]["top1_st_owl_p_m4"]["value"], "grades": gcheck}
    tk, pk = pools_thr(UK[tr].ravel(), sk[tr].ravel())
    tv, pv = pools_thr(UV[tr].ravel(), sv[tr].ravel())
    assert tk == km["t_f1_train"]["st_owl_p"] and tv == vm["t_f1_train"]["st_owl_p_m4"]
    # side exact / top-1 recomputed with the peer's definitions (val)
    yc = K["ycase"]
    vi = np.where(va & K["present"].any(1))[0]
    side = [n for n in vi if yc[n] == 1 and not np.isnan(UK[n]).any() and K["present"][n].all()]
    ok = []
    for n in side:
        p = sk[n] >= tk
        if not p.any():
            p = np.array([sk[n, 0] >= sk[n, 1], sk[n, 1] > sk[n, 0]])
        ok.append(bool((p == (UK[n] == 1)).all()))
    check["kidney_side_exact"] = float(np.mean(ok))
    ycv = V["ycase"]
    lev = [n for n in np.where(va & V["present"].any(1))[0] if ycv[n] == 1 and (np.nan_to_num(UV[n]) == 1).any()
           and not np.isnan(sv[n]).all()]
    check["vertebra_top1"] = float(np.mean([UV[n, np.nanargmax(np.where(np.isnan(sv[n]), -np.inf, sv[n]))] == 1
                                            for n in lev]))
    for k1, k2 in (("kidney_val_unit_auroc", "kidney_option_a"), ("vertebra_val_unit_auroc", "vertebra_option_a"),
                   ("kidney_side_exact", "kidney_side_exact_option_a"), ("vertebra_top1", "vertebra_top1_option_a")):
        assert abs(check[k1] - check[k2]) < 1e-9, (k1, check[k1], check[k2])
    for t, c in gcheck.items():
        assert abs(c["val_auroc"] - c["option_a_val_auroc"]) < 1e-9, (t, c)
    check["all_match_option_a"] = True

    # ---- probe_scores_pillar.csv (same columns as the peer's probe_scores.csv + 3 grade columns)
    cols = ["case_id", "split", "kid_right", "kid_left"] + oa.LEVELS + list(GRADES)
    fmt = lambda v: "" if np.isnan(v) else f"{v:.5f}"  # noqa: E731
    with open(os.path.join(a.out, "probe_scores_pillar.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n in range(len(ids)):
            w.writerow([ids[n], split[n]] + [fmt(v) for v in list(sk[n]) + list(sv[n])] + [fmt(Gs[t][n]) for t in GRADES])

    # ---- unit_cue_pillar.json (peer structure; grade keys added in the same style)
    thr = {"kidney": tk, "vertebra": tv}
    pools = {"kidney": pk, "vertebra": pv}
    for t in GRADES:
        thr[t], pools[t] = pools_thr(Gy[t][tr], Gs[t][tr])
    iv = lambda v: None if np.isnan(v) else int(v)  # noqa: E731
    lab = {}
    for n in np.where(tr)[0]:
        e = {"kid": [iv(v) for v in UK[n]], "vert": [iv(v) for v in UV[n]]}
        for t in GRADES:
            e[t] = iv(Gy[t][n])
        lab[str(ids[n])] = e
    json.dump({"labels_train": lab, "thresholds": thr, "pools": pools, "levels": oa.LEVELS, "sides": ["right", "left"],
               "grades": {t: {"disease": d, "score": "P(moderate or worse | disease reported)",
                              "label": "1 = moderate/severe (ascites: moderate/large), 0 = mild (ascites: trace/small), "
                                       "null = disease not reported positive or grade not written"}
                          for t, d in GRADES.items()}},
              open(os.path.join(a.out, "unit_cue_pillar.json"), "w"))

    # ---- coverage
    cov = {}
    for sp, m in (("train", tr), ("val", va)):
        cov[sp] = {"cases": int(m.sum()),
                   "kid_right": int((~np.isnan(sk[m, 0])).sum()), "kid_left": int((~np.isnan(sk[m, 1])).sum()),
                   "any_vertebra": int((~np.isnan(sv[m])).any(1).sum()),
                   "vertebra_units": int((~np.isnan(sv[m])).sum()),
                   **{t: int((~np.isnan(Gs[t][m])).sum()) for t in GRADES},
                   **{f"{t}_labelled": int((~np.isnan(Gy[t][m])).sum()) for t in GRADES}}
    check["coverage"] = cov
    check["thresholds"] = thr
    check["pool_sizes"] = {k: {s: len(v[s]) for s in ("pos", "neg")} for k, v in pools.items()}
    json.dump(check, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
