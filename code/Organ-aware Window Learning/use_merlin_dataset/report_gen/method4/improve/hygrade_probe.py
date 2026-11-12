#!/usr/bin/env python3
"""Agent I (11-12): hydronephrosis GRADE probe (mild vs moderate / severe) for the method-4 grade line. CPU only.

labels  : report-extracted structured attribute of the evaluation targets (m4/inputs/targets_v2, diseases.hydronephrosis:
          status == pos, attributes.grade in mild / moderate / severe, attributes.side). ONE grade per case (the report
          rule gives no per-side grade), so the label is per case; rows without a written grade are never trained on.
          No report text is read or printed. targets_test is never opened; every id comes from the train / val meta.
units   : a model scores ONE kidney. Training rows = the affected kidney(s) of the graded positive cases (side right /
          left -> that kidney, bilateral -> both, side not stated -> the "predicted" kidneys below), label = case grade.
          Case score (what goes into the prompt, no report information) = max over the predicted kidneys = kidneys whose
          presence score (probe_pillar kid_right / kid_left, train OOF / val refit) >= its train threshold, else the
          higher one. The oracle-side case score (max over the report side) is reported for reference only.
feats   : per kidney  owl_pool  OWL kidney unit tokens, masked means: organ3 x5 | ru mean-pool x20 | ru max-pool x20 (1344)
                      owl_flat  the 45 x 448 tokens flattened (20160)
                      pil       Pillar-0 vec64 | vec16 composite of that kidney (768)        (pillar_optA/packed)
                      m4        9 kidney measurements (signed log1p)                         (pillar_optA/in/attrs.csv)
          -> standardised LR, C by nested CV (the peer's pillar/optionA optA.nested, imported read-only);
          stacks (LR, C=1, on logits): st_owl = OWL case prob + OWL unit presence + owl probe;
          st_owl_p = OWL case prob + stacked presence + owl probe + pil probe; st_owl_p_m4 = + m4.
folds   : unit_probe.folds() hydronephrosis folds (region_units/hydronephrosis oof_train; crc32 % 5 fallback).
          TRAIN = 5-fold OOF (every train kidney is scored by the model of its own fold, labelled or not);
          VAL = full-train refit. The model (owl feature set, stack) is chosen by train OOF AUROC only.
report  : val AUROC / accuracy with bootstrap 95% CIs (2,000) vs (a) the majority class and (b) the hydronephrosis
          presence probability alone (OWL case prob = the prompt's disease cue; also the stricter stacked per-kidney
          presence that the A+ unit line already shows); 3-class variant (second model: severe vs mild / moderate)
          reported only. decide() records the task rule and the stricter rule.
writes  : <out>/probe_scores_hygrade.csv (case_id, split, hy_grade, hy_grade_right, hy_grade_left, hy_severe),
          unit_cue_hygrade.json (= probe_pillar/unit_cue_pillar.json + key hy_grade: labels_train, threshold from train
          OOF F1, pos / neg pools), check.json (aggregates only), fit.log. Hold case ids: stay on A6000, never in git.
usage (A6000, radar env): python hygrade_probe.py --out improve/probe_hygrade [--threads 4]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
H = Path(os.path.expanduser("~/Desktop/Tony/VLM/OWL"))
M4 = H / "report_gen/m4"
OPTA = H / "pillar_optA"
KEY = "hy_grade"
GRADE3 = {"mild": 0, "moderate": 1, "severe": 2}
B = 2000


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def read_targets(ids, split):
    """-> pos [N] bool, g3 [N] (nan / 0 / 1 / 2), side [N] str ('' when not positive). Structured fields only."""
    idx = {c: i for i, c in enumerate(ids)}
    pos = np.zeros(len(ids), bool)
    g3 = np.full(len(ids), np.nan)
    side = np.array([""] * len(ids), dtype=object)
    for sp in ("train", "val"):                                 # the test targets are never opened
        for line in open(M4 / f"inputs/targets_v2/targets_{sp}.jsonl"):
            r = json.loads(line)
            assert r["split"] == sp
            i = idx.get(r["case_id"])
            if i is None:
                continue
            assert split[i] == sp
            x = r["diseases"]["hydronephrosis"]
            if x["status"] != "pos":
                continue
            pos[i] = True
            side[i] = x["attributes"].get("side") or "not_stated"
            g3[i] = GRADE3.get(x["attributes"].get("grade"), np.nan)
    return pos, g3, side


def read_csv_cols(path, ids, cols):
    by = {r["case_id"]: r for r in csv.DictReader(open(path))}
    out = np.full((len(ids), len(cols)), np.nan)
    for n, c in enumerate(ids):
        r = by.get(c)
        if r is not None:
            out[n] = [float(r[k]) if r[k] not in ("", None) else np.nan for k in cols]
    return out


def stack(Z, y, fold, tr, va, C=1.0):
    """LR on low-dim Z (nan -> train median): train rows OOF by fold (all rows of the fold scored), val rows from
    the all-train fit. Returns probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Z = np.where(np.isnan(Z), np.nanmedian(Z[tr], 0), Z)
    lab = ~np.isnan(y)
    s = np.full(len(y), np.nan)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))  # noqa: E731
    for k in range(5):
        a = tr & lab & (fold != k)
        b = tr & (fold == k)
        s[b] = mk().fit(Z[a], y[a]).predict_proba(Z[b])[:, 1]
    s[va] = mk().fit(Z[tr & lab], y[tr & lab]).predict_proba(Z[va])[:, 1]
    return s


def best_thr(y, s, crit):
    best = (-1, 0.5)
    for t in np.unique(np.round(s, 2)):
        p = s >= t
        v = (2 * (p & (y == 1)).sum() / max(p.sum() + (y == 1).sum(), 1)) if crit == "f1" else float((p == (y == 1)).mean())
        if v > best[0]:
            best = (v, float(t))
    return best[1], float(best[0])


def boot(y, S, base_auc=(), acc_thr=None, seed=7, b=B):
    """case bootstrap (stratified for AUROC, plain for accuracy). S {name: score}; base_auc: names every other name is
    compared with (paired delta AUROC); acc_thr {name: threshold} -> accuracy and accuracy - majority accuracy."""
    from scipy.stats import rankdata

    def auc(yy, ss):
        n1 = yy.sum()
        if n1 == 0 or n1 == len(yy):
            return np.nan
        r = rankdata(ss)
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1)))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = len(y)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    out = {}
    A = {k: [] for k in S}
    for _ in range(b):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k in S:
            A[k].append(auc(y[i], S[k][i]))
    for k in S:
        out[k] = {"auroc": auc(y, S[k]), "auroc_ci": ci(A[k])}
        for bn in base_auc:
            if bn != k:
                d = np.array(A[k]) - np.array(A[bn])
                out[k][f"d_auroc_vs_{bn}"] = {"delta": out[k]["auroc"] - auc(y, S[bn]), "ci": ci(d)}
    maj = max(y.mean(), 1 - y.mean())
    out["_majority"] = {"acc": float(maj), "auroc": 0.5, "class": int(y.mean() > 0.5)}
    if acc_thr:
        I = [rng.choice(n, n) for _ in range(b)]
        mj = np.array([max(y[i].mean(), 1 - y[i].mean()) for i in I])
        out["_majority"]["acc_ci"] = ci(mj)
        for k, t in acc_thr.items():
            hit = ((S[k] >= t) == (y == 1)).astype(float)
            a = np.array([hit[i].mean() for i in I])
            out[k].update(acc=float(hit.mean()), acc_thr=t, acc_ci=ci(a),
                          d_acc_vs_majority={"delta": float(hit.mean() - maj), "ci": ci(a - mj)})
            p = S[k] >= t
            out[k].update(sens=float(p[y == 1].mean()), spec=float((~p)[y == 0].mean()))
    return out


def decide(res, name, bases):
    """per presence baseline: the probe's val AUROC CI lower bound is above 0.5 and above the baseline's val AUROC, the
    paired delta has CI > 0, and the val accuracy CI lower bound is above the majority-class accuracy.
    strict (the rule coded before val was read) = every baseline in `bases`; task = the task's two baselines, i.e. the
    majority class and bases[0] (the OWL classifier's hydronephrosis probability = the prompt's disease cue). The
    task / strict split of the record was added after the first val read; the model, thresholds and numbers are the
    same run (nothing was re-chosen)."""
    r = res[name]
    per = {b: {"auroc_lo_gt_0.5": r["auroc_ci"][0] > 0.5, "auroc_lo_gt_baseline": r["auroc_ci"][0] > res[b]["auroc"],
               "paired_delta_gt_0": r[f"d_auroc_vs_{b}"]["ci"][0] > 0,
               "acc_lo_gt_majority": r["acc_ci"][0] > res["_majority"]["acc"]} for b in bases}
    ok = {b: all(c.values()) for b, c in per.items()}
    return {"task_rule_majority_and_classifier_presence": bool(ok[bases[0]]), "strict_rule_every_presence_baseline":
            bool(all(ok.values())), "per_presence_baseline": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    sys.path.insert(0, str(M4 / "code_f/improve"))
    sys.path.insert(1, str(M4 / "code_f"))
    sys.path.insert(2, str(OPTA / "code"))
    import unit_probe as up                                     # folds / f1 threshold (peer, read-only)
    import optA as oa                                           # nested LR, Pillar shards, M4 unit features (peer, read-only)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)
    except ImportError:
        pass

    meta = np.load(M4 / "improve/probe/meta.npz")
    ids, split = meta["ids"].astype(str), meta["split"].astype(str)
    assert set(split) <= {"train", "val"}
    N = len(ids)
    tr_c, va_c = split == "train", split == "val"
    pos, g3, side = read_targets(ids, split)
    fold_c, _, miss = up.folds(ids, split)
    log(f"[{time.strftime('%m-%d %H:%M')}] hygrade: {N} cases (train {int(tr_c.sum())}, val {int(va_c.sum())}); "
        f"train cases without official fold: {miss[0]}")
    counts = {}
    for sp, m in (("train", tr_c), ("val", va_c)):
        p = m & pos
        c = {"cases": int(m.sum()), "positive": int(p.sum()), "grade_written": int((p & ~np.isnan(g3)).sum()),
             "grade": {g: int((p & (g3 == v)).sum()) for g, v in GRADE3.items()},
             "bilateral": int((p & (side == "bilateral")).sum()),
             "bilateral_grade_written": int((p & (side == "bilateral") & ~np.isnan(g3)).sum()),
             "side_of_graded": {s: int((p & ~np.isnan(g3) & (side == s)).sum())
                                for s in ("right", "left", "bilateral", "not_stated", "transplant")},
             "grade_by_side": {s: {g: int((p & (g3 == v) & (side == s)).sum()) for g, v in GRADE3.items()}
                               for s in ("right", "left", "bilateral", "not_stated", "transplant")},
             "grades_per_case": 1}
        counts[sp] = c
        log(f"  {sp}: {json.dumps(c)}")

    # ---------------------------------------------------------------- presence scores (train OOF / val refit; peers)
    pres = read_csv_cols(M4 / "improve/probe_pillar/probe_scores_pillar.csv", ids, ["kid_right", "kid_left"])
    pres_owl = read_csv_cols(M4 / "improve/probe/probe_scores.csv", ids, ["kid_right", "kid_left"])
    t_pres = json.load(open(M4 / "improve/probe_pillar/unit_cue_pillar.json"))["thresholds"]["kidney"]
    import pandas as pd
    C = pd.read_csv(OPTA / "in/cases.csv", usecols=["study_id", "split", "owl_hydronephrosis"]).set_index("study_id")
    C = C.reindex(ids)
    assert (C["split"].values == split).all()
    owl_case = C["owl_hydronephrosis"].values.astype(float)     # OWL class prob: train OOF, val inference
    pf = oa.PF()
    r, have = pf.rows(ids)
    present = meta["kid_mask"].any(-1) & have[:, None] & ~np.isnan(pres) & ~np.isnan(pres_owl)
    # predicted kidneys (no report information): presence >= threshold, else the higher one
    ps = np.where(present, pres, -1)
    sel_pred = (ps >= t_pres) & present
    none = ~sel_pred.any(1)
    am = ps.argmax(1)
    sel_pred[none, am[none]] = present[none, am[none]]
    # report side (training rows / oracle reference); not stated -> predicted
    sel_gt = np.zeros((N, 2), bool)
    sel_gt[:, 0] = np.isin(side, ["right", "bilateral"])
    sel_gt[:, 1] = np.isin(side, ["left", "bilateral"])
    ns = pos & ~np.isin(side, ["right", "left", "bilateral"])
    sel_gt[ns] = sel_pred[ns]
    sel_gt &= present

    # ---------------------------------------------------------------- kidney rows
    rn, ru = np.where(present)
    R = len(rn)
    tr, va = tr_c[rn], va_c[rn]
    fold = np.where(tr, fold_c[rn], -1)
    lab_row = sel_gt[rn, ru] & ~np.isnan(g3[rn])
    y2 = np.where(lab_row, (g3[rn] >= 1).astype(float), np.nan)
    y3 = np.where(lab_row, (g3[rn] >= 2).astype(float), np.nan)
    log(f"  kidney rows {R}; labelled rows train {int((tr & lab_row).sum())} ({int(np.nansum(y2[tr]))} moderate+, "
        f"{int(np.nansum(y3[tr]))} severe), val {int((va & lab_row).sum())}")

    def case_score(s_row, sel):
        """[N] max of the row scores over the selected kidneys (nan when none)."""
        S = np.full((N, 2), np.nan)
        S[rn, ru] = s_row
        S = np.where(sel, S, np.nan)
        o = np.full(N, np.nan)
        ok = ~np.isnan(S).all(1)
        o[ok] = np.nanmax(S[ok], 1)
        return o

    # ---------------------------------------------------------------- features
    K = np.load(M4 / "improve/probe/kid.npy", mmap_mode="r")
    KM = meta["kid_mask"]
    Xf = np.empty((R, 45 * 448), np.float32)
    Xp = np.empty((R, 3 * 448), np.float32)
    grp = [np.arange(0, 5), np.arange(5, 45, 2), np.arange(6, 45, 2)]   # organ3 | ru mean-pool | ru max-pool tokens
    for i0 in range(0, R, 4096):
        sl = slice(i0, i0 + 4096)
        x = np.asarray(K[rn[sl], ru[sl]], np.float32)           # [b, 45, 448]
        m = KM[rn[sl], ru[sl]].astype(np.float32)[..., None]
        x = x * m
        Xf[sl] = x.reshape(len(x), -1)
        Xp[sl] = np.concatenate([x[:, g].sum(1) / np.maximum(m[:, g].sum(1), 1) for g in grp], 1)
    del K
    Xpil = np.concatenate([np.stack([pf.comp(r, lb, res)[0] for lb in ([2], [3])], 1)[rn, ru] for res in ("64", "16")], 1)
    Zm, _, m4cols = oa.m4_unit_feats(ids, "kidney")
    Zm = Zm[rn, ru]
    log(f"  features: owl_flat {Xf.shape}, owl_pool {Xp.shape}, pil {Xpil.shape}, m4 {Zm.shape}")

    def auc_rows(y, s, m):
        return oa.auc(y[m & ~np.isnan(y)], s[m & ~np.isnan(y)])

    def probes(y, tag):
        P, info = {}, {}
        for nm, X in (("owl_pool", Xp), ("owl_flat", Xf), ("pil", Xpil)):
            t0 = time.time()
            s, inf = oa.nested(X, y, fold, tr, va, oa.CS, "lr")
            P[nm] = s
            info[nm] = {"C_outer": inf["C_outer"], "C_refit": inf["C_refit"], "train_oof_row_auroc": auc_rows(y, s, tr)}
            log(f"  [{tag}] probe {nm}: train OOF row AUROC {info[nm]['train_oof_row_auroc']:.4f}  C {inf['C_outer']} / "
                f"{inf['C_refit']}  ({time.time() - t0:.0f}s)")
        return P, info

    lo_case = logit(owl_case)[rn]
    lp, lpo = logit(pres[rn, ru]), logit(pres_owl[rn, ru])

    def stacks(P, y, owl_nm):
        return {"st_owl": stack(np.c_[lo_case, lpo, P[owl_nm]], y, fold, tr, va),
                "st_owl_p": stack(np.c_[lo_case, lp, P[owl_nm], P["pil"]], y, fold, tr, va),
                "st_owl_p_m4": stack(np.c_[lo_case, lp, P[owl_nm], P["pil"], Zm], y, fold, tr, va)}

    # ---------------------------------------------------------------- binary: moderate / severe vs mild
    P2, info2 = probes(y2, "mod+")
    owl_nm = max(("owl_pool", "owl_flat"), key=lambda k: info2[k]["train_oof_row_auroc"])
    S2 = stacks(P2, y2, owl_nm)
    y2c = np.where(pos & ~np.isnan(g3), (g3 >= 1).astype(float), np.nan)      # case label
    cand = {}
    for k, s in S2.items():
        d = case_score(s, sel_pred)
        cand[k] = {"train_oof_case_auroc": oa.auc(y2c[tr_c], d[tr_c]),
                   "train_oof_case_auroc_oracle_side": oa.auc(y2c[tr_c], case_score(s, sel_gt)[tr_c]),
                   "train_oof_row_auroc": auc_rows(y2, s, tr)}
        log(f"  stack {k}: {json.dumps(cand[k])}")
    # single-feature references (train OOF, case level, predicted kidneys)
    for k in ("owl_pool", "owl_flat", "pil"):
        cand["probe_" + k] = {"train_oof_case_auroc": oa.auc(y2c[tr_c], case_score(P2[k], sel_pred)[tr_c])}
    best = max(S2, key=lambda k: cand[k]["train_oof_case_auroc"])
    log(f"  chosen by train OOF only: owl features {owl_nm}, stack {best}")
    s_row = S2[best]
    hy = case_score(s_row, sel_pred)                            # the prompt score
    hy_or = case_score(s_row, sel_gt)
    hy_or = np.where(np.isnan(hy_or), hy, hy_or)                # report-side kidney without tokens -> the prompt score
    HYK = np.full((N, 2), np.nan)
    HYK[rn, ru] = s_row

    # thresholds / pools: train OOF only
    lt = tr_c & ~np.isnan(y2c) & ~np.isnan(hy)
    t_f1 = up.f1_threshold(y2c[lt], hy[lt])[0]
    t_acc, acc_tr = best_thr(y2c[lt], hy[lt], "acc")
    pools = {"pos": sorted(round(float(v), 2) for v in hy[lt][(y2c[lt] == 1) & (hy[lt] >= t_f1)]),
             "neg": sorted(round(float(v), 2) for v in hy[lt][(y2c[lt] == 0) & (hy[lt] < t_f1)])}
    log(f"  thresholds (train OOF): F1 {t_f1}, accuracy {t_acc} (train OOF acc {acc_tr:.3f}); pools pos {len(pools['pos'])} "
        f"neg {len(pools['neg'])}")

    # ---------------------------------------------------------------- baselines + val report
    pres_case = case_score(pres[rn, ru], sel_pred)              # stacked per-kidney presence (max over predicted kidneys)
    base = {"presence_owl_case": owl_case, "presence_unit_max": pres_case}
    names = {"probe": hy, "probe_oracle_side": hy_or, **base,
             **{"alt_" + k: case_score(s, sel_pred) for k, s in S2.items() if k != best}}
    res = {}
    for sp, m in (("train_oof", tr_c), ("val", va_c)):
        ev = m & ~np.isnan(y2c) & ~np.isnan(hy)
        thr = {"probe": t_acc, "probe_oracle_side": t_acc}
        for bn, bs in base.items():                             # baselines get their own train-OOF accuracy threshold
            thr[bn] = best_thr(y2c[lt], bs[lt], "acc")[0]
        res[sp] = boot(y2c[ev], {k: v[ev] for k, v in names.items()}, base_auc=list(base), acc_thr=thr)
        res[sp]["_n"] = {"cases": int(ev.sum()), "moderate_plus": int(y2c[ev].sum()),
                         "graded_without_score": int((m & ~np.isnan(y2c) & np.isnan(hy)).sum())}
        hit = (hy[ev] >= t_f1) == (y2c[ev] == 1)
        res[sp]["probe"]["acc_at_f1_thr"] = float(hit.mean())
        log(f"  {sp}: n {res[sp]['_n']}  majority acc {res[sp]['_majority']['acc']:.3f}")
        for k in names:
            q = res[sp][k]
            log(f"    {k}: AUROC {q['auroc']:.3f} [{q['auroc_ci'][0]:.3f}, {q['auroc_ci'][1]:.3f}]"
                + (f"  acc {q['acc']:.3f} [{q['acc_ci'][0]:.3f}, {q['acc_ci'][1]:.3f}] (thr {q['acc_thr']})" if "acc" in q else "")
                + "".join(f"  d vs {b} {q[f'd_auroc_vs_{b}']['delta']:+.3f} [{q[f'd_auroc_vs_{b}']['ci'][0]:+.3f}, "
                          f"{q[f'd_auroc_vs_{b}']['ci'][1]:+.3f}]" for b in base if f"d_auroc_vs_{b}" in q))
    dec = decide(res["val"], "probe", list(base))
    log(f"  decision: {json.dumps(dec)}")

    # ---------------------------------------------------------------- 3-class (reported only)
    P3, info3 = probes(y3, "severe")
    S3 = stacks(P3, y3, owl_nm)[best]
    sev = case_score(S3, sel_pred)
    y3c = np.where(pos & ~np.isnan(g3), (g3 >= 2).astype(float), np.nan)
    lt3 = lt & ~np.isnan(sev)
    bt = (-1, 0.5, 0.5)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for t2 in grid:
        for t3 in grid:
            pr = (hy[lt3] >= t2).astype(int) + ((hy[lt3] >= t2) & (sev[lt3] >= t3)).astype(int)
            v = float((pr == g3[lt3]).mean())
            if v > bt[0]:
                bt = (v, float(t2), float(t3))
    three = {"thr_train_oof": {"moderate_plus": bt[1], "severe": bt[2], "train_oof_acc3": bt[0]}}
    from scipy.stats import spearmanr
    for sp, m in (("train_oof", tr_c), ("val", va_c)):
        ev = m & ~np.isnan(y2c) & ~np.isnan(hy) & ~np.isnan(sev)
        g = g3[ev].astype(int)
        pr = (hy[ev] >= bt[1]).astype(int) + ((hy[ev] >= bt[1]) & (sev[ev] >= bt[2])).astype(int)
        rng = np.random.default_rng(5)
        n = int(ev.sum())
        I = [rng.choice(n, n) for _ in range(B)]
        hit = (pr == g).astype(float)
        majc = np.bincount(g, minlength=3).argmax()
        mj = (g == majc).astype(float)
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        bs = boot(y3c[ev], {"severe_probe": sev[ev], "moderate_plus_probe": hy[ev], "presence_owl_case": owl_case[ev]},
                  base_auc=["presence_owl_case"])
        three[sp] = {"n": n, "grade_counts": np.bincount(g, minlength=3).tolist(), "acc3": float(hit.mean()),
                     "acc3_ci": ci([hit[i].mean() for i in I]), "majority_acc3": float(mj.mean()),
                     "d_acc3_vs_majority_ci": ci([hit[i].mean() - mj[i].mean() for i in I]),
                     "confusion_rows_true_cols_pred": [[int(((g == t) & (pr == p)).sum()) for p in range(3)] for t in range(3)],
                     "spearman_ordinal_score": float(spearmanr(hy[ev] + sev[ev], g).correlation),
                     "severe_vs_rest": {k: v for k, v in bs.items() if not k.startswith("_")}}
        log(f"  3-class {sp}: {json.dumps(three[sp])}")

    # ---------------------------------------------------------------- outputs
    f = lambda v: "" if np.isnan(v) else f"{v:.5f}"             # noqa: E731
    with open(out / "probe_scores_hygrade.csv", "w") as fo:
        w = csv.writer(fo)
        w.writerow(["case_id", "split", KEY, "hy_grade_right", "hy_grade_left", "hy_severe"])
        for n in range(N):
            w.writerow([ids[n], split[n], f(hy[n]), f(HYK[n, 0]), f(HYK[n, 1]), f(sev[n])])
    uc = json.load(open(M4 / "improve/probe_pillar/unit_cue_pillar.json"))      # peer file, read-only; copied + one key
    assert set(uc["labels_train"]) == set(ids[tr_c])
    for n in np.where(tr_c)[0]:
        uc["labels_train"][ids[n]][KEY] = None if np.isnan(y2c[n]) else int(y2c[n])
    uc["thresholds"][KEY] = t_f1
    uc["pools"][KEY] = pools
    uc["grades"][KEY] = {"disease": "hydronephrosis", "score": "P(moderate or severe | hydronephrosis reported)",
                         "label": "1 = report grade moderate / severe, 0 = mild, null = hydronephrosis not reported "
                                  "positive or grade not written (one grade per case)"}
    (out / "unit_cue_hygrade.json").write_text(json.dumps(uc))
    cov = {sp: {"cases": int(m.sum()), "hy_grade_scored": int((m & ~np.isnan(hy)).sum()),
                "hy_grade_labelled": int((m & ~np.isnan(y2c)).sum()),
                "hy_grade_labelled_scored": int((m & ~np.isnan(y2c) & ~np.isnan(hy)).sum())}
           for sp, m in (("train", tr_c), ("val", va_c))}
    chk = {"counts": counts, "label": "one grade per case (targets_v2 hydronephrosis.attributes.grade); moderate+ = 1",
           "test_rows_read": 0, "splits_in_scores": sorted(set(split)), "fold_fallback_train_cases": int(miss[0]),
           "presence_threshold_kidney": t_pres, "owl_features_chosen": owl_nm, "stack_chosen": best,
           "probes": info2, "candidates_train_oof": cand, "probes_severe": info3, "m4_cols": m4cols,
           "thresholds": {KEY: t_f1, "accuracy_thr_train_oof": t_acc}, "pool_sizes": {k: len(v) for k, v in pools.items()},
           "coverage": cov, "binary": res, "three_class": three,
           "decision": dict(dec, rule=" ".join(decide.__doc__.split())),
           "bootstrap": B, "note": "aggregates only; thresholds / pools / model choice from train OOF; val reported once"}
    (out / "check.json").write_text(json.dumps(chk, indent=1))
    log(f"[{time.strftime('%m-%d %H:%M')}] wrote {out}/probe_scores_hygrade.csv, unit_cue_hygrade.json, check.json")


if __name__ == "__main__":
    main()
