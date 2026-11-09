#!/usr/bin/env python3
"""S6: evaluate the test-set drafts.

Inputs: decisions.csv (S5), labels_drafts_test.jsonl (same Qwen disease14 extraction = run_cg10k.py +
prompt.md, unchanged, run on the drafts), disease14_qwen.csv (report labels, def B), osteo HU label.

Outputs (report_gen/):
  eval_phase1.csv          per class: draft (Qwen re-extracted) vs report def-B labels, strict & lenient
  s6_assembly_test.csv     per class: assembly decisions vs labels on the full test set (no Qwen needed)
  s6_consistency.csv       rendered state (pos / verify / neg / none) x extracted category
  s6_coverage.csv          report-negative coverage and critical errors (report positive, draft negative)
  side_by_side_samples.md  10 test exams per LOC, original report vs draft
"""
import json

import numpy as np
import pandas as pd

from common import DIS, NEG_EXPLICIT, OUT, POS, ZH, defb_from_cat, defb_labels, read_reports

LAB_FILE = OUT / "labels_drafts_test.jsonl"


def prf(pred, y):
    keep = ~np.isnan(y)
    p, yy = pred[keep], y[keep] == 1
    tp, fp = int((p & yy).sum()), int((p & ~yy).sum())
    fn, tn = int((~p & yy).sum()), int((~p & ~yy).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(N=int(keep.sum()), n_pos=int(yy.sum()), TP=tp, FP=fp, FN=fn, TN=tn, sens=sens, spec=spec, PPV=ppv,
                F1=f1)


def load_extracted():
    rows = {}
    for l in open(LAB_FILE):
        try:
            j = json.loads(l)
        except Exception:  # noqa: BLE001
            continue
        if not j.get("parse_ok"):
            continue
        rows[j["study_id"]] = {k: j["answer"][k]["category"] for k in DIS}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    dec = pd.read_csv(OUT / "decisions.csv").set_index("EXAMID")
    test = dec[dec.split == "test"]
    lab = defb_labels()
    rcat = pd.read_csv(OUT.parent / "disease14_qwen.csv", encoding="utf-8-sig").set_index("EXAMID")
    ex = load_extracted()
    ids = [i for i in test.index if i in ex.index]
    print(f"test {len(test)}, extracted ok {len(ids)}")
    E, T, L = ex.loc[ids], test.loc[ids], lab.loc[ids]

    # ---- 1. draft (re-extracted) vs report labels
    rows = []
    for k in DIS:
        y = L[k].values
        cat = E[k].astype(str)
        for mode, s in (("strict", POS), ("lenient", POS | {"unclear"})):
            rows.append(dict(disease=k, 疾病=ZH[k], mode=mode, **prf(cat.isin(s).values, y)))
        if k == "osteopenia":
            yr = defb_from_cat(rcat.loc[ids, "osteopenia_cat"]).values
            rows.append(dict(disease=k, 疾病=ZH[k] + "（報告標籤）", mode="strict", **prf(cat.isin(POS).values, yr)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "eval_phase1.csv", index=False)

    # ---- 2. assembly decisions vs labels (full test and the extracted subset)
    arows = []
    for subset, frame in (("test_all", test), ("test_extracted", T)):
        ll = lab.loc[frame.index]
        for k in DIS:
            d = frame[f"{k}_dec"].values
            y = ll[k].values
            r = prf(np.isin(d, ["pos", "verify"]), y)
            keep = ~np.isnan(y)
            r.update(neg_stmt_on_pos=int(((d == "neg") & (y == 1)).sum()),
                     crit_rate=float(((d == "neg") & (y == 1)).sum() / max((y == 1).sum(), 1)),
                     frac_neg=float(np.mean(d == "neg")), frac_verify=float(np.mean(d == "verify")),
                     frac_pos=float(np.mean(d == "pos")), frac_omit=float(np.mean(d == "omit")),
                     neg_coverage_assembly=float(((d == "neg") & (y == 0)).sum() / max((y == 0).sum(), 1)))
            arows.append(dict(subset=subset, disease=k, 疾病=ZH[k], **r))
    A = pd.DataFrame(arows)
    A.to_csv(OUT / "s6_assembly_test.csv", index=False)

    # ---- 3. consistency rendered state x extracted category
    cats = ["yes", "probably_yes", "unclear", "probably_no", "no", "not_mentioned"]
    expect = {"pos": POS, "verify": POS | {"unclear"}, "neg": NEG_EXPLICIT, "none": {"not_mentioned"},
              "implied_neg": NEG_EXPLICIT | {"not_mentioned"}}
    crows = []
    for k in DIS:
        for stt in ("pos", "verify", "neg", "none", "implied_neg"):
            m = T[f"{k}_rendered"].values == stt
            if m.sum() == 0:
                continue
            vc = E.loc[m, k].value_counts()
            crows.append(dict(disease=k, 疾病=ZH[k], rendered=stt, n=int(m.sum()),
                              consistent=float(E.loc[m, k].isin(expect[stt]).mean()),
                              **{c: int(vc.get(c, 0)) for c in cats}))
    Cn = pd.DataFrame(crows)
    Cn.to_csv(OUT / "s6_consistency.csv", index=False)

    # ---- 4. coverage of report negatives + critical errors (report positive, draft states negative)
    vrows = []
    tot_pos = tot_crit = tot_crit_asm = 0
    for k in DIS:
        rc = rcat.loc[ids, k + "_cat"].astype(str)
        if k == "osteopenia":  # HU label: pos <=110, neg >160
            ypos, yneg = (L[k] == 1).values, (L[k] == 0).values
        else:
            ypos, yneg = rc.isin(POS).values, rc.isin(NEG_EXPLICIT).values
        dneg = E[k].astype(str).isin(NEG_EXPLICIT).values
        aneg = np.isin(T[f"{k}_rendered"].values, ["neg", "implied_neg"])  # implied: S/P cholecystectomy written, no GB
        crit, crit_a = int((ypos & dneg).sum()), int((ypos & aneg).sum())
        tot_pos += int(ypos.sum())
        tot_crit += crit
        tot_crit_asm += crit_a
        vrows.append(dict(disease=k, 疾病=ZH[k], report_neg=int(yneg.sum()),
                          neg_covered=int((yneg & dneg).sum()), neg_coverage=float((yneg & dneg).sum() / max(yneg.sum(), 1)),
                          report_pos=int(ypos.sum()), critical=crit, critical_rate=crit / max(ypos.sum(), 1),
                          critical_assembly=crit_a, critical_rate_assembly=crit_a / max(ypos.sum(), 1)))
    V = pd.DataFrame(vrows)
    V.loc[len(V)] = dict(disease="ALL", 疾病="合計", report_neg=V.report_neg.sum(), neg_covered=V.neg_covered.sum(),
                         neg_coverage=V.neg_covered.sum() / V.report_neg.sum(), report_pos=tot_pos, critical=tot_crit,
                         critical_rate=tot_crit / tot_pos, critical_assembly=tot_crit_asm,
                         critical_rate_assembly=tot_crit_asm / tot_pos)
    V.to_csv(OUT / "s6_coverage.csv", index=False)

    # ---- 5. side-by-side samples
    rep = read_reports().set_index("EXAMID")
    drafts = {json.loads(l)["EXAMID"]: json.loads(l)["draft"] for l in open(OUT / "drafts_test.jsonl")}
    rng = np.random.default_rng(11)
    md = ["# 原報告 vs 草稿（test，每院區 10 份）", "",
          "草稿標記：`[please verify]` = 模型不確定，寫陽性句請醫師確認；`[NOT ASSESSED]` = 模型未判讀的器官，放院區常用正常句。", ""]
    for loc in sorted(T.LOC.unique()):
        pool = [i for i in ids if T.loc[i, "LOC"] == loc]
        for sid in rng.choice(pool, size=min(10, len(pool)), replace=False):
            pos = [ZH[k] for k in DIS if lab.loc[sid, k] == 1]
            md += [f"## {loc} — {sid}", "", f"原報告陽性（定義 B）：{'、'.join(pos) if pos else '無'}", "",
                   "### 原報告", "", "```", str(rep.loc[sid, "Report"]).replace("\r", "").strip(), "```", "",
                   "### 草稿", "", "```", drafts[sid].strip(), "```", ""]
    (OUT / "side_by_side_samples.md").write_text("\n".join(md), encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(R.round(3).to_string(index=False))
    print(A[A.subset == "test_all"].round(3).to_string(index=False))
    print(Cn.round(3).to_string(index=False))
    print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
