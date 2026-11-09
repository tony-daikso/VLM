#!/usr/bin/env python3
"""Method 1, step S5: assemble template findings on VAL (default) from the train-built sentence bank + train thresholds.

Per disease decision from the calibrated OWL score p (train-OOF thresholds, see m1_thresholds.py):
  variant A (two thresholds): p > t_pos -> pos ; t_neg <= p <= t_pos -> verify (positive sentence + "[please verify]") ;
                              p < t_neg -> neg ; no score -> omit
  variant B (single F1 cut):  p >= t_f1 -> pos ; else neg ; no score -> omit
Per organ group: silent-negative diseases (osteopenia, atherosclerosis) drop out when not positive; the joint sentence
for the group's polarity combination is used when the bank has one, otherwise one sentence per disease.
Measured attributes (segmentation-based, not label-trained): AAA positive sentence gets the auto-measured maximal
abdominal aortic diameter, splenomegaly positive sentence gets the auto-measured craniocaudal length; both linearly
calibrated to the report values on TRAIN (cases where the report states them).
Outputs (method1/): val_drafts_{A,B}.jsonl  {case_id, findings: {disease: sentence}, text, text_marked, decisions,
                     scores, measured}, val_decisions_{A,B}.csv, measure_calibration.json
"""
import json
import sys

import numpy as np
import pandas as pd

from m1_common import D14, OUT, PROB, ROOT, load_labels

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
assert SPLIT in ("train", "val"), "test is locked"
MARK = " [please verify]"


def calib_measure():
    L = load_labels(("train",))
    M = pd.read_csv(ROOT / "oof" / "out" / "measure_trval.csv").set_index("study_id")
    dia = L["abdominal_aortic_aneurysm__detail"].map(lambda x: json.loads(x).get("diameter_cm"))
    sl = L["splenomegaly__detail"].map(lambda x: json.loads(x).get("length_cm"))
    out = {}
    for name, ref, col in (("aaa_cm", dia, "abd_eq_rmax"), ("spleen_cm", sl, "sp_cc_main_cm")):
        df = pd.DataFrame({"ref": ref}).join(M[[col]]).dropna()
        df = df[df.ref > 0]
        x = df[col].values / (10.0 if name == "aaa_cm" else 1.0); y = df.ref.values.astype(float)
        a, b = np.polyfit(x, y, 1)
        out[name] = dict(col=col, a=float(a), b=float(b), n=int(len(df)), mae_train=float(np.mean(np.abs(a * x + b - y))))
    json.dump(out, open(OUT / "measure_calibration.json", "w"), indent=1)
    return out, M


def main():
    bank = json.load(open(OUT / "sentence_bank.json"))
    groups = bank["groups"]; silent = bank["silent_negative"]
    single = {k: v[0][0] for k, v in bank["single"].items() if v}
    joint = {k: v[0][0] for k, v in bank["joint"].items()}
    th = pd.read_csv(OUT / "thresholds.csv").set_index("disease")
    P = pd.read_csv(PROB).set_index("study_id"); P = P[P.split == SPLIT]
    mc, M = calib_measure()
    M = M.reindex(P.index)
    for var in ("A", "B"):
        dec = {}
        for d in D14:
            p = P[d].values
            if var == "A":
                x = np.where(p > th.loc[d, "t_pos"], "pos", np.where(p >= th.loc[d, "t_neg"], "verify", "neg"))
            else:
                x = np.where(p >= th.loc[d, "t_f1"], "pos", "neg")
            dec[d] = np.where(np.isnan(p), "omit", x)
        dec = pd.DataFrame(dec, index=P.index)
        dec.to_csv(OUT / f"{SPLIT}_decisions_{var}.csv")
        with open(OUT / f"{SPLIT}_drafts_{var}.jsonl", "w") as f:
            for sid in P.index:
                row = dec.loc[sid]; findings = {}; lines, lines_m = [], []; meas = {}
                for g, ds in groups:
                    act = [d for d in ds if row[d] != "omit" and not (silent[d] and row[d] == "neg")]
                    if not act:
                        continue
                    pol = {d: ("pos" if row[d] in ("pos", "verify") else "neg") for d in act}
                    k = g + "|" + ",".join(f"{d}:{pol[d]}" for d in act)
                    ver = any(row[d] == "verify" for d in act)
                    if k in joint and not any(d in ("abdominal_aortic_aneurysm", "splenomegaly") and pol[d] == "pos" for d in act):
                        sents = [(joint[k], act)]
                    else:
                        sents = []
                        for d in act:
                            s = single.get(f"{d}|{pol[d]}")
                            if not s:
                                continue
                            if pol[d] == "pos" and d == "abdominal_aortic_aneurysm" and pd.notna(M.loc[sid, "abd_eq_rmax"]):
                                c = mc["aaa_cm"]; v = c["a"] * M.loc[sid, "abd_eq_rmax"] / 10 + c["b"]; meas["aaa_cm"] = round(float(v), 1)
                                s = f"Abdominal aortic aneurysm measuring up to {v:.1f} cm."
                            if pol[d] == "pos" and d == "splenomegaly" and pd.notna(M.loc[sid, "sp_cc_main_cm"]):
                                c = mc["spleen_cm"]; v = c["a"] * M.loc[sid, "sp_cc_main_cm"] + c["b"]; meas["spleen_cm"] = round(float(v), 1)
                                s = f"Splenomegaly, measuring {v:.1f} cm in craniocaudal length."
                            sents.append((s, [d]))
                    for s, ds_ in sents:
                        v = any(row[d] == "verify" for d in ds_)
                        if s not in lines:
                            lines.append(s); lines_m.append(s[:-1] + MARK + "." if v and s.endswith(".") else s + (MARK if v else ""))
                        for d in ds_:
                            findings[d] = (findings.get(d, "") + " " + s).strip()
                f.write(json.dumps({"case_id": sid, "findings": findings, "text": "\n".join(lines), "text_marked": "\n".join(lines_m),
                                    "decisions": row.to_dict(), "scores": {d: (None if np.isnan(P.loc[sid, d]) else round(float(P.loc[sid, d]), 4)) for d in D14},
                                    "measured": meas}) + "\n")
        print(var, dec.apply(lambda c: c.value_counts(normalize=True)).T.round(3).to_string())
    print(json.dumps(mc, indent=1))


if __name__ == "__main__":
    main()
