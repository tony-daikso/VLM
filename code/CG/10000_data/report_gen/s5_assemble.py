#!/usr/bin/env python3
"""S5: assemble per-site (LOC) English draft reports from S1 canonical sentences, S3 thresholds and S4 measurements.

Per class decision (S3): pos -> positive sentence; verify -> positive sentence + " [please verify]" (no negative
sentence); neg -> negative sentence. Organs the model does not assess get the site's normal sentence +
" [NOT ASSESSED]". Clinical information / comparison / tumour lesions / IMP are empty headers for the radiologist.

Outputs (report_gen/): decisions.csv (raw score, decision, rendered state per class), drafts/<EXAMID>.txt,
drafts_test.jsonl, drafts_dev.jsonl, measurements.csv (S4 values used in the drafts).
"""
import json

import numpy as np
import pandas as pd

from common import DIS, OUT, PROBS
from s3_thresholds import apply_guards, decide, load_meas

L8, L3, L2 = "??(8)", "??(3)", "??(2)"
VERIFY, NA = " [please verify]", " [NOT ASSESSED]"  # placed before the final period
# organ groups in reading order; each item is a model class, a NA slot, "gallbladder" (combined logic) or "measure"
GROUPS = [["liver_lesion", "hepatic_steatosis", "biliary_ductal_dilation"],
          ["gallbladder", "portal_vein"],
          ["splenomegaly", "pancreas", "adrenal"],
          ["hydronephrosis"],
          ["abdominal_aortic_aneurysm", "atherosclerosis", "lymph_nodes"],
          ["ascites", "free_air", "bowel_obstruction", "appendicitis", "mesentery"],
          ["urinary_bladder"],
          ["lower_chest"],
          ["osteopenia", "fracture", "bone_lesion"],
          ["measure"]]
TODO = "[To be completed by radiologist]"


def canon():
    c = pd.read_csv(OUT / "canonical_sentences.csv")
    return {(r.LOC, r.slot, r.polarity): r.sentence.rstrip(".") for r in c.itertuples()}


def measure_line(m):
    parts = []
    if pd.notna(m.get("L1_hu")):
        parts.append(f"{m['hu_level']} trabecular bone attenuation {m['L1_hu']:.0f} HU")
    if pd.notna(m.get("spleen_ml")):
        parts.append(f"spleen volume {m['spleen_ml']:.0f} mL")
    if pd.notna(m.get("aorta_eqd_mm")):
        parts.append(f"maximal abdominal aortic diameter {m['aorta_eqd_mm'] / 10:.1f} cm")
    if not parts:
        return None
    return "Auto-measurements (segmentation-based, please verify): " + "; ".join(parts) + "."


def render(loc, dec, meas, C):
    """returns list of groups (list of sentences) and rendered-state dict per class."""
    st = {k: "none" for k in DIS}

    def model(k):
        d = dec[k]
        st[k] = d
        if d == "neg":
            return C[(loc, k, "neg")] + "."
        if d == "omit":
            st[k] = "none"
            return None
        return C[(loc, k, "pos")] + (VERIFY if d == "verify" else "") + "."

    out = []
    for g in GROUPS:
        sents = []
        for item in g:
            if item == "measure":
                ml = measure_line(meas)
                if ml:
                    sents.append(ml)
            elif item == "gallbladder":
                c, s = dec["surgically_absent_gallbladder"], dec["gallstones"]
                if c in ("pos", "verify"):
                    sents.append(model("surgically_absent_gallbladder"))
                    if c == "pos":  # gallbladder absent -> no gallstone sentence; Qwen reads it as "no gallstones"
                        st["gallstones"] = "implied_neg"
                    if c == "verify":
                        sents.append(model("gallstones"))
                elif s == "neg":
                    sents.append(C[(loc, "gallbladder_normal", "neg")] + ".")
                    st["surgically_absent_gallbladder"], st["gallstones"] = "neg", "neg"
                else:
                    sents.append(model("gallstones"))
            elif item in DIS:
                sents.append(model(item))
            else:
                sents.append(C[(loc, item, "neg")] + NA + ".")
        sents = [x for x in sents if x]
        if sents:
            out.append(sents)
    return out, st


def assemble(loc, groups):
    if loc == L8:
        body = "\n".join(s for g in groups for s in g)
        return ("Indications:\n\n\nTechniques:\n\n\nComparison:\n\n\nFindings:\n" + TODO + "\n" + body +
                "\n\nIMP:\n")
    if loc == L2:
        body = "\n".join(s for g in groups for s in g)
        return ("Clinical information:\n\n\nTechnique:\n\nComparison exam:\n\n\n<INDEX LESIONS>\n" + TODO +
                "\n\n<OTHER FINDINGS>\n" + body + "\n\nIMPRESSION:\n")
    body = "\n".join(f"{i}. " + " ".join(g) for i, g in enumerate(groups, 1))
    return ("Clinical Information:\n\n\nComparison:\n\n\n<Main findings>\n" + TODO + "\n\n<Other findings>\n" + body +
            "\n\nIMP:\n")


def main():
    C = canon()
    split = pd.read_csv(OUT / "split.csv").set_index("EXAMID")
    probs = pd.read_csv(PROBS).set_index("study_id").loc[split.index]
    th = pd.read_csv(OUT / "thresholds.csv").set_index("disease")
    guards = json.load(open(OUT / "guards.json"))
    meas = load_meas().reindex(split.index)
    meas[["L1_hu", "hu_level", "spleen_ml", "aorta_eqd_mm"]].rename_axis("EXAMID").to_csv(OUT / "measurements.csv")

    raw = pd.DataFrame({k: decide(probs[k].values, th.loc[k, "t_neg"], th.loc[k, "t_pos"], th.loc[k, "rule"])
                        for k in DIS},
                       index=split.index)
    dec, changed = apply_guards(raw, meas, guards)
    print("guard changes (all exams):", changed)

    (OUT / "drafts").mkdir(exist_ok=True)
    rows, fh = [], {s: open(OUT / f"drafts_{s}.jsonl", "w") for s in ("dev", "test")}
    for sid, r in split.iterrows():
        groups, st = render(r.LOC, dec.loc[sid], meas.loc[sid], C)
        txt = assemble(r.LOC, groups)
        (OUT / "drafts" / f"{sid}.txt").write_text(txt)
        rec = {"EXAMID": sid, "PID": r.PID, "LOC": r.LOC, "split": r.split, "draft": txt,
               "decision": dec.loc[sid].to_dict(), "rendered": st}
        fh[r.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        row = {"EXAMID": sid, "LOC": r.LOC, "split": r.split}
        for k in DIS:
            row[f"{k}_score"] = probs.loc[sid, k]
            row[f"{k}_dec_raw"] = raw.loc[sid, k]
            row[f"{k}_dec"] = dec.loc[sid, k]
            row[f"{k}_rendered"] = st[k]
        rows.append(row)
    for f in fh.values():
        f.close()
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "decisions.csv", index=False)
    for k in DIS:
        print(k, D[f"{k}_rendered"].value_counts().to_dict())


if __name__ == "__main__":
    main()
