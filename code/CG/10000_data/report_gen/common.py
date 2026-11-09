"""Shared paths, class list and def-B label logic for the CG report-draft pipeline (Phase 1)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
OUT = ROOT / "report_gen"
REPORTS = ROOT / "deliver_10k_reports_v1(in).csv"
LABELS = ROOT / "disease14_qwen.csv"
PROBS = ROOT / "owl_infer/best/probs_best_main.csv"
OSTEO = ROOT / "osteo_hu_label_main.csv"
MEAS_TS = OUT / "measurements_ts.csv"

DIS = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "gallstones",
       "biliary_ductal_dilation", "hydronephrosis", "abdominal_aortic_aneurysm", "atherosclerosis",
       "osteopenia", "fracture", "bowel_obstruction", "appendicitis", "ascites", "free_air"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "hydronephrosis": "腎水腫",
      "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化", "osteopenia": "骨質減少",
      "fracture": "骨折", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體"}
# same sets as code/CG/10000_data/eval_owl_cg10k.py (definition B)
POS = {"yes", "probably_yes"}
NEG_EXPLICIT = {"no", "probably_no"}
OSTEO_POS_HU, OSTEO_NEG_HU = 110, 160  # <=110 positive; >160 negative; 110-160 excluded (def-B analogue)


def read_reports():
    return pd.read_csv(REPORTS, encoding="cp1252")


def defb_from_cat(cat: pd.Series) -> pd.Series:
    """1 = yes/probably_yes, 0 = no/probably_no, NaN = not_mentioned/unclear/missing (excluded)."""
    c = cat.astype(str)
    y = pd.Series(np.nan, index=cat.index)
    y[c.isin(NEG_EXPLICIT)] = 0.0
    y[c.isin(POS)] = 1.0
    return y


def defb_labels():
    """EXAMID-indexed frame with def-B label (1/0/NaN) per class. Osteopenia = image L1-HU label."""
    lab = pd.read_csv(LABELS, encoding="utf-8-sig").set_index("EXAMID")
    out = pd.DataFrame(index=lab.index)
    for k in DIS:
        if k != "osteopenia":
            out[k] = defb_from_cat(lab[k + "_cat"])
    hu = pd.read_csv(OSTEO).set_index("study_id")["hu"]
    y = pd.Series(np.nan, index=hu.index)
    y[hu > OSTEO_NEG_HU] = 0.0
    y[hu <= OSTEO_POS_HU] = 1.0
    out["osteopenia"] = y.reindex(out.index)
    out["osteopenia_report"] = defb_from_cat(lab["osteopenia_cat"])
    return out
