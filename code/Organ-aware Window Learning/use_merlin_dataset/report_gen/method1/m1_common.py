"""Shared constants / loaders for Merlin method 1 (template baseline). Test rows are never parsed."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# data side (DUA-restricted, gitignored): reports, labels, scores, outputs. Code lives in the git repo, data never does.
MERLIN = Path(os.environ.get("MERLIN_DATA", "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin"))
ROOT = MERLIN / "report_gen"
LABELS = MERLIN / "disease14" / "labels_disease14.jsonl"
REPORTS = MERLIN / "disease14" / "reports.json"
PROB = ROOT / "oof" / "out" / "scores_trval_prob.csv"   # Platt-calibrated; train = OOF, val = full-train model
OUT = ROOT / "method1"

D14 = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "hydronephrosis", "abdominal_aortic_aneurysm",
       "atherosclerosis", "osteopenia", "bowel_obstruction", "appendicitis", "ascites", "free_air", "gallstones",
       "biliary_ductal_dilation", "fracture"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊切除術後", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no"}


def load_labels(splits=("train", "val")):
    """category per study x disease (strings) + split; rows of other splits are skipped before json parsing."""
    rows = []
    for line in open(LABELS):
        sp = line[line.index('"split": "') + 10:].split('"', 1)[0]
        if sp not in splits:
            continue
        r = json.loads(line)
        a = r.get("answer") or {}
        rows.append({"study_id": r["study_id"], "split": sp,
                     **{d: (a.get(d) or {}).get("category", "not_mentioned") for d in D14},
                     **{f"{d}__detail": json.dumps((a.get(d) or {}).get("detail", {})) for d in D14}})
    return pd.DataFrame(rows).set_index("study_id")


def defb(cat):
    """definition B: 1 = yes/probably_yes, 0 = no/probably_no, NaN = unclear/not_mentioned."""
    return cat.map(lambda c: 1.0 if c in POS else (0.0 if c in NEG else np.nan))


def defa(cat):
    """definition A (= OWL training label): 1 = yes/probably_yes, 0 = no/probably_no/not_mentioned, NaN = unclear."""
    return cat.map(lambda c: 1.0 if c in POS else (np.nan if c == "unclear" else 0.0))
