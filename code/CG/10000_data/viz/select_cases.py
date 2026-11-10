"""Pick CG main cases for unit-level attribution (Mac, CPU).

ids_attr.txt : every report-positive ('yes') case of the 6 target findings + top-40 'no'/'not_mentioned' by prob (FP pool)
Writes into viz/work/.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data")
OUT = Path(__file__).resolve().parent / "work"
TARGETS = ["fracture", "hydronephrosis", "abdominal_aortic_aneurysm", "gallstones", "splenomegaly", "ascites"]


def main():
    OUT.mkdir(exist_ok=True)
    pr = pd.read_csv(ROOT / "owl_infer/probs_main.csv", usecols=lambda c: "__" not in c).set_index("study_id")
    lab = {}
    for l in open(ROOT / "disease14_qwen.jsonl"):
        x = json.loads(l)
        if x["study_id"] in pr.index and x.get("parse_ok"):
            lab[x["study_id"]] = x["answer"]
    ids = set()
    for f in TARGETS:
        cat = pd.Series({k: v[f]["category"] for k, v in lab.items()})
        pos = cat[cat == "yes"].index
        neg = cat[cat.isin(["no", "not_mentioned"])].index
        fp = pr.loc[neg, f].sort_values(ascending=False).index[:40]
        ids |= set(pos) | set(fp)
        print(f, "pos", len(pos), "fp-pool", len(fp))
    (OUT / "ids_attr.txt").write_text("\n".join(sorted(ids)) + "\n")
    print("total ids", len(ids))


if __name__ == "__main__":
    main()
