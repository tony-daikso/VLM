#!/usr/bin/env python3
"""Build the std14 input JSON and the validation id list (stratified by LOC, half enriched for positives).

Writes into the Mac output folder:
  reports_std_in.json   {EXAMID: {report, PID, LOC, EDATE, row}}
  ids_val100.txt        ~100 EXAMIDs: per LOC, half random, half with any disease14 yes/probably_yes/unclear
  ids_all.txt           all EXAMIDs by row
"""
import json
import os

import pandas as pd

DATA = "/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data"
OUT = os.path.join(DATA, "report_gen", "standardize")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "deliver_10k_reports_v1(in).csv"), encoding="cp1252")
rep = {r.EXAMID: {"report": r.Report, "PID": r.PID, "LOC": r.LOC, "EDATE": r.EDATE, "row": int(i)}
       for i, r in df.iterrows()}
json.dump(rep, open(os.path.join(OUT, "reports_std_in.json"), "w"), ensure_ascii=False)
open(os.path.join(OUT, "ids_all.txt"), "w").write("\n".join(df.EXAMID) + "\n")

lab = pd.read_csv(os.path.join(DATA, "disease14_qwen.csv"))
cats = [c for c in lab.columns if c.endswith("_cat")]
lab["anypos"] = lab[cats].isin(["yes", "probably_yes", "unclear"]).sum(axis=1) >= 2
df = df.merge(lab[["EXAMID", "anypos"]], on="EXAMID")
ids = []
for loc, n in [("??(8)", 34), ("??(3)", 33), ("??(2)", 33)]:
    sub = df[df.LOC == loc]
    pos = sub[sub.anypos].sample(n // 2, random_state=0)
    rnd = sub.drop(pos.index).sample(n - n // 2, random_state=0)
    ids += list(pos.EXAMID) + list(rnd.EXAMID)
open(os.path.join(OUT, "ids_val100.txt"), "w").write("\n".join(ids) + "\n")
print(len(rep), len(ids))
