"""
Compute per-finding AUC for Pillar0-AbdomenCT zero-shot scores against the CG2
abdomen72 ground-truth labels. Same methodology as RADAR's calc_metrics_cg2.py,
so the two models' results are directly comparable.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS_PATH = "/datadrive/VLM/data/CT/CG/abdomen_labels/abdomen72_labels_by_finding.json"
RESULTS_CSV = "/root/Desktop/VLM/Pillar0/CT/results/Pillar0_infer_results_CG2_abdomen72.csv"
OUT_CSV = "/root/Desktop/VLM/Pillar0/CT/results/CG2_abdomen72_auc_per_finding.csv"

labels = json.load(open(LABELS_PATH, encoding="utf-8"))
results = pd.read_csv(RESULTS_CSV)
results["patient_id"] = results["patient_id"].astype(str)

rows = []
for col in results.columns:
    if col == "patient_id":
        continue
    if col not in labels:
        continue
    label_map = labels[col]

    gt, pd_scores = [], []
    for pid, score in zip(results["patient_id"], results[col]):
        if pid not in label_map:
            continue
        score = pd.to_numeric(score, errors="coerce") if not isinstance(score, float) else score
        if pd.isna(score):
            continue
        gt.append(int(label_map[pid]))
        pd_scores.append(float(score))

    n = len(gt)
    n_pos = sum(gt)
    n_neg = n - n_pos
    auc = np.nan
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(gt, pd_scores)

    rows.append({
        "finding_key": col,
        "n_labeled": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
    })

df = pd.DataFrame(rows).sort_values(["n_labeled", "finding_key"], ascending=[False, True])
df.to_csv(OUT_CSV, index=False)

evaluable = df.dropna(subset=["auc"])
print(f"Findings with any labels: {len(df)}")
print(f"Findings with both pos+neg (AUC computable): {len(evaluable)}")
print(f'Total (patient, finding) label pairs: {df["n_labeled"].sum()}')
if len(evaluable):
    print(f'Mean AUC (unweighted, over evaluable findings): {evaluable["auc"].mean():.4f}')
print()
print(evaluable[["finding_key", "n_labeled", "n_pos", "n_neg", "auc"]].to_string(index=False))
print()
print("--- findings with labels but not evaluable (all-pos or all-neg) ---")
print(df[df["auc"].isna()][["finding_key", "n_labeled", "n_pos", "n_neg"]].to_string(index=False))
