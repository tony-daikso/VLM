import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS_PATH = "/datadrive/VLM/data/CT/CG/abdomen_labels/abdomen72_labels_by_finding.json"
RESULTS_CSV = "/root/Desktop/VLM/TotalFM/CT/results/TotalFM_infer_results_CG2_abdomen72.csv"
OUT_CSV = "/root/Desktop/VLM/TotalFM/CT/results/CG2_abdomen72_auc_per_finding.csv"

labels = json.load(open(LABELS_PATH, encoding="utf-8"))
results = pd.read_csv(RESULTS_CSV)
results["patient_id"] = results["patient_id"].astype(str)

rows = []
for col in results.columns:
    if col == "patient_id" or col not in labels:
        continue
    label_map = labels[col]
    gt, pd_scores = [], []
    for pid, score in zip(results["patient_id"], results[col]):
        if pid not in label_map or pd.isna(score) or score == "":
            continue
        gt.append(int(label_map[pid]))
        pd_scores.append(float(score))
    n = len(gt)
    n_pos = sum(gt)
    n_neg = n - n_pos
    auc = np.nan
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(gt, pd_scores)
    rows.append({"finding_key": col, "n_labeled": n, "n_pos": n_pos, "n_neg": n_neg, "auc": auc})

df = pd.DataFrame(rows).sort_values(["n_labeled", "finding_key"], ascending=[False, True])
df.to_csv(OUT_CSV, index=False)
evaluable = df.dropna(subset=["auc"])
print(f"Findings with any labels: {len(df)}")
print(f"Findings with both pos+neg (AUC computable): {len(evaluable)}")
if len(evaluable):
    print(f'Mean AUC (unweighted): {evaluable["auc"].mean():.4f}')
print(evaluable[["finding_key", "n_labeled", "n_pos", "n_neg", "auc"]].to_string(index=False))
