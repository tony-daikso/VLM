import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS_PATH = '/datadrive/VLM/data/CT/CG/abdomen_labels/abdomen72_labels_by_finding.json'
RESULTS_CSV = '/root/Desktop/VLM/RADAR/CT/results/RADAR_infer_results_CG2_abdomen72.csv'
OUT_CSV = '/root/Desktop/VLM/RADAR/CT/results/CG2_abdomen72_auc_per_finding.csv'

labels = json.load(open(LABELS_PATH, encoding='utf-8'))
results = pd.read_csv(RESULTS_CSV)
results['patient_id'] = results['file_name'].str.replace('.nii.gz', '', regex=False)

rows = []
for col in results.columns:
    if col in ('file_name', 'patient_id'):
        continue
    cn_key = col.split(' (')[0]
    if cn_key not in labels:
        continue
    label_map = labels[cn_key]

    gt, pd_scores = [], []
    for pid, prob in zip(results['patient_id'], results[col]):
        if pid not in label_map:
            continue
        prob = pd.to_numeric(prob, errors='coerce')
        if pd.isna(prob):
            prob = 0.0
        gt.append(int(label_map[pid]))
        pd_scores.append(float(prob))

    n = len(gt)
    n_pos = sum(gt)
    n_neg = n - n_pos
    auc = np.nan
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(gt, pd_scores)

    rows.append({
        'finding_key': cn_key,
        'column': col,
        'n_labeled': n,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'auc': auc,
    })

df = pd.DataFrame(rows).sort_values(['n_labeled', 'finding_key'], ascending=[False, True])
df.to_csv(OUT_CSV, index=False)

evaluable = df.dropna(subset=['auc'])
print(f'Findings with any labels: {len(df)}')
print(f'Findings with both pos+neg (AUC computable): {len(evaluable)}')
print(f'Total (patient, finding) label pairs: {df["n_labeled"].sum()}')
if len(evaluable):
    print(f'Mean AUC (unweighted, over evaluable findings): {evaluable["auc"].mean():.4f}')
print()
print(evaluable[['finding_key', 'n_labeled', 'n_pos', 'n_neg', 'auc']].to_string(index=False))
print()
print('--- findings with labels but not evaluable (all-pos or all-neg) ---')
print(df[df['auc'].isna()][['finding_key', 'n_labeled', 'n_pos', 'n_neg']].to_string(index=False))
