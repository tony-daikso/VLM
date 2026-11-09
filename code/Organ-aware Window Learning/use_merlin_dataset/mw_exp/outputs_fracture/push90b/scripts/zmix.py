"""push90: z-average of head score dirs (oof_train.csv + val_scores.csv, column ft): each input z-scored with its TRAIN
mean/sd, then averaged over the inputs present (rows need all inputs; else mean of available). Train-only stats.
usage: zmix.py OUT DIR [DIR ...]"""
import sys, os, pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True); TR, VA = [], []
for d in sys.argv[2:]:
    t = pd.read_csv(f"{d}/oof_train.csv").set_index("study_id")["ft"]; v = pd.read_csv(f"{d}/val_scores.csv").set_index("study_id")["ft"]
    mu, sd = t.mean(), t.std(); TR.append((t - mu) / sd); VA.append((v - mu) / sd)
tr = pd.concat(TR, axis=1).mean(1); va = pd.concat(VA, axis=1).mean(1)
pd.DataFrame({"study_id": tr.index, "ft": tr.values}).to_csv(f"{out}/oof_train.csv", index=False)
pd.DataFrame({"study_id": va.index, "ft": va.values}).to_csv(f"{out}/val_scores.csv", index=False)
open(f"{out}/inputs.txt", "w").write("\n".join(sys.argv[2:]) + "\n"); print(out, len(tr), len(va))
