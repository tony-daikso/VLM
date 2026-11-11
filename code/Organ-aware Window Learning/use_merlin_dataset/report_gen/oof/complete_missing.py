"""Step 0: list the empty cells of out/scores_trval_prob.csv -> complete/missing_cells.csv (run in report_gen/oof/complete on A6000)."""
# list missing cells of scores_trval_prob.csv (train/val only; test never read)
import pandas as pd
P = pd.read_csv("../out/scores_trval_prob.csv"); assert set(P.split) <= {"train", "val"}
r = []
for d in P.columns[2:]:
    m = P[P[d].isna()]
    r.append(pd.DataFrame({"disease": d, "study_id": m.study_id, "split": m.split}))
pd.concat(r).to_csv("missing_cells.csv", index=False); print(pd.concat(r).groupby(["disease","split"]).size())
