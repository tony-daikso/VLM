"""GS2 pilot: stratified TRAIN-only sample (~600) with inverse-probability weights."""
import pandas as pd, numpy as np
d = pd.read_csv("in/analysis_gallstones/per_case_train_oof.csv")[["study_id","y_gs","y_chole","gb_vox","vis100"]]
lab = pd.read_csv("in/analysis_gallstones/labels_trval.csv"); lab = lab[lab.split=="train"]  # split filter first
d = d[d.study_id.isin(lab.study_id)]
miss = d.gb_vox.fillna(0) < 100
cat = np.select([(d.y_gs==1)&miss, (d.y_gs==1)&(d.vis100==1), d.y_gs==1,
                 (d.y_gs==0)&(d.y_chole==1), (d.y_gs==0)&miss, (d.y_gs==0)&(d.vis100==1)],
                ["pos_missingGB","pos_visible3mm","pos_nonvisible3mm","neg_cholecystectomy","neg_missingGB_other","neg_hyperdense3mm"], "neg_other")
d["stratum"] = cat
want = {"pos_visible3mm":100,"pos_nonvisible3mm":160,"pos_missingGB":50,"neg_cholecystectomy":40,"neg_missingGB_other":20,"neg_hyperdense3mm":80,"neg_other":150}
rows = []
for s,n in want.items():
    g = d[d.stratum==s]; k = min(n, len(g)); smp = g.sample(k, random_state=0).copy(); smp["w"] = len(g)/k; rows.append(smp)
    print(s, len(g), k)
P = pd.concat(rows); P.to_csv("out/gs2_pilot_ids.csv", index=False); P.study_id.to_csv("out/gs2_pilot_ids.txt", index=False, header=False)
print(len(P), P.y_gs.sum())
