"""Per-level (T8..L5) vertebra-head logits, OOF on train (5 folds, same folds/HP as bone_label_exp ref site_max/mean/all3).
Only train rows are scored here (val predictions not saved). Output: perunit_oof.npz (ids, logit_any, vert_lvl [N,10], P)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/src"))
import numpy as np, torch
from bone_probe import load, prep_X, site_targets, VERT
from bone_label_exp import make_targets, raw_fracture, train_e
from sklearn.model_selection import StratifiedKFold, train_test_split
torch.set_num_threads(6)
out = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones"
ids, S, E, P, units, lab, T = load(out)
y, ys = T["any"], site_targets(T)
raw = raw_fracture(ids)
t, w, yss = make_targets("ref", raw, y, ys)
M = torch.from_numpy(P)
hp = json.loads((out / "agg" / "site_max__mean__all3.json").read_text())["hp"]
tr_all = np.where(S == "train")[0]
oa = np.full(len(ids), np.nan, np.float32); ol = np.full((len(ids), 10), np.nan, np.float32)
for seed_off in (0, 1, 2):
  for k, (fp, hpos) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
    fit, hold = tr_all[fp], tr_all[hpos]
    inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
    X = prep_X(E, P, "all3", "mean", np.isin(np.arange(len(S)), fit))
    m, s, ep = train_e("site_max", X, M, t, w, yss, y, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * seed_off, "cpu")
    with torch.no_grad():
        Xh, Mh = X[hold], M[hold]
        a, _ = m(Xh, Mh)
        lv = (m.heads[0](Xh[:, VERT])[..., 0] + m.ub[VERT]).masked_fill(~Mh[:, VERT], float("nan"))
    if seed_off == 0:
        oa[hold] = a.numpy(); ol[hold] = lv.numpy()
    else:
        oa[hold] += a.numpy(); ol[hold] += lv.numpy()
    print(seed_off, k, s, ep, flush=True)
  np.savez(Path(sys.argv[1]) / "perunit_oof.npz", ids=np.array(ids), split=S, logit_any=oa / (seed_off + 1), vert_lvl=ol / (seed_off + 1), P=P, units=np.array(units), nseeds=seed_off + 1)
print("done")
