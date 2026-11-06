"""闌尾炎拆分：probe 5-fold OOF (TRAIN) + full-train -> val scores, for targets y_ap_def / y_ap_unc / y_ap (reference).
Recipe = analysis_appendicitis/scripts/ap_oof.py (probe.py Probe, standardize per unit on train, class-balanced BCE,
lr 1e-3 / wd 1e-4, early stopping on an inner 15% TRAIN holdout). Folds = analysis_appendicitis/oof_train.csv folds
(stratified on the original combined label; every split target is a subset of those rows).
Masking: rows with NaN target (other group / unclear) are dropped from training AND evaluation of that target.
De-aging variants (training only, score at inference = image probe only):
  hnK   : negatives with age < 35 get loss weight xK (hard-negative weighting)
  trap  : logit_train = probe(E) + age_branch(age, sex); age_branch = linear on [age z, age z^2, 5 age bins, female];
          at inference the age branch is dropped, early stopping also uses the image-only logit.
Exploratory: soft (definite=1, uncertain=0.5, neg=0; trained on all three, evaluated on the definite masking).
Test is never loaded: the embedding matrix is the *_notest cache and labels_trval_split.csv has no test rows.
CPU only, torch threads 4.
"""
import sys, time, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from probe import Probe, auroc
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
A0 = R + "analysis_appendicitis/"; O = R + "appendicitis_split/work/"
ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="y_ap_def,y_ap_unc,y_ap")
ap.add_argument("--variants", default="")
a = ap.parse_args()

z = np.load(R + "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz")
ids = list(z["ids"]); units = list(z["units"]); ui = [units.index("colon"), units.index("small_bowel")]
E = torch.from_numpy(z["E"][:, ui].astype(np.float32)); P = torch.from_numpy(z["P"][:, ui].astype(np.float32)); del z
lab = pd.read_csv(R + "appendicitis_split/labels_trval_split.csv").set_index("study_id")
assert set(lab.split) <= {"train", "val"}
lab = lab[lab.y_ap.notna()]                                     # unclear dropped everywhere (as before)
keep = [i for i, s in enumerate(ids) if s in lab.index]
ids = [ids[i] for i in keep]; E, P = E[keep], P[keep]
lab = lab.loc[ids]
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").set_index("study id")
age = meta.Age.reindex(ids).astype(float).values; fem = (meta.Gender.reindex(ids) == "Female").astype(float).values
age_f = np.where(np.isnan(age), np.nanmedian(age), age)
S = lab.split.values; tr = S == "train"
for u in range(2):
    pres = P[torch.from_numpy(tr), u] > 0
    rows = E[torch.from_numpy(tr)][pres][:, u]
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
    E[:, u] = ((E[:, u] - mu) / sd) * (P[:, u, None, None] > 0)
W = E.shape[2]
az = (age_f - age_f[tr].mean()) / age_f[tr].std()
bins = np.stack([(age_f < 30), (age_f >= 30) & (age_f < 45), (age_f >= 45) & (age_f < 60), (age_f >= 60) & (age_f < 75), age_f >= 75], 1)
AGEX = torch.from_numpy(np.c_[az, az ** 2, bins, fem].astype(np.float32))
YOUNG = torch.from_numpy((age_f < 35).astype(np.float32))

folds = pd.read_csv(A0 + "oof_train.csv")[["study_id", "fold"]].set_index("study_id").fold
FOLD = folds.reindex(ids).values
assert not np.isnan(FOLD[tr]).any()


def fit(idx_tr, idx_es, idx_pred, Y, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    us = spec["units"]; Eu, Pu = E[:, us], P[:, us]
    m = Probe("late", len(us), W, E.shape[3], [list(range(len(us)))], None)
    agehead = nn.Linear(AGEX.shape[1], 1) if spec.get("trap") else None
    params = list(m.parameters()) + (list(agehead.parameters()) if agehead is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    Et, Pt, Yt = Eu[idx_tr], Pu[idx_tr], torch.from_numpy(Y[idx_tr].astype(np.float32))
    Xa, Yg = AGEX[idx_tr], YOUNG[idx_tr]
    hard = (Yt >= 0.5).float()
    npos = hard.sum(); nneg = len(Yt) - npos
    wpos = len(Yt) / (2 * npos.clamp(min=1)); wneg = len(Yt) / (2 * nneg.clamp(min=1))
    yes_hard = (Y[idx_es] >= 0.5).astype(float)
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Et))
        for i in range(0, len(Et), bs):
            j = perm[i:i + bs]; y = Yt[j]
            lg = m(Et[j], Pt[j])[:, 0]
            if agehead is not None:
                lg = lg + agehead(Xa[j])[:, 0]
            w = torch.where(y >= 0.5, wpos, wneg)
            if spec.get("hn"):
                w = w * torch.where((y < 0.5) & (Yg[j] > 0), torch.tensor(float(spec["hn"])), torch.tensor(1.0))
            loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Eu[idx_es], Pu[idx_es])[:, 0].numpy()            # image-only logit
        sc = auroc(yes_hard, pv)
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        return m(Eu[idx_pred], Pu[idx_pred])[:, 0].numpy()


SPECS = {
    "E3_late": dict(units=[0], seeds=[42, 43, 44]),
    "E3_late_colon_sb": dict(units=[0, 1], seeds=[42, 43, 44]),
    "E3cs_hn2": dict(units=[0, 1], hn=2, seeds=[42, 43]),
    "E3cs_hn3": dict(units=[0, 1], hn=3, seeds=[42, 43]),
    "E3cs_trap": dict(units=[0, 1], trap=True, seeds=[42, 43]),
    "E3cs_trap_hn2": dict(units=[0, 1], trap=True, hn=2, seeds=[42, 43]),
    "E3cs_soft05": dict(units=[0, 1], soft=True, seeds=[42]),
}
for tgt in a.targets.split(","):
    yv_all = lab[tgt].values.astype(float)
    if tgt == "y_ap_def":
        names = [n for n in SPECS]
    elif tgt == "y_ap":
        names = ["E3cs_hn2", "E3cs_hn3", "E3cs_trap", "E3cs_trap_hn2"]      # E3_late / colon_sb reused from analysis_appendicitis
    else:
        names = [n for n in SPECS if n != "E3cs_soft05"]
    if a.variants:
        names = [n for n in names if n in a.variants.split(",")]
    of, vf = O + f"oof_{tgt}.csv", O + f"val_{tgt}.csv"
    try:
        oof = pd.read_csv(of); val = pd.read_csv(vf)
    except FileNotFoundError:
        oof = pd.DataFrame({"study_id": [ids[i] for i in np.where(tr)[0]], "fold": FOLD[tr].astype(int), tgt: yv_all[tr]})
        val = pd.DataFrame({"study_id": [ids[i] for i in np.where(~tr)[0]], tgt: yv_all[~tr]})
    tri_all = np.where(tr)[0]; vai = np.where(~tr)[0]
    for name in names:
        if name in oof.columns:
            print(tgt, name, "exists, skip", flush=True); continue
        spec = SPECS[name]; t0 = time.time()
        Y = lab["y_ap_soft"].values.astype(float) if spec.get("soft") else yv_all
        trainable = tr & ~np.isnan(Y)
        col = np.full(len(tri_all), np.nan)
        for k in range(5):
            a_ = np.where(trainable & (FOLD != k))[0]                 # train rows of other folds, target not masked
            b_ = tri_all[FOLD[tri_all] == k]                          # predict the whole fold (masking at evaluation)
            a_ = np.random.default_rng(k).permutation(a_); nes = int(0.15 * len(a_))
            es, a2 = a_[:nes], a_[nes:]
            if spec.get("soft"):                                      # early stop on the definite masking
                es = es[~np.isnan(yv_all[es])]
            ps = [fit(a2, es, b_, Y, spec, sd) for sd in spec["seeds"]]
            col[np.searchsorted(tri_all, b_)] = np.mean(ps, 0)
        oof[name] = col
        a_ = np.random.default_rng(99).permutation(np.where(trainable)[0]); nes = int(0.15 * len(a_))
        es = a_[:nes]
        if spec.get("soft"):
            es = es[~np.isnan(yv_all[es])]
        val[name] = np.mean([fit(a_[nes:], es, vai, Y, spec, sd) for sd in spec["seeds"]], 0)
        m_tr = ~np.isnan(oof[tgt].values); m_va = ~np.isnan(val[tgt].values)
        print(f"{tgt} {name}: train OOF {auroc(oof[tgt].values[m_tr], oof[name].values[m_tr]):.4f} | val {auroc(val[tgt].values[m_va], val[name].values[m_va]):.4f} | {time.time()-t0:.0f}s", flush=True)
        oof.to_csv(of, index=False); val.to_csv(vf, index=False)
print("PROBE_DONE")
