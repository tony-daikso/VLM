"""Ingest findings whose stacks were already built with train-OOF model selection (gallstones GS1/GS4/GS5,
biliary BD1-BD5, fracture Exp I / Exp J). Among all variants the one with the highest train OOF AUROC is taken
(val never used for the choice); metrics are recomputed with ib_core.evaluate on the official labels.
usage: ib_external.py <ib_root> gallstones|biliary_ductal_dilation|fracture"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_core import evaluate, auc, sens_at_spec

R = sys.argv[1].rstrip("/") + "/"
lab = pd.read_csv(R + "base/labels14_trval.csv"); age = pd.read_csv(R + "base/age_trval.csv")
exb = pd.read_csv(R + "base/expB_E3late_val.csv")


def finish(fd, T, V, var_oof, chosen, src, notes, e3_oof_col="E3_late", e3_val_rerun=None, fdlab=None):
    """T: train frame (study_id, y, E3 OOF, variant OOF cols); V: val frame (study_id, y, variant cols)."""
    fl = fdlab or fd
    T = T.merge(age, on="study_id", how="left"); V = V.merge(exb[["study_id", fl]].rename(columns={fl: "expB_E3"}), on="study_id").merge(age, on="study_id", how="left")
    _, thr = sens_at_spec(T.y.values.astype(int), T[chosen].values)
    r = evaluate(fd, T.y.values, T[chosen].values, T[e3_oof_col].values, T.Age.values, V.y.values, V[chosen].values,
                 V.expB_E3.values, V[e3_val_rerun].values if e3_val_rerun else None, V.Age.values, thr_oof=thr)
    r.update(featset=chosen, cand=src, cv=var_oof, notes=notes)
    os.makedirs(R + "scores", exist_ok=True); os.makedirs(R + "results", exist_ok=True)
    pd.DataFrame({"study_id": T.study_id, "y": T.y.astype(int), "E3_late_oof": T[e3_oof_col], "stack_oof": T[chosen]}).to_csv(R + f"scores/{fd}_oof.csv", index=False)
    pd.DataFrame({"study_id": V.study_id, "y": V.y.astype(int), "expB_E3_late": V.expB_E3, "stack": V[chosen]}).to_csv(R + f"scores/{fd}_val.csv", index=False)
    json.dump(r, open(R + f"results/{fd}.json", "w"), indent=1, default=float)
    print(fd, chosen, f"OOF {r['oof']:.4f} val {r['val']:.4f} {r['val_ci']} expB {r['val_E3_expB']:.4f} d {r['d_val_expB']}", flush=True)


def official(fd, split):
    l = lab[lab.split == split][["study_id", fd]].rename(columns={fd: "y"}); return l[l.y.notna()]


def gallstones():
    fd = "gallstones"; G = R + "gs/"
    files = [("gs1/gs1_oof_train.csv", "gs1/gs1_val_scores.csv", ["quickwins", "V0", "V1", "V2", "V3", "V4"]),
             ("gs4/oof_train.csv", "gs4/val_scores.csv", ["G4", "G3b", "G4_3b"]),
             ("gs4/noage/oof_train.csv", "gs4/noage/val_scores.csv", ["G4na"]),
             ("gs5/stack/oof_train.csv", "gs5/stack/val_scores.csv", ["G5", "G5b", "G5c"])]
    T = official(fd, "train"); V = official(fd, "val"); first = True
    for fo, fv, cols in files:
        o = pd.read_csv(G + fo); v = pd.read_csv(G + fv)
        keep = cols + (["E3_late"] if first else [])
        T = T.merge(o[["study_id"] + keep], on="study_id", how="inner"); V = V.merge(v[["study_id"] + keep], on="study_id", how="inner")
        first = False
    V = V.rename(columns={"E3_late": "E3_late_rerun"})
    var = {c: auc(T.y, T[c]) for c in sum([f[2] for f in files], [])}
    ch = max(var, key=var.get)
    finish(fd, T, V, var, ch, "gallstones_improve (gs_stack2/gs1_stack, same 7-candidate grid, chosen by train OOF)",
           "GS2 native-resolution not scaled up; GS6 not available", e3_val_rerun="E3_late_rerun")


def biliary():
    fd = "biliary_ductal_dilation"; B = R + "bd/"
    T = official(fd, "train"); V = official(fd, "val"); var = {}; src = {}
    for d in ["stack_bd1", "stack_full", "stack_noart", "stack_bd5"]:
        if not os.path.exists(B + d + "/stack_oof_train.csv"): continue
        o = pd.read_csv(B + d + "/stack_oof_train.csv"); v = pd.read_csv(B + d + "/stack_val.csv")
        for c in [c for c in o.columns if "|" in c and c in v.columns]:
            k = f"{d}:{c}"
            if k in var: continue
            m = T[["study_id", "y"]].merge(o[["study_id", c]], on="study_id")
            if len(m) < 0.99 * len(T): continue
            var[k] = auc(m.y, m[c]); src[k] = (o[["study_id", c, "E3_late"]], v[["study_id", c, "E3_late"]])
    ch = max(var, key=var.get); o, v = src[ch]; c = ch.split(":", 1)[1]
    T = T.merge(o.rename(columns={c: ch}), on="study_id"); V = V.merge(v.rename(columns={c: ch, "E3_late": "E3_late_rerun"}), on="study_id")
    finish(fd, T, V, var, ch, "biliary_improve bd_stack.py (LR/GBM, chosen by train OOF)", "", e3_val_rerun="E3_late_rerun")


def fracture():
    fd = "fracture"; T = official(fd, "train"); V = official(fd, "val"); var = {}; tabs = {}
    z = np.load(R + "fx/ens_greedy_all.npz", allow_pickle=True)
    tabs["expI_ens"] = pd.DataFrame({"study_id": z["ids"], "split": z["split"], "s": z["logit"]})
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        z = np.load(R + f"fx/J/{s}/scores.npz", allow_pickle=True)
        tabs[f"J_{s}"] = pd.DataFrame({"study_id": z["ids"], "split": z["split"], "s": z["stack"]})
        if s == "S1":
            tabs["J_add_member"] = pd.DataFrame({"study_id": z["ids"], "split": z["split"], "s": z["m_J_add"]})
            e3 = pd.DataFrame({"study_id": z["ids"], "E3_late": z["m_e3_oof"]})
    for k, t in tabs.items():
        m = T.merge(t[t.split == "train"], on="study_id"); var[k] = auc(m.y, m.s)
    # coordinator decision (2026-11-06): fracture row = Exp J add 10-seed (train OOF .853; the S3/S4 stackers have a
    # marginally higher stacker-CV .855 but lower val); Exp I ensemble kept as the 3 mm-only deployable alternative.
    T0, V0 = T, V
    for key, ch in (("fracture", "J_add_member"), ("fracture_expI", "expI_ens")):
        t = tabs[ch]
        T = T0.merge(t[t.split == "train"][["study_id", "s"]].rename(columns={"s": ch}), on="study_id").merge(e3, on="study_id")
        V = V0.merge(t[t.split == "val"][["study_id", "s"]].rename(columns={"s": ch}), on="study_id")
        finish(key if key == "fracture" else key, T, V, var, ch, "outputs_vert15 final_eval (Exp J add 10 seed) / outputs_fracture expI",
               "E3_late OOF = bone E3_late OOF (m_e3_oof) of Exp J final_eval; max train-OOF variant = " + max(var, key=var.get), fdlab="fracture")


if __name__ == "__main__":
    for a in sys.argv[2:]: {"gallstones": gallstones, "biliary_ductal_dilation": biliary, "fracture": fracture}[a]()
