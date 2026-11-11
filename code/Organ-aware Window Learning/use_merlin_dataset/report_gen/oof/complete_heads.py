"""Complete the report-gen OWL score table, step 1: probe heads for studies whose label is 'unclear' (no stored head score).
Re-runs the ORIGINAL integrated_best analysis head scripts (same seeds/folds/recipe, trained on Merlin train only) in a
sandbox and additionally scores the missing studies with the FULL-TRAIN refit head (the same model that produced the stored
val head scores). Test is never read (labels_trval / notest matrix only). Nothing under mw_exp or cg10k_best is modified.

Two ways of adding the missing studies (chosen per script so that training rows / standardisation stay bit-identical):
  A  ('cg' split) - scripts that DROP unclear rows and standardise with a fixed stats file (hs, hs_cd, sp):
     unclear rows get split 'cg' + dummy label 0 and are scored together with val (never trained on).
  C  (prediction wrapper) - scripts that keep unclear train rows for train-row standardisation (hn, asc, op, bd):
     data/labels unchanged; fit() is wrapped so that the full-train refit call (idx_pred is vai) also predicts the missing
     rows; their scores are pickled (per spec, per seed) and averaged over seeds exactly as the script averages val.
FASTFOLD=1: inside the 5-fold OOF loop fit() runs 1 epoch only (the OOF columns are then meaningless and are not used);
the refit is unchanged because its rng permutation is drawn after the fold permutations, which do not depend on fit()
results, and every fit() reseeds torch. The stored val head scores must be reproduced (checked in 'check').
usage (A6000, radar env):  complete_heads.py prep | run <job> [...] | check"""
import os, sys, re, glob, pickle, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/")
OF = H + "mw_exp/outputs_final/"; SRC = H + "mw_exp/src"; AC = OF + "integrated_best/scripts/analysis_code/"
SB = H + "report_gen/oof/complete/heads/"; R = SB + "R/"
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
MISS = H + "report_gen/oof/complete/missing_cells.csv"
FD = {"hs": "hepatic_steatosis", "hs_cd": "hepatic_steatosis", "sp": "splenomegaly", "hn": "hydronephrosis", "asc": "ascites",
      "op": "osteopenia", "bd": "biliary_ductal_dilation"}
MODE = {"hs": "A", "hs_cd": "A", "sp": "A", "hn": "C", "asc": "C", "op": "C", "bd": "C"}
JOBS = {  # job -> (script, argv, sandbox analysis dir, stored val file)
    "hs": ("steatosis/hs_oof.py", [SRC, R + "analysis_steatosis"], "analysis_steatosis", "scores_val_descriptive.csv"),
    "hs_cd": ("steatosis/hs_oof_spleen.py", [SRC, R + "analysis_steatosis", "concat_diff"], "analysis_steatosis", "scores_val_concat_diff.csv"),
    "sp": ("splenomegaly/sp_oof.py", [SRC, R + "analysis_splenomegaly"], "analysis_splenomegaly", "scores_val.csv"),
    "hn": ("hydronephrosis/hn_oof.py", [R + "analysis_hydronephrosis/", SRC, "E3_late", "E3_late_max"], "analysis_hydronephrosis", "scores_val_descriptive.csv"),
    "asc": ("ascites/asc_oof.py", ["E3_late", "E3_late_all12"], "analysis_ascites", "scores_val_descriptive.csv"),
    "op": ("osteopenia/op_oof.py", [], "analysis_osteopenia", "scores_val_descriptive.csv"),
    "bd": ("biliary/bd_oof.py", ["E0", "E3_late", "E3late_cat_+gb_duod"], "analysis_biliary", "scores_val_descriptive.csv"),
}
HEADS = {"hs": ["E3_late"], "hs_cd": ["E3_late_concat_diff"], "sp": ["E3_late"], "hn": ["E3_late", "E3_late_max"],
         "asc": ["E3_late", "E3_late_all12"], "op": ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"],
         "bd": ["E0", "E3_late", "E3late_cat_+gb_duod"]}
TARGET = {"analysis_steatosis": "y_hs", "analysis_splenomegaly": "y_sp"}


def ln(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst): os.symlink(src, dst)


def lab_cg(src, dst, tg):
    """mode A: unclear rows (target NaN) -> split 'cg', dummy label 0 on every NaN y_ column (never trained on)."""
    lab = pd.read_csv(src); assert set(lab.split) <= {"train", "val"}
    u = lab[tg].isna()
    lab.loc[u, "split"] = "cg"
    for c in [c for c in lab.columns if c.startswith("y_")]: lab.loc[u, c] = lab.loc[u, c].fillna(0.0)
    lab.to_csv(dst, index=False); print(dst, "cg rows", int(u.sum()))


def prep():
    ln(OF + MAT, R + MAT)
    M = pd.read_csv(MISS)
    for j, f in FD.items():                                         # missing study ids per job (train unclear + val missing)
        open(SB + f"xids_{j}.txt", "w").write("\n".join(M[M.disease == f].study_id))
    for a, sub, files in (("steatosis", "data/", ("liver_emb_trval.npz", "spleen_emb_trval.npz", "std_stats_main12.npz")),
                          ("splenomegaly", "data/", ("spleen_emb_trval.npz", "std_stats_main12.npz"))):
        srcd = OF + f"analysis_{a}/" + ("" if a == "steatosis" else "data/")
        for f in files: ln(srcd + f, R + f"analysis_{a}/{sub}{f}")
        lab_cg(srcd + "labels_trval.csv", R + f"analysis_{a}/{sub}labels_trval.csv", TARGET[f"analysis_{a}"])
    ln(OF + "analysis_hydronephrosis/kidney_emb_trval.npz", R + "analysis_hydronephrosis/kidney_emb_trval.npz")
    ln(OF + "analysis_osteopenia/emb_lumbar_L1_trval.npz", R + "analysis_osteopenia/emb_lumbar_L1_trval.npz")
    for a in ("hydronephrosis", "ascites", "osteopenia", "biliary"):
        ln(OF + f"analysis_{a}/labels_trval.csv", R + f"analysis_{a}/labels_trval.csv")      # mode C: labels unchanged
    ln(SRC + "/probe.py", R + "analysis_ascites/scripts/probe.py")
    if not os.path.exists(R + "analysis_ascites/emb12_trval.npz"):                          # same slice as asc_dump_emb / cg_heads
        z = np.load(OF + MAT); units = list(z["units"]); ui = list(range(12))
        np.savez(R + "analysis_ascites/emb12_trval.npz", E=z["E"][:, ui].astype(np.float16), P=z["P"][:, ui], ids=z["ids"],
                 units=np.array(units[:12]))
    print("prep done", flush=True)


WRAP = '''
import inspect as _insp, pickle as _pk, os as _os, atexit as _ax
_XIDS = set(open(_os.environ["XIDS"]).read().split()); _REC = []
_ax.register(lambda: _pk.dump(_REC, open(_os.environ["XOUT"], "wb")))
def _nm(spec):
    for n, v in specs.items():
        if v is spec or (isinstance(v, tuple) and any(x is spec for x in v)): return n
def _cut(r, n):
    if isinstance(r, np.ndarray): return r[:n], r[n:]
    if isinstance(r, tuple): a, b = _cut(r[0], n); return (a,) + tuple(r[1:]), b
    if isinstance(r, dict):
        o, x = dict(r), {}
        for k, v in r.items():
            if k != "gate" and isinstance(v, np.ndarray) and v.ndim >= 1: o[k], x[k] = v[:n], v[n:]
        return o, x
    raise TypeError(type(r))
def fit(*a, **k):
    b = _insp.signature(_fit_orig).bind(*a, **k)
    if b.arguments["idx_pred"] is vai:                       # full-train refit call -> also predict the missing rows
        vs = set(np.asarray(vai).tolist())
        xi = np.array([i for i, s in enumerate(ids) if s in _XIDS and i not in vs], dtype=np.asarray(vai).dtype)
        if len(xi) == 0: return _fit_orig(*b.args, **b.kwargs)
        b.arguments["idx_pred"] = np.concatenate([np.asarray(vai), xi])
        r, x = _cut(_fit_orig(*b.args, **b.kwargs), len(vai))
        _REC.append(dict(name=_nm(b.arguments["spec"]), seed=b.arguments["seed"], ids=[ids[i] for i in xi], x=x))
        return r
    if _os.environ.get("FASTFOLD") == "1": b.arguments["max_ep"] = 1       # OOF fold fits (unused)
    return _fit_orig(*b.args, **b.kwargs)
'''


def patch(code, j):
    code = code.replace("/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/", R)
    code = code.replace('sys.path.insert(0, A + "scripts")', f'sys.path.insert(0, "{SRC}")')
    if MODE[j] == "A":
        code = code.replace('S == "val"', 'np.isin(S, ("val", "cg"))')
    code = code.replace('specs["L1unit_E0"] = (1, {"mode": "single", "win_idx": 0}); specs["L1unit_E3_late"] = (1, {"mode": "late"})', "")  # op: unused trailing specs (as cg_heads)
    assert code.count("\ndef fit(") == 1, j
    code = code.replace("\ndef fit(", WRAP + "\n\ndef _fit_orig(")
    return code


def run(jobs):
    os.makedirs(SB + "patched", exist_ok=True); os.makedirs(SB + "logs", exist_ok=True)
    for j in jobs:
        sc, argv = JOBS[j][0], JOBS[j][1]
        p = SB + "patched/" + j + "_" + os.path.basename(sc); open(p, "w").write(patch(open(AC + sc).read(), j))
        env = dict(os.environ, ADIR=R + "analysis_ascites/", OP_DIR=R + "analysis_osteopenia/", OMP_NUM_THREADS="6", NT="6",
                   XIDS=SB + f"xids_{j}.txt", XOUT=SB + f"x_{j}.pkl", FASTFOLD="1", CUDA_VISIBLE_DEVICES="")
        print(j, "start", flush=True)
        r = subprocess.run(["nice", "-n", "19", PY, p] + argv, stdout=open(SB + f"logs/{j}.log", "w"), stderr=subprocess.STDOUT, env=env)
        print(j, "exit", r.returncode, flush=True)


def check():
    """val reproduction vs the stored head scores + export heads of the missing studies -> heads/out/<job>.csv"""
    os.makedirs(SB + "out", exist_ok=True); rows = []
    for j, (sc, argv, d, fv) in JOBS.items():
        new = R + d + "/" + fv
        if not os.path.exists(new): print(j, "no output"); continue
        n = pd.read_csv(new); o = pd.read_csv(OF + d + "/" + fv)
        m = o.merge(n, on="study_id", suffixes=("_old", "_new"))
        for h in HEADS[j]:
            df = (m[h + "_old"] - m[h + "_new"]).abs()
            rows.append(dict(job=j, head=h, n_val_stored=len(o), n_common=len(m), max_abs_diff=df.max()))
        xids = set(open(SB + f"xids_{j}.txt").read().split())
        if MODE[j] == "A":
            x = n[n.study_id.isin(xids)][["study_id"] + HEADS[j]]
        else:
            rec = pickle.load(open(SB + f"x_{j}.pkl", "rb")); parts = []
            for h in HEADS[j]:
                rs = [r for r in rec if r["name"] == h]; assert rs, (j, h)
                assert all(r["ids"] == rs[0]["ids"] for r in rs)
                v = np.mean([r["x"]["s"] if isinstance(r["x"], dict) else r["x"] for r in rs], 0)
                parts.append(pd.DataFrame({"study_id": rs[0]["ids"], h: v}).set_index("study_id"))
            x = pd.concat(parts, axis=1).reset_index()
            x = pd.concat([x, n[n.study_id.isin(xids - set(x.study_id))][["study_id"] + HEADS[j]]])   # (val rows already scored)
        x.to_csv(SB + f"out/{j}.csv", index=False)
        print(j, "missing studies", len(xids), "scored", len(x), "with NaN", int(x[HEADS[j]].isna().any(axis=1).sum()))
    r = pd.DataFrame(rows); print(r.to_string()); r.to_csv(SB + "heads_check.csv", index=False)


if __name__ == "__main__":
    {"prep": lambda: prep(), "run": lambda: run(sys.argv[2:]), "check": lambda: check()}[sys.argv[1]]()
