"""Gallstones (formal best = Exp N 'N_U' = G5c features + GB/GB-fossa unit head gbn_U, val .881) for CG10k.
Inputs per study (same code as Merlin): gs_hu (hu), ag_feat (ag_*), gs1_feat (g1_*), chole_logit (= absent_gb E3_late head),
E3_late (gallstone head, gs_oof.py), g3b_livfb (gs3b_oof.py), gbx_* (gs5_probe.py on gs5 crops), gbn_U (nu_probe.py gs on
the Exp N unit crops with GS6 centring). Heads are refit on Merlin train by the ORIGINAL scripts with CG rows appended
(split 'cg', dummy y, never trained on). The saved stack_N_U.joblib (columns, medians, GBM) is applied unchanged.
steps: heads | g3b | gbx | gbn | stack <set>"""
import os, sys, re, subprocess, joblib, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; G = OF + "gallstones_improve/"; RU = OF + "region_units/"
B = H + "cg10k_best/"; R = B + "heads/R/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python"); W = B + "gs/"
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
os.makedirs(W, exist_ok=True)

def cg_rows(f, pre):
    d = pd.read_csv(f); d["study_id"] = pre + d.study_id.astype(str); return d

def run_patched(src, code_fn, argv, log):
    code = code_fn(open(src).read()); p = W + "patched_" + os.path.basename(src); open(p, "w").write(code)
    r = subprocess.run(["nice", "-n", "15", PY, p] + argv, stdout=open(W + log, "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, NT="4", OMP_NUM_THREADS="4"), cwd=H + "mw_exp"); print(os.path.basename(src), "exit", r.returncode)

def common(c):
    c = c.replace("/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/", R)
    c = c.replace('S == "val"', 'np.isin(S, ("val", "cg"))').replace('isin(["train", "val"])', 'isin(["train", "val", "cg"])')
    return c

def heads():
    """gs_oof.py: specs E0, E3_late only (shared rng: the E1 specs come after E3_late, so E3_late is unchanged)."""
    os.makedirs(R + "analysis_gallstones", exist_ok=True)
    lab = pd.read_csv(OF + "analysis_gallstones/labels_trval.csv")
    cg = [str(i) for i in np.load(R + MAT, mmap_mode="r")["ids"] if not str(i).startswith("AC")]
    add = pd.DataFrame({"study_id": cg, "split": "cg"})
    for c in lab.columns:
        if c.startswith("y_"): add[c] = 0.0
    pd.concat([lab, add], ignore_index=True).to_csv(R + "analysis_gallstones/labels_trval.csv", index=False)
    f = lambda c: common(c).replace('    specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}', "    pass")
    run_patched(OF + "integrated_best/scripts/analysis_code/gallstones/gs_oof.py", f, [], "gs_oof.log")
    v = pd.read_csv(R + "analysis_gallstones/scores_val_descriptive.csv"); o = pd.read_csv(OF + "analysis_gallstones/scores_val_descriptive.csv")
    m = o.merge(v, on="study_id", suffixes=("_o", "_n")); print("E3_late val max|d|", np.abs(m.E3_late_o - m.E3_late_n).max(), len(m))

def g3b():
    os.makedirs(R + "gallstones_improve", exist_ok=True)
    g1 = [pd.read_csv(G + "gs1_feat3mm.csv")] + [cg_rows(B + f"feat/{s}/gs1_feat.csv", p) for s, p in (("main", ""), ("delay", "D_"))]
    pd.concat(g1, ignore_index=True).to_csv(R + "gallstones_improve/gs1_feat3mm.csv", index=False)
    def f(c):
        c = common(c)
        return re.sub(r'(VIEWS = \{"g3b_base".*?\n.*?\n.*?\n.*?"g3b_gbliv_max".*?\}\n)', r'\1VIEWS = {"g3b_livfb": VIEWS["g3b_livfb"]}  # CG: only the stacked view (per-view rng)\n', c, flags=re.S)
    run_patched(G + "scripts/gs3b_oof.py", f, [W + "g3b"], "g3b.log")
    v = pd.read_csv(W + "g3b/gs3b_val.csv"); o = pd.read_csv(G + "gs3b/gs3b_val.csv"); m = o.merge(v, on="study_id", suffixes=("_o", "_n"))
    print("g3b_livfb val max|d|", np.abs(m.g3b_livfb_o - m.g3b_livfb_n).max(), len(m))

def gbx():
    ed = W + "gs5_emb/"; os.makedirs(ed, exist_ok=True)
    for src, pre in ((G + "gs5/emb/", ""), (B + "bemb/gs5/main/", ""), (B + "bemb/gs5/delay/", "D_")):
        for f in os.listdir(src):
            t = ed + pre + f
            if not os.path.exists(t): os.symlink(src + f, t)
    run_patched(G + "scripts/gs5_probe.py", common, [ed, W + "gbx"], "gbx.log")
    v = pd.read_csv(W + "gbx/gbx_val.csv"); o = pd.read_csv(G + "gs5/gbx_val.csv"); m = o.merge(v, on="study_id", suffixes=("_o", "_n"))
    for c in ("gbx_r0", "gbx_r1", "gbx_cat"): print(c, "val max|d|", np.abs(m[c + "_o"] - m[c + "_n"]).max(), len(m))

def stack(s):
    pre = "D_" if s == "delay" else ""
    M = joblib.load(RU + "gallstones/stack/stack_N_U.joblib")
    hu = pd.read_csv(B + f"feat/{s}/gs_hu.csv"); ag = pd.read_csv(B + f"feat/{s}/absent_gb_feat.csv").drop(columns=["gb_vox"])
    ag = ag.rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"}); g1 = pd.read_csv(B + f"feat/{s}/gs1_feat.csv")
    d = hu.merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left")
    d["study_id"] = pre + d.study_id.astype(str)
    ch = pd.read_csv(R + "analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
    e3 = pd.read_csv(R + "analysis_gallstones/scores_val_descriptive.csv")[["study_id", "E3_late"]]
    g3 = pd.read_csv(W + "g3b/gs3b_val.csv")[["study_id", "g3b_livfb"]]
    gx = pd.read_csv(W + "gbx/gbx_val.csv")[["study_id", "gbx_r0", "gbx_r1", "gbx_cat"]]
    gn = pd.read_csv(B + "bprobe/gs/out/val_scores.csv")[["study_id", "U"]].rename(columns={"U": "gbn_U"})
    for t in (ch, e3, g3, gx, gn): d = d.merge(t, on="study_id", how="left")
    sys.argv = [sys.argv[0], W + "_nu_dummy", "x", "--gbn", RU + "gallstones/out"]     # nu_gs_stack defs only (exec up to the data section)
    src = open(RU + "scripts/nu_gs_stack.py").read(); src = src[:src.index("# ---------------- data")]
    g = {}; exec(compile(src, "nu_gs_stack_defs", "exec"), g)
    g["EXTRA_COLS"].update({"gbx": ["gbx_r0", "gbx_r1", "gbx_cat"]})
    g["G1"] = [c for c in pd.read_csv(G + "gs1_feat3mm.csv", nrows=1).columns if c.startswith("g1_")]   # defined in the data section
    X = g["feats"](d.set_index("study_id"), M["sets"])[M["columns"]].fillna(M["medians"])
    sc = g["score"](M["model"], X.values)
    # Merlin val check: same frame recipe from the ORIGINAL Merlin files -> must equal stored val N_U
    I = OF
    hu0 = pd.concat([pd.read_csv(I + f"analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
    ag0 = pd.concat([pd.read_csv(I + f"analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id").drop(columns=["gb_vox"])
    ag0 = ag0.rename(columns={c: "ag_" + c for c in ag0.columns if c != "study_id"}); g10 = pd.read_csv(G + "gs1_feat3mm.csv").drop_duplicates("study_id")
    sv = pd.read_csv(RU + "gallstones/stack/val_scores.csv"); v = sv[["study_id"]]
    for t in (hu0, ag0, g10, pd.read_csv(I + "analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"}),
              pd.read_csv(I + "analysis_gallstones/scores_val_descriptive.csv")[["study_id", "E3_late"]], pd.read_csv(G + "gs3b/gs3b_val.csv")[["study_id", "g3b_livfb"]],
              pd.read_csv(G + "gs5/gbx_val.csv")[["study_id", "gbx_r0", "gbx_r1", "gbx_cat"]], pd.read_csv(RU + "gallstones/out/val_scores.csv")[["study_id", "U"]].rename(columns={"U": "gbn_U"})):
        v = v.merge(t, on="study_id", how="left")
    Xv = g["feats"](v.set_index("study_id"), M["sets"])[M["columns"]].fillna(M["medians"])
    print("Merlin val N_U max|d| vs stored", np.abs(g["score"](M["model"], Xv.values) - sv.set_index("study_id").loc[Xv.index, "N_U"].values).max())
    out = pd.DataFrame({"study_id": X.index.str.replace("^D_", "", regex=True), "gallstones": sc})
    out.to_csv(W + f"cg_scores_{s}.csv", index=False); print(s, len(out), "missing gbn", int(d.gbn_U.isna().sum()), "gbx", int(d.gbx_r0.isna().sum()))

if __name__ == "__main__":
    {"heads": heads, "g3b": g3b, "gbx": gbx}.get(sys.argv[1], lambda: stack(sys.argv[2]))()
