"""Tier B biliary (BD5 = 'cat4+E3_late+E0+porta_emb+HU+tube+porta|gbm', val .911).
step 'bd5'  : refit bd5_probe.py (porta crop heads P5_porta / P5_cat4_porta; full original run incl. OOF because the refit
              permutation follows the fold permutations) with CG rows added (split 'cg', dummy y=0, never trained on).
step 'stack': run the ORIGINAL bd_stack.py on a sandbox WORK dir = biliary_improve/work + CG rows appended as split 'val'
              (heads from cg_heads.py 'bd' + bd5 refit, features from bd3_feat.py on CG). Original Merlin val rows are kept
              as stored, so the val output must equal stack_bd5/stack_val.csv; the CG rows of stack_val.csv are the CG scores.
usage: cg_biliary.py bd5 <cg_porta_emb_main> | stack <set>"""
import os, sys, re, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; BI = OF + "biliary_improve/"
B = H + "cg10k_best/"; SB = B + "bprobe/bd5/"; HR = B + "heads/R/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
COL = "cat4+E3_late+E0+porta_emb+HU+tube+porta|gbm"

def bd5(embdir, embdir_delay=None):
    os.makedirs(SB + "emb", exist_ok=True)
    for f in os.listdir(BI + "emb_porta"):
        t = SB + "emb/" + f
        if not os.path.exists(t): os.symlink(BI + "emb_porta/" + f, t)
    for pre, e in (("", embdir), ("D_", embdir_delay)):
        if not e: continue
        ed = os.path.abspath(e)
        for f in os.listdir(ed):
            t = SB + "emb/" + pre + f
            if not os.path.exists(t): os.symlink(os.path.join(ed, f), t)
    code = open(BI + "scripts/bd5_probe.py").read()
    code = code.replace('lab = lab[lab.split.isin(["train", "val"])]', 'lab = lab[lab.split.isin(["train", "val", "cg"])]')
    code = code.replace('vai = np.where(S == "val")[0]', 'vai = np.where(np.isin(S, ("val", "cg")))[0]')
    assert '"cg"' in code
    open(SB + "patched_bd5_probe.py", "w").write(code)
    os.makedirs(SB + "out", exist_ok=True)
    os.system(f"cp {BI}scripts/probe.py {BI}scripts/common.py {SB} 2>/dev/null")
    r = subprocess.run(["nice", "-n", "15", PY, SB + "patched_bd5_probe.py", HR + MAT, SB + "emb", HR + "analysis_biliary/labels_trval.csv",
                        OF + "analysis_biliary/oof_train.csv", SB + "out"], stdout=open(SB + "probe.log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, NT="4", OMP_NUM_THREADS="4"))
    print("bd5 exit", r.returncode)
    v = pd.read_csv(SB + "out/bd5_val.csv"); o = pd.read_csv(BI + "out/bd5_val.csv"); m = o.merge(v, on="study_id", suffixes=("_o", "_n"))
    for c in [c for c in o.columns if c != "study_id"]:
        print(c, len(m), "max|d|", np.abs(m[c + "_o"] - m[c + "_n"]).max())

def stack(s):
    W = B + f"bprobe/bd_work_{s}/"; O = B + f"bprobe/bd_stack_{s}/"; os.makedirs(W, exist_ok=True)
    os.system(f"cp {BI}work/*.csv {W}")
    pre = "D_" if s == "delay" else ""
    hd = pd.read_csv(HR + "analysis_biliary/scores_val_descriptive.csv"); hd = hd[hd.study_id.str.startswith(pre + "EX")] if pre else hd[hd.study_id.str.startswith("EX")]
    b5 = pd.read_csv(SB + "out/bd5_val.csv"); b5 = b5[b5.study_id.isin(hd.study_id)]
    ids = hd.study_id.str.replace("^D_", "", regex=True); hd = hd.assign(study_id=ids); b5 = b5.assign(study_id=b5.study_id.str.replace("^D_", "", regex=True))
    lab = pd.read_csv(W + "labels_trval.csv"); add = pd.DataFrame({"study_id": ids, "split": "val", "y_bdd": 0.0})
    pd.concat([lab, add], ignore_index=True).to_csv(W + "labels_trval.csv", index=False)
    for f, d in (("scores_val_descriptive.csv", hd), ("bd5_val.csv", b5)):
        x = pd.read_csv(W + f); pd.concat([x, d[[c for c in d.columns if c in x.columns]]], ignore_index=True).to_csv(W + f, index=False)
    F = pd.read_csv(B + f"feat/{s}/bd3_feat.csv")
    for f in ("hu_all.csv", "bd3_all.csv"):
        x = pd.read_csv(W + f); pd.concat([x, F[[c for c in x.columns if c in F.columns]]], ignore_index=True).to_csv(W + f, index=False)
    # only the formally selected configuration (bd_stack.py's own ONLY filter); the others were model-selection candidates
    r = subprocess.run(["nice", "-n", "15", PY, BI + "scripts/bd_stack.py", W, O], stdout=open(O.rstrip("/") + ".log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, ONLY="cat4+E3_late+E0+porta_emb+HU+tube+porta"))
    print("stack exit", r.returncode)
    out = pd.read_csv(O + "stack_val.csv"); ref = pd.read_csv(BI + "stack_bd5/stack_val.csv")
    m = ref[["study_id", COL]].merge(out[["study_id", COL]], on="study_id", suffixes=("_o", "_n"))
    print("val rows", len(m), "max|d|", np.abs(m[COL + "_o"] - m[COL + "_n"]).max())
    c = out[out.study_id.isin(ids)][["study_id", COL]].rename(columns={COL: "biliary_ductal_dilation"})
    c.to_csv(B + f"bprobe/cg_scores_biliary_{s}.csv", index=False); print("CG", len(c))

if __name__ == "__main__":
    bd5(*sys.argv[2:4]) if sys.argv[1] == "bd5" else stack(sys.argv[2])
