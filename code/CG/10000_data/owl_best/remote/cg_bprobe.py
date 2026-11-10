"""Tier B (region-crop units): refit the unit-probe heads on Merlin train with the ORIGINAL probe scripts and score Merlin val
(reproduction) + CG rows. Only the full-train refit is run (the 5-fold OOF loop is skipped: the refit uses its own
rng(0) permutation, independent of the folds, so the val/CG scores are identical to a full run).
usage: cg_bprobe.py prep <code> <cg_emb_dir_main> [<cg_emb_dir_delay>]   (code: hn fa asc ath ...)
       cg_bprobe.py run  <code> <script> <cg_emb_root> [--device cuda --vram-gb 6]
Merlin emb dir = region_units/<unit>/emb ; CG embs are symlinked next to them in a sandbox dir."""
import os, sys, re, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; RU = OF + "region_units/"
SB = H + "cg10k_best/bprobe/"; CGM = {"main": H + "cg10k_best/mat/cg_main.npz", "delay": H + "cg10k_best/mat/cg_delay.npz"}
UNITDIR = {"hn": "hydronephrosis", "fa": "free_air", "asc": "ascites", "ath": "atherosclerosis", "gs": "gallstones",
           "frp": "fracture_pelvis_r2", "frpp": "fracture_pelvis_r2"}
OUTSUB = {"frpp": "out_pel"}                      # stored Exp O pelvis-label head
PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")

def prep(code, embdirs):
    d = SB + code + "/"; os.makedirs(d + "emb", exist_ok=True)
    m = np.load(RU + f"probe_inputs/main_{code}.npz"); units = [str(u) for u in m["units"]]
    Es, Ps, Is = [m["E"]], [m["P"]], [m["ids"].astype("<U12")]
    for s, p in CGM.items():
        z = np.load(p); cu = [str(u) for u in z["units"]]; ui = [cu.index(u) for u in units]
        Es.append(z["E"][:, ui].astype(m["E"].dtype)); Ps.append(z["P"][:, ui].astype(m["P"].dtype))
        Is.append(np.array([("D_" if s == "delay" else "") + str(i) for i in z["ids"]], "<U12"))
    np.savez(d + "main.npz", E=np.concatenate(Es), P=np.concatenate(Ps), ids=np.concatenate(Is), units=m["units"])
    lab = pd.read_csv(RU + f"probe_inputs/labels_{code}.csv")
    cg = pd.DataFrame({"study_id": np.concatenate(Is[1:]), "split": "cg", "y": 0.0})
    pd.concat([lab, cg], ignore_index=True).to_csv(d + "labels.csv", index=False)
    src = RU + UNITDIR[code] + "/emb/"
    for f in os.listdir(src):                                        # Merlin train/val unit embeddings
        t = d + "emb/" + f
        if not os.path.exists(t): os.symlink(src + f, t)
    for s, ed in zip(("main", "delay"), [os.path.abspath(e) for e in embdirs]):
        for f in os.listdir(ed):
            t = d + "emb/" + ("D_" if s == "delay" else "") + f
            if not os.path.exists(t): os.symlink(os.path.join(ed, f), t)
    print(code, "prep", np.concatenate(Es).shape, len(os.listdir(d + "emb")))

def run(code, script, extra):
    d = SB + code + "/"
    code_txt = open(script).read()
    code_txt = code_txt.replace('assert set(lab.split) <= {"train", "val"}', 'assert set(lab.split) <= {"train", "val", "cg"}')
    code_txt = re.sub(r"for k in sorted\(np\.unique\(fid\)\):", "for k in []:  # CG: OOF skipped (refit is fold-independent)", code_txt)
    assert "for k in []:" in code_txt
    p = d + "patched_" + os.path.basename(script); open(p, "w").write(code_txt)
    env = dict(os.environ, MW_SRC=H + "mw_exp/src", NT="4", OMP_NUM_THREADS="4")
    r = subprocess.run(["nice", "-n", "15", PY, p, code, d + "emb", d + "main.npz", d + "labels.csv", d + "out"] + extra,
                       stdout=open(d + "probe.log", "w"), stderr=subprocess.STDOUT, env=env)
    print(code, "exit", r.returncode)
    v = pd.read_csv(d + "out/val_scores.csv"); o = pd.read_csv(RU + UNITDIR[code] + "/" + OUTSUB.get(code, "out") + "/val_scores.csv")
    m = o.merge(v, on="study_id", suffixes=("_old", "_new"))
    for c in [c for c in o.columns if c != "study_id" and c + "_new" in m]:
        print(f"  {c}: n {len(m)} max|d| {np.abs(m[c+'_old']-m[c+'_new']).max():.4f} corr {np.corrcoef(m[c+'_old'], m[c+'_new'])[0,1]:.5f}")

if __name__ == "__main__":
    if sys.argv[1] == "prep": prep(sys.argv[2], sys.argv[3:])
    else: run(sys.argv[2], sys.argv[3], sys.argv[4:])
