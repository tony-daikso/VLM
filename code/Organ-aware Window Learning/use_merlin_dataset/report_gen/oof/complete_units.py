"""Complete the report-gen OWL score table, step 2: region-unit / crop probe heads for the 'unclear' studies.
ORIGINAL probe scripts, trained on Merlin train only, run in a sandbox (report_gen/oof/complete/units/<job>/); the missing
studies are scored by the FULL-TRAIN refit (same model as the stored val scores). Test never read; mw_exp / cg10k_best untouched.
  ru/nu probes (hn fa asc ath gs) : script drops y-NaN rows before standardising -> missing studies appended / relabelled as
        split 'cg' (dummy y 0, never trained on) and scored with val; 5-fold OOF loop skipped (refit uses its own rng(0)
        permutation, fold-independent - as cg10k_best/cg_bprobe.py).
  rlq (appendicitis RLQ unit)      : same ('cg' rows, OOF loop skipped; refit rng(0) is fold-independent).
  gbx (gallstones gs5 crop probe)  : drops y-NaN rows -> 'cg' rows; refit permutation follows the fold permutations, so the
        fold loop is kept but its fits run 1 epoch (OOF unused; rng consumption unchanged).
  bd5 (biliary porta probe)        : standardises on ALL train rows (incl. y-NaN) -> labels unchanged; the refit call
        predicts vai + missing rows (pickled); fold fits 1 epoch.
usage: complete_units.py run <job> [...]  |  check"""
import os, sys, re, pickle, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; RU = OF + "region_units/"
C = H + "report_gen/oof/complete/"; SB = C + "units/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
M = pd.read_csv(C + "missing_cells.csv")
UNIT = {"hn": ("hydronephrosis", "hydronephrosis", "ru_probe.py"), "fa": ("free_air", "free_air", "ru_probe.py"),
        "asc": ("ascites", "ascites", "ru_probe.py"), "ath": ("atherosclerosis", "atherosclerosis", "nu_probe.py"),
        "gs": ("gallstones", "gallstones", "nu_probe.py")}
DEV = ["--device", "cuda", "--vram-gb", "5.5"]


def xids(fd): return list(M[M.disease == fd].study_id)


def run_py(p, argv, log, env=None, cwd=None):
    e = dict(os.environ, MW_SRC=H + "mw_exp/src", NT="6", OMP_NUM_THREADS="6", **(env or {}))
    r = subprocess.run(["nice", "-n", "19", PY, p] + argv, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=e, cwd=cwd)
    print(os.path.basename(p), "exit", r.returncode, flush=True); return r.returncode


def probe(code):
    unit, fd, script = UNIT[code]; d = SB + code + "/"; os.makedirs(d, exist_ok=True)
    lab = pd.read_csv(RU + f"probe_inputs/labels_{code}.csv"); assert set(lab.split) <= {"train", "val"}
    x = set(xids(fd)); have = lab.study_id.isin(x)
    lab.loc[have, "split"] = "cg"; lab.loc[have, "y"] = 0.0                       # unclear rows already listed (y NaN)
    add = pd.DataFrame({"study_id": sorted(x - set(lab.study_id)), "split": "cg", "y": 0.0})
    pd.concat([lab, add], ignore_index=True).to_csv(d + "labels.csv", index=False)
    print(code, "cg rows", int(have.sum()) + len(add), flush=True)
    code_txt = open(RU + "scripts/" + script).read()
    code_txt = code_txt.replace('assert set(lab.split) <= {"train", "val"}', 'assert set(lab.split) <= {"train", "val", "cg"}')
    code_txt = re.sub(r"for k in sorted\(np\.unique\(fid\)\):", "for k in []:  # OOF skipped (refit is fold-independent)", code_txt)
    assert "for k in []:" in code_txt and '"cg"}' in code_txt
    p = d + "patched_" + script; open(p, "w").write(code_txt)
    if os.path.exists(d + "out/val_scores.csv"): return
    rc = run_py(p, [code, RU + unit + "/emb", RU + f"probe_inputs/main_{code}.npz", d + "labels.csv", d + "out"] + DEV, d + "probe.log")
    assert rc == 0


def rlq():
    U = OF + "appendicitis_rlq_unit/"; d = SB + "rlq/"; os.makedirs(d, exist_ok=True)
    lab = pd.read_csv(OF + "appendicitis_split/labels_trval_split.csv"); assert set(lab.split) <= {"train", "val"}
    u = lab.study_id.isin(set(xids("appendicitis"))); assert lab.loc[u, "y_ap"].isna().all()
    lab.loc[u, "split"] = "cg"
    for c in ("y_ap", "y_ap_def", "y_ap_unc"): lab.loc[u, c] = 0.0
    lab.to_csv(d + "labels.csv", index=False); print("rlq cg rows", int(u.sum()))
    code = open(U + "scripts/rlq_probe.py").read()
    code = code.replace('R + "outputs_final/appendicitis_split/labels_trval_split.csv"', f'"{d}labels.csv"')
    code = code.replace('assert set(lab.split) <= {"train", "val"}', 'assert set(lab.split) <= {"train", "val", "cg"}')
    code = code.replace('vai_all = np.where(S == "val")[0]', 'vai_all = np.where(np.isin(S, ("val", "cg")))[0]')
    code = code.replace("for k in range(5):", "for k in []:  # OOF skipped (refit rng(0) is fold-independent)")
    assert f"{d}labels.csv" in code and "for k in []" in code and '("val", "cg")' in code
    open(d + "patched_rlq_probe.py", "w").write(code)
    if not os.path.exists(d + "out/val_scores.csv"):
        assert run_py(d + "patched_rlq_probe.py", [U + "emb_rlq", d + "out", "--device", "cuda", "--targets", "y_ap",
                                                   "--variants", "E3_late,E3cs,U_rlq_cs"], d + "probe.log") == 0


def gbx():
    G = OF + "gallstones_improve/"; d = SB + "gbx/"; R2 = d + "R/"; os.makedirs(R2 + "analysis_gallstones", exist_ok=True)
    lab = pd.read_csv(OF + "analysis_gallstones/labels_trval.csv"); assert set(lab.split) <= {"train", "val"}
    u = lab.y_gs.isna(); assert set(lab.study_id[u]) == set(xids("gallstones"))
    lab.loc[u, "split"] = "cg"
    for c in [c for c in lab.columns if c.startswith("y_")]: lab.loc[u, c] = lab.loc[u, c].fillna(0.0)
    lab.to_csv(R2 + "analysis_gallstones/labels_trval.csv", index=False)
    t = R2 + "analysis_gallstones/oof_train.csv"
    if not os.path.lexists(t): os.symlink(OF + "analysis_gallstones/oof_train.csv", t)
    code = open(G + "scripts/gs5_probe.py").read()
    code = code.replace('R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"', f'R = "{R2}"')
    code = code.replace('S == "val"', 'np.isin(S, ("val", "cg"))').replace('isin(["train", "val"])', 'isin(["train", "val", "cg"])')
    n0 = code.count("tri[b], sd) for sd in SEEDS]")
    code = code.replace("tri[b], sd) for sd in SEEDS]", "tri[b], sd, max_ep=1) for sd in SEEDS]")   # OOF fits: 1 epoch (unused)
    assert n0 == 1 and f'R = "{R2}"' in code
    open(d + "patched_gs5_probe.py", "w").write(code)
    if not os.path.exists(d + "out/gbx_val.csv"):
        assert run_py(d + "patched_gs5_probe.py", [G + "gs5/emb", d + "out"], d + "probe.log", cwd=H + "mw_exp") == 0


def bd5():
    BI = OF + "biliary_improve/"; d = SB + "bd5/"; os.makedirs(d + "out", exist_ok=True)
    open(d + "xids.txt", "w").write("\n".join(xids("biliary_ductal_dilation")))
    code = open(BI + "scripts/bd5_probe.py").read()
    a = 'tri[b], s) for s in (42, 43)], 0)'
    b = 'vv = np.mean([fit(X, Pm, tri[a[nes:]], tri[a[:nes]], vai, s) for s in (42, 43)], 0)'
    assert code.count(a) == 1 and code.count(b) == 1
    code = code.replace(a, 'tri[b], s, max_ep=1) for s in (42, 43)], 0)')                    # OOF fits: 1 epoch (unused)
    code = code.replace(b, "_XI = np.array([i for i, s_ in enumerate(ids) if s_ in _XS and i not in set(vai.tolist())], dtype=vai.dtype)\n"
                           "    _o = np.mean([fit(X, Pm, tri[a[nes:]], tri[a[:nes]], np.concatenate([vai, _XI]), s) for s in (42, 43)], 0)\n"
                           "    vv = _o[:len(vai)]; _REC[name] = pd.Series(_o[len(vai):], index=[ids[i] for i in _XI])\n"
                           f"    pickle.dump(_REC, open('{d}x_bd5.pkl', 'wb'))")
    code = code.replace("\nEMB, PD, LAB, REF, OUT = sys.argv[1:6]",
                        f"\nimport pickle\n_XS = set(open('{d}xids.txt').read().split()); _REC = {{}}\nEMB, PD, LAB, REF, OUT = sys.argv[1:6]")
    assert "_REC = {}" in code
    for f in ("probe.py", "common.py"):
        if os.path.exists(BI + "scripts/" + f) and not os.path.lexists(d + f): os.symlink(BI + "scripts/" + f, d + f)
    open(d + "patched_bd5_probe.py", "w").write(code)
    if not os.path.exists(d + "out/bd5_val.csv"):
        assert run_py(d + "patched_bd5_probe.py", [OF + MAT, BI + "emb_porta", OF + "analysis_biliary/labels_trval.csv",
                                                   OF + "analysis_biliary/oof_train.csv", d + "out"], d + "probe.log") == 0


def check():
    rows = []
    def cmp(job, new, old):
        n = pd.read_csv(new); o = pd.read_csv(old); m = o.merge(n, on="study_id", suffixes=("_o", "_n"))
        for c in [c for c in o.columns if c not in ("study_id", "fold") and c + "_n" in m]:
            rows.append(dict(job=job, col=c, n_val_stored=len(o), n_common=len(m), max_abs_diff=float(np.abs(m[c + "_o"] - m[c + "_n"]).max())))
    for code, (unit, fd, s) in UNIT.items():
        if os.path.exists(SB + code + "/out/val_scores.csv"): cmp(code, SB + code + "/out/val_scores.csv", RU + unit + "/out/val_scores.csv")
    if os.path.exists(SB + "rlq/out/val_scores.csv"): cmp("rlq", SB + "rlq/out/val_scores.csv", OF + "appendicitis_rlq_unit/out/val_scores.csv")
    if os.path.exists(SB + "gbx/out/gbx_val.csv"): cmp("gbx", SB + "gbx/out/gbx_val.csv", OF + "gallstones_improve/gs5/gbx_val.csv")
    if os.path.exists(SB + "bd5/out/bd5_val.csv"): cmp("bd5", SB + "bd5/out/bd5_val.csv", OF + "biliary_improve/out/bd5_val.csv")
    r = pd.DataFrame(rows); print(r.to_string()); r.to_csv(SB + "units_check.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        for j in sys.argv[2:]:
            print(j, "start", flush=True)
            {"rlq": rlq, "gbx": gbx, "bd5": bd5}.get(j, lambda: probe(j))()
