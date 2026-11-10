"""C-tier fracture (formal best C3 = LR(expI + J_add + L + U_org_pel + U_pel + U_org_out), val .854) for CG10k.
All member scripts are the ORIGINAL ones; CG rows are appended to their input matrices with split 'val' (dummy label 0,
never fitted / early-stopped on): every member's 'val' logit = mean of its 5 fold models (or the full-train refit for
Exp O), so the CG rows get exactly the score Merlin val got.  Merlin val rows must reproduce the stored members.
  prep        : sandbox bones dir (bone_matrix / rib_matrix / expI meta with CG rows, hp json links), emb15 dirs
  e3late      : bone_probe.py modes_oof E3_late std seeds 0-9           (Exp I group E3late10)
  expI V S    : bone_expI.py --variant V --seed-off S                      (hn_b2, lv1, age_full, hn_m3, r18)
  ens         : Exp I greedy ensemble (fixed multiset from ens_greedy_all.json; bone_expI_ens.py z-scoring)
  v15 / L     : vert15_probe.py add (10 seeds) / rib15_probe.py L_repl+rl03+lrc (5 seeds) on matrices incl. CG
  stack       : final_stack_combo C3 LR stacker refit on the stored Merlin members -> CG score
Labels: bone_probe.load_labels is wrapped so that CG ids get y=0 / no sites (only ever used as 'val')."""
import os, sys, json, glob, runpy, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); MW = H + "mw_exp/"; SRC = MW + "src"; FR = MW + "outputs_fracture/"
B = H + "cg10k_best/"; FB = B + "frac/bones/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")

def prep_rib(cg_ids, ids):
    zr = dict(np.load(FR + "bones/rib_matrix.npz", allow_pickle=True))
    rr = [dict(np.load(B + f"frac/cg_{s}_ribs3.npz")) for s in ("main", "delay")]   # materialise once (npz keys re-read per access)
    rid = {str(i): (k, j) for k, r in enumerate(rr) for j, i in enumerate(r["ids"])}
    U = zr["E"].shape[1]; Ec = np.zeros((len(cg_ids), U, 3, 2, 448), np.float16); Pc = np.zeros((len(cg_ids), U), bool); miss = []
    for n, i in enumerate(cg_ids):
        if i in rid: k, j = rid[i]; Ec[n] = rr[k]["E"][j]; Pc[n] = rr[k]["P"][j]
        else: miss.append(i)
    for r in rr: assert list(r["units"]) == list(zr["units"])
    np.savez(FB + "rib_matrix.npz", E=np.concatenate([zr["E"], Ec]), P=np.concatenate([zr["P"], Pc]), ids=ids, units=zr["units"],
             missing=np.concatenate([zr["missing"].astype("<U12"), np.array(miss, "<U12")]))
    print("rib_matrix: CG missing", len(miss))


def prep(what="all"):
    os.makedirs(FB + "expI", exist_ok=True); os.makedirs(FB + "rerun/std", exist_ok=True)
    zb = np.load(FR + "bones/bone_matrix.npz", allow_pickle=True)
    cgs = [np.load(B + f"frac/cg_{s}_bones.npz") for s in ("main", "delay")]
    for c in cgs: assert list(c["units"]) == list(zb["units"])
    cg_ids = np.concatenate([c["ids"].astype("<U12") for c in cgs])
    ids = np.concatenate([zb["ids"].astype("<U12"), cg_ids])
    if what in ("all", "bones"):
        np.savez(FB + "bone_matrix.npz", E=np.concatenate([zb["E"]] + [c["E"] for c in cgs]), P=np.concatenate([zb["P"]] + [c["P"] for c in cgs]),
                 ids=ids, split=np.concatenate([zb["split"].astype("<U5"), np.array(["val"] * len(cg_ids))]), units=zb["units"])
        m = pd.read_csv(FR + "bones/expI/meta.csv", index_col=0, keep_default_na=False, na_values=[""])
        add = pd.DataFrame(index=cg_ids, columns=m.columns)
        for c in m.columns:
            add[c] = "False" if (c.startswith("hn_") or c in ("fov_out",)) else np.nan
        add["split"] = "val"; add["levels"] = ""
        if "y" in m: add["y"] = 0
        pd.concat([m, add]).to_csv(FB + "expI/meta.csv")
        for f in glob.glob(FR + "bones/rerun/std/*.json"):
            t = FB + "rerun/std/" + os.path.basename(f)
            if not os.path.exists(t): os.symlink(f, t)
    if what in ("all", "ribs"): prep_rib(cg_ids, ids)
    if what in ("all", "emb15"):
        for kind in ("v15", "r15"):     # CG emb15 dirs with Merlin-style ids (delay -> D_ prefix)
            d = B + f"frac/emb_{kind}/"; os.makedirs(d, exist_ok=True)
            for s, pre in (("main", ""), ("delay", "D_")):
                sd = B + f"frac/{kind}_{s}/"
                if not os.path.isdir(sd): print("missing", sd); continue
                for f in os.listdir(sd):
                    t = d + pre + f
                    if not os.path.exists(t): os.symlink(sd + f, t)
    print("prep", what, ": N", len(ids), "CG", len(cg_ids))

WRAP = r'''
import sys, numpy as np, pandas as pd
sys.path.insert(0, "%s")
import bone_probe as BP
_ll = BP.load_labels
def ll(path, ids):
    ids = [str(i) for i in ids]; df = _ll(path, [i for i in ids if i.startswith("AC")])
    cg = [i for i in ids if not i.startswith("AC")]
    add = pd.DataFrame({"y": 0.0, "compression": False, "sites": [[] for _ in cg], "acute": "not_stated"}, index=cg)
    return pd.concat([df, add])
BP.load_labels = ll
'''

def run_script(script, argv, mod_patch="", log=None):
    mod = os.path.basename(script)[:-3]          # import the (patched-labels) module and call its own main()
    code = WRAP % SRC + mod_patch + f"\nimport {mod} as _S\nsys.argv = {[script] + argv!r}\n_S.main()\n"
    p = B + "frac/_run_" + os.path.basename(log or script) + ".py"; open(p, "w").write(code)
    r = subprocess.run(["nice", "-n", "10", PY, p], cwd=SRC, stdout=open(log or p + ".log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="4")); return r.returncode

def patch_paths(name):
    """module-level path constants of bone_expI / bone_probe -> sandbox"""
    return f'''
import pathlib
import {name} as _M
_M.B = pathlib.Path("{FB}"); _M.OUT = pathlib.Path("{FB}expI")
'''

GREEDY = ["E3late10", "hn_b2", "lv1", "age_full", "E3late10", "hn_m3", "r18"]
GROUPS = {"E3late10": FB + "rerun/std/oof/oof_bone_E_ref__E3_late_s[0-9].npz", "hn_b2": FB + "expI/members/oof_bone_E_hn_b2_s*.npz",
          "lv1": FB + "expI/members/oof_bone_E_lv1_s*.npz", "age_full": FB + "expI/members/oof_bone_E_age_full_s*.npz",
          "hn_m3": FB + "expI/members/oof_bone_E_hn_m3_s*.npz", "r18": FB + "expI/members/oof_bone_E_r18_s*.npz"}

def ens():
    """bone_expI_ens.py val formula for the fixed greedy multiset: mean_s z_train(seed-mean raw logit of group s)."""
    RAW, ids0 = {}, None
    for n, p in GROUPS.items():
        fs = sorted(glob.glob(p)); assert fs, p
        L = []
        for f in fs:
            z = np.load(f, allow_pickle=True); i = [str(x) for x in z["ids"]]
            if ids0 is None: ids0, split, y = i, z["split"], z["y"]
            assert i == ids0; L.append(np.asarray(z["logit"], float))
        RAW[n] = np.mean(L, 0); print(n, len(fs), "files")
    tr = split == "train"
    ZS = {n: (v - v[tr].mean()) / v[tr].std() for n, v in RAW.items()}
    e = np.mean([ZS[g] for g in GREEDY], 0)
    ref = np.load(FR + "bones/expI/groups/ens_greedy_all.npz", allow_pickle=True)
    r = pd.Series(ref["logit"], index=[str(i) for i in ref["ids"]]); va = [i for i, sp in zip(ids0, split) if sp == "val" and i.startswith("AC")]
    mine = pd.Series(e, index=ids0)
    print("Merlin val rows", len(va), "max|d| vs stored ens", np.abs(r[va].values - mine[va].values).max())
    np.savez(B + "frac/expI_ens.npz", ids=np.array(ids0), split=split, logit=e.astype(np.float32))

def member_mean(globs):
    S = None
    for f in sorted(sum([glob.glob(g) for g in globs], [])):
        z = np.load(f, allow_pickle=True); s = pd.Series(np.asarray(z["logit"], float), index=[str(i) for i in z["ids"]])
        S = s.to_frame() if S is None else S.join(s.rename(len(S.columns)), how="outer")
    return S.mean(1)

def stack():
    """C3 stacker (final_stack_combo 'lr' = StandardScaler + LogisticRegression(balanced, max_iter 3000), no features)
    refit on the STORED Merlin train member logits, applied to Merlin val (check) and to the CG member logits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load(FR + "final_combo/C3_ILJ_pel_all/scores.npz", allow_pickle=True)
    M = ["m_expI", "m_J_add", "m_L", "m_U_org_pel", "m_U_pel", "m_U_org_out"]; assert str(z["stack_kind"]) == "lr"
    tr, va = z["split"] == "train", z["split"] == "val"
    X = np.c_[[z[m] for m in M]].T; sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), z["y"][tr])
    print("Merlin val max|d| vs stored C3 stack", np.abs(lr.decision_function(sc.transform(X[va])) - z["stack"][va]).max())
    cg = {}
    cg["m_expI"] = (lambda q: pd.Series(q["logit"], index=[str(i) for i in q["ids"]]))(np.load(B + "frac/expI_ens.npz", allow_pickle=True))
    cg["m_J_add"] = member_mean([B + "frac/v15/oof/full/oof_v15_add_s*.npz"])
    cg["m_L"] = member_mean([B + "frac/L/oof/full/oof_L_L_repl-rl03-lrc_s*.npz"])
    for col, (code, c) in {"m_U_org_pel": ("frpp", "U_org"), "m_U_pel": ("frpp", "U"), "m_U_org_out": ("frp", "U_org")}.items():
        v = pd.read_csv(B + f"bprobe/{code}/out/val_scores.csv").set_index("study_id")[c]; cg[col] = v
    D = pd.DataFrame(cg); D = D[~D.index.str.startswith("AC")]
    for m in M: print(m, "CG non-missing", int(D[m].notna().sum()))
    # Merlin members of the CG pipeline must equal the stored members on Merlin val
    ids = [str(i) for i in z["ids"]]; ref = pd.DataFrame({m: z[m] for m in M}, index=ids)[va]
    chk = pd.DataFrame(cg).reindex(ref.index)
    for m in M: print("  Merlin val", m, "max|d|", float(np.nanmax(np.abs(chk[m].values - ref[m].values))))
    D = D.dropna(); s = lr.decision_function(sc.transform(D[M].values))
    out = pd.DataFrame({"study_id": D.index, "fracture": s})
    for st, pre in (("main", ""), ("delay", "D_")):
        o = out[out.study_id.str.startswith("D_")] if pre else out[~out.study_id.str.startswith("D_")]
        o.assign(study_id=o.study_id.str.replace("^D_", "", regex=True)).to_csv(B + f"frac/cg_scores_{st}.csv", index=False); print(st, len(o))

if __name__ == "__main__":
    if sys.argv[1] == "ens": ens(); sys.exit()
    if sys.argv[1] == "stack": stack(); sys.exit()
    cmd = sys.argv[1]; os.makedirs(B + "frac/logs", exist_ok=True)
    if cmd == "prep": prep(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "e3late":
        rc = run_script(SRC + "/bone_probe.py", ["--stage", "modes_oof", "--model", "E3_late", "--sub", "std", "--out", FB,
                        "--seed-offs"] + sys.argv[2:], log=B + "frac/logs/e3late_" + "_".join(sys.argv[2:]) + ".log"); print("rc", rc)
    elif cmd == "expI":
        v, s = sys.argv[2], sys.argv[3]
        # bone_expI imports B/OUT at module level: run it as a module with patched constants, same main()
        code = WRAP % SRC + patch_paths("bone_expI") + f"\nsys.argv = ['bone_expI.py', '--variant', {v!r}, '--seed-off', {s!r}, '--outdir', '{FB}expI/members']\n_M.main()\n"
        p = B + f"frac/_run_expI_{v}_{s}.py"; open(p, "w").write(code)
        r = subprocess.run(["nice", "-n", "10", PY, p], cwd=SRC, stdout=open(B + f"frac/logs/expI_{v}_s{s}.log", "w"), stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="4")); print("rc", r.returncode)
    elif cmd == "v15":
        if not os.path.exists(B + "frac/v15/v15_matrix.npz"):
            os.makedirs(B + "frac/v15", exist_ok=True)
            run_script(SRC + "/vert15_probe.py", ["--stage", "build", "--out", B + "frac/v15", "--bone-matrix", FB + "bone_matrix.npz",
                       "--emb-dirs", MW + "outputs_vert15/merged_final/emb15", B + "frac/emb_v15"], log=B + "frac/logs/v15_build.log")
        hp = json.load(open(MW + "outputs_vert15/final_eval/oof/full/hp_add.json"))["hp"]
        rc = run_script(SRC + "/vert15_probe.py", ["--stage", "oof", "--out", B + "frac/v15", "--cfgs", "add", "--lr", str(hp["lr"]), "--wd", str(hp["wd"]),
                        "--seeds"] + sys.argv[2:] + ["--device", "cuda", "--threads", "3"], log=B + "frac/logs/v15_oof_" + "_".join(sys.argv[2:]) + ".log"); print("rc", rc)
    elif cmd == "L":
        if not os.path.exists(B + "frac/L/l_matrix.npz"):
            os.makedirs(B + "frac/L", exist_ok=True)
            run_script(SRC + "/rib15_probe.py", ["--stage", "build", "--out", B + "frac/L", "--bone-matrix", FB + "bone_matrix.npz",
                       "--rib-matrix", FB + "rib_matrix.npz", "--v15-dirs", MW + "outputs_vert15/merged_final/emb15", B + "frac/emb_v15",
                       "--r15-dirs", MW + "outputs_ribs15/emb15", B + "frac/emb_r15"], log=B + "frac/logs/L_build.log")
        rc = run_script(SRC + "/rib15_probe.py", ["--stage", "oof", "--out", B + "frac/L", "--variants", "L_repl+rl03+lrc", "--seeds"] + sys.argv[2:] + ["--device", "cuda", "--threads", "3"],
                        log=B + "frac/logs/L_oof_" + "_".join(sys.argv[2:]) + ".log"); print("rc", rc)
