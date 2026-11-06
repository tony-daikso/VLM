"""Exp G paired comparisons (same config, rib18 vs 2 big rib units; 10-seed groups). train OOF + val, rib subgroup bootstrap."""
import sys, numpy as np
sys.path.insert(0, "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/src")
from final_stack import split_trval, load_labels_trval, SITES
from probe import auroc
B = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/"
G = {"F1 ref (2 rib)": B + "final_F1/groups/ref__site_max__mean__all3.npz",
     "F1 excl (2 rib)": B + "final_F1/groups/excl_amb__site_max__mean__all3.npz",
     "F1 lse (2 rib)": B + "final_F1/groups/ref__site_lse__mean__all3.npz",
     "F1 bone_ens": B + "final_F1/groups/bone_ens.npz",
     "G ref rib18": B + "final_G/ens_G/groups/ref__site_max-rib18__mean__all3.npz",
     "G excl rib18": B + "final_G/ens_G/groups/excl_amb__site_max-rib18__mean__all3.npz",
     "G lse rib18": B + "final_G/ens_G/groups/ref__site_lse-rib18__mean__all3.npz",
     "G ref rib18big": B + "final_G/ens_G/groups/ref__site_max-rib18big__mean__all3.npz",
     "G ens_all": B + "final_G/ens_all/groups/bone_ens.npz"}
spl = split_trval(); lab = load_labels_trval(set(spl.index)); lab = lab[~lab.y.isna()]
z0 = np.load(G["F1 ref (2 rib)"], allow_pickle=True); ids = [str(i) for i in z0["ids"]]; S = z0["split"]
L = {k: np.load(v, allow_pickle=True) for k, v in G.items()}
for k, z in L.items(): assert [str(i) for i in z["ids"]] == ids
lab = lab.loc[ids]; y = lab.y.values.astype(int); rib = SITES["肋骨"](lab).values.astype(bool) & (y == 1)
def ribauc(s, m):
    sel = m & (rib | (y == 0)); return auroc(rib[sel].astype(float), s[sel])
rng = np.random.default_rng(0); va = np.where(S == "val")[0]
boots = [rng.choice(va, len(va)) for _ in range(1000)]
def bdiff(a, b, rib_only):
    out = []
    for bi in boots:
        m = np.zeros(len(y), bool); yy = y[bi]; rr = rib[bi]
        if rib_only:
            sel = rr | (yy == 0); out.append(auroc(rr[sel].astype(float), a[bi][sel]) - auroc(rr[sel].astype(float), b[bi][sel]))
        else:
            out.append(auroc(yy.astype(float), a[bi]) - auroc(yy.astype(float), b[bi]))
    return np.percentile(out, [2.5, 97.5])
tr, vm = S == "train", S == "val"
print(f"N train {tr.sum()} (rib pos {rib[tr].sum()}), val {vm.sum()} (rib pos {rib[vm].sum()})")
for k, z in L.items():
    s = z["logit"]
    print(f"{k:16s} train OOF any {auroc(y[tr].astype(float), s[tr]):.4f} rib {ribauc(s, tr):.4f} | val any {auroc(y[vm].astype(float), s[vm]):.4f} rib {ribauc(s, vm):.4f}")
for a, b in [("G ref rib18", "F1 ref (2 rib)"), ("G excl rib18", "F1 excl (2 rib)"), ("G lse rib18", "F1 lse (2 rib)"), ("G ref rib18big", "F1 ref (2 rib)"), ("G ref rib18", "F1 bone_ens")]:
    sa, sb = L[a]["logit"], L[b]["logit"]
    d_any = auroc(y[vm].astype(float), sa[vm]) - auroc(y[vm].astype(float), sb[vm]); d_rib = ribauc(sa, vm) - ribauc(sb, vm)
    print(f"val {a} - {b}: any {d_any:+.4f} {np.round(bdiff(sa, sb, False), 4)} | rib {d_rib:+.4f} {np.round(bdiff(sa, sb, True), 4)}")
