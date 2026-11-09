"""Native-resolution HU inside the (3 mm TS) GB mask interior for selected VAL cases. Read-only on native CTs."""
import sys, numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
S = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-tony-tu-Desktop------VLM-code-Organ-aware-Window-Learning-use-merlin-dataset/23e4ea44-f6c4-4626-ba65-ca9a6b22eb63/scratchpad/gs2"; NAT = "/Volumes/My Passport/Tony/VLM/data/CT/merlin/merlin_data/"
X = pd.read_csv(S + "/out/native_sel.csv")
def run(sid):
    try:
        z = np.load(f"{S}/npz_nat/{sid}.npz"); names = list(z["unit_names"]); A3 = z["affine"]
        gb = z["units"] == names.index("gallbladder") + 1
        inner = ndimage.binary_erosion(gb, iterations=1)
        if inner.sum() < 5: inner = gb
        ct = nib.load(NAT + sid + ".nii.gz"); An = ct.affine
        # bounding box of GB in native index space
        zz, yy, xx = np.nonzero(gb)
        w = lambda a, i: A3[a, a] * i + A3[a, 3]
        lo = [(w(0, xx.min() - 1) - An[0, 3]) / An[0, 0], (w(1, yy.min() - 1) - An[1, 3]) / An[1, 1], (w(2, zz.min() - 1) - An[2, 3]) / An[2, 2]]
        hi = [(w(0, xx.max() + 1) - An[0, 3]) / An[0, 0], (w(1, yy.max() + 1) - An[1, 3]) / An[1, 1], (w(2, zz.max() + 1) - An[2, 3]) / An[2, 2]]
        sl = []
        for a in range(3):
            a0, a1 = sorted([lo[a], hi[a]]); sl.append(slice(max(int(np.floor(a0)), 0), min(int(np.ceil(a1)) + 1, ct.shape[a])))
        vol = np.asarray(ct.dataobj[sl[0], sl[1], sl[2]], dtype=np.float32)
        idx = [np.round((An[a, a] * np.arange(sl[a].start, sl[a].stop) + An[a, 3] - A3[a, 3]) / A3[a, a]).astype(int) for a in range(3)]
        m_xyz = inner.transpose(2, 1, 0); g_xyz = gb.transpose(2, 1, 0)
        cl = [np.clip(idx[a], 0, m_xyz.shape[a] - 1) for a in range(3)]
        mi = m_xyz[np.ix_(*cl)]; mg = g_xyz[np.ix_(*cl)]
        v = vol[mi]; vg = vol[mg]
        b130 = ndimage.label((vol > 130) & mi)[0]; sizes = np.bincount(b130.ravel())[1:] if b130.max() else np.array([0])
        zooms = np.abs(np.diag(An)[:3])
        return dict(study_id=sid, nat_thick=zooms[2], inner_vox=int(mi.sum()), in_max=float(v.max()), in_p99=float(np.percentile(v, 99)), in_med=float(np.median(v)),
                    in_n130=int((v > 130).sum()), in_n200=int((v > 200).sum()), in_blob130_max_mm3=float(sizes.max() * zooms.prod()),
                    full_max=float(vg.max()), full_n200=int((vg > 200).sum()))
    except Exception as e:
        return dict(study_id=sid, err=str(e))
if __name__ == "__main__":
  with ProcessPoolExecutor(3) as ex:
    R = list(ex.map(run, X.study_id, chunksize=4))
  pd.DataFrame(R).merge(X, on="study_id").to_csv(S + "/out/native_hu_val.csv", index=False); print("done", len(R))
