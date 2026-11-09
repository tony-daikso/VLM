"""Export val error cases: native CT copy, TS 3 mm GB/liver mask resampled to native grid, gallery PNG. Read-only on sources."""
import sys, os, re, shutil, json, textwrap, numpy as np, pandas as pd, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import ndimage
S = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
NAT = "/Volumes/My Passport/Tony/VLM/data/CT/merlin/merlin_data/"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"; fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["axes.unicode_minus"] = False
D = pd.read_csv(S + "/out/per_case_trval.csv"); TX = pd.read_csv(S + "/out/text_trval_private.csv")
sel = pd.read_csv(S + "/out/sel.csv"); thr = json.load(open(S + "/out/summary.json"))["thr_logit"]
D = D.merge(TX, on="study_id", how="left")
GBS = re.compile(r"gallbladder|gall bladder|cholelith|gallstone|sludge|cholecyst|pericholecystic", re.I)
recs = []; cnt = {}
for _, s in sel.iterrows():
    sid, tag = s.study_id, s.tag; cnt[tag] = cnt.get(tag, 0) + 1; key = f"{tag}-{cnt[tag]}"
    r = D[D.study_id == sid].iloc[0]
    z = np.load(f"{S}/npz/{sid}.npz"); names = list(z["unit_names"]); u = z["units"]; hu3 = z["hu"]; A3 = z["affine"]
    gb3 = u == names.index("gallbladder") + 1; lv3 = u == names.index("liver") + 1
    lab3 = np.zeros(u.shape, np.uint8); lab3[lv3] = 2; lab3[gb3] = 1
    ct = nib.load(NAT + sid + ".nii.gz"); An = ct.affine
    assert np.allclose(An[:3, :3], np.diag(np.diag(An[:3, :3]))), "non-diagonal affine"
    # per-axis nearest-neighbour index map native(i,j,k) -> 3mm (x,y,z)
    idx = []
    for ax in range(3):
        w = An[ax, ax] * np.arange(ct.shape[ax]) + An[ax, 3]
        ii = np.round((w - A3[ax, 3]) / A3[ax, ax]).astype(int); idx.append(ii)
    lab_xyz = lab3.transpose(2, 1, 0)
    valid = [(ii >= 0) & (ii < lab_xyz.shape[a]) for a, ii in enumerate(idx)]
    cl = [np.clip(ii, 0, lab_xyz.shape[a] - 1) for a, ii in enumerate(idx)]
    labn = lab_xyz[np.ix_(cl[0], cl[1], cl[2])] * (valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :])
    labn = labn.astype(np.uint8)
    pre = f"{key}_{sid}"
    (None if os.path.exists(f"{OUT}/{pre}_ct.nii.gz") else shutil.copyfile(NAT + sid + ".nii.gz", f"{OUT}/{pre}_ct.nii.gz"))
    mimg = nib.Nifti1Image(labn, An); mimg.set_qform(An, 1); mimg.set_sform(An, 1)
    nib.save(mimg, f"{OUT}/{pre}_gb_mask.nii.gz")
    vol = np.asanyarray(ct.dataobj).astype(np.float32)
    gbn = labn == 1; zooms = np.abs(np.diag(An)[:3])
    st = dict(gb_ml_native=float(gbn.sum() * zooms.prod() / 1000))
    if gbn.sum() >= 5:
        st.update(nat_mean=float(vol[gbn].mean()), nat_max=float(vol[gbn].max()), nat_n130=int((vol[gbn] > 130).sum()), nat_n200=int((vol[gbn] > 200).sum()))
        c = np.array(ndimage.center_of_mass(gbn)); area = gbn.sum((0, 1)); kA = int(area.argmax())
        h = np.where(gbn, vol, -3000); mi = np.unravel_index(h.argmax(), h.shape); src = "膽囊遮罩"
    else:
        # fossa fallback = expected GB-fossa point used by GS5/Exp N (liver bbox fraction [0.3355, 0.7002, 0.6623], 3 mm z,y,x)
        lz, ly, lx = [np.where(lv3.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
        lo = np.array([lz[0], ly[0], lx[0]], float); hi = np.array([lz[-1], ly[-1], lx[-1]], float)
        pz, py, px = lo + np.array([0.3355, 0.7002, 0.6623]) * (hi - lo)
        w = np.array([A3[0, 0] * px + A3[0, 3], A3[1, 1] * py + A3[1, 3], A3[2, 2] * pz + A3[2, 3]])
        c = (w - An[:3, 3]) / np.diag(An)[:3]; kA = int(round(c[2])); mi = None; src = "膽囊窩推估點（遮罩缺失；與 Exp N crop 相同規則）"
    ci, cj = int(round(c[0])), int(round(c[1]))
    half_i = int(70 / zooms[0]); half_k = int(70 / zooms[2])
    def ax_img(k):
        a = vol[max(ci - half_i, 0):ci + half_i, max(cj - half_i, 0):cj + half_i, k]
        g = gbn[max(ci - half_i, 0):ci + half_i, max(cj - half_i, 0):cj + half_i, k]
        return a.T[::-1], g.T[::-1]
    def cor_img():
        a = vol[max(ci - half_i, 0):ci + half_i, cj, max(kA - half_k, 0):kA + half_k]
        g = gbn[max(ci - half_i, 0):ci + half_i, cj, max(kA - half_k, 0):kA + half_k]
        return a.T[::-1], g.T[::-1]
    # 3 mm model-resolution axial at same world z
    wz = An[2, 2] * kA + An[2, 3]; z3 = int(np.clip(round((wz - A3[2, 3]) / A3[2, 2]), 0, hu3.shape[0] - 1))
    wx = An[0, 0] * ci + An[0, 3]; wy = An[1, 1] * cj + An[1, 3]
    x3 = int(round((wx - A3[0, 3]) / 3)); y3 = int(round((wy - A3[1, 3]) / 3)); h3 = 23
    a3 = hu3[z3, max(y3 - h3, 0):y3 + h3, max(x3 - h3, 0):x3 + h3]; g3 = gb3[z3, max(y3 - h3, 0):y3 + h3, max(x3 - h3, 0):x3 + h3]
    a3, g3 = a3[::-1, ::-1], g3[::-1, ::-1]
    axA, gA = ax_img(kA); co, gC = cor_img()
    fig, axs = plt.subplots(2, 3, figsize=(15, 11.5))
    WINS = [("腹部窗 W400/L40", 40, 400), ("結石窄窗 W200/L50", 50, 200)]
    for row, (wn, L, W) in enumerate(WINS):
        for col, (img, g, ttl, asp) in enumerate([(axA, gA, f"原始解析度 軸切面（z={kA}）", 1), (co, gC, "原始解析度 冠狀切面", zooms[2] / zooms[0]), (a3, g3, "3 mm 軸切面（模型輸入解析度）", 1)]):
            axx = axs[row, col]; axx.imshow(img, cmap="gray", vmin=L - W / 2, vmax=L + W / 2, aspect=asp, interpolation="nearest" if col == 2 else "bilinear")
            if g.any(): axx.contour(g.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
            if col == 0 and mi is not None and abs(mi[2] - kA) <= 0:
                pass
            axx.set_title(f"{ttl}｜{wn}", fontsize=10); axx.axis("off")
    if mi is not None:
        # max-HU point on its own axial slice, inset text
        st["maxpt_slice"] = int(mi[2])
    gbtxt = " ".join(x for x in re.split(r" \|\| | \| ", str(r.gb_text)) if GBS.search(x))[:600] or "（報告中沒有膽囊相關句子）"
    lab_s = {"yes": "陽性（yes）", "probably_yes": "陽性（probably_yes）", "no": "陰性（no）", "probably_no": "陰性（probably_no）", "not_mentioned": "陰性（not_mentioned，報告未提及）"}.get(r.gs_cat, r.gs_cat)
    p = 1 / (1 + np.exp(-r.N_U))
    cat = r["cat"] if isinstance(r["cat"], str) and r["cat"] else ("正確偵測（TP）｜明顯高密度結石" if tag == "TP" else "")
    hdr = (f"{key}  {sid}（val）｜標籤：{lab_s}｜Exp N 分數 logit {r.N_U:+.2f}（sigmoid {p:.3f}；閾值 logit {thr:+.2f}）\n"
           f"類別：{cat}\n膽囊（3 mm 遮罩）{int(r.gb_vox) if r.gb_vox == r.gb_vox else 0} vox；最大 HU {r.gb_max if r.gb_max == r.gb_max else '—'}；>130 HU {r.gb_n130 if r.gb_n130 == r.gb_n130 else '—'}；>200 HU {r.gb_n200 if r.gb_n200 == r.gb_n200 else '—'}"
           f"\n原始解析度（3 mm 遮罩最近鄰放大，邊緣含部分容積）：{st['gb_ml_native']:.1f} mL" + (f"，最大 {st['nat_max']:.0f} HU，>130 HU {st['nat_n130']} vox，>200 HU {st['nat_n200']} vox" if "nat_max" in st else "") +
           f"｜中心：{src}｜{r.phase}")
    fig.suptitle(hdr, fontsize=10.5, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.01, "報告：" + "\n".join(textwrap.wrap(gbtxt, 190)), fontsize=8.5, ha="left", va="bottom", family="DejaVu Sans" if False else None)
    plt.tight_layout(rect=(0, 0.06, 1, 0.92)); fig.savefig(f"{OUT}/{pre}_gallery.png", dpi=110); plt.close(fig)
    rec = dict(key=key, study_id=sid, label=lab_s, score_logit=round(float(r.N_U), 3), score_p=round(float(p), 4), category=cat,
               gb_vox3=r.gb_vox, gb_max3=r.gb_max, gb_n130_3=r.gb_n130, gb_n200_3=r.gb_n200, phase=r.phase, center=src, report_gb=gbtxt, **st)
    recs.append(rec); print(key, sid, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()}, flush=True)
    del vol
json.dump(recs, open(S + "/out/export_recs.json", "w"), ensure_ascii=False, indent=1, default=str)
