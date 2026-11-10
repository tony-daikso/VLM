"""(Mac) from owl_viz/<疾病>/<EXAMID>/{ct,units_label,contrib_heatmap}.nii.gz + meta.json (make_itk.py on CT-rate):
  labels.txt, workspace.itksnap, overview.png per case; copy to ~/Desktop/OWL_ITK/<疾病>/<EXAMID>/ (own workspace);
  meta.json (viewer schema) per case, written case by case; nibabel self-check (work/nifti_checks.json).
"""
import json
import shutil
import textwrap
from pathlib import Path
from xml.sax.saxutils import quoteattr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from common_viz import UNIT_ZH  # noqa: E402

plt.rcParams["font.family"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
HERE = Path(__file__).resolve().parent
VIZ = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/owl_viz")
ITK = Path("/Users/tony.tu/Desktop/OWL_ITK")
WARM = [(165, 0, 38), (215, 48, 39), (244, 109, 67), (253, 174, 97), (254, 224, 144)]   # rank 1..5
COOL = [(116, 173, 209), (69, 117, 180), (49, 54, 149)]
GREY = (150, 150, 150)
FILES = ["ct.nii.gz", "units_label.nii.gz", "contrib_heatmap.nii.gz", "labels.txt", "workspace.itksnap", "overview.png"]
COPY = ["ct.nii.gz", "units_label.nii.gz", "contrib_heatmap.nii.gz", "labels.txt", "overview.png", "grid.json"]


def ranked(s):
    """[(label_id, unit, value)] sorted by contribution desc; unused/absent units after (value None)."""
    r = [(i + 1, u, v) for i, (u, v) in enumerate(zip(s["units"], s["contrib"]))]
    used = sorted([x for x in r if x[2] is not None], key=lambda x: -x[2])
    return used + [x for x in r if x[2] is None]


def label_table(s, meta):
    """-> list of (id, rgb, alpha 0-1, vis, name)."""
    out = []
    present = {int(k) for k in meta["centroids"]}
    used_rank = 0
    for lid, u, v in ranked(s):
        if v is not None:
            if used_rank < 5:
                rgb, a = WARM[used_rank], 1.0
            else:
                rgb, a = COOL[min(used_rank - 5, 2)], 0.35
            used_rank += 1
            name = f"{UNIT_ZH[u]} ({v:+.3f})"
        else:
            rgb, a = GREY, 0.15
            name = f"{UNIT_ZH[u]} (未用於此病{'' if lid in present else '／影像中不存在'})"
        out.append((lid, rgb, a, 1, name))
    return out


def write_labels_txt(path, table):
    lines = ["################################################",
             "# ITK-SnAP Label Description File",
             "# Columns: IDX -R- -G- -B- -A-- VIS MSH LABEL",
             "################################################",
             '    0     0    0    0        0  0  0    "Clear Label"']
    for lid, (r, g, b), a, vis, name in sorted(table):
        lines.append(f'{lid:5d} {r:5d} {g:4d} {b:4d} {a:8.2f} {vis:2d} {vis:2d}    "{name}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def curve(lo, hi, vmin, vmax, ind):
    t0 = (lo - vmin) / max(vmax - vmin, 1e-6)
    t1 = (hi - vmin) / max(vmax - vmin, 1e-6)
    pts = [(t0, 0.0), ((t0 + t1) / 2, 0.5), (t1, 1.0)]
    s = [f'{ind}<folder key="Curve" >', f'{ind}  <entry key="NumberOfControlPoints" value="3" />']
    for i, (t, x) in enumerate(pts):
        s += [f'{ind}  <folder key="ControlPoint[{i}]" >', f'{ind}    <entry key="tValue" value="{t:.6f}" />',
              f'{ind}    <entry key="xValue" value="{x}" />', f'{ind}  </folder>']
    return s + [f"{ind}</folder>"]


def workspace_xml(d, s, meta, table, hmin, hmax):
    """ITK-SNAP workspace: main CT (W400/L40), segmentation units_label + label table, heatmap overlay (Hot)."""
    ind = "        "
    L = ['<?xml version="1.0" encoding="UTF-8" ?>', "<!--ITK-SNAP (itksnap.org) Project File-->", "<!DOCTYPE registry [",
         "<!ELEMENT registry (entry*,folder*)>", "<!ELEMENT folder (entry*,folder*)>", "<!ELEMENT entry EMPTY>",
         "<!ATTLIST folder key CDATA #REQUIRED>", "<!ATTLIST entry key CDATA #REQUIRED>",
         "<!ATTLIST entry value CDATA #REQUIRED>", "]>", "<registry>",
         f'  <entry key="SaveLocation" value={quoteattr(str(d))} />', '  <entry key="Version" value="20190612" />',
         '  <folder key="Layers" >']
    # layer 0: main
    L += ['    <folder key="Layer[000]" >', f'      <entry key="AbsolutePath" value={quoteattr(str(d / "ct.nii.gz"))} />',
          '      <entry key="Role" value="MainRole" />', '      <entry key="Tags" value="" />',
          '      <folder key="LayerMetaData" >', '        <entry key="Alpha" value="255" />',
          '        <entry key="CustomNickName" value="CT" />', '        <entry key="Sticky" value="0" />',
          '        <entry key="Tags" value="" />', '        <folder key="DisplayMapping" >',
          '          <folder key="ColorMap" >', '            <entry key="Preset" value="Grayscale" />', "          </folder>"]
    L += curve(-160, 240, meta["hu_min"], meta["hu_max"], ind + "  ")
    L += ["        </folder>", "      </folder>", '      <folder key="ProjectMetaData" >',
          '        <entry key="GaussianBlurScale" value="1" />', '        <entry key="RemappingExponent" value="3" />',
          '        <entry key="RemappingSteepness" value="0.04" />', '        <folder key="IRIS" >',
          '          <entry key="SliceViewLayerLayout" value="Stacked" />', '          <folder key="LabelTable" >',
          f'            <entry key="NumberOfElements" value="{len(table)}" />']
    for i, (lid, (r, g, b), a, vis, name) in enumerate(sorted(table)):
        L += [f'            <folder key="Element[{i}]" >', f'              <entry key="Alpha" value="{int(round(a * 255))}" />',
              f'              <entry key="Color" value="{r} {g} {b}" />', f'              <entry key="Flags" value="{vis} {vis}" />',
              f'              <entry key="Index" value="{lid}" />', f"              <entry key=\"Label\" value={quoteattr(name)} />",
              "            </folder>"]
    L += ["          </folder>", "        </folder>", "      </folder>", "    </folder>"]
    # layer 1: segmentation
    L += ['    <folder key="Layer[001]" >', f'      <entry key="AbsolutePath" value={quoteattr(str(d / "units_label.nii.gz"))} />',
          '      <entry key="Role" value="SegmentationRole" />', '      <entry key="Tags" value="" />',
          '      <folder key="LayerMetaData" >', '        <entry key="Alpha" value="0.5" />',
          '        <entry key="CustomNickName" value="單元標籤" />', '        <entry key="Sticky" value="1" />',
          '        <entry key="Tags" value="" />', "      </folder>", "    </folder>"]
    # layer 2: overlay heatmap
    L += ['    <folder key="Layer[002]" >', f'      <entry key="AbsolutePath" value={quoteattr(str(d / "contrib_heatmap.nii.gz"))} />',
          '      <entry key="Role" value="OverlayRole" />', '      <entry key="Tags" value="" />',
          '      <folder key="LayerMetaData" >', '        <entry key="Alpha" value="0.5" />',
          '        <entry key="CustomNickName" value="貢獻熱圖" />', '        <entry key="Sticky" value="1" />',
          '        <entry key="Tags" value="" />', '        <folder key="DisplayMapping" >',
          '          <folder key="ColorMap" >', '            <entry key="Preset" value="Hot" />', "          </folder>"]
    L += curve(0.0, max(hmax, 1e-3), hmin, hmax, ind + "  ")
    L += ["        </folder>", "      </folder>", "    </folder>", "  </folder>", "</registry>"]
    return "\n".join(L) + "\n"


def png(d, s, meta, table):
    ct = np.asanyarray(nib.load(d / "ct.nii.gz").dataobj)
    lab = np.asanyarray(nib.load(d / "units_label.nii.gz").dataobj)
    rk = [x for x in ranked(s) if x[2] is not None and str(x[0]) in meta["centroids"]]
    top_id = rk[0][0]
    cx, cy, cz = meta["centroids"][str(top_id)]
    frac = s["disease"] == "fracture"
    lo, hi = (-450, 1050) if frac else (-160, 240)
    vals = np.full(256, np.nan)
    for lid, u, v in rk:
        vals[lid] = v
    if frac:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        cmap, norm, clab = plt.get_cmap("Oranges"), mcolors.Normalize(vmin, vmax), "MIL 單元分數 (logit)"
    else:
        m = np.nanmax(np.abs(vals)) or 1.0
        cmap, norm, clab = plt.get_cmap("RdBu_r"), mcolors.TwoSlopeNorm(0, -m, m), "單元對 logit 的貢獻"
    fig = plt.figure(figsize=(13, 7.6), dpi=110)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55])
    for j, (title, sl_ct, sl_lab) in enumerate((
            (f"軸切 z={cz}", ct[::-1, :, cz].T, lab[::-1, :, cz].T),
            (f"冠狀切 y={cy}", ct[::-1, cy, :].T, lab[::-1, cy, :].T))):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.clip(sl_ct, lo, hi), cmap="gray", origin="lower", vmin=lo, vmax=hi)
        v = vals[sl_lab]
        rgba = cmap(norm(np.nan_to_num(v, nan=0.0)))
        rgba[..., 3] = np.where(np.isnan(v), 0.0, 0.5)
        ax.imshow(rgba, origin="lower")
        for r, (lid, u, val) in enumerate(rk[:5]):
            mk = sl_lab == lid
            if mk.any():
                ax.contour(mk, levels=[0.5], colors=[np.array(WARM[r]) / 255], linewidths=1.2, origin="lower")
                yy, xx = np.argwhere(mk).mean(0)
                ax.text(xx, yy, str(r + 1), color="white", fontsize=10, ha="center", va="center", weight="bold",
                        bbox=dict(boxstyle="circle,pad=0.15", fc=np.array(WARM[r]) / 255, ec="none"))
        ax.set_title(title + "（右側＝病人左側）", fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    txt = [f"前 5 單元（{s['metric']}）:"]
    for r, (lid, u, val) in enumerate(rk[:5]):
        txt.append(f"  {r + 1}. {UNIT_ZH[u]}  {val:+.3f}")
    txt.append(f"\nlogit = {s['logit']:+.3f}")
    ax.text(0, 0.98, "\n".join(txt), va="top", fontsize=10.5, family=plt.rcParams["font.family"])
    cax = fig.add_axes([0.70, 0.12, 0.22, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(clab, fontsize=9)
    q = textwrap.fill("報告：" + s["quote"], 120)
    kind = "真陽性" if s["kind"] == "TP" else f"偽陽性（報告 {s['category']}）"
    fig.suptitle(f"{s['id']}  {s['disease_zh']}  模型機率 {s['prob']:.3f}  {kind}\n{q}", fontsize=11, x=0.02, ha="left")
    fig.savefig(d / "overview.png", bbox_inches="tight")
    plt.close(fig)


def unit_entries(s, meta):
    aff = np.array(meta["affine"])
    out = []
    for lid, u, v in ranked(s):
        k = str(lid)
        if k not in meta["centroids"]:
            continue
        c_mm = (aff @ np.r_[meta["centroids"][k], 1])[:3]
        b0, b1 = meta["bbox"][k]
        cs = np.array([(aff @ np.r_[x, y, z, 1])[:3] for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])])
        out.append(dict(id=lid, key=u, name_zh=UNIT_ZH[u], contrib=None if v is None else round(v, 4),
                        centroid_mm=[round(float(x), 1) for x in c_mm],
                        bbox_mm=[cs.min(0).round(1).tolist(), cs.max(0).round(1).tolist()],
                        centroid_vox=meta["centroids"][k], n_vox=meta["nvox"][k]))
    return out


def summary(s):
    q = s["quote"]
    return q if len(q) <= 120 else q[:117] + "..."


def main():
    spec = json.load(open(HERE / "work/viz_spec.json"))
    index, checks = {"diseases": []}, []
    by_dis = {}
    for s in spec:
        d = VIZ / s["disease_zh"] / s["id"]
        meta = json.load(open(d / "grid.json"))
        table = label_table(s, meta)
        h = np.asanyarray(nib.load(d / "contrib_heatmap.nii.gz").dataobj)
        hmin, hmax = float(min(h.min(), 0)), float(max(h.max(), 1e-3))
        write_labels_txt(d / "labels.txt", table)
        (d / "workspace.itksnap").write_text(workspace_xml(d, s, meta, table, hmin, hmax), encoding="utf-8")
        png(d, s, meta, table)
        units = unit_entries(s, meta)
        m = dict(examid=s["id"], disease_zh=s["disease_zh"], disease_key=s["disease"], prob=round(s["prob"], 4),
                 label_category=s["category"], report_quote=s["quote"], kind=s["kind"], logit=round(s["logit"], 4),
                 metric=s["metric"], heat_range=[hmin, hmax], units=units)
        (d / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        # desktop copy with its own absolute paths (written per case)
        t = ITK / s["disease_zh"] / s["id"]
        t.mkdir(parents=True, exist_ok=True)
        for fn in COPY:
            shutil.copyfile(d / fn, t / fn)
        (t / "workspace.itksnap").write_text(workspace_xml(t, s, meta, table, hmin, hmax), encoding="utf-8")
        (t / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        # self-check
        imgs = [nib.load(x / f) for x in (d, t) for f in FILES[:3]]
        ok = all(i.shape == imgs[0].shape and np.allclose(i.affine, imgs[0].affine) for i in imgs)
        ok &= np.asanyarray(imgs[0].dataobj).dtype == np.int16 and np.asanyarray(imgs[2].dataobj).dtype == np.float32
        lab = np.asanyarray(imgs[1].dataobj); h = np.asanyarray(imgs[2].dataobj)
        ok &= bool((h[lab == 0] == 0).all())
        checks.append((s["disease_zh"], s["id"], bool(ok), list(imgs[0].shape)))
        print(s["disease_zh"], s["id"], "ok" if ok else "CHECK FAILED", flush=True)
    json.dump(checks, open(HERE / "work/nifti_checks.json", "w"), ensure_ascii=False)
    bad = [c for c in checks if not c[2]]
    print(f"{len(checks)} cases, nibabel check failures: {bad}")


if __name__ == "__main__":
    main()
