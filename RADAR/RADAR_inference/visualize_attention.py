import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference_demo import initialize, DataFolder

ROI = (96, 256, 384)

ORGAN_EN = {
    "肾上腺": "Adrenal gland", "主动脉": "Aorta", "竖脊肌": "Erector spinae muscle", "脑": "Brain",
    "锁骨": "Clavicle", "大肠": "Large bowel", "十二指肠": "Duodenum", "食管": "Esophagus", "面部": "Face",
    "股骨": "Femur", "胆囊": "Gallbladder", "臀肌": "Gluteus muscle", "心脏": "Heart", "髋关节": "Hip joint",
    "肱骨": "Humerus", "髂动脉": "Iliac artery", "髂静脉": "Iliac vena", "髂腰肌": "Iliopsoas muscle",
    "下腔静脉": "Inferior vena cava", "肾": "Kidney", "肝": "Liver", "肺": "Lung", "胰腺": "Pancreas",
    "门静脉": "Portal vein", "肺动脉": "Pulmonary artery", "肋骨": "Rib", "骶骨": "Sacrum",
    "肩胛骨": "Scapula", "小肠": "Small bowel", "脾": "Spleen", "胃": "Stomach", "气管": "Trachea",
    "膀胱": "Bladder", "颈椎": "Cervical vertebrae", "腰椎": "Lumbar vertebrae", "胸椎": "Thoracic vertebrae",
}


def fit_range(sz, roi):
    if sz >= roi:
        start = (sz - roi) // 2
        return start, start + roi
    return 0, sz


def get_center_window(img):
    d, h, w = img.shape[-3:]
    zs, ze = fit_range(d, ROI[0])
    ys, ye = fit_range(h, ROI[1])
    xs, xe = fit_range(w, ROI[2])
    window = img[:, zs:ze, ys:ye, xs:xe]
    pad = [0, ROI[2] - window.shape[-1], 0, ROI[1] - window.shape[-2], 0, ROI[0] - window.shape[-3]]
    window = F.pad(window, pad)
    return window


@torch.no_grad()
def run_visual_encoder(model, x):
    skips, segs = model.visual_encoder.UNet(x)

    scale1 = model.visual_encoder.proj1(skips[-1])
    scale2 = model.visual_encoder.proj2(skips[-2])
    scale3 = model.visual_encoder.proj3(skips[-3])

    shape1 = tuple(scale1.shape[-3:])
    shape2 = tuple(scale2.shape[-3:])
    shape3 = tuple(scale3.shape[-3:])

    pred_logit = segs[0]
    target_size = [pred_logit.shape[-3], pred_logit.shape[-2] * 2, pred_logit.shape[-1] * 2]
    pred_logit = F.interpolate(pred_logit, size=target_size, mode="trilinear", align_corners=False)
    seg_probs = torch.softmax(pred_logit, 1)
    pred_mask = seg_probs.argmax(1)  # (B, D, H, W) at "target_size" resolution (matches roi here)

    res_x1 = scale1.flatten(2).transpose(1, 2)
    res_x2 = scale2.flatten(2).transpose(1, 2)
    res_x3 = scale3.flatten(2).transpose(1, 2)

    organs = model.organs
    B = x.size(0)
    L1, L2, L3 = res_x1.shape[1], res_x2.shape[1], res_x3.shape[1]
    organ_token_flags1 = torch.zeros(B, len(organs), L1, dtype=torch.bool, device=x.device)
    organ_token_flags2 = torch.zeros(B, len(organs), L2, dtype=torch.bool, device=x.device)
    organ_token_flags3 = torch.zeros(B, len(organs), L3, dtype=torch.bool, device=x.device)

    y = pred_mask
    for i in range(B):
        unique_values = torch.unique(y[i])
        unique_values = unique_values[unique_values != 0]
        if unique_values.numel() == 0:
            continue
        masks = torch.stack([torch.eq(y[i], uv) for uv in unique_values]).float()
        h3 = F.max_pool3d(masks.unsqueeze(1), kernel_size=(2, 8, 8), stride=(2, 8, 8)).flatten(1) > 0
        h2 = F.max_pool3d(masks.unsqueeze(1), kernel_size=(4, 16, 16), stride=(4, 16, 16)).flatten(1) > 0
        h1 = F.max_pool3d(masks.unsqueeze(1), kernel_size=(8, 32, 32), stride=(8, 32, 32)).flatten(1) > 0
        organ_token_flags1[i][unique_values.long() - 1] = h1
        organ_token_flags2[i][unique_values.long() - 1] = h2
        organ_token_flags3[i][unique_values.long() - 1] = h3

    return {
        "seg_probs": seg_probs, "pred_mask": pred_mask,
        "res_x1": res_x1, "res_x2": res_x2, "res_x3": res_x3,
        "shape1": shape1, "shape2": shape2, "shape3": shape3,
        "flags1": organ_token_flags1, "flags2": organ_token_flags2, "flags3": organ_token_flags3,
    }


@torch.no_grad()
def organ_attention_map(model, enc_out, organ_id, roi_size=ROI):
    f1 = enc_out["flags1"][0, organ_id]
    f2 = enc_out["flags2"][0, organ_id]
    f3 = enc_out["flags3"][0, organ_id]
    if f1.sum() == 0 and f2.sum() == 0 and f3.sum() == 0:
        return None, None

    key1 = enc_out["res_x1"][0][f1].unsqueeze(0)
    key2 = enc_out["res_x2"][0][f2].unsqueeze(0)
    key3 = enc_out["res_x3"][0][f3].unsqueeze(0)
    key = value = torch.cat([key1, key2, key3], dim=1)
    query = model.query_tokens[organ_id].unsqueeze(0).unsqueeze(0)

    _, attn_weights = model.attention(query, key, value, need_weights=True)
    attn_weights = attn_weights[0, 0]  # (n1+n2+n3,)

    n1, n2, n3 = f1.sum().item(), f2.sum().item(), f3.sum().item()
    a1, a2, a3 = attn_weights[:n1], attn_weights[n1:n1 + n2], attn_weights[n1 + n2:]

    def scatter_grid(flag, attn, shape):
        grid = torch.zeros(shape[0] * shape[1] * shape[2], device=flag.device)
        grid[flag] = attn
        return grid.reshape(1, 1, *shape)

    g1 = scatter_grid(f1, a1, enc_out["shape1"])
    g2 = scatter_grid(f2, a2, enc_out["shape2"])
    g3 = scatter_grid(f3, a3, enc_out["shape3"])

    up1 = F.interpolate(g1, size=roi_size, mode="nearest")[0, 0]
    up2 = F.interpolate(g2, size=roi_size, mode="nearest")[0, 0]
    up3 = F.interpolate(g3, size=roi_size, mode="nearest")[0, 0]

    combined = up1 + up2 + up3
    return combined.cpu().numpy(), (up1.cpu().numpy(), up2.cpu().numpy(), up3.cpu().numpy())


def make_figure(patient_id, window_np, pred_mask_np, attn_map, organ_id, organ_name_cn, finding_label, finding_score, out_path):
    organ_mask = (pred_mask_np == organ_id + 1)
    if organ_mask.sum() == 0:
        z = window_np.shape[0] // 2
    else:
        zs = np.where(organ_mask.any(axis=(1, 2)))[0]
        # slice with the largest organ cross-section
        areas = organ_mask.sum(axis=(1, 2))
        z = int(np.argmax(areas))

    ct_slice = window_np[z]
    mask_slice = pred_mask_np[z]
    attn_slice = attn_map[z] if attn_map is not None else np.zeros_like(ct_slice)

    organ_en = ORGAN_EN.get(organ_name_cn, organ_name_cn)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"{patient_id[:8]}…  axial z={z}\nCT (windowed)")
    axes[0].axis("off")

    axes[1].imshow(ct_slice, cmap="gray")
    seg_show = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[1].imshow(seg_show, cmap="tab20", alpha=0.5, vmin=0, vmax=36)
    axes[1].contour(mask_slice == (organ_id + 1), colors="red", linewidths=1.2)
    axes[1].set_title(f"Segmentation overlay\n(red = {organ_en})")
    axes[1].axis("off")

    axes[2].imshow(ct_slice, cmap="gray")
    if attn_map is not None and attn_slice.max() > 0:
        norm = attn_slice / (attn_slice.max() + 1e-8)
        alpha = np.clip(norm * 0.9, 0, 0.9)
        heat = axes[2].imshow(norm, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        fig.colorbar(heat, ax=axes[2], fraction=0.04, pad=0.02, label="norm. attn weight")
    axes[2].contour(mask_slice == (organ_id + 1), colors="white", linewidths=0.8)
    axes[2].set_title(f"Cross-attention on {organ_en}\nquery token")
    axes[2].axis("off")

    fig.suptitle(f"{patient_id}\nTop finding: {finding_label}  (p={finding_score:.3f})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="For each patient, render the multi-organ segmentation and the "
                     "organ-query cross-attention map for their top-scoring finding, "
                     "as a sanity check on what the vision encoder has learned."
    )
    parser.add_argument("--img_dir", required=True, help="Folder of per-patient NIfTI files (see prepare_dicom_series.py).")
    parser.add_argument("--results_csv", required=True, help="CSV produced by inference_demo.py for the same img_dir.")
    parser.add_argument("--out_dir", required=True, help="Folder to save <patient_id>_attn.png into.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pad_func, model = initialize()
    ds = DataFolder(args.img_dir)
    df = pd.read_csv(args.results_csv)

    organs_list = model.organs

    for i in range(len(ds)):
        img, test_items, meta = ds[i]
        patient_id = meta["patient_id"]
        file_name = meta["file_name"]
        print(f"\n=== {file_name} ===")

        row = df[df["file_name"] == file_name]
        if len(row) == 0:
            print("  no result row found, skipping")
            continue
        row = row.iloc[0]
        scores = pd.to_numeric(row.drop("file_name"), errors="coerce").dropna()
        if len(scores) == 0:
            print("  no scored findings, skipping")
            continue
        top_col = scores.sort_values(ascending=False).index[0]
        top_score = scores[top_col]
        # column format: "organ_finding (Organ_Finding)"
        cn_part = top_col.split(" (")[0]
        organ_cn = cn_part.split("_")[0]
        finding_label = top_col.split(" (")[1].rstrip(")") if " (" in top_col else top_col

        if organ_cn not in organs_list:
            print(f"  top organ {organ_cn} not in organ list, skipping")
            continue
        organ_id = organs_list.index(organ_cn)

        window = get_center_window(img)
        x = window[None].cuda()

        enc_out = run_visual_encoder(model, x)
        pred_mask_np = enc_out["pred_mask"][0].cpu().numpy()
        window_np = window[0].cpu().numpy()

        present_organs = set((torch.unique(enc_out["pred_mask"][0]).long() - 1).tolist())
        if organ_id not in present_organs:
            areas = {oid: (pred_mask_np == oid + 1).sum() for oid in present_organs if oid >= 0}
            if areas:
                organ_id = max(areas, key=areas.get)
                organ_cn = organs_list[organ_id]
                print(f"  top organ not visible in sampled window; falling back to largest visible organ: {organ_cn}")
            else:
                print("  no organs visible in sampled window, skipping")
                continue

        attn_map, _ = organ_attention_map(model, enc_out, organ_id)

        out_path = os.path.join(args.out_dir, f"{patient_id}_attn.png")
        make_figure(patient_id, window_np, pred_mask_np, attn_map, organ_id, organ_cn, finding_label, top_score, out_path)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
